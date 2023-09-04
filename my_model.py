import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from whisper.whisper.decoding import decode as decode_function
from whisper.whisper.decoding import detect_language as detect_language_function
from whisper.whisper.transcribe import transcribe as transcribe_function


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_ref_text_state: int
    n_ref_text_ctx: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int
    n_ref_encoder_layer: int
    n_ref_decoder_layer: int
    ref_encoder_type: str
    use_self_atn_in_ref_dec: bool 
    use_mlp_in_ref_dec: bool
    use_double_cross_atn_block: bool
    use_audio_first: bool
    


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        apply_cross_mask: bool = False,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask, apply_cross_mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, apply_cross_mask: bool = False
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25 # scale = sqrt(1 / n_state)
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale # (n_batch, n_head, n_ctx, n_state // n_head)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale # (n_batch, n_head, n_state // n_head, n_ctx)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) # 

        qk = q @ k
        if mask is not None:
            if apply_cross_mask:
                adding_mask = torch.zeros_like(mask, dtype=torch.float32)
                adding_mask.masked_fill_(mask==0, -torch.inf)
                qk = qk + adding_mask[:, None, None, :]
            else:
                if mask.shape[0] == qk.shape[0]:
                    qk = qk + mask[:, None, None, :n_ctx]
                else:
                    qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()



class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)
        
    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None, # cross attention에서 사용되는 인코딩된 오디오 특성 텐서(크로스 어텐션이 아닐 때는 None)
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0] #residual connection
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x
      
class ResidualAttentionCrossMaskBlock(ResidualAttentionBlock):
  def __init__(self, n_state: int, n_head: int, self_attention: bool=True, use_mlp: bool=True) -> None:
      super().__init__(n_state, n_head, cross_attention=True)
      self.self_attention = self_attention
      self.use_mlp = use_mlp
      if not self_attention:
        del self.attn, self.attn_ln
      if not use_mlp:
        del self.mlp, self.mlp_ln

  def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        self_mask: Optional[Tensor] = None,
        cross_mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        if self.self_attention:
            x = x + self.attn(self.attn_ln(x), mask=self_mask, kv_cache=kv_cache)[0] #residual connection
        x = x + self.cross_attn(self.cross_attn_ln(x), xa, mask=cross_mask, kv_cache=kv_cache, apply_cross_mask=True)[0]
        if self.use_mlp:
            x = x + self.mlp(self.mlp_ln(x))
        return x
      
class DoubleCrossAttentionBlock(ResidualAttentionBlock):
    def __init__(self, n_state: int, n_head: int, use_audio_first: bool = True):
        super().__init__(n_state, n_head, cross_attention=True)

        self.ref_cross_attn = MultiHeadAttention(n_state, n_head)
        self.ref_cross_attn_ln = LayerNorm(n_state)
        self.use_audio_first = use_audio_first
        
    def load_state_dict_from_pretrained(self, ablock):
        copy_keys = ["attn", "attn_ln", "cross_attn", "cross_attn_ln", "mlp", "mlp_ln"]
        for key in copy_keys:
            getattr(self, key).load_state_dict(getattr(ablock, key).state_dict())

    def forward(
        self,
        x: Tensor,
        xa: Tensor, # cross attention에서 사용되는 인코딩된 오디오 특성 텐서(크로스 어텐션이 아닐 때는 None)
        x_ref: Tensor,
        mask: Optional[Tensor] = None,
        ref_mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0] #residual connection
        if self.use_audio_first:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
            x = x + self.ref_cross_attn(self.ref_cross_attn_ln(x), x_ref, mask=ref_mask, kv_cache=kv_cache, apply_cross_mask=True)[0]
        else:
            x = x + self.ref_cross_attn(self.ref_cross_attn_ln(x), x_ref, mask=ref_mask, kv_cache=kv_cache, apply_cross_mask=True)[0]
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1) # (batch_size, n_ctx, n_state)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x
    
class RefTextEncoderOld(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_in_state:int, n_state: int, n_head: int, n_layer: int
    ): # n_in_state = 1024, n_state = 240
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_in_state)
        self.positional_embedding = nn.Parameter(torch.randn(n_ctx, n_in_state))
        self.compressor = nn.Linear(n_in_state, n_state)
        self.expander = nn.Linear(n_state, n_in_state)        
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_in_state)

    def forward(self, x: Tensor): #, mask: Optional[Tensor] = None):
        """
    
        """
        x = self.token_embedding(x)
        x = (x + self.positional_embedding).to(x.dtype)
        x = self.compressor(x) 
        for block in self.blocks:
            x = block(x)
        
        x = self.expander(x)
        x = self.ln_post(x)
        return x
    
    
class RefTextEncoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ): # n_in_state = 1024, n_state = 240
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.randn(n_ctx, n_state))
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        """
    
        """
        x = self.token_embedding(x)
        x = (x + self.positional_embedding).to(x.dtype)
        adding_mask = None
        if mask is not None:
            adding_mask = torch.zeros_like(mask, dtype=torch.float32)
            adding_mask.masked_fill_(mask==0, -torch.inf)
        for block in self.blocks:
            x = block(x, mask=adding_mask)
        x = self.ln_post(x)
        return x

class CNNBlock(nn.Module):
    def __init__(self, n_state) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(n_state, n_state, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(n_state, n_state, kernel_size=3, padding=1),
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        x_out = self.layers(x)
        x = x + x_out
        return self.gelu(x)
        
    
class RefTextEncoderCNN(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_state, n_head, n_layer):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.randn(n_ctx, n_state))
        self.blocks = nn.ModuleList(
            [CNNBlock(n_state) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        # print("337: RefTextEncoderCNN called!!")
        x = self.token_embedding(x)
        x = (x + self.positional_embedding).to(x.dtype)
        x = x.permute(0, 2, 1) # N x C x T
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1) # N x T x C
        x = self.ln_post(x)
        return x



class TextRefDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, 
        n_ref_layer: int, 
        ref_self: bool=True,
        ref_mlp: bool=True,
        use_double_cross_atn_block: bool=False,
        use_audio_first: bool=True,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        self.use_double_cross_atn_block = use_double_cross_atn_block
        self.use_audio_first = use_audio_first

        if use_double_cross_atn_block:
          n_layer = n_layer - n_ref_layer
          
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        
        if use_double_cross_atn_block:
            self.ref_blocks: Iterable[DoubleCrossAttentionBlock] = nn.ModuleList(
                  [
                    DoubleCrossAttentionBlock(n_state, n_head, use_audio_first=use_audio_first)
                    for _ in range(n_ref_layer)
                ]
            )    
        else:
            self.ref_blocks: Iterable[ResidualAttentionCrossMaskBlock] = nn.ModuleList(
                [
                    ResidualAttentionCrossMaskBlock(n_state, n_head, ref_self, ref_mlp)
                    for _ in range(n_ref_layer)
                ]
            ) 
            self.final_ln = LayerNorm(n_state)
   
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, xb: Tensor, ref_mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, audio_ctx, n_state) 
            the encoded audio features to be attended on
        xb : torch.Tensor, shape = (batch_size, n_ctx, n_state)
            the encoded lyrics text features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
        # mask = self.mask * ref_text_mask # TODO
        xb = xb.to(x.dtype)

        if self.use_double_cross_atn_block:
            for block in self.ref_blocks:
                x = block(x, xa, xb, mask=self.mask, ref_mask=ref_mask, kv_cache=kv_cache)
            x = self.ln(x)
        else:  
            x = self.ln(x)
            for block in self.ref_blocks:
                x = block(x, xb, self_mask=self.mask, cross_mask=ref_mask, kv_cache=kv_cache)
            x = self.final_ln(x)
            
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits
    
    def get_logit_without_ref(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        return logits 

class Mymodel(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        # self.decoder = TextDecoder(
        #     self.dims.n_vocab,
        #     self.dims.n_text_ctx,
        #     self.dims.n_text_state,
        #     self.dims.n_text_head,
        #     self.dims.n_text_layer,
        # )
        if dims.ref_encoder_type == 'transformer':
            ref_encoder_class = RefTextEncoder
        elif dims.ref_encoder_type == 'cnn':
            ref_encoder_class = RefTextEncoderCNN
        self.ref_encoder = ref_encoder_class(
            self.dims.n_vocab, # add new pad token
            self.dims.n_ref_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_ref_encoder_layer,
        )
        
        self.decoder = TextRefDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            self.dims.n_ref_decoder_layer,
            ref_self=self.dims.use_self_atn_in_ref_dec,
            ref_mlp=self.dims.use_mlp_in_ref_dec,
            use_double_cross_atn_block=self.dims.use_double_cross_atn_block,
            use_audio_first=self.dims.use_audio_first,
        )
        
        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor, ref_features: torch.Tensor):
        ref_features = self.ref_encoder(ref_features)
        return self.decoder(tokens, audio_features, ref_features)

    def forward(
        self, mel: torch.Tensor, ref_text: torch.Tensor, tokens: torch.Tensor, ref_mask: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # audio_features = self.encoder(mel)
        # ref_features = self.ref_encoder(ref_text, mask=ref_mask)
        # logits = self.decoder(tokens, audio_features, ref_features, ref_text_mask=ref_mask)
        # return logits
        if mel.ndim == 3 and mel[0].shape == torch.Size([1500, 1280]): ## mel is audio features
            audio_input = mel
        else:
            audio_input = self.encoder(mel)
        ref_features = self.ref_encoder(ref_text, ref_mask)
    
        return self.decoder(tokens, audio_input, ref_features, ref_mask)
    
    def get_pretrained_only_result(self, mel, tokens):
        if mel.ndim == 3 and mel[0].shape == torch.Size([1500, 1280]): ## mel is audio features
            audio_input = mel
        else:
            audio_input = self.encoder(mel)
        return self.decoder.get_logit_without_ref(tokens, audio_input)


    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function




# class reference_encoder(nn.Module):
#   def __init__(self, model, num_layers = 2, max_len = 1024, dim = 1280 ):
#     super().__init__()
#     self.layers = nn.ModuleList([reference_encoder_block(model) for _ in range(num_layers)])
#     self.final_layer_norm = nn.LayerNorm(model.config.d_model)
#     self.input_emb = model.get_input_embeddings()
#     # self.pos_emb = model.get_decoder().embed_positions
#     self.pos_emb = torch.nn.Parameter(torch.randn(max_len, dim))
    
#   def forward(self, hidden_states, key_padding_mask = None):
#     hidden_states = self.input_emb(hidden_states) + self.pos_emb
    
#     for layer in self.layers:
#       hidden_states, attn_weight = layer(hidden_states, key_padding_mask = key_padding_mask )
#     hidden_states = self.final_layer_norm(hidden_states)
    
#     return hidden_states





# class TextDecoder(nn.Module):
#     def __init__(
#         self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
#     ):
#         super().__init__()

#         self.token_embedding = nn.Embedding(n_vocab, n_state)
#         self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

#         self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
#             [
#                 ResidualAttentionBlock(n_state, n_head, cross_attention=True)
#                 for _ in range(n_layer)
#             ]
#         )
#         self.ln = LayerNorm(n_state)

#         mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
#         self.register_buffer("mask", mask, persistent=False)

#     def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
#         """
#         x : torch.LongTensor, shape = (batch_size, <= n_ctx)
#             the text tokens
#         xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
#             the encoded audio features to be attended on
#         """
#         offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
#         x = (
#             self.token_embedding(x)
#             + self.positional_embedding[offset : offset + x.shape[-1]]
#         )
#         x = x.to(xa.dtype)

#         for block in self.blocks:
#             x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

#         x = self.ln(x)
#         logits = (
#             x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
#         ).float()
#         return logits



# class Reference_encoder_block(nn.Module):
#   def __init__(self, model, n_state=128): #hidden_size, num_heads
#     super().__init__()
    
#     # whisper config 
#     # self.d_model = model.config.d_model
#     # self.num_heads = model.config.decoder_attention_heads
    
#     #self.input_emb = model.get_input_embeddings()
#     # self.pos_emb = model.get_decoder().embed_positions
#     self.n_state = n_state
#     self.self_attn = MultiHeadAttention(self.d_model, self.num_heads, dropout=0.0, batch_first=True)
#     self.input_layer_norm = nn.LayerNorm(self.n_state)
#     self.attn_layer_norm = nn.LayerNorm(self.n_state)
    
#     self.mlp = nn.Sequential(
#         nn.Linear(n_state, n_state * 4),  
#         nn.GELU(), 
#         nn.Linear(n_state * 4, n_state)
#     )
    
#   def forward(self, hidden_states, key_padding_mask = None):
#     residual = hidden_states
#     hidden_states = self.input_layer_norm(hidden_states)
    
#     if key_padding_mask is not None:
#       hidden_states, attn_weight = self.self_attn(hidden_states, hidden_states, hidden_states, key_padding_mask = (key_padding_mask == 0))
#     else:
#       hidden_states, attn_weight = self.self_attn(hidden_states,hidden_states,hidden_states)  
    
#     hidden_states = hidden_states + residual 
    
#     residual = hidden_states
#     hidden_states = self.attn_layer_norm(hidden_states)
    
#     hidden_states = self.mlp(hidden_states) 
#     hidden_states = hidden_states + residual
#     return hidden_states, attn_weight

    # x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0] #residual connection
    #     if self.cross_attn:
    #         x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
    #     x = x + self.mlp(self.mlp_ln(x))
    #     return x

