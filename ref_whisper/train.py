import os 

import torch 
import torch.nn 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pickle
from pathlib import Path
import pandas as pd
import ast

from transformers import WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import nlptutti as metrics
import torch.optim as optim
from tqdm.auto import tqdm 
import wandb
import argparse
import datetime
import time
import random

import ref_whisper.whisper.whisper as whisper
from ref_whisper.data_utils import MinyoDataset, custom_collate_fn, get_wer, get_wer_from_string_pairs
from ref_whisper.my_model import Mymodel
from ref_whisper.trainer import inference, testset_inference, validate, train, get_argument_parser, make_experiment_name_with_date


def main():
  args = get_argument_parser().parse_args()
  save_log = not args.debug
  if save_log:
    wandb.init(
      project="whisper-korean_folksong",  
      name = make_experiment_name_with_date(args), 
      config = args
    )
    save_dir = Path(wandb.run.dir) / 'checkpoints/'
  else:
    save_dir = Path('debug/')
  save_dir.mkdir(exist_ok=True)

  audio_path = '/home/daewoong/userdata/danbi/final_tts_audio'
  lyric_path = '/home/daewoong/userdata/danbi/final_lyrics_data/'
  each_lyric = '/home/daewoong/userdata/danbi/each_song_lyrics.txt'
  result_path = '/home/daewoong/userdata/danbi/encoder_result'
  filtered_id_list = pickle.load(open('/home/daewoong/userdata/danbi/thirty_second_filtered_id.pkl', 'rb'))
  
  audio_parent_pth = Path('/home/daewoong/userdata/test_set')
  test_dataset_mid = pd.read_csv('/home/daewoong/userdata/test_set/test_dataset_mid.csv')
  audio_path_list = list(audio_parent_pth.glob('*.wav'))
  audio_path_list = sorted(audio_path_list, key=lambda x: int(x.stem.split('_audio')[-1]))

  print('download token now')

  processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="ko", task="transcribe", predict_timestamps=False)
  processor.tokenizer.set_prefix_tokens(language="ko", task="transcribe", predict_timestamps=False)

  dataset = MinyoDataset(result_path, lyric_path, processor, filtered_id_list, each_lyric, max_len = args.n_ref_text_ctx, random_ratio=args.random_ratio)
  
  print('token download complete')

  train_size = int(len(dataset) * 0.9)
  valid_size = len(dataset) - train_size

  train_data, valid_data = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
  
  if args.debug:
    num_workers = 0
  else:
    num_workers = args.num_workers
  train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True)
  valid_dataloader = DataLoader(valid_data, batch_size=16, shuffle=False, collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True)

  #next(iter(train_dataloader))


  print('load model now')
  #pre_model = whisper.load_model("large-v2")
  pre_model = whisper.load_model("/home/daewoong/userdata/danbi/whisper_pretrain/large-v2.pt", device='cpu')  
  #WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
  print('load model complete')

  # pre_model.config.forced_decoder_ids = None
  # pre_model.config.suppress_tokens = []


  criterion = nn.CrossEntropyLoss(ignore_index = -100)

  device = args.device
  epoch = args.num_epochs
  model_dims = pre_model.dims
  model_dims.n_ref_encoder_layer = args.n_ref_encoder_layer
  model_dims.n_ref_decoder_layer = args.n_ref_decoder_layer
  model_dims.n_ref_text_ctx = args.n_ref_text_ctx
  model_dims.n_ref_text_state = args.n_ref_text_state
  model_dims.use_self_atn_in_ref_dec = args.use_self_atn_in_ref_dec
  model_dims.use_mlp_in_ref_dec = args.use_mlp_in_ref_dec
  model_dims.use_double_cross_atn_block = args.use_double_cross_atn_block
  model_dims.use_audio_first = args.use_audio_first
  
  if args.ref_encoder_type == 'transformer':
    model_dims.ref_encoder_type = 'transformer'
  elif args.ref_encoder_type == 'cnn':
    model_dims.ref_encoder_type = 'cnn'
  else:
    raise ValueError("ref_encoder_type should be transformer or cnn")

  model = Mymodel(model_dims)

  model.encoder.load_state_dict(pre_model.encoder.state_dict())

  model.decoder.token_embedding.load_state_dict(pre_model.decoder.token_embedding.state_dict())
  if model_dims.use_double_cross_atn_block:
    num_load_layers = model_dims.n_text_layer-model_dims.n_ref_decoder_layer
    model.decoder.blocks[:num_load_layers].load_state_dict(pre_model.decoder.blocks[:num_load_layers].state_dict())
    for i, block in enumerate(model.decoder.ref_blocks):
        block.load_state_dict_from_pretrained(pre_model.decoder.blocks[num_load_layers+i])
  else:
      model.decoder.blocks.load_state_dict(pre_model.decoder.blocks.state_dict())
  model.decoder.positional_embedding.data=pre_model.decoder.positional_embedding.data.clone()
  model.decoder.ln.load_state_dict(pre_model.decoder.ln.state_dict())

  model.ref_encoder.token_embedding.load_state_dict(pre_model.decoder.token_embedding.state_dict())
  # model.ref_encoder.token_embedding.weight.data[:-1] = pre_model.decoder.token_embedding.weight.data.clone()
  if len(pre_model.decoder.positional_embedding.data) > args.n_ref_text_ctx:
    model.ref_encoder.positional_embedding.data = pre_model.decoder.positional_embedding.data.clone()[:args.n_ref_text_ctx]
  else:
    model.ref_encoder.positional_embedding.data[:len(pre_model.decoder.positional_embedding.data)] = pre_model.decoder.positional_embedding.data.clone()
  
  # if args.ref_encoder_type == 'transformer':
  #   for i in range(model_dims.n_ref_encoder_layer):
  #     model.ref_encoder.blocks[i].attn.load_state_dict(pre_model.decoder.blocks[i].attn.state_dict())
  #     model.ref_encoder.blocks[i].attn_ln.load_state_dict(pre_model.decoder.blocks[i].attn_ln.state_dict())
  #     model.ref_encoder.blocks[i].mlp.load_state_dict(pre_model.decoder.blocks[i].mlp.state_dict())
  #     model.ref_encoder.blocks[i].mlp_ln.load_state_dict(pre_model.decoder.blocks[i].mlp_ln.state_dict())

    
  model = model.to(device)
  
  for param in model.encoder.parameters():
    param.requires_grad = False
  for param in model.decoder.blocks.parameters():
    param.requires_grad = False
    
  # for param in model.decoder.blocks[31].parameters():
  #   param.requires_grad = True
  # for param in model.decoder.blocks[30].parameters():
  #   param.requires_grad = True
    
  # optim_param_list = list(model.decoder.ref_blocks.parameters()) \
  #                  + list(model.ref_encoder.parameters())
  
  new_param_list = []       
  pretrained_param_list = list(model.decoder.ln.parameters()) \
                          + [param for block in model.decoder.ref_blocks for param in block.attn.parameters()] \
                          + [param for block in model.decoder.ref_blocks for param in block.attn_ln.parameters() ] \
                          + [param for block in model.decoder.ref_blocks for param in block.mlp.parameters() ] \
                          + [param for block in model.decoder.ref_blocks for param in block.mlp_ln.parameters()]
  if args.finetune_audio_cross_attn:
    pretrained_param_list += [param for block in model.decoder.ref_blocks for param in block.cross_attn.parameters()] 
    pretrained_param_list += [param for block in model.decoder.ref_blocks for param in block.cross_attn_ln.parameters()] 

  if args.ref_encoder_type == 'cnn':
    new_param_list += list(model.ref_encoder.blocks.parameters())
  if args.finetune_token:
    pretrained_param_list += list(model.decoder.token_embedding.parameters()) 
  
  if args.use_double_cross_atn_block:
    new_param_list += [param for block in model.decoder.ref_blocks for param in block.ref_cross_attn.parameters()]
    new_param_list += [param for block in model.decoder.ref_blocks for param in block.ref_cross_attn_ln.parameters()]
    
  else:
    new_param_list += list(model.decoder.final_ln.parameters())
    
    # num_load_layers = model_dims.n_text_layer-model_dims.n_ref_decoder_layer
    # model.decoder.blocks[:num_load_layers].load_state_dict(pre_model.decoder.blocks[:num_load_layers].state_dict())
    # for i, block in enumerate(model.decoder.ref_blocks):
    #     block.load_state_dict_from_pretrained(pre_model.decoder.blocks[num_load_layers+i])
    

  # if args.finetune_token:
  #   pretrained_param_list += list(model.decoder.token_embedding.parameters())

  #   pretrained_param_list += list(model.decoder.ln.parameters())      
  # else:
  #   pretrained_param_list += list(model.decoder.final_ln.parameters())

  # else: 
  # if args.finetune_token:
  #   pretrained_param_list += list(model.decoder.token_embedding.parameters())  
  
  # new_param_list = []
  # pretrained_param_list = []
  optimizer = AdamW([ {'params': pretrained_param_list, 'lr':args.pre_trained_lr}, {'params': new_param_list, 'lr':args.new_lr} ])
  scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.total_planend_steps)
  # optimizer = torch.optim.AdamW(optim_param_list, 
  #                               lr=args.lr)

  best_val_wer = 1.0
  step_i = 0
  
  for i in tqdm(range(epoch)):
    train_loss_record, step_i = train(model, train_dataloader, criterion, optimizer, scheduler, device, save_log, step_i)
    
    # for idx, batch in (enumerate(tqdm(train_dataloader, leave=False))):
    #   audio, input_text, input_txt_attn, around_text, around_txt_attn = batch
    #   x_input_text = input_text[:,:-1] #[batch, seq_len-1]
    #   true_input_text = input_text[:,4:] #[batch, seq_len-1]
      
    #   pred = model(audio.to(device), around_text.to(device), tokens=x_input_text.to(device))
    #   pred = pred[:, 3:]
    #   # pred = model(audio.to(device), input_text.to(device), tokens=x_input_text.to(device))
      
    #   true_input_text_with_mask = true_input_text.masked_fill(input_txt_attn[:, 4:].ne(1) , -100)
      
    #   #pred [batch, seq_len, ]
    #   loss = criterion(pred.reshape(-1, pred.size(-1)), true_input_text_with_mask.reshape(-1).to(device))
    #   if save_log:
    #     wandb.log({"train_loss": loss.item()}, step=step_i)
    #   # pred_list, target_list, wers_n, wers_s, wers_d, wers_i, cers_n, cers_s, cers_d, cers_i = get_wer(pred, true_input_text, processor)

    #   train_loss_record.append(loss.item())
    #   loss.backward()
    #   optimizer.step()
    #   optimizer.zero_grad()
    #   step_i += 1
    inf_texts, inf_targets = testset_inference(model, processor, audio_path_list, test_dataset_mid, args.num_test_samples)
    wer, cer = get_wer_from_string_pairs(inf_texts, inf_targets)
    result_table = wandb.Table(columns=["target", "pred"], data=list(zip(inf_targets, inf_texts)))
    if save_log:
      wandb.log({"Inference result": result_table,
                 "Test WER":  wer,
                 "Test CER": cer
                 }, step=step_i)
      
    valid_loss, mean_valid_acc = validate(model, processor, valid_dataloader, criterion, device)
    print('valid_loss', valid_loss, 'valid_acc', mean_valid_acc)
    if save_log:
      wandb.log({"valid_loss": valid_loss,
                 "valid_acc": mean_valid_acc},
                step=step_i)

    
    if (i+1) % 3 == 0:
        torch.save({'model':model.state_dict(), 'optim':optimizer.state_dict(), 'dims': model.dims}, save_dir / f'epoch_{i}.pt')

  wandb.finish()  
if __name__ == '__main__':
  main()