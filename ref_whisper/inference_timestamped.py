#!/usr/bin/env python3

__author__ = "Jérôme Louradour"
__credits__ = ["Jérôme Louradour"]
__license__ = "GPLv3"
__version__ = "1.12.20"

# Set some environment variables
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Remove warning "This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)..."
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # GPU in the right order
from pathlib import Path

# Whisper and Torch
from .whisper import whisper
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from .my_model import Mymodel, ModelDimensions
from .whisper.whisper.tokenizer import RefProcessor
from .whisper.whisper.audio import load_audio, N_SAMPLES
from .whisper.whisper import inference_hack as hack

# For alignment
import numpy as np
import dtw
# from scipy.signal import medfilt as median_filter
from scipy.ndimage import median_filter # faster owing to https://github.com/openai/whisper/commit/f0083e7eb20d032390e42f6f6039947fa8669c93
from scipy.signal import find_peaks

# Additional
import string
import csv
import sys
import gzip, base64
import copy
import re

# Constant variables
from .whisper.whisper.utils import format_timestamp
from .whisper.whisper.audio import N_FRAMES, HOP_LENGTH, SAMPLE_RATE  # 3000, 160, 16000
AUDIO_SAMPLES_PER_TOKEN = HOP_LENGTH * 2                     # 320
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / SAMPLE_RATE # 0.02 (sec)
SEGMENT_DURATION = N_FRAMES * HOP_LENGTH / SAMPLE_RATE       # 30.0 (sec)

# Logs
import logging
logger = logging.getLogger("whisper_timestamped")

USE_EFFICIENT_BY_DEFAULT = True
TRUST_WHISPER_TIMESTAMP_BY_DEFAULT = True
DISFLUENCY_MARK = "[*]"

try:
    whisper_version = whisper.__version__
except NameError:
    whisper_version = ""
WHIPSER_GE_20230306 = whisper_version >= "20230306"
WHIPSER_GE_20230308 = whisper_version >= "20230308"


import pickle

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'whisper':
            return getattr(whisper, name)
        return super().find_class(module, name)


def transcribe_timestamped(
    # Main Whisper options
    model,
    audio,
    txt_path:str,
    language=None,
    task="transcribe",

    # Additional options for word alignment
    remove_punctuation_from_words=False,
    compute_word_confidence=True,
    include_punctuation_in_confidence=False,
    refine_whisper_precision=0.5,
    min_word_duration=0.02, # Was 0.04 before 1.11
    plot_word_alignment=False,
    word_alignement_most_top_layers=None, # Was 6 before 1.9
    remove_empty_words=False,

    # Reproducibility
    seed=1234,

    vad=False,
    detect_disfluencies=False,
    trust_whisper_timestamps=TRUST_WHISPER_TIMESTAMP_BY_DEFAULT,
    naive_approach=False,

    # Other Whisper options
    temperature=0.0 if USE_EFFICIENT_BY_DEFAULT else (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    best_of=None,
    beam_size=None,
    patience=None,
    length_penalty=None,
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
    fp16=None,
    condition_on_previous_text=True,
    initial_prompt=None,
    suppress_tokens="-1",
    sample_len=None,
    verbose=False,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance.

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform in 16kHz.

    language: str
        The language to use for the transcription. If None, the language is detected automatically.

    task: str
        The task to perform: either "transcribe" or "translate".

    remove_punctuation_from_words: bool
        If False, words will be glued with the next punctuation mark (if any).
        If True, there will be no punctuation mark in the `words[:]["text"]` list.
        It only affects these strings; This has no influence on the computation of the word confidence, whatever the value of `include_punctuation_in_confidence` is.
    
    include_punctuation_in_confidence: bool
        Whether to include proba of punctuation in the computation of the (previous) word confidence.

    compute_word_confidence: bool
        Whether to compute word confidence.
        If True, a finer confidence for each segment will be computed as well.

    vad: bool
        Whether to perform voice activity detection (VAD) on the audio file, to remove silent parts before transcribing with Whisper model.
        This should decrease hallucinations from the Whisper model.

    detect_disfluencies: bool
        Whether to detect disfluencies (i.e. hesitations, filler words, repetitions, corrections, etc.) that Whisper model might have omitted in the transcription.
        This should make the word timestamp prediction more accurate.
        And probable disfluencies will be marked as special words "[*]".

    trust_whisper_timestamps: bool
        Whether to rely on Whisper's timestamps to get approximative first estimate of segment positions (up to refine_whisper_precision).

    refine_whisper_precision: float
        How much can we refine Whisper segment positions, in seconds. Must be a multiple of 0.02.

    min_word_duration: float
        Minimum duration of a word, in seconds. If a word is shorter than this, timestamps will be adjusted.

    plot_word_alignment: bool
        Whether to plot the word alignment for each segment. matplotlib must be installed to use this option.

    remove_empty_words: bool
        Whether to remove words with no duration occuring at the end of segments (probable Whisper hallucinations).

    seed: int
        Random seed to use for temperature sampling, for the sake of reproducibility.
        Choose None for unpredictable randomness.

    naive_approach: bool
        Force the naive approach that consists in decoding twice the audio file, once to get the transcritpion and once with the decoded tokens to get the alignment.
        Note that this approach is used anyway when beam_size is not None and/or when the temperature is a list with more than one element.

    temperature: float
        Temperature for sampling.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed.

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed.

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent.

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    initial_prompt: str
        Optional text to provide as a prompt for the first window.

    suppress_tokens: str
        Comma-separated list of token ids to suppress during sampling;
        '-1' will suppress most special characters except common punctuations.

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Check input options
    assert refine_whisper_precision >= 0 and refine_whisper_precision / AUDIO_TIME_PER_TOKEN == round(refine_whisper_precision / AUDIO_TIME_PER_TOKEN), f"refine_whisper_precision must be a positive multiple of {AUDIO_TIME_PER_TOKEN}"
    refine_whisper_precision_nframes = round(refine_whisper_precision / AUDIO_TIME_PER_TOKEN)
    assert min_word_duration >= 0, f"min_word_duration must be a positive number"
    assert word_alignement_most_top_layers is None or word_alignement_most_top_layers > 0, f"word_alignement_most_top_layers must be a strictly positive number"

    if isinstance(temperature, (list, tuple)) and len(temperature) == 1:
        temperature = temperature[0]
    if isinstance(temperature, (list, tuple)):
        # temperature fallback
        naive_approach = True
    elif temperature > 0 and best_of is not None and best_of > 1:
        naive_approach = True
    if beam_size is not None:
        # beam-search
        naive_approach = True

    # Input options
    if fp16 is None:
        fp16 = model.device != torch.device("cpu")

    # Safety check
    input_stride = N_FRAMES // model.dims.n_audio_ctx
    time_precision = input_stride * HOP_LENGTH / SAMPLE_RATE
    assert time_precision == AUDIO_TIME_PER_TOKEN

    alignment_options = dict(
            remove_punctuation_from_words=remove_punctuation_from_words,
            compute_word_confidence=compute_word_confidence,
            include_punctuation_in_confidence=include_punctuation_in_confidence,
            detect_disfluencies=detect_disfluencies,
            refine_whisper_precision_nframes=refine_whisper_precision_nframes,
            plot_word_alignment=plot_word_alignment,
            word_alignement_most_top_layers=word_alignement_most_top_layers,
            alignment_heads=get_alignment_heads(model) if word_alignement_most_top_layers is None else None,
    )
    whisper_options = dict(
            language=language,
            task=task,
            fp16=fp16,
            temperature=temperature,
            best_of=best_of,
            beam_size=beam_size,
            patience=patience,
            length_penalty=length_penalty,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            suppress_tokens=suppress_tokens,
            sample_len=sample_len,
            verbose=verbose if (not vad or verbose is not True) else False,
    )
    other_options = dict(
        no_speech_threshold=no_speech_threshold,
        logprob_threshold=logprob_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
    )

    if vad:
        audio = get_audio_tensor(audio)
        audio, convert_timestamps = remove_non_speech(audio, plot=plot_word_alignment)
        

    global num_alignment_for_plot
    num_alignment_for_plot = 0

    if naive_approach:
        (transcription, words, ref_align_segments) = _transcribe_timestamped_naive(model, audio, txt_path, 
                                                               min_word_duration=0.0, # Was 0.04 before 1.11
                                                               trust_whisper_timestamps=trust_whisper_timestamps,
                                                               **alignment_options, **whisper_options, **other_options)
    else:
        (transcription, words, ref_align_segments) = _transcribe_timestamped_efficient(model, audio, txt_path, 
                                                                   trust_whisper_timestamps=trust_whisper_timestamps,
                                                                   **alignment_options, **whisper_options, **other_options)
    if remove_empty_words:
        # Remove words with empty duration happening at the end of segments, to remove some hallucinations
        transcription, words = remove_last_null_duration_words(transcription, words, recompute_text=True)

    # Refine word positions
    ensure_increasing_positions(words, min_duration=min_word_duration if trust_whisper_timestamps else 0)
    
    # Combine words and segments
    whisper_segments = transcription["segments"]
    for word in words:
        if verbose and not naive_approach and not vad:
            print_timestamped(word)
        word.pop("tokens")
        word.pop("tokens_indices")
        if "avg_logprob_reliable" in word:
            word.pop("avg_logprob_reliable")
        idx_segment = word.pop("idx_segment")
        assert idx_segment < len(whisper_segments), f"Fatal error: Got unexpected segment index {idx_segment} >= {len(whisper_segments)}"
        segment = whisper_segments[idx_segment]
        if "words" in segment:
            segment["words"].append(word)
        else:
            segment["words"] = [word]
            if refine_whisper_precision:
                segment["start"] = word["start"]
        if refine_whisper_precision:
            segment["end"] = word["end"]

    if vad:
        # Recompute timestamps to match the original audio
        for segment in whisper_segments:
            for word in segment.get("words", []):
                word["start"], word["end"] = convert_timestamps(word["start"], word["end"])
                if verbose:
                    print_timestamped(word)
            if refine_whisper_precision and len(segment.get("words", [])):
                segment["start"] = segment["words"][0]["start"]
                segment["end"] = segment["words"][-1]["end"]
            else:
                segment["start"], segment["end"] = convert_timestamps(segment["start"], segment["end"])
    transcription['ref_align_segments'] = ref_align_segments
    return transcription

def _transcribe_timestamped_efficient(
    model,
    audio,
    text,
    remove_punctuation_from_words,
    compute_word_confidence,
    include_punctuation_in_confidence,
    refine_whisper_precision_nframes,
    alignment_heads,
    plot_word_alignment,
    word_alignement_most_top_layers,
    detect_disfluencies,
    trust_whisper_timestamps,
    use_timestamps_for_alignment = True,
    # Whisper specific options
    **whisper_options,
):

    # Get options
    sample_len = whisper_options["sample_len"]
    temperature = whisper_options["temperature"]
    no_speech_threshold = whisper_options["no_speech_threshold"]
    logprob_threshold = whisper_options["logprob_threshold"]
    verbose = whisper_options["verbose"]
    # Note: "on-the-fly" verbose is not implementable in the current state (we don't know the absolute position of the current chunk). See issue #18
    verbose_bugged = False
    whisper_options["verbose"] = None if whisper_options["verbose"] is True else whisper_options["verbose"]  # We will print intermediate results ourselves

    logit_filters = get_logit_filters(model, whisper_options)
    language = whisper_options["language"]
    
    ref_text_processor = RefProcessor(model.dims.n_ref_text_ctx, model.is_multilingual, task=whisper_options["task"], language=language)
    tokenizer = ref_text_processor.tokenizer
    num_mel_segs = -(-load_audio(audio).shape[0] // N_SAMPLES)
    ref_input_tokens = []
    ref_batch = ref_text_processor(text, num_mel_segments=num_mel_segs)
    ref_text_seg_offset = ref_text_processor.get_segment_offset(text, num_segments=num_mel_segs)
    text_sliced = [tokenizer.decode(ref_batch[i].tolist()) for i in range(len(ref_batch))]
    
    # len_ref_text = (ref_text_processor(text) != ref_text_processor.pad_idx).sum().item() + 1
    # tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, task=whisper_options["task"], language=language)

    max_sample_len = sample_len or model.dims.n_text_ctx // 2 
    n_ctx = model.dims.n_text_ctx

    debug = logger.getEffectiveLevel() >= logging.DEBUG

    word_alignement_most_top_layers = float("inf") if word_alignement_most_top_layers is None else word_alignement_most_top_layers

    # The main outcome
    timestamped_word_segments = []  # list of timestamped word segments that have been collected so far
    ref_align_segments = []
    # Main variables to be accumulated
    segment_tokens = [[]]              # list of lists of token indices that have been collected so far (one list per segment)
    segment_attweights = [[] for _ in range(min(word_alignement_most_top_layers, len(model.decoder.blocks) + len(model.decoder.ref_blocks)))]
                                    # attention weights on the last segments
    ref_segment_attweights = [[] for _ in range(min(word_alignement_most_top_layers, len(model.decoder.ref_blocks)))]
    segment_avglogprobs = []        # average log probability for each segment (actually of the corresponding chunk, as computed by whisper)
    segment_logprobs = []           # token log probabilities for each segment
    # Variables related to options that can skip some segments
    sot_index = None                # index of the SOT token in the current set of processed tokens
    no_speech_prob = None           # no speech probability for the current 30 sec chunk
    chunk_logprobs = []             # log probabilities for the current 30 sec chunk
    chunk_tokens = []               # tokens for the current 30 sec chunk (list of Torch tensors)
    chunk_tokens_nosot = []         # tokens for the current 30 sec chunk, without the SOT tokens (list of indices)
    last_chunk_token = None         # last token of the current chunk, that may be needed for corner cases
    last_token_fallback = None      # last token to use as a fallback if the model gets stuck
    has_started = False             # whether we have started decoding
    mfcc = None                     # MFCC features for the current 30 sec chunk
    new_mfcc = None                 #
    num_inference_steps = 0         # number of inference steps performed so far (for debugging only)
    num_alignment_called = 0
    prev_segment_idx = 0

    def is_sot(curr_tokens):
        return curr_tokens is None or len(curr_tokens) > 1 or curr_tokens[0] == tokenizer.sot
    
    def has_reached_decoding_limit():
        n = len(chunk_tokens_nosot) + 1
        m = n + (len(chunk_tokens[0]) if len(chunk_tokens) > 0 else 0)
        return n + 1 >= max_sample_len or m > n_ctx

    def reset(add_segment, keep_last_token=True):
        """ Reset the list of tokens for the current speech segment, and corresponding cross-attention weights """
        nonlocal segment_tokens, segment_attweights, ref_segment_attweights
        if add_segment:
            if keep_last_token:
                segment_tokens.append([segment_tokens[-1][-1]])
                segment_attweights = [w[-1:] for w in segment_attweights]
                ref_segment_attweights = [w[-1:] for w in ref_segment_attweights]
            else:
                segment_tokens.append([])
                segment_attweights = [[] for w in segment_attweights]
                ref_segment_attweights = [[] for w in ref_segment_attweights]
            segment_tokens[-2].pop(0)
        elif len(segment_tokens[-1]) > 0:
            if debug:
                logger.debug(f"Reset last segment: {tokenizer.decode_with_timestamps(segment_tokens[-1])}")
            segment_tokens[-1] = []
            segment_attweights = [[] for w in segment_attweights]
            ref_segment_attweights = [[] for w in ref_segment_attweights]

    saw_consecutive_timestamps = False
    def must_flush_segment(curr_tokens):
        """ Return whether or not the previously collected tokens must be used to add a new speech segment """
        nonlocal segment_tokens, saw_consecutive_timestamps, chunk_tokens_nosot

        if not is_sot(curr_tokens):
            is_timestamp = curr_tokens[0] >= tokenizer.timestamp_begin
            is_previous_timestamp = segment_tokens[-1][-1] >= tokenizer.timestamp_begin if len(segment_tokens[-1]) > 0 else False
            consecutive_timestamps = is_timestamp and is_previous_timestamp
            if consecutive_timestamps:
                saw_consecutive_timestamps = True
            return consecutive_timestamps
        else: # Several tokens as a prompt or must flush last segments

            must_flush = len(segment_tokens[-1]) > 1 and not saw_consecutive_timestamps
            if not must_flush and WHIPSER_GE_20230306: # If the last token is a timestamp, the last segment is used
                if last_chunk_token is None:
                    must_flush = (len(segment_tokens[-1]) > 2 and segment_tokens[-1][-1] >= tokenizer.timestamp_begin)
                else:
                    must_flush = (last_chunk_token >= tokenizer.timestamp_begin)
            if not must_flush and trust_whisper_timestamps:
                # Discard the end of the last transcription
                reset(False)
            saw_consecutive_timestamps = False
            return must_flush

    index_begin_30sec_chunck = 0
    def get_index_begin_30sec_chunck(curr_tokens):
        nonlocal index_begin_30sec_chunck, has_started

        if is_sot(curr_tokens) and has_started:
            if trust_whisper_timestamps:
                res = index_begin_30sec_chunck
                index_begin_30sec_chunck = len(segment_tokens)-1
            else:
                res = len(segment_tokens)-1
            return res

    def align_last_segment(curr_tokens=None):
        nonlocal segment_tokens, segment_attweights, ref_segment_attweights, timestamped_word_segments, has_started, no_speech_prob, chunk_tokens, chunk_tokens_nosot, chunk_logprobs, mfcc, new_mfcc, logit_filters, index_begin_30sec_chunck, last_token_fallback, num_inference_steps, ref_batch, ref_input_tokens, num_alignment_called, prev_segment_idx

        if debug and trust_whisper_timestamps:
            logger.debug(f"Add segment {len(timestamped_word_segments)+1} at step {num_inference_steps}:\n\t{tokenizer.decode_with_timestamps(segment_tokens[-1])}")

        tokens = segment_tokens[-1][1:]

        # When the decoding hit the max limit (number of tokens) -- usually when the language model gets stuck --
        # then we have to recover the last token from what is send to the decoder
        unfinished_decoding = has_reached_decoding_limit()
        last_is_not_timestamp = len(tokens) and tokens[-1] < tokenizer.timestamp_begin
        last_token_reliable = True

        if unfinished_decoding:
            logger.debug(f"WARNING: decoding hit the max limit for segment {segment_tokens[-1]} (It usually happens when the language model gets stuck)")
            # The last token chosen is in the prompt for the new chunk
            if curr_tokens is not None and curr_tokens[0] == tokenizer.sot_prev:
                index_sot =  (curr_tokens == tokenizer.sot).nonzero(as_tuple=True)
                assert len(index_sot) == 1
                index_sot = index_sot[0].item()
                assert index_sot > 0 
                last_token_fallback = curr_tokens[index_sot-1].item()
                logger.debug(f"         Guessed last token from the prompt for the new chunk: {last_token_fallback}")
            # Fallback for the last segment, or without prompt: Assume greedy decoding
            else:
                last_token_fallback = torch.argmax(chunk_logprobs[-1]).item() if last_chunk_token is None else last_chunk_token
                last_token_reliable = (temperature == 0)
                logger.debug(f"         Guess last token using probas (assuming greedy decoding): {last_token_fallback}")
            if debug:
                logger.debug(f"WARNING: also add last token: {tokenizer.decode_with_timestamps([last_token_fallback])}")

            tokens.append(last_token_fallback)
            segment_tokens[-1].append(last_token_fallback)
            attention_weights = [torch.cat(w, dim=-2) for w in segment_attweights]
            ref_attention_weights = [torch.cat(w, dim=-2) for w in ref_segment_attweights]
            last_logprobs = chunk_logprobs[-1]
        elif last_is_not_timestamp: # <eot> was emitted early, without a timestamp before
            logger.debug(f"WARNING: end timestamp not produced. Adding <|endoftext|>")
            tokens.append(tokenizer.eot)
            segment_tokens[-1].append(tokenizer.eot)
            attention_weights = [torch.cat(w, dim=-2) for w in segment_attweights]
            ref_attention_weights = [torch.cat(w, dim=-2) for w in ref_segment_attweights]
            last_logprobs = chunk_logprobs[-1]
        else:
            attention_weights = [torch.cat(w[:-1], dim=-2) for w in segment_attweights]
            ref_attention_weights = [torch.cat(w[:-1], dim=-2) for w in ref_segment_attweights]
            last_logprobs = chunk_logprobs[-2]

        # Check prediction of last token
        end_token = tokens[-1]
        if end_token >= tokenizer.timestamp_begin:
            start_token = tokens[0]
            assert start_token >= tokenizer.timestamp_begin
            # If Whisper prediction of the end is obviously wrong, we predict it again (constrained)
            if end_token <= start_token:
                new_end_token = last_logprobs[start_token+1:].argmax() + start_token + 1
                tokens[-1] = new_end_token.item()
                if debug:
                    logger.debug(f"Re-estimated end token {tokenizer.decode_with_timestamps([new_end_token])} (was {tokenizer.decode_with_timestamps([end_token])}) to be after start token {tokenizer.decode_with_timestamps([start_token])}")

        
        if len(tokens) <= 1:
            # Corner case: nothing in between timestamps
            ws = []
            ref_align_result = []
        else:
            # segment_idx = torch.nonzero((ref_batch == ref_input_tokens.to(ref_batch.device)).all(dim=1))[0,0].item()
            # segment_idx = hack.SEEK // N_FRAMES
            if hack.NUM_DECODED >= len(hack.SEEK_LIST):
                hack.NUM_DECODED = len(hack.SEEK_LIST) - 1 
            segment_idx = hack.SEEK_LIST[hack.NUM_DECODED] // N_FRAMES
            # if num_alignment_called == 0 and segment_idx != prev_segment_idx:
            #     prev_segment_idx
            # print("532: perform_word_alignment called!!")
            ws, ref_align_result = perform_word_alignment(
                tokens,
                attention_weights,
                ref_attention_weights,
                ref_batch[segment_idx],
                # ref_input_tokens,
                tokenizer,
                ref_text_seg_offset = ref_text_seg_offset[segment_idx],
                use_space=should_use_space(language),
                alignment_heads=alignment_heads,
                remove_punctuation_from_words=remove_punctuation_from_words,
                refine_whisper_precision_nframes=refine_whisper_precision_nframes,
                detect_disfluencies=detect_disfluencies,
                unfinished_decoding=unfinished_decoding,
                mfcc=mfcc,
                plot=plot_word_alignment,
                debug=debug,
            )
            hack.NUM_DECODED += 1
            # num_alignment_called += 1

        add_segment = len(ws) > 0
        if add_segment:
            timestamped_word_segments.append(ws)
            ref_align_segments.append(ref_align_result)
        else:
            logger.debug(f"Not added!")
        reset(add_segment, not is_sot(curr_tokens))

        return add_segment, unfinished_decoding, last_token_reliable

    def may_flush_segment(curr_tokens = None):
        """ Add a speech segment with the new tokens if necessary.
            May also remove the last collected segments if filtered out by Whisper (no_speech_prob <= no_speech_threshold)
        """
        nonlocal segment_tokens, segment_attweights, ref_segment_attweights, timestamped_word_segments, segment_logprobs, has_started, no_speech_prob, chunk_tokens, chunk_tokens_nosot, chunk_logprobs, mfcc, new_mfcc, logit_filters, index_begin_30sec_chunck, last_token_fallback, num_inference_steps, last_chunk_token

        # Check if a new segment should be added
        unfinished_decoding = False
        last_token_reliable = True
        
        if must_flush_segment(curr_tokens) and trust_whisper_timestamps:
            # print("575: Align last segment called by must_flush_segment!!")
            _, unfinished_decoding, last_token_reliable = align_last_segment(curr_tokens)

        i_start = get_index_begin_30sec_chunck(curr_tokens)

        # All segments from previous 30sec chunck have been collected
        if i_start is not None:

            if not trust_whisper_timestamps:

                tokens = torch.Tensor(segment_tokens[-1]).int()
                idx_task = torch.where(tokens==tokenizer.sot_sequence[-1])[0][0].item() # index of <|transcribe|>

                is_special = tokens.ge(tokenizer.eot)
                # Remove prompt
                is_special[:idx_task] = True
                # Keep begin timestamp
                is_special[idx_task:idx_task+2] = False

                is_timestamp = tokens.ge(tokenizer.timestamp_begin)
                consecutive = torch.where(is_timestamp[1:] & is_timestamp[:-1])[0]
                if (WHIPSER_GE_20230306 or has_reached_decoding_limit()) and (
                    (is_timestamp[-1] and not is_timestamp[-2]) if last_chunk_token is None else
                    last_chunk_token >= tokenizer.timestamp_begin and not is_timestamp[-2]
                ):
                    consecutive = torch.cat([consecutive, torch.Tensor([len(tokens)-1]).int()])
                last_is_timestamp = True
                if len(consecutive):
                    # Remove last tokens
                    is_special[consecutive[-1]+1:] = True 
                    # Keep end timestamp
                    is_special[consecutive[-1]] = False
                elif is_timestamp[-1]:
                    # Keep end timestamp
                    is_special[-1] = False
                else:
                    last_is_timestamp = False

                if use_timestamps_for_alignment and len(consecutive):
                    # Keep all timestamps
                    is_special[idx_task+2:consecutive[-1]] = False

                # Do remove what has to be removed
                is_next_achar = ~torch.cat([is_special[1:], torch.Tensor([False]).bool()])
                for i, weights in enumerate(segment_attweights):
                    assert len(weights) == len(tokens), f"{len(weights)} attention weights != {len(tokens)}"
                    # We must remove attention weights used to predict timestamp tokens
                    segment_attweights[i] = [w for s, w in zip(is_next_achar, weights) if s]
                for i, weights in enumerate(ref_segment_attweights):
                    assert len(weights) == len(tokens), f"{len(weights)} attention weights != {len(tokens)}"
                    # We must remove attention weights used to predict timestamp tokens
                    ref_segment_attweights[i] = [w for s, w in zip(is_next_achar, weights) if s]
                tokens_filtered = tokens[~is_special]                        
                assert len(segment_attweights[0]) == len(tokens_filtered), f"{len(segment_attweights[0])} attention weights != {len(tokens_filtered)} "

                # Replace first and last timestamp
                orig_start, orig_end = tokens_filtered[1].item(), tokens_filtered[-1].item()
                tokens_filtered[1] = tokenizer.timestamp_begin # <|0.00|>
                if last_is_timestamp:
                    tokens_filtered[-1] = tokenizer.timestamp_begin + N_FRAMES // 2 # <|30.00|>
                segment_tokens[-1] = tokens_filtered.tolist()

                # Do alignement
                print("638: Align last segment called by get_index_begin_30sec_chunck!!")
                added, unfinished_decoding, last_token_reliable = align_last_segment()

                # Re-split into segments (if necessary)
                if added:
                    if len(consecutive) > 1:
                        segments_timestamped_concat = timestamped_word_segments[-1]
                        new_segments_timestamped = []
                        new_segment_tokens = []
                        start = idx_task+1
                        i_word = 0
                        for i, end in enumerate(consecutive):
                            end = end.item()
                            new_segment_tokens.append(tokens[start:end+1].tolist())
                            if debug:
                                logger.debug(f"Add segment {len(timestamped_word_segments)+i}:\n\t{tokenizer.decode_with_timestamps(new_segment_tokens[-1])}")
                            total_length = end - start - 1
                            start = end+1
                            length = 0
                            new_segments_timestamped.append([])
                            while length < total_length:
                                if not use_timestamps_for_alignment and i_word == len(segments_timestamped_concat):
                                    # This can happen in the case of "..."
                                    assert total_length == 1 and i == len(consecutive)-1, "Unexpected situation!"
                                    break
                                assert i_word < len(segments_timestamped_concat), f"i_word={i_word} < len(segments_timestamped_concat)={len(segments_timestamped_concat)}"
                                word = segments_timestamped_concat[i_word]
                                new_segments_timestamped[-1].append(word)
                                length += len(word["tokens_indices"])
                                i_word += 1
                            # This can be non zero, when a punctuation (alone in a segment) is glued to the previous segment
                            if use_timestamps_for_alignment:
                                assert length == total_length, f"length={length} != total_length={total_length}"
                            elif length > total_length:
                                delta = length - total_length
                                word = new_segments_timestamped[-1][-1]
                                word_tokindices = word["tokens_indices"]
                                word_tokens = word["tokens"]
                                word["tokens_indices"] = word_tokindices[:-delta]
                                word["tokens"] = word_tokens[:-delta]
                                word["word"] = "".join(word_tokens[:-delta])
                                i_word -= 1
                                t = segments_timestamped_concat[i_word]["end"]
                                segments_timestamped_concat[i_word] = dict(
                                    text="".join(word_tokens[-delta:]),
                                    start=t, end=t, # Word without timestamp
                                    tokens=word_tokens[-delta:],
                                    tokens_indices=word_tokindices[-delta:],
                                )

                        assert i_word == len(segments_timestamped_concat)

                        segment_tokens = segment_tokens[:-2] + new_segment_tokens + [segment_tokens[-1]]
                        timestamped_word_segments = timestamped_word_segments[:-1] + new_segments_timestamped

                    else:

                        # Recover start and end token
                        segment = segment_tokens[-2]
                        tokenizer.decode_with_timestamps([orig_start,orig_end])
                        segment[0] = orig_start
                        if last_is_timestamp:
                            segment[-1] = orig_end

                        if debug:
                            logger.debug(f"Add segment {len(timestamped_word_segments)}:\n\t{tokenizer.decode_with_timestamps(segment)}")
                        
                    if unfinished_decoding:
                        timestamped_word_segments[-1][-1]["avg_logprob_reliable"] = last_token_reliable

                reset(False)

            mfcc = new_mfcc

            n_segments = len(segment_tokens)-1

            # Get word confidence and/or check if previous segments shoud have been skipped
            should_skip = False
            if compute_word_confidence or no_speech_threshold is not None:

                # no voice activity check
                should_skip = (no_speech_prob > no_speech_threshold) if (no_speech_threshold is not None) else False
                if compute_word_confidence or (should_skip and logprob_threshold is not None):
                    n = len(chunk_logprobs)
                    if n == len(chunk_tokens_nosot):
                        chunk_tokens_nosot = chunk_tokens_nosot[1:]
                    if unfinished_decoding:
                        assert last_token_fallback is not None
                        last_tokens = [last_token_fallback]
                        timestamped_word_segments[-1][-1]["avg_logprob_reliable"] = last_token_reliable
                        n += 1
                    elif has_reached_decoding_limit():
                        # there were segments in the 30sec chunck, and then the LM got stuck
                        last_tokens = [torch.argmax(chunk_logprobs[-1]).item()]
                        timestamped_word_segments[-1][-1]["avg_logprob_reliable"] = (temperature == 0)
                    else:
                        last_tokens = [tokenizer.eot]
                    chunck_indices = chunk_tokens_nosot + last_tokens
                    assert len(chunk_logprobs) == len(chunck_indices), f"{len(chunk_logprobs)} != {len(chunck_indices)}"
                    logprobs = torch.cat([logprob[i].unsqueeze(0) for (logprob, i) in zip(chunk_logprobs, chunck_indices)])
                    assert min([p.isfinite().item() for p in logprobs]), \
                        f"Got infinite logprob among ({len(logprobs)}) {[(i, tokenizer.decode_with_timestamps([i]), v.item()) for (i,v) in zip(chunck_indices, logprobs)]}"
                    sum_logprob = sum(logprobs)
                    avg_logprob = sum_logprob/n
                    # don't skip if the logprob is high enough, whatever the no_speech_prob is
                    if logprob_threshold is not None and avg_logprob > logprob_threshold:
                        should_skip = False

                if should_skip:
                    logger.debug(f"Skipping last {n_segments-i_start} segments (no_speech_prob {no_speech_prob} > {no_speech_threshold} and avg_logprob {avg_logprob} < {logprob_threshold})")
                    index_begin_30sec_chunck -= n_segments-i_start
                    segment_tokens = segment_tokens[:i_start] + [segment_tokens[-1]]
                    timestamped_word_segments = timestamped_word_segments[:i_start]
                elif compute_word_confidence:
                    avg_logprob = avg_logprob.item()
                    i_token_end = -1
                    for i in range(i_start, n_segments):
                        tokens = segment_tokens[i]
                        i_token_start = i_token_end + 1
                        i_token_end = i_token_start + len(tokens)
                        assert chunck_indices[i_token_start:i_token_end] == tokens, f"Inconsistent token list {tokenizer.decode_with_timestamps(chunck_indices[i_token_start:i_token_end])} != {tokenizer.decode_with_timestamps(tokens)}"
                        i_token_start += 1 # skip sos (start time)
                        if not unfinished_decoding or i != n_segments-1:
                            i_token_end -= 1 # skip eos (end time)
                        segment_logprobs.append(logprobs[i_token_start:i_token_end])
                        segment_avglogprobs.append(avg_logprob)
                else:
                    for i in range(i_start, n_segments):
                        segment_logprobs.append(None)
                        segment_avglogprobs.append(None)
               
            else:
                for i in range(i_start, n_segments):
                    segment_logprobs.append(None)
                    segment_avglogprobs.append(None)

            if verbose_bugged and not should_skip:
                for segment in timestamped_word_segments[i_start:]:
                    for word in segment:
                        print_timestamped(word)

            # Reset counters
            chunk_tokens = []
            chunk_tokens_nosot = []
            chunk_logprobs = []
            no_speech_prob = None

    def hook_attention_weights(layer, ins, outs, index):
        nonlocal segment_attweights
        # In old version of whisper, output is a single tensor
        assert isinstance(outs, tuple) and len(outs) == 2, "whisper seems to be outdated, please update it (pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git)"
        if not has_started:
            return
        w = outs[-1]
        # Only the last attention weights is useful
        if w.shape[-2] > 1:
            w = w[:, :, -1:, :]
        segment_attweights[index].append(w.cpu())
        
    def hook_ref_attention_weights(layer, ins, outs, index):
        nonlocal ref_segment_attweights
        if not has_started:
            return
        w = outs[-1]
        # Only the last attention weights is useful
        if w.shape[-2] > 1:
            w = w[:, :, -1:, :]
        ref_segment_attweights[index].append(w.cpu())
        
    def hook_mfcc(layer, ins, outs):
        nonlocal new_mfcc, mfcc
        new_mfcc = ins[0]
        if mfcc is None:
            mfcc = new_mfcc
            
    def hook_ref_tokens(layer, ins, outs):
        nonlocal ref_input_tokens
        ref_input_tokens = ins[0]
        ref_input_tokens.squeeze_()

    def hook_input_tokens(layer, ins, outs):
        nonlocal segment_tokens, sot_index, chunk_tokens, chunk_tokens_nosot, logit_filters, has_started, language, num_inference_steps
        num_inference_steps += 1

        curr_tokens = ins[0]
        assert curr_tokens.shape[0] == 1, "Batch decoding is not supported"
        curr_tokens = curr_tokens.squeeze(0)

        if is_sot(curr_tokens):
            chunk_prompt = curr_tokens.tolist()
            if language is None:
                if len(curr_tokens) > 1:
                    language = tokenizer.decode(curr_tokens[-2:-1])
                    language = language[2:-2] # remove trailing "<|" and "|>"
                    whisper_options["language"] = language

                    if verbose and not whisper_options["verbose"] and len(curr_tokens) > 1:
                        # Reproduce whisper verbose (2/2)
                        print(f"Detected language: {whisper.tokenizer.LANGUAGES[language].title()}")
                        sys.stdout.flush()

            logit_filters = get_logit_filters(model, whisper_options, prompt = chunk_prompt[1:-len(tokenizer.sot_sequence)])
        
        may_flush_segment(curr_tokens)

        # Get the index of the <|startoftranscript|> tokens (to get proba of silence later)
        if is_sot(curr_tokens):
            has_started = len(curr_tokens) > 1 or not model.is_multilingual
            if no_speech_threshold is not None:
                sot_index = curr_tokens.tolist().index(tokenizer.sot)
        else:
            sot_index = None

        # Keep the last token only
        if has_started:
            segment_tokens[-1].append(curr_tokens[-1].item())

        # Accumulate tokens
        if has_started:
            chunk_tokens.append(curr_tokens)
            if not is_sot(curr_tokens):
                chunk_tokens_nosot.append(curr_tokens[-1].item())
        else:
            if verbose and not whisper_options["verbose"]:
                # Reproduce whisper verbose (1/2)
                print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")

    embedding_weights = None 
    def hook_output_logits(layer, ins, outs):
        nonlocal no_speech_prob, chunk_logprobs, segment_tokens, chunk_tokens, chunk_tokens_nosot, last_chunk_token, embedding_weights, has_started
        
        if embedding_weights is None:
            embedding_weights = torch.transpose(model.decoder.token_embedding.weight, 0, 1).to(outs[0].dtype)

        # Get the probability of silence
        if sot_index is not None:
            logits = (outs[0][sot_index,:] @ embedding_weights).float()
            logits = logits.softmax(dim=-1)
            no_speech_prob = logits[tokenizer.no_speech].item()
        
        # Get the log-probabilities of tokens (we don't know yet which one will be chosen)
        if has_started:
            logits = (outs[0][-1:,:] @ embedding_weights).float()
            tokens = torch.cat(chunk_tokens).unsqueeze(0)
            for logit_filter in logit_filters:
                logit_filter.apply(logits, tokens)
            logits = F.log_softmax(logits.squeeze(0), dim=-1)
            chunk_logprobs.append(logits)

            if WHIPSER_GE_20230306 and has_reached_decoding_limit():
                last_chunk_token = torch.argmax(logits).item()
            else:
                last_chunk_token = None
                
    try:

        # Add hooks to the model, to get tokens and attention weights on the fly
        all_hooks = []
        all_hooks.append(model.encoder.conv1.register_forward_hook(hook_mfcc))
        all_hooks.append(model.ref_encoder.token_embedding.register_forward_hook(hook_ref_tokens))
        all_hooks.append(model.decoder.token_embedding.register_forward_hook(hook_input_tokens))
        nblocks = len(model.decoder.blocks) + len(model.decoder.ref_blocks)
        j = 0
        for i, block in enumerate(model.decoder.blocks):
            if i < nblocks - word_alignement_most_top_layers:
                continue
            all_hooks.append(
                block.cross_attn.register_forward_hook(
                    lambda layer, ins, outs, index=j: hook_attention_weights(layer, ins, outs, index))
            )
            j += 1
        for i, block in enumerate(model.decoder.ref_blocks, start=len(model.decoder.blocks)):
            if i < nblocks - word_alignement_most_top_layers:
                continue
            all_hooks.append(
                block.cross_attn.register_forward_hook(
                    lambda layer, ins, outs, index=j: hook_attention_weights(layer, ins, outs, index))
            )
            all_hooks.append(
                block.ref_cross_attn.register_forward_hook(
                   lambda layer, ins, outs, index=i-len(model.decoder.blocks): hook_ref_attention_weights(layer, ins, outs, index))
            )
            j += 1
        if compute_word_confidence or no_speech_threshold is not None:
            all_hooks.append(model.decoder.ln.register_forward_hook(hook_output_logits))
        hack.NUM_DECODED = 0
        transcription = model.transcribe(audio, text, **whisper_options)

    finally:

        # Remove hooks
        for hook in all_hooks:
            hook.remove()

    # Finalize (collect last segment)
    may_flush_segment()
    segment_tokens.pop(-1)

    token_special_idx = min(tokenizer.sot, tokenizer.eot)
    def filter_tokens(tokens):
        while len(tokens) and tokens[0] >= token_special_idx:
            tokens = tokens[1:]
        while len(tokens) and tokens[-1] >= token_special_idx:
            tokens = tokens[:-1]
        return tokens

    assert len(segment_tokens) == len(timestamped_word_segments), f"Inconsistent number of segments: tokens ({len(segment_tokens)}) != timestamped_word_segments ({len(timestamped_word_segments)})"
    assert len(segment_avglogprobs) == len(segment_tokens), f"Inconsistent number of segments: avg logprobs ({len(segment_avglogprobs)}) != tokens ({len(segment_tokens)})"
    assert len(segment_logprobs) == len(segment_tokens), f"Inconsistent number of segments: logprobs ({len(segment_logprobs)}) != tokens ({len(segment_tokens)})"

    whisper_segments = transcription["segments"]
    l1 = len(whisper_segments)
    l2 = len(timestamped_word_segments)
    if l1 != l2 and l1 != 0:
        logger.warning(f"Inconsistent number of segments: whisper_segments ({l1}) != timestamped_word_segments ({l2})")
    assert l1 == l2 or l1 == 0, f"Inconsistent number of segments: whisper_segments ({l1}) != timestamped_word_segments ({l2})"

    logger.debug("Compile results")
    words = []
    for i, (segment, timestamped_words, token, avglogprob, logprobs) in enumerate(zip(whisper_segments, timestamped_word_segments, segment_tokens, segment_avglogprobs, segment_logprobs)):
        timestamped_tokens = filter_tokens(token)
        whisper_tokens = filter_tokens(segment["tokens"])
        if timestamped_tokens != whisper_tokens:
            if len(timestamped_tokens) == len(whisper_tokens) + 1:
                logger.warning(f"An additional token was added on segment {i}")
            elif WHIPSER_GE_20230306 and len(whisper_tokens) == 0:
                logger.warning(f"Whisper has empty segment {i}")
                assert segment["end"] == segment["start"], f"Fatal Error: Got empty segment {i} with non-zero duration"
                segment["tokens"] = timestamped_tokens
                segment["text"] = tokenizer.decode(timestamped_tokens)
            else:
                assert len(timestamped_tokens) < len(whisper_tokens) and timestamped_tokens == whisper_tokens[:len(timestamped_tokens)], \
                    f"Fatal Error: Got inconsistent text for segment {i}:\n({len(timestamped_tokens)})\n{tokenizer.decode_with_timestamps(timestamped_tokens)}\n{timestamped_tokens}\n!=\n({len(whisper_tokens)})\n{tokenizer.decode_with_timestamps(whisper_tokens)}\n{whisper_tokens[:len(timestamped_tokens)]}"
                segment["tokens"] = token if WHIPSER_GE_20230306 else timestamped_tokens # tokens include special timestamp tokens since 20230306
                segment["text"] = tokenizer.decode(segment["tokens"])
                logger.warning(f"Text had to be shortned on segment {i}:\n{tokenizer.decode(timestamped_tokens)}\n!=\n{tokenizer.decode(whisper_tokens)}")
            timestamped_words[-1]["avg_logprob_reliable"] = False

        offset = segment["seek"] * HOP_LENGTH / SAMPLE_RATE
        for timestamped_word in timestamped_words:
            timestamped_word["start"] += offset
            timestamped_word["end"] += offset
            timestamped_word["idx_segment"] = i

        if compute_word_confidence:
            if "avg_logprob_reliable" not in timestamped_words[-1] or timestamped_words[-1]["avg_logprob_reliable"]:
                # assert abs(segment["avg_logprob"] - avglogprob) < 1e-2, f"Fatal Error: Got inconsistent logprob for segment {i}: {segment['avg_logprob']} != {avglogprob}"
                if abs(segment["avg_logprob"] - avglogprob) >= 1e-2:
                    logger.warning(f"Recomputed different logprob for segment {i}: {avglogprob} != {segment['avg_logprob']}")
            if include_punctuation_in_confidence:
                segment["confidence"] = round_confidence(logprobs.mean().exp().item())
            else:
                logprobs_nopunc = []
            i_end = 0
            for timestamped_word in timestamped_words:
                i_start = i_end
                tokens = timestamped_word["tokens"]
                i_end += len(tokens)

                assert i_end <= len(logprobs), f"Fatal Error: Got out-of-bound index for segment {i}: {i_end} > {len(logprobs)}"
                if include_punctuation_in_confidence:
                    word_logprobs = logprobs[i_start:i_end]
                else:
                    while len(tokens) > 1 and len(tokens[-1]) and tokens[-1][-1] in _punctuation: # Note: look at the last character of token, to take into account "...", "!!", etc.
                        tokens = tokens[:-1]
                    word_logprobs = logprobs[i_start:i_start + len(tokens)]
                    logprobs_nopunc.append(word_logprobs)

                timestamped_word["confidence"] = round_confidence(word_logprobs.mean().exp().item() if len(word_logprobs) else 0.0)

            if i_end not in [len(logprobs), len(logprobs)-1]:
                logger.warning(f"Got inconsistent length for segment {i} ({len(logprobs)} != {i_end}). Some words have been ignored.")
            if not include_punctuation_in_confidence:   
                logprobs_nopunc = torch.cat(logprobs_nopunc)
                segment["confidence"] = round_confidence(logprobs_nopunc.mean().exp().item())

        words.extend(timestamped_words)
    
    assert len(ref_align_segments) == len(whisper_segments)
    for i, seg in enumerate(ref_align_segments):
      offset = whisper_segments[i]["seek"] * HOP_LENGTH / SAMPLE_RATE
      for line in seg:
        line["start"] += offset
        line["end"] += offset
    
    # select best align
    ref_align_lines = [line for seg in ref_align_segments for line in seg]
    unique_ref_align_lines = {} 
    for line in ref_align_lines:
      line_id = tuple(line['ref_token_slices'])
      if line_id not in unique_ref_align_lines:
        unique_ref_align_lines[line_id] = line
      else:
        if line['norm_distance'] < unique_ref_align_lines[line_id]['norm_distance']:
          unique_ref_align_lines[line_id] = line
    unique_ref_align_lines = list(unique_ref_align_lines.values())
    unique_ref_align_lines.sort(key=lambda x:x['ref_token_slices'])
    return transcription, words, unique_ref_align_lines

def _transcribe_timestamped_naive(
    model,
    audio,
    text,
    remove_punctuation_from_words,
    compute_word_confidence,
    include_punctuation_in_confidence,
    refine_whisper_precision_nframes,
    alignment_heads,
    plot_word_alignment,
    word_alignement_most_top_layers,
    detect_disfluencies,
    trust_whisper_timestamps,
    min_word_duration,
    **whisper_options,
):
    verbose = whisper_options["verbose"]
    whisper_options["verbose"] = None if whisper_options["verbose"] is True else whisper_options["verbose"]  # We will print intermediate results ourselves
    language = whisper_options["language"]
    refine_whisper_precision_sec = refine_whisper_precision_nframes * AUDIO_TIME_PER_TOKEN

    word_alignement_most_top_layers = float("inf") if word_alignement_most_top_layers is None else word_alignement_most_top_layers

    audio = get_audio_tensor(audio)
    text = get_ref_text_tensor(text)
    audio_duration = audio.shape[-1] / SAMPLE_RATE

    if verbose and language is None and not whisper_options["verbose"]:
        # Reproduce whisper verbose (1/2)
        print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")

    transcription = model.transcribe(audio, text, **whisper_options)

    if verbose and language is None and not whisper_options["verbose"]:
        # Reproduce whisper verbose (2/2)
        print(f"Detected language: {whisper.tokenizer.LANGUAGES[transcription['language']].title()}")
        sys.stdout.flush()

    language = norm_language(transcription["language"])

    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, task=whisper_options["task"], language=language)
    use_space = should_use_space(language)

    attention_weights = [[] for _ in range(min(word_alignement_most_top_layers,len(model.decoder.blocks) + len(model.decoder.ref_blocks) ))]

    try:

        all_hooks = []

        # Hook the model
        nblocks = len(model.decoder.blocks) + len(model.decoder.ref_blocks)
        j = 0
        for i, block in enumerate(model.decoder.blocks):
            if i < nblocks - word_alignement_most_top_layers:
                continue
            all_hooks.append(
                block.cross_attn.register_forward_hook(
                    lambda layer, ins, outs, index=j: attention_weights.__setitem__(index, outs[-1])
                )
            )
            j += 1
            
        for i, block in enumerate(model.decoder.ref_blocks, start=len(model.decoder.blocks)):
            if i < nblocks - word_alignement_most_top_layers:
                continue
            all_hooks.append(
                block.cross_attn.register_forward_hook(
                    lambda layer, ins, outs, index=j: attention_weights.__setitem__(index, outs[-1])
                )
            )
            j += 1

        # When not relying on Whisper timestamps
        current_tokens = []
        token_to_idx_segment = []

        words = []
        previous_end = 0
        whisper_segments = transcription["segments"]
        for i_segment, segment in enumerate(whisper_segments):

            # Note: this could also be a fix to issue #61 where a "<|te|>" token was predicted
            # segment["tokens"] = [t for t in segment["tokens"] if t < tokenizer.eot or t >= tokenizer.timestamp_begin]

            start = end = tokens = None
            if trust_whisper_timestamps:

                start = segment["start"]
                end = segment["end"]
                if end < start:
                    # Whisper is wrong on the prediction of segment end
                    end = min(audio_duration, start + SEGMENT_DURATION)

                start_margin_min = start - refine_whisper_precision_sec
                start_margin_max = start + refine_whisper_precision_sec
                if start >= audio_duration - min_word_duration or (previous_end >= start_margin_min and previous_end <= start_margin_max):
                    # Make start as accurate as possible (as the decoding will start with timestamp <|0|>)
                    start = previous_end
                else:
                    # Fallback
                    start = start_margin_min

                if start > audio_duration - min_word_duration:
                    # Skip last segment if too short
                    logger.warning(f"Skipping segment outside of audio duration {audio_duration} (original: {segment['start']}-{segment['end']}, new: {start}-XXX)")
                    continue

                end_margin_min = end - refine_whisper_precision_sec
                end_margin_max = end + refine_whisper_precision_sec
                if i_segment < len(whisper_segments) - 1:
                    # Try to enforce:
                    #   end + min_word_duration <= next start + refine_whisper_precision_sec
                    end_margin_max2 = whisper_segments[i_segment + 1]["start"] + refine_whisper_precision_sec - min_word_duration
                    if end_margin_max2 >= end_margin_min:
                        end_margin_max = min(end_margin_max2, end_margin_max)
                end = min(audio_duration, end_margin_max)

                if end < start + min_word_duration:
                    logger.warning(f"Got super short segment (original from whisper: {segment['start']}-{segment['end']}, new: {start, end})")
                    end = min(audio_duration, start + min_word_duration)
                    if end <= start:
                        logger.warning(f"Skipping this short segment occuring too close to the end of the audio")
                        continue

                tokens = segment["tokens"]

            else:

                seek = segment["seek"]
                new_tokens = segment["tokens"]
                if not len(new_tokens):
                    continue
                # Add timestamps that will be needed after
                if new_tokens[0] < tokenizer.timestamp_begin:
                    relative_start = segment["start"] - (seek * HOP_LENGTH / SAMPLE_RATE)
                    start_token = round(relative_start * SAMPLE_RATE / AUDIO_SAMPLES_PER_TOKEN) + tokenizer.timestamp_begin
                    new_tokens = [start_token] + new_tokens
                if new_tokens[-1] < tokenizer.timestamp_begin:
                    relative_end = segment["end"] - (seek * HOP_LENGTH / SAMPLE_RATE)
                    end_token = round(relative_end * SAMPLE_RATE / AUDIO_SAMPLES_PER_TOKEN) + tokenizer.timestamp_begin
                    new_tokens = new_tokens + [end_token]

                current_tokens.extend(new_tokens)
                token_to_idx_segment.extend([i_segment] * len(new_tokens))

                next_seek = whisper_segments[i_segment+1]["seek"] if i_segment < len(whisper_segments) - 1 else None
                if seek != next_seek:
                    start = float(seek * HOP_LENGTH / SAMPLE_RATE)
                    assert start < audio_duration, f"Got start {start} which is outside of audio duration {audio_duration}"
                    end = min(start + SEGMENT_DURATION, audio_duration)
                    tokens = current_tokens

            if tokens is None or not len(tokens):
                continue

            start_sample = min(round(start * SAMPLE_RATE), audio.shape[-1])
            end_sample = min(round(end * SAMPLE_RATE), audio.shape[-1])

            # Extract features on the audio segment
            sub_audio = audio_minimum_padding(audio[start_sample:end_sample])

            mfcc = whisper.log_mel_spectrogram(sub_audio).to(model.device)
            mfcc = whisper.pad_or_trim(mfcc, N_FRAMES)
            mfcc = mfcc.unsqueeze(0)

            segment_tokens_check = []
            if tokens[0] >= tokenizer.timestamp_begin:
                segment_tokens_check.append(tokens[0])
            while tokens[0] >= tokenizer.timestamp_begin:
                tokens = tokens[1:]
                assert len(tokens), "Got transcription with only timestamps!"
            last_token_check = None
            while tokens[-1] >= tokenizer.timestamp_begin:
                last_token_check = tokens[-1]
                tokens = tokens[:-1]

            tokens = [
                    *tokenizer.sot_sequence,
                    tokenizer.timestamp_begin,
                ] + tokens

            i_start = len(tokenizer.sot_sequence)

            with torch.no_grad():
                logprobs = model(mfcc, torch.Tensor(tokens).int().to(model.device).unsqueeze(0))
                logprobs = F.log_softmax(logprobs, dim=-1)

            end_token = tokenizer.timestamp_begin + round(min(N_FRAMES * HOP_LENGTH, end_sample - start_sample) // AUDIO_SAMPLES_PER_TOKEN)
            tokens = tokens[i_start:] + [end_token]
            attention_weights = [w[:, :, i_start-1:, :] for w in attention_weights]
            ref_attention_weights = [w[:, :, i_start-1:, :] for w in ref_attention_weights]

            ws = perform_word_alignment(
                tokens,
                attention_weights,
                ref_attention_weights,
                text,
                tokenizer,
                use_space=use_space,
                alignment_heads=alignment_heads,
                remove_punctuation_from_words=remove_punctuation_from_words,
                refine_whisper_precision_nframes=refine_whisper_precision_nframes,
                detect_disfluencies=detect_disfluencies,
                mfcc=mfcc,
                plot=plot_word_alignment,
            )

            segment_logprobs = []
            i_token = 1
            
            for word in ws:

                word["start"] = round(word["start"] + start, 2)
                word["end"] = round(word["end"] + start, 2)
                
                if trust_whisper_timestamps:
                    word.update({"idx_segment": i_segment})
                else:
                    assert i_token < len(tokens)
                    assert not len(word["tokens_indices"]) or word["tokens_indices"][0] == tokens[i_token]
                    word.update({"idx_segment": token_to_idx_segment[i_token]})
                    i_token += len(word["tokens"])
                    while i_token < len(tokens) and tokens[i_token] >= tokenizer.timestamp_begin:
                        i_token += 1
                
                tok_indices = word["tokens_indices"]
                segment_tokens_check.extend(tok_indices)

                if compute_word_confidence:
                    tok = word["tokens"]
                    i_end = i_start + len(tok)
                    if include_punctuation_in_confidence:
                        while len(tok) > 1 and len(tok[-1]) and tok[-1][-1] in _punctuation: # Note: look at the last character of token, to take into account "...", "!!", etc.
                            tok = tok[:-1]
                            tok_indices = tok_indices[:-1]
                    word_logprobs = [logprobs[:, step, tok] for (step, tok) in zip(range(i_start, i_start + len(tok_indices)), tok_indices)]
                    i_start = i_end
                    if len(word_logprobs):
                        word_logprobs = torch.cat(word_logprobs)
                        segment_logprobs.append(word_logprobs)
                        word_confidence = word_logprobs.mean().exp().item()
                    else:
                        word_confidence = 0
                    word.update({"confidence": round_confidence(word_confidence)})

                words.append(word)

                if verbose:
                    print_timestamped(word)

            if last_token_check is not None:
                segment_tokens_check.append(last_token_check)
            if trust_whisper_timestamps:
                if segment_tokens_check != segment["tokens"]:
                    assert len(segment_tokens_check) < len(segment["tokens"]) and segment_tokens_check[:-1] == segment["tokens"][:len(segment_tokens_check)-1], \
                        f"Got inconsistent tokens: {tokenizer.decode(segment_tokens_check)} != {tokenizer.decode(segment['tokens'])}"
                    segment["tokens"] = segment_tokens_check
                    segment["text"] = tokenizer.decode(segment["tokens"])
            # else: TODO

            if len(segment_logprobs):
                segment.update({"confidence": round_confidence(torch.cat(segment_logprobs).mean().exp().item())})

            if len(ws):
                previous_end = ws[-1]["end"]

            if not trust_whisper_timestamps:
                current_tokens = []
                token_to_idx_segment = []

    finally:

        # Remove hooks
        for hook in all_hooks:
            hook.remove()

    return (transcription, words)

def get_audio_tensor(audio, device="cpu"):
    if isinstance(audio, str):
        audio = whisper.load_audio(audio)
    if isinstance(audio, np.ndarray):
        audio = torch.Tensor(audio)
    else:
        assert isinstance(audio, torch.Tensor), f"Got unexpected audio of type {type(audio)}"
    return audio.to(device)


def get_ref_text_tensor(text: str, device="cpu"):
    
    # if isinstance(text, str):
    #     text = whisper.load_text(text)
    # if isinstance(text, list):
    #     text = " ".join(text)
    # if isinstance(text, str):
    #     text = torch.Tensor([ord(c) for c in text])
    # else:
    #     assert isinstance(text, torch.Tensor), f"Got unexpected text of type {type(text)}"
    return text.to(device)
  
# def read_text(text_fn):
#     with open(text_fn, "r") as f:
#         return f.read().strip()


def audio_minimum_padding(audio):
    if audio.shape[-1] <= 200:
        return whisper.pad_or_trim(audio, 201)
    return audio


def should_use_space(language):
    return norm_language(language) not in ["zh", "ja", "th", "lo", "my"]

def norm_language(language):
    if language is None:
        return "en"
    return whisper.tokenizer.TO_LANGUAGE_CODE.get(language.lower(), language)

def print_timestamped(w):
    line = f"[{format_timestamp(w['start'])} --> {format_timestamp(w['end'])}] {w['text']}\n"
    # compared to just `print(line)`, this replaces any character not representable using
    # the system default encoding with an '?', avoiding UnicodeEncodeError.
    sys.stdout.buffer.write(line.encode(sys.getdefaultencoding(), errors="replace"))
    sys.stdout.flush()


def get_logit_filters(model, whisper_options, prompt = None):
    decoding_options = get_decoding_options(whisper_options)
    if "initial_prompt" in decoding_options:
        prompt0 = decoding_options.pop("initial_prompt")
        if prompt is None:
            prompt = prompt0
    if prompt is not None:
        decoding_options["prompt"] = prompt
    decoding_options = whisper.DecodingOptions(
        without_timestamps=False,
        max_initial_timestamp=1.0,
        prefix=None,
        suppress_blank=True,
        **decoding_options
    )

    # This performs some checks on the options
    decoding_task = whisper.decoding.DecodingTask(model, decoding_options)
    return decoding_task.logit_filters

def get_decoding_options(whisper_options):
    return dict([(k,v) for (k,v) in whisper_options.items()
        if k not in [
            "no_speech_threshold",
            "logprob_threshold",
            "compression_ratio_threshold",
            "condition_on_previous_text",
            "verbose",
        ]
    ])


def perform_word_alignment(
    tokens,
    attention_weights,
    ref_attention_weights,
    ref_text_tensor: torch.LongTensor,
    tokenizer,
    ref_text_seg_offset = 0,
    use_space=True,
    mfcc=None,
    refine_whisper_precision_nframes=0,
    remove_punctuation_from_words=False,
    include_punctuation_in_timing=False, # Was True before 1.9
    unfinished_decoding=False,
    alignment_heads=None, 
    medfilt_width=9,
    qk_scale=1.0,
    detect_disfluencies=True,
    subwords_can_be_empty=True, # Was False before 1.11
    plot=False,
    debug=False,
):
    """
    Perform word alignment on the given tokens and attention weights.
    Returns a list of (word, start_time, end_time) tuples.

    tokens: list of tokens (integers)
    attention_weights: list of attention weights (torch tensors)
    tokenizer: tokenizer used to tokenize the text
    use_space: whether to use spaces to split the tokens into words (should be true for all languages except Japanese, Chinese, ...)
    mfcc: MFCC features (used to identify padded region, and for plotting)
    refine_whisper_precision_nframes: precision time
    remove_punctuation_from_words: whether to remove punctuation from words
    include_punctuation_in_timing: whether to include punctuation in the timing of (previous) words
    unfinished_decoding: whether the decoding is unfinished (e.g. because the model is stuck)
    alignment_heads: list of attention heads to use for alignment
    medfilt_width: width of the median filter used to smooth the attention weights
    qk_scale: scale factor applied to the attention weights
    plot: whether to plot the word alignment
    debug: whether to print debug information
    """

    assert len(tokens) > 1, f"Got unexpected sequence of tokens of length {len(tokens)} {tokenizer.decode_with_timestamps(tokens)}"
    start_token = tokens[0] - tokenizer.timestamp_begin
    end_token = tokens[-1] - tokenizer.timestamp_begin
    
    # if Path(ref_text).exists():
    #     ref_text = Path(ref_text).read_text()
    # tokenized_ref_text = tokenizer.encode(ref_text)
    PAD_IDX = 50257
    len_ref_text = (ref_text_tensor != PAD_IDX).sum().item()
    

    # Check start / end tokens
    if start_token < 0:
        raise RuntimeError(f"Missing start token in: {tokenizer.decode_with_timestamps(tokens)}")
    if len(tokens) == 1 or end_token < 0:
        # This can happens when Whisper is stucked as a Language Model
        if debug:
            logger.debug(f"Missing end token in {tokenizer.decode_with_timestamps(tokens)}")
        end_token = N_FRAMES // 2
    if end_token == start_token and refine_whisper_precision_nframes == 0:
        if debug:
            logger.debug(f"Got empty segment in {tokenizer.decode_with_timestamps(tokens)}")
        return []

    # Let a minimal duration given the number of tokens (see https://github.com/linto-ai/whisper-timestamped/issues/67)
    end_token = min(N_FRAMES // 2, max(end_token, start_token + len(tokens)))

    # Put some margin around the segment
    if refine_whisper_precision_nframes > 0:
        start_token = max(start_token - refine_whisper_precision_nframes, 0)
        end_token = min(end_token + refine_whisper_precision_nframes, N_FRAMES // 2)

    if end_token <= start_token:
        raise RuntimeError(f"Got segment with null or negative duration {tokenizer.decode_with_timestamps(tokens)}: {start_token} {end_token}")

    start_time = start_token * AUDIO_TIME_PER_TOKEN
    # end_time = end_token * AUDIO_TIME_PER_TOKEN

    split_tokens = split_tokens_on_spaces if use_space else split_tokens_on_unicode
    words, word_tokens, word_tokens_indices = split_tokens(tokens, tokenizer, remove_punctuation_from_words=remove_punctuation_from_words)

    # If the last token is a punctuation that comes after a word
    # group this final punctuation with the final timestamp
    # This is to avoid assigning the final punctuation to a big silence or a noise/music background coming after
    num_punctuations_per_tokens = [
        0 if len(w) == 1 or w[-1] not in _punctuation else 1
        for w in word_tokens
    ]
    if include_punctuation_in_timing:
        num_punctuations_per_tokens[:-2]=[0]*(len(num_punctuations_per_tokens)-2)

    for i, w in enumerate(attention_weights):
        assert w.shape[-2] == len(tokens), f"Attention weights have wrong shape: {w.shape[-2]} (expected {len(tokens)})."
    weights = torch.cat(attention_weights) # layers * heads * tokens * frames
    ref_weights = torch.cat(ref_attention_weights) # layers * heads * tokens * frames

    num_tokens = weights.shape[-2]
    num_frames = end_token - start_token
    if num_tokens > num_frames:
        logger.warning(f"Too much text ({num_tokens} tokens) for the given number of frames ({num_frames}) in: {tokenizer.decode_with_timestamps(tokens)}\nThe end of the text will be removed.")
        return perform_word_alignment(
            tokens[:num_frames-1] + [tokens[-1]],
            [torch.cat([w[:, :, :num_frames-1, :], w[:, :, -1:, :]], dim=-2)
                for w in attention_weights],
            [torch.cat([w[:, :, :num_frames-1, :], w[:, :, -1:, :]], dim=-2)
                for w in ref_attention_weights],
            ref_text_tensor,
            tokenizer,
            use_space=use_space,
            refine_whisper_precision_nframes=refine_whisper_precision_nframes,
            medfilt_width=medfilt_width,
            qk_scale=qk_scale,
            alignment_heads=alignment_heads,
            mfcc=mfcc,
            plot=plot,
            remove_punctuation_from_words=remove_punctuation_from_words,
            detect_disfluencies=detect_disfluencies,
            subwords_can_be_empty=subwords_can_be_empty,
            unfinished_decoding=True,
            debug=debug,
        )

    assert end_token <= weights.shape[-1]
    assert len(tokens) == num_tokens

    weights = weights[..., start_token: end_token].cpu()                        # layers * heads * tokens * frames
    ref_weights = ref_weights[..., :len_ref_text].cpu()                # layers * heads * tokens * frames
    
    if alignment_heads is None:
        weights = weights.reshape(-1, *weights.shape[-2:])                      # N * tokens * frames
    else:
        weights = torch.stack([weights[l][h] for l, h in alignment_heads.indices().T])
    weights = median_filter(weights, (1, 1, medfilt_width))
    weights = torch.tensor(weights * qk_scale).softmax(dim=-1)
    weights = weights.mean(axis=(0))  # average over layers and heads           # tokens * frames
    weights = weights / weights.norm(dim=-2, keepdim=True)  # This was before the mean before 1.9
    weights = -weights.double().numpy()
    
    ref_alignment_heads = [-4, -15, -18, -19, -25, -30, -31, -32, -34,-39]
    ref_weights = ref_weights.reshape(-1, *ref_weights.shape[-2:])
    ref_weights = ref_weights[ref_alignment_heads]
    # ref_weights = median_filter(ref_weights, (1, 1, medfilt_width//3))
    ref_weights_dtw = (ref_weights * qk_scale).softmax(dim=-1)
    ref_weights_dtw = ref_weights_dtw.mean(axis=(0))
    ref_weights_dtw = ref_weights_dtw / ref_weights_dtw.norm(dim=-2, keepdim=True)
    ref_weights_dtw = -ref_weights_dtw.double().numpy()
    

    
    worse_weight = 0

    # Get the limit of audio duration
    max_duration = None
    if mfcc is not None:
        max_duration = find_start_padding(mfcc)
        if max_duration is not None:
            max_duration = max_duration // 2

    # Enforce the max duration
    if max_duration:
        if start_token >= max_duration:
            logger.warning(f"Got start time outside of audio boundary")
        else:
            weights[:-1, max_duration:] = worse_weight

    # Encourage to start early
    weights[0, 0] = weights.min()
    # ref_weights[0, 0] = ref_weights.min()
    # weights[0, refine_whisper_precision_nframes*2:] = worse_weight

    if subwords_can_be_empty:
        step_pattern = dtw.stepPattern.symmetric1
    else:
        # Similar as "symmetric1" but without the possibility to have the same timestamp for two tokens
        step_pattern = dtw.stepPattern.StepPattern(dtw.stepPattern._c(
            1, 1, 1, -1,
            1, 0, 0, 1,
            2, 0, 1, -1,
            2, 0, 0, 1,
        ))
    alignment = dtw.dtw(weights, step_pattern=step_pattern)
    ref_alignment = dtw.dtw(ref_weights_dtw, step_pattern=step_pattern )

    global num_alignment_for_plot
    num_alignment_for_plot += 1

    plot_ref = False
    if plot_ref:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import koreanize_matplotlib

        nplots = 1
        plt.subplots(1, 1, figsize=(32, 18), gridspec_kw={'height_ratios': [3] + [1] * (nplots - 1)})
        plt.subplot(1, 1, 1, frameon=False)
        plt.imshow(-ref_weights_dtw, aspect="auto")
        plt.plot(ref_alignment.index2s, ref_alignment.index1s, color="red")

        xticks = np.arange(0, ref_weights_dtw.shape[1])
        ylims = plt.gca().get_ylim()
        
        
        ax = plt.gca()
        ax.tick_params('both', length=0, width=0, which='minor', pad=6)

        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_label_position("left")
        ax.invert_yaxis()
        ax.set_ylim(ylims)

        major_ticks = [-0.5]
        minor_ticks = []
        current_y = 0

        for word, word_token in zip(words, word_tokens):
            minor_ticks.append(current_y + len(word_token) / 2 - 0.5)
            current_y += len(word_token)
            major_ticks.append(current_y - 0.5)

        # words_with_subwords = ["|".join(s).strip() for (w, s) in zip(words, word_tokens)]

        ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
        ax.yaxis.set_minor_formatter(
            ticker.FixedFormatter(words))
        ax.set_yticks(major_ticks)
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        for y in major_ticks:
            plt.axhline(y, color="black", linestyle="dashed")

        plt.ylabel("Prediction", fontsize=40)
        
        
        ref_words, ref_word_tokens, _ = split_tokens_on_spaces(ref_text_tensor[:len_ref_text].tolist(), 
                                                     tokenizer, 
                                                     remove_punctuation_from_words=False)
        
        major_ticks = [-0.5]
        minor_ticks = []
        current_x = 0
        for word, word_token in zip(ref_words, ref_word_tokens):
            minor_ticks.append(current_x + len(word_token) / 2 - 0.5)
            current_x += len(word_token)
            major_ticks.append(current_x - 0.5)
        ref_words_with_subwords = ["".join(s).strip() for (w, s) in zip(ref_words, ref_word_tokens)]
        ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
        ax.xaxis.set_minor_formatter(
            ticker.FixedFormatter(ref_words_with_subwords))
        ax.set_xticks(major_ticks)
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        for x in major_ticks:
            plt.axvline(x, color="black", linestyle="dashed")

        plt.xlabel("Reference", fontsize=40)
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=45, fontsize=20)
        plt.setp(ax.yaxis.get_minorticklabels(), fontsize=20)

        plt.savefig(f"{plot}.ref_alignment{num_alignment_for_plot:03d}.jpg", bbox_inches='tight', pad_inches=0)
        plt.close()


    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import koreanize_matplotlib
        
        plot_mfcc = 1 if mfcc is not None else 0
        plot_disfluencies = 1 if detect_disfluencies else 0
        nplots = (1 + plot_mfcc + plot_disfluencies)

        plt.subplots(nplots, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [3] + [1] * (nplots - 1)})
        plt.subplot(nplots, 1, 1, frameon=False)

        plt.imshow(-weights, aspect="auto")
        plt.plot(alignment.index2s, alignment.index1s, color="red")

        xticks = np.arange(0, weights.shape[1], 1 / AUDIO_TIME_PER_TOKEN)
        xticklabels = [round_timestamp(x) for x in xticks * AUDIO_TIME_PER_TOKEN + start_time]

        ylims = plt.gca().get_ylim()

        ax = plt.gca()
        ax.tick_params('both', length=0, width=0, which='minor', pad=6)

        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_label_position("left")
        ax.invert_yaxis()
        ax.set_ylim(ylims)

        major_ticks = [-0.5]
        minor_ticks = []
        current_y = 0

        for word, word_token in zip(words, word_tokens):
            minor_ticks.append(current_y + len(word_token) / 2 - 0.5)
            current_y += len(word_token)
            major_ticks.append(current_y - 0.5)

        words_with_subwords = ["|".join(s).strip() for (w, s) in zip(words, word_tokens)]

        ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
        ax.yaxis.set_minor_formatter(
            ticker.FixedFormatter(words_with_subwords))
        ax.set_yticks(major_ticks)
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        for y in major_ticks:
            plt.axhline(y, color="black", linestyle="dashed")

        plt.ylabel("Words")

        if plot_mfcc:
            plt.xticks(xticks)
            plt.setp(plt.gca().get_xticklabels(), visible=False)

            xticks *= 2

            plt.subplot(nplots, 1, 2, frameon=False)
            plt.imshow(mfcc[0, :, start_token * 2: end_token * 2].cpu(), aspect="auto", origin="lower")
            plt.yticks([])
            plt.ylabel("MFCC")

        plt.xticks(xticks, xticklabels)
        plt.xlabel("Time (s)")
    


    jumps = np.diff(alignment.index1s)
    jumps = np.pad(jumps, (1, 0), constant_values=1)
    jumps = jumps.astype(bool)
    jumps = alignment.index2s[jumps]
    jumps = np.pad(jumps, (0, 1), constant_values=alignment.index2s[-1])
    
    ref_jumps = np.diff(ref_alignment.index1s)
    ref_jumps = np.pad(ref_jumps, (1, 0), constant_values=1)
    ref_jumps = ref_jumps.astype(bool)
    ref_jumps = ref_alignment.index2s[ref_jumps]
    ref_jumps = np.pad(ref_jumps, (0, 1), constant_values=ref_alignment.index2s[-1])

    jumps_start = jumps
    disfluences = {}
    if detect_disfluencies:
        jumps_start = copy.copy(jumps)

        for (i_token, (tok, begin, end)) in enumerate(zip(tokens, jumps[:-1], jumps[1:])):

            # Find local maxima in the portion of attention weights
            attention_weights = -weights[i_token, begin:end]
            peaks, properties = find_peaks(attention_weights,
                width=3,
                prominence=0.02,
            )
            # If more than 
            if len(peaks) > 1:
                if "left_ips" in properties:
                    left = [round(x) for x in properties["left_ips"]]
                else:
                    left = properties["left_bases"]

                new_begin = left[-1] + begin

                jumps_start[i_token] = new_begin

                if new_begin != begin:
                    is_punctuation = tokenizer.decode_with_timestamps([tok]) in _punctuation
                    if not is_punctuation:
                        disfluences[i_token] = (begin, jumps_start[i_token])
                    else:
                        disfluences[i_token+1] = (begin, end)

            if plot:
                plt.subplot(nplots, 1, 2 + plot_mfcc, frameon=False)
                plt.plot(range(begin,end), attention_weights)
                plt.xlim(0, end)
                
                for i, p in enumerate(peaks):
                    color = 'red' if (len(peaks)>1 and i<len(peaks)-1) else 'green'
                    plt.vlines(begin+p, 0, 1, color=color, linestyle="--")

                if "left_bases" in properties:
                    def barxxy(start, end, y, **kwargs):
                        middle = (start + end) / 2
                        plt.bar(middle, y, width=end-start, **kwargs)
                    color = 'red' if len(peaks)>1 else 'green'
                    barxxy(begin+properties["left_bases"], begin+properties["right_bases"], properties.get("prominences",[1]*len(properties["left_bases"])), alpha=0.5,
                        # put a line with a custom color
                        linewidth=1, edgecolor=color
                    )
                if "left_ips" in properties:
                    for left in properties["left_ips"]:
                        plt.vlines(begin+left, 0, 0.5, color='green', linestyle=':')
                    for right in properties["right_ips"]:  
                        plt.vlines(begin+right, 0, 0.5, color='red', linestyle=':')


    # display the word-level timestamps in a table
    word_boundaries = np.cumsum([len(t) for t in word_tokens])
    word_boundaries = np.pad(word_boundaries, (1, 0))
    begin_times = jumps_start[word_boundaries[:-1]]
    end_times = jumps[word_boundaries[1:] - num_punctuations_per_tokens]
    
    begin_ref_words_idx = ref_jumps[word_boundaries[:-1]]
    end_ref_words_idx = ref_jumps[word_boundaries[1:] - num_punctuations_per_tokens]

    begin_times = begin_times * AUDIO_TIME_PER_TOKEN
    end_times = end_times * AUDIO_TIME_PER_TOKEN

    if detect_disfluencies:
        to_be_added = []
        i_start = 0
        for i_word, toks in enumerate(word_tokens[:-1]):
            i_end = i_start + len(toks)
            if i_start in disfluences and i_word > 0:
                begin, end = disfluences[i_start]
                begin *= AUDIO_TIME_PER_TOKEN
                end *= AUDIO_TIME_PER_TOKEN
                to_be_added.append((i_word, begin, end))
            i_start = i_end
        # Add from the end to avoid messing up the indices
        for (i_word, begin, end) in to_be_added[-1::-1]:
            words.insert(i_word, DISFLUENCY_MARK)
            word_tokens.insert(i_word, [])
            word_tokens_indices.insert(i_word, [])
            begin_times = np.insert(begin_times, i_word, begin)
            end_times = np.insert(end_times, i_word, end)

    # Ignore start / end tokens
    if not refine_whisper_precision_nframes:
        begin_times[1] = begin_times[0]
    if not refine_whisper_precision_nframes:
        end_times[-2] = end_times[-1]
    if unfinished_decoding:
        words = words[1:]
        word_tokens = word_tokens[1:]
        word_tokens_indices = word_tokens_indices[1:]
        begin_times = begin_times[1:]
        end_times = end_times[1:]
    else:
        words = words[1:-1]
        word_tokens = word_tokens[1:-1]
        word_tokens_indices = word_tokens_indices[1:-1]
        begin_times = begin_times[1:-1]
        end_times = end_times[1:-1]
        begin_ref_words_idx = begin_ref_words_idx[1:-1]
        end_ref_words_idx = end_ref_words_idx[1:-1]


    if plot:
        ymin = 1

        plt.subplot(nplots, 1, 1)
        for i, (w, ws, begin, end) in enumerate(zip(words, word_tokens, begin_times, end_times)):
            ymax = ymin + len(ws)
            if mfcc is None:
                plt.text(begin / AUDIO_TIME_PER_TOKEN, num_tokens-0.5, w, ha="left", va="top", color="red")
            for x in [begin, end,]:
                plt.axvline(x / AUDIO_TIME_PER_TOKEN, color="red", linestyle="dotted",
                            ymin=1-ymin/num_tokens,
                            ymax=0,  # 1-ymax/num_tokens,
                            )
            ymin = ymax

        if plot_mfcc:
            plt.subplot(nplots, 1, 2)
            for i, (w, begin, end) in enumerate(zip(words, begin_times, end_times)):
                plt.text(begin * 2 / AUDIO_TIME_PER_TOKEN, mfcc.shape[-2]*1.05, w, ha="left", va="bottom", color="red")
                for x in [begin, end,]:
                    plt.axvline(x * 2 / AUDIO_TIME_PER_TOKEN, color="red", linestyle="dotted")

        if isinstance(plot, str):
            plt.savefig(f"{plot}.alignment{num_alignment_for_plot:03d}.jpg", bbox_inches='tight', pad_inches=0)
        else:
            plt.show()

    
    def get_align_by_ref_line():
      line_split_points = torch.where(ref_text_tensor==198)[0].tolist() # 198 := '\n'
      # add last_index
      if line_split_points[-1] != len_ref_text-1:
        line_split_points.append(len_ref_text-1)
      line_slices = [(0, line_split_points[0])] + [(line_split_points[i]+1, line_split_points[i+1]) for i in range(len(line_split_points)-1)]
      line_slices = [x for x in line_slices if x[1]-x[0]>1]
      
      align_by_ref_line = []
      for slc in line_slices:
        sliced_ref_weights = ref_weights_dtw[:,slc[0]:slc[1]].transpose().copy()
        sliced_align = dtw.dtw(sliced_ref_weights, open_end=True, open_begin=True, step_pattern=dtw.stepPattern.asymmetric)
        token_start, token_end = sliced_align.index2s[0], sliced_align.index2s[-1] + 1
        corresp_transcription = tokenizer.decode(tokens[token_start:token_end])
        norm_distance = sliced_align.normalizedDistance
        strt, endt = jumps[token_start] * AUDIO_TIME_PER_TOKEN, jumps[token_end] * AUDIO_TIME_PER_TOKEN
        
        slc_result = {'ref_token_slices': [slc[0]+ref_text_seg_offset, slc[1]+ref_text_seg_offset],
                      'ref_token_text': tokenizer.decode(ref_text_tensor[slc[0]:slc[1]]),
                      'aligned_transcription': corresp_transcription,
                      'norm_distance': norm_distance,
                      'start': strt + start_time,
                      'end': endt + start_time
                      }
        align_by_ref_line.append(slc_result)
      return align_by_ref_line
    
    align_by_ref_line = get_align_by_ref_line()
    
    return [
        dict(
            text=word,
            start=round_timestamp(begin + start_time),
            end=round_timestamp(end + start_time),
            ref_word=tokenizer.decode(ref_text_tensor[ref_begin:ref_end]),
            ref_word_tokens = ref_text_tensor[ref_begin:ref_end].tolist(),
            ref_timestamp = (ref_text_seg_offset + int(ref_begin), ref_text_seg_offset + int(ref_end)),
            tokens=tokens,
            tokens_indices=tokens_indices,
        )
        for word, begin, end, ref_begin, ref_end, tokens, tokens_indices in zip(words, begin_times, end_times, begin_ref_words_idx, end_ref_words_idx, word_tokens, word_tokens_indices)
        if not word.startswith("<|")
    ], align_by_ref_line

def find_start_padding(mfcc):
    """ Return start of padding given the mfcc, or None if there is no padding """
    last_mfcc = mfcc[0, :, -1]
    if torch.min(last_mfcc) == torch.max(last_mfcc) == 0:
        candidate_index = mfcc.shape[-1] - 2
        while candidate_index > 0:
            candidate = mfcc[0, :, candidate_index]
            if not torch.equal(candidate, last_mfcc):
                return candidate_index + 1
            candidate_index -= 1
        return 0 # WTF!?

def round_confidence(x):
    return round(x, 3)

def round_timestamp(x):
    return round(x, 2)

_punctuation = "".join(c for c in string.punctuation if c not in ["-", "'"]) + "。，！？：”、…"

def split_tokens_on_unicode(tokens: list, tokenizer, remove_punctuation_from_words=False, isolate_punctuations=False):
    words = []
    word_tokens = []
    word_tokens_indices = []
    current_tokens = []

    for token in tokens:
        current_tokens.append(token)
        decoded = tokenizer.decode_with_timestamps([t for t in current_tokens if t < tokenizer.eot or t >= tokenizer.timestamp_begin])
        if "\ufffd" not in decoded:
            empty_tokens = [""] * (len(current_tokens)-1)
            punctuation = not isolate_punctuations and (decoded.strip() and decoded.strip() in _punctuation)
            previous_special = len(word_tokens_indices) > 0 and (word_tokens_indices[-1][-1] >= tokenizer.timestamp_begin)
            if punctuation and not previous_special:
                if len(words) == 0:
                    words = [""]
                    word_tokens = [[]]
                if not remove_punctuation_from_words:
                    words[-1] += decoded
                word_tokens[-1].extend(empty_tokens + [decoded])
                word_tokens_indices[-1].extend(current_tokens)
            else:
                words.append(decoded)
                word_tokens.append(empty_tokens + [decoded])
                word_tokens_indices.append(current_tokens)
            current_tokens = []

    return words, word_tokens, word_tokens_indices


def split_tokens_on_spaces(tokens: torch.Tensor, tokenizer, remove_punctuation_from_words=False):
    subwords, subword_tokens_list, subword_tokens_indices_list = split_tokens_on_unicode(tokens, tokenizer, remove_punctuation_from_words=remove_punctuation_from_words)
    words = []
    word_tokens = []
    word_tokens_indices = []

    for i, (subword, subword_tokens, subword_tokens_indices) in enumerate(zip(subwords, subword_tokens_list, subword_tokens_indices_list)):
        special = (subword_tokens_indices[0] >= tokenizer.timestamp_begin)
        previous_special = (i > 0) and (subword_tokens_indices_list[i-1][0] >= tokenizer.timestamp_begin)
        next_special = (i < len(subword_tokens_indices_list)-1) and (subword_tokens_indices_list[i+1][0] >= tokenizer.timestamp_begin)
        previous_space = (i > 0) and (not subwords[i-1].strip())
        is_space = not subword.strip()
        with_space = subword.startswith(" ") and not is_space
        punctuation = not is_space and subword.strip() in _punctuation
        if special or (not previous_space and (previous_special or (with_space and not punctuation) or (is_space and not next_special))):
            words.append(subword.strip())
            word_tokens.append(subword_tokens)
            word_tokens_indices.append(subword_tokens_indices)
        elif len(words) == 0:
            words = [subword.strip()]
            word_tokens = [subword_tokens]
            word_tokens_indices = [subword_tokens_indices]
        else:
            words[-1] = words[-1] + subword.strip()
            word_tokens[-1].extend(subword_tokens)
            word_tokens_indices[-1].extend(subword_tokens_indices)

    return words, word_tokens, word_tokens_indices

silero_vad_model = None
def get_vad_segments(audio,
    output_sample=False,
    min_speech_duration=0.1,
    min_silence_duration=0.1,
    dilatation=0.5,
    ):
    """
    Get speech segments from audio using Silero VAD
    parameters:
        audio: torch.Tensor
            audio data *in 16kHz*
        output_sample: bool
            if True, return start and end in samples instead of seconds
        min_speech_duration: float
            minimum duration (in sec) of a speech segment
        min_silence_duration: float
            minimum duration (in sec) of a silence segment
        dilatation: float
            how much (in sec) to enlarge each speech segment detected by the VAD
    """
    global silero_vad_model, silero_get_speech_ts

    if silero_vad_model is None:
        import onnxruntime
        onnxruntime.set_default_logger_severity(3) # Remove warning "Removing initializer 'XXX'. It is not used by any node and should be removed from the model."
        repo_or_dir = os.path.expanduser("~/.cache/torch/hub/snakers4_silero-vad_master")
        source = "local"
        if not os.path.exists(repo_or_dir):
            repo_or_dir = "snakers4/silero-vad"
            source = "github"
        silero_vad_model, utils = torch.hub.load(repo_or_dir=repo_or_dir, model="silero_vad", onnx=True, source=source)
        silero_get_speech_ts = utils[0]

    # Cheap normalization of the volume
    audio = audio / max(0.1, audio.abs().max())

    segments = silero_get_speech_ts(audio, silero_vad_model,
        min_speech_duration_ms = round(min_speech_duration * 1000),
        min_silence_duration_ms = round(min_silence_duration * 1000),
        return_seconds = False,
    )

    if dilatation > 0:
        dilatation = round(dilatation * SAMPLE_RATE)
        new_segments = []
        for seg in segments:
            new_seg = {
                "start": max(0, seg["start"] - dilatation),
                "end": min(len(audio), seg["end"] + dilatation)
            }
            if len(new_segments) > 0 and new_segments[-1]["end"] >= new_seg["start"]:
                new_segments[-1]["end"] = new_seg["end"]
            else:
                new_segments.append(new_seg)
        segments = new_segments

    ratio = 1 if output_sample else 1 / SAMPLE_RATE

    if ratio != 1:
        for seg in segments:
            seg["start"] *= ratio
            seg["end"] *= ratio
    if output_sample:
        for seg in segments:
            seg["start"] = round(seg["start"])
            seg["end"] = round(seg["end"])
    return segments

def remove_non_speech(audio,
    use_sample=False,
    min_speech_duration=0.1,
    min_silence_duration=1,
    plot=False,
    ):
    """
    Remove non-speech segments from audio (using Silero VAD),
    glue the speech segments together and return the result along with
    a function to convert timestamps from the new audio to the original audio
    """

    segments = get_vad_segments(
        audio,
        output_sample=True,
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
    )

    segments = [(seg["start"], seg["end"]) for seg in segments]
    if len(segments) == 0:
        segments = [(0, audio.shape[-1])]

    audio_speech = torch.cat([audio[..., s:e] for s,e in segments], dim=-1)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(audio)
        for s,e in segments:
            plt.axvspan(s, e, color='red', alpha=0.1)
        if isinstance(plot, str):
            plt.savefig(f"{plot}.VAD.jpg", bbox_inches='tight', pad_inches=0)
        else:
            plt.show()

    if not use_sample:
        segments = [(float(s)/SAMPLE_RATE, float(e)/SAMPLE_RATE) for s,e in segments]
 
    return audio_speech, lambda t, t2 = None: do_convert_timestamps(segments, t, t2)

def do_convert_timestamps(segments, t, t2 = None):
    """
    Convert timestamp from audio without non-speech segments to original audio (with non-speech segments)

    parameters:
        segments: list of tuple (start, end) corresponding to non-speech segments in original audio
        t: timestamp to convert
        t2: second timestamp to convert (optional), when the two timestamps should be in the same segment
    """
    assert len(segments)
    ioffset = 0 # Input offset
    ooffset = 0 # Output offset
    ipreviousend = 0
    result = []
    for istart, iend in segments:
        ostart = ooffset
        oend = ostart + (iend - istart)
        ooffset = oend
        ioffset += istart - ipreviousend
        ipreviousend = iend
        t_in = t <= oend
        t2_in = t_in if t2 is None else t2 <= oend
        if t_in or t2_in:
            result.append([
                max(istart, min(iend, ioffset + t)),
                max(istart, min(iend, ioffset + t2)) if t2 is not None else None
            ])
            if t_in and t2_in:
                break
    if not len(result):
        result.append(
            [ioffset + t, ioffset + t2 if t2 is not None else None]
        )
        
    if len(result) > 1:
        # Minimize difference between durations
        result = sorted(result, key=lambda x: abs(abs(t2-t) - abs(x[1]-x[0])))
    result = result[0]
    if t2 is None:
        result = round(result[0], 2)
    else:
        result = [round(x, 2) for x in result]
    return result

def remove_last_null_duration_words(transcription, words, recompute_text=False):
    """
    Remove words with null duration happening at the end of a chunk (probable Whisper hallucinations)
    """
    # First group segments by audio chunk
    segments_groups = {}
    seek = None
    current_chunk = -1
    for i, segment in enumerate(transcription["segments"]):
        if segment["seek"] != seek:
            current_chunk += 1
            seek = segment["seek"]
        segments_groups[i] = current_chunk

    # Remove words with null duration happening at the end of a chunk
    current_chunk = -1
    is_last_empty = False
    to_remove = []
    for i, word in enumerate(words[::-1]): # Reverse order
        i = len(words) - i - 1
        empty = (word["start"] == word["end"])
        idx_segment = word["idx_segment"]
        group = segments_groups[idx_segment]
        if current_chunk != group:
            is_last_empty = empty
            current_chunk = group
        elif not empty:
            is_last_empty = False
        if is_last_empty:
            # Remove word
            to_remove.append(i)
            # Shorten text of segment
            full_word = "".join(word["tokens"])
            logger.debug(f"Removing word {i+1}/{len(words)} \"{full_word}\" with empty duration at the end of segment {idx_segment+1}/{len(transcription['segments'])}")
            segment = transcription["segments"][idx_segment]
            text = segment["text"]
            if not text.endswith(full_word): # see issue #62
                if text.endswith(full_word[:-1]):
                    full_word = full_word[:-1]
                elif text[:-1].endswith(full_word):
                    text = text[:-1]
                else:
                    raise RuntimeError(f"\"{text}\" not ending with \"{full_word}\"")
            text = text[:-len(full_word)]
            if i > 0 and words[i-1]["idx_segment"] == idx_segment:
                segment["text"] = text
            else:
                logger.debug(f"Removing empty segment {idx_segment}")
                # Remove segment with no more words
                transcription["segments"].pop(idx_segment)
                for j in range(i+1, len(words)):
                    words[j]["idx_segment"] -= 1
            recompute_text = True

    for i in to_remove:
        words.pop(i) # Warning: inplace modification

    if recompute_text:
        transcription["text"] = "".join([s["text"] for s in transcription["segments"]])

    return transcription, words


def ensure_increasing_positions(segments, min_duration=0):
    """
    Ensure that "start" and "end" come in increasing order
    """
    has_modified_backward = False
    previous_end = 0
    for i, seg in enumerate(segments):
        if seg["start"] < previous_end:
            assert i > 0
            new_start = round_timestamp((previous_end + seg["start"]) / 2)
            if new_start < segments[i-1]["start"] + min_duration:
                new_start = previous_end
            else:
                segments[i-1]["end"] = new_start
                has_modified_backward = True
            seg["start"] = new_start
        if seg["end"] <= seg["start"] + min_duration:
            seg["end"] = seg["start"] + min_duration
        previous_end = seg["end"]
    if has_modified_backward:
        return ensure_increasing_positions(segments, min_duration)

    previous_end = 0
    for seg in segments:
        seg["start"] = round_timestamp(seg["start"])
        seg["end"] = round_timestamp(seg["end"])
        assert seg["start"] >= previous_end, f"Got segment {seg} coming before the previous finishes ({previous_end} > {seg['start']})"
        assert seg["end"] >= seg["start"], f"Got segment {seg} with end < start"
        previous_end = seg["end"]

    return segments

## Some utilities for writing transcripts to files

def flatten(list_of_lists, key = None):
    for sublist in list_of_lists:
        for item in sublist.get(key, []) if key else sublist:
            yield item

def remove_keys(list_of_dicts, key):
    for d in list_of_dicts:
        yield {k: d[k] for k in d.keys() - {key}}
        

def write_csv(transcript, file, sep = ",", text_first=True, format_timestamps=None, header=False):
    writer = csv.writer(file, delimiter=sep)
    if format_timestamps is None: format_timestamps = lambda x: x
    if header is True:
        header = ["text", "start", "end"] if text_first else ["start", "end", "text"]
    if header:
        writer.writerow(header)
    if text_first:
        writer.writerows(
            [[segment["text"].strip(), format_timestamps(segment["start"]), format_timestamps(segment["end"])] for segment in transcript]
        )
    else:
        writer.writerows(
            [[format_timestamps(segment["start"]), format_timestamps(segment["end"]), segment["text"].strip()] for segment in transcript]
        )

# https://stackoverflow.com/questions/66588715/runtimeerror-cudnn-error-cudnn-status-not-initialized-using-pytorch
# CUDA initialization may fail on old GPU card
def force_cudnn_initialization(device=None, s=32):
    if device is None:
        device = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=device), torch.zeros(s, s, s, s, device=device))

# base85-encoded (n_layers, n_heads) boolean arrays indicating the cross-attention heads that are
# highly correlated to the word-level timing, i.e. the alignment between audio and text tokens.
_ALIGNMENT_HEADS = {
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
    "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
    "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
    "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
    "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
    "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
    "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
    "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
    "large-v2": b'ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj',
    # "large": b'ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj',
}

_PARAMETERS_TO_MODEL_NAME = {
    37184256 : "tiny.en",
    37184640 : "tiny",
    71825408 : "base.en",
    71825920 : "base",
    240582144 : "small.en",
    240582912 : "small",
    762320896 : "medium.en",
    762321920 : "medium",
    1541384960 : "large",
}

def get_alignment_heads(model):
    if hasattr(model, "alignment_heads"): # Since version 20230306
        return model.alignment_heads
    model_name = "large-v2"
    # model_name = _PARAMETERS_TO_MODEL_NAME[_get_number_of_parameters(model)]
    # if model_name == "large":
    #     if next(model.parameters())[0,0,0] > 0:
    #         model_name = "large-v1"
    #     else:
    #         model_name = "large-v2"
    num_layers = model.dims.n_text_layer
    num_heads = model.dims.n_text_head
    return _get_alignment_heads(model_name, num_layers, num_heads)

def _get_alignment_heads(model_name, num_layers, num_heads):
    dump = _ALIGNMENT_HEADS[model_name]
    array = np.frombuffer(gzip.decompress(base64.b85decode(dump)), dtype=bool).copy()
    mask = torch.from_numpy(array).reshape(num_layers, num_heads)
    alignment_heads = mask.to_sparse()
    return alignment_heads

def _get_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters())

from typing import Optional, Union

def load_model(path: str, 
               device: Optional[Union[str, torch.device]] = None):
    
    try:
        ckpt = torch.load(path, map_location='cpu')
    except:
        with open(path, 'rb') as f:
            unpickler =  CustomUnpickler(f)
            ckpt = unpickler.load()
    state_dict = ckpt['model']
    model_dims = ckpt['dims']
    model = Mymodel(model_dims)
    model.load_state_dict(state_dict)
    model.to(device)
    if hasattr(model, "alignment_heads"):
        del model.alignment_heads # Will be recomputed later
    
    return model

'''
def load_model(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    download_root: str = None,
    in_memory: bool = False,
):
    extension = os.path.splitext(name)[-1] if os.path.isfile(name) else None

    if name in whisper.available_models() or extension == ".pt":
        return whisper.load_model(name, device=device, download_root=download_root, in_memory=in_memory)
    
    # Otherwise, assume transformers
    if extension in [".ckpt", ".bin"]:
        model_path = name
    else:
        # Search for the cached file (download if necessary)
        try:
            import transformers
        except ImportError:
            raise ImportError(f"If you are trying to download a HuggingFace model with {name}, please install first the transformers library")
        from transformers.utils import cached_file

        try:
            model_path = cached_file(name, "pytorch_model.bin", cache_dir=download_root, use_auth_token=None, revision=None)
        except Exception as e:
            try:
                if isinstance(e, OSError):
                    model_path = cached_file(name, "whisper.ckpt", cache_dir=download_root, use_auth_token=None, revision=None)
                else:
                    raise e
            except:
                raise RuntimeError(f"Original error: {e}\nCould not find model {name} from HuggingFace nor local folders.")
    # Load HF Model
    hf_state_dict = torch.load(model_path, map_location="cpu")
    # Rename layers
    for key in list(hf_state_dict.keys())[:]:
        new_key = hf_to_whisper_states(key)
        hf_state_dict[new_key] = hf_state_dict.pop(key)
    
    # Remove useless key (Speechbrain
    if "_mel_filters" in hf_state_dict:
        hf_state_dict.pop("_mel_filters")

    # Init Whisper Model and replace model weights
    dims = whisper.model.ModelDimensions(**states_to_dim(hf_state_dict))
    whisper_model = whisper.model.Whisper(dims)
    whisper_model.load_state_dict(hf_state_dict)
    del hf_state_dict
    if hasattr(whisper_model, "alignment_heads"):
        del whisper_model.alignment_heads # Will be recomputed later
    whisper_model = whisper_model.to(device)
    return whisper_model
'''
# Credit: https://github.com/openai/whisper/discussions/830
def hf_to_whisper_states(text):
    text = re.sub('.layers.', '.blocks.', text)
    text = re.sub('.self_attn.', '.attn.', text)
    text = re.sub('.q_proj.', '.query.', text)
    text = re.sub('.k_proj.', '.key.', text)
    text = re.sub('.v_proj.', '.value.', text)
    text = re.sub('.out_proj.', '.out.', text)
    text = re.sub('.fc1.', '.mlp.0.', text)
    text = re.sub('.fc2.', '.mlp.2.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.encoder_attn.', '.cross_attn.', text)
    text = re.sub('.cross_attn.ln.', '.cross_attn_ln.', text)
    text = re.sub('.embed_positions.weight', '.positional_embedding', text)
    text = re.sub('.embed_tokens.', '.token_embedding.', text)
    text = re.sub('model.', '', text)
    text = re.sub('attn.layer_norm.', 'attn_ln.', text)
    text = re.sub('.final_layer_norm.', '.mlp_ln.', text)
    text = re.sub('encoder.layer_norm.', 'encoder.ln_post.', text)
    text = re.sub('decoder.layer_norm.', 'decoder.ln.', text)
    return text

def states_to_dim(state_dict):
    n_audio_state = len(state_dict['encoder.ln_post.bias'])
    n_text_state = len(state_dict["decoder.ln.bias"])
    return {
        "n_mels":        state_dict["encoder.conv1.weight"].shape[1],           # 80
        "n_vocab":       state_dict["decoder.token_embedding.weight"].shape[0], # 51864 / 51865
        "n_audio_ctx":   state_dict["encoder.positional_embedding"].shape[0],   # 1500
        "n_audio_state": n_audio_state,         # 384 / 512 / 768 / 1024 / 1280
        "n_audio_head":  n_audio_state // 64,   # 6 / 8 / 12 / 16 / 20
        "n_audio_layer": len(set([".".join(k.split(".")[:3]) for k in state_dict.keys() if "encoder.blocks." in k])), # 4 / 6 / 12 / 24 / 32
        "n_text_ctx":    state_dict["decoder.positional_embedding"].shape[0],   # 448
        "n_text_state":  n_text_state,          # 384 / 512 / 768 / 1024 / 1280
        "n_text_head":   n_text_state // 64,    # 6 / 8 / 12 / 16 / 20
        "n_text_layer":  len(set([".".join(k.split(".")[:3]) for k in state_dict.keys() if "decoder.blocks." in k])), # 4 / 6 / 12 / 24 / 32
    }

def cli():

    import os
    import sys
    import argparse
    import json

    from .whisper.whisper.utils import str2bool, optional_float, optional_int
    
    try:
        # Old whisper version # Before https://github.com/openai/whisper/commit/da600abd2b296a5450770b872c3765d0a5a5c769
        from .whisper.whisper.utils import write_txt, write_srt, write_vtt
        write_tsv = lambda transcript, file: write_csv(transcript, file, sep="\t", header=True, text_first=False, format_timestamps=lambda x: round(1000 * x))
    
    except ImportError:
        # New whisper version
        from .whisper.whisper.utils import get_writer

        def do_write(transcript, file, output_format):
            writer = get_writer(output_format, os.path.curdir)
            try:
                return writer.write_result({"segments": transcript}, file)
            except TypeError:
                # Version > 20230314
                return writer.write_result({"segments": list(transcript)}, file, {
                    "highlight_words": False,
                    "max_line_width": None,
                    "max_line_count": None,
                })
        def get_do_write(output_format):
            return lambda transcript, file: do_write(transcript, file, output_format)

        write_txt = get_do_write("txt")
        write_srt = get_do_write("srt")
        write_vtt = get_do_write("vtt")
        write_tsv = get_do_write("tsv")

    parser = argparse.ArgumentParser(
        description='Transcribe a single audio with whisper and compute word timestamps',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-v', '--version', help="show version and exit", action='version', version=f'{__version__}')
    parser.add_argument('--versions', help="show versions (of whisper-timestamped and whisper) and exit", action='version',
                        version=f'{__version__} -- Whisper {whisper.__version__} in {os.path.realpath(os.path.dirname(whisper.__file__))}')

    parser.add_argument('audio', help="audio file(s) to transcribe", nargs='+')
    parser.add_argument('text', help="text file(s) to use as a reference", nargs='+')

    
    parser.add_argument('--model', help=f"name of the Whisper model to use. Examples: {', '.join(whisper.available_models())}", default="small")
    parser.add_argument("--model_dir", default=None, help="the path to save model files; uses ~/.cache/whisper by default", type=str)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-o", default=None, help="directory to save the outputs", type=str)
    valid_formats = ["txt", "vtt", "srt", "tsv", "csv", "json"]
    def str2output_formats(string):
        if string == "all":
            return valid_formats
        formats = string.split(",")
        for format in formats:
            if format not in valid_formats:
                raise ValueError(f"Expected one of {valid_formats}, got {format}")
        return formats
    parser.add_argument("--output_format", "-f", default="all", help=f"Format(s) of the output file(s). Possible formats are: {', '.join(valid_formats)}. Several formats can be specified by using commas (ex: \"json,vtt,srt\"). By default (\"all\"), all available formats will be produced", type=str2output_formats)

    parser.add_argument("--task", default="transcribe", help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')", choices=["transcribe", "translate"], type=str)
    parser.add_argument('--language', help=f"language spoken in the audio, specify None to perform language detection.", choices=sorted(whisper.tokenizer.LANGUAGES.keys()) + sorted([k.title() for k in whisper.tokenizer.TO_LANGUAGE_CODE.keys()]), default=None)
    # f"{', '.join(sorted(k+'('+v+')' for k,v in whisper.tokenizer.LANGUAGES.items()))}

    parser.add_argument('--vad', default=False, help="whether to run Voice Activity Detection (VAD) to remove non-speech segment before applying Whisper model (removes hallucinations)", type=str2bool)
    parser.add_argument('--detect_disfluencies', default=False, help="whether to try to detect disfluencies, marking them as special words [*]", type=str2bool)
    parser.add_argument('--recompute_all_timestamps', default=not TRUST_WHISPER_TIMESTAMP_BY_DEFAULT, help="Do not rely at all on Whisper timestamps (Experimental option: did not bring any improvement, but could be useful in cases where Whipser segment timestamp are wrong by more than 0.5 seconds)", type=str2bool)
    parser.add_argument("--punctuations_with_words", default=True, help="whether to include punctuations in the words", type=str2bool)
        
    parser.add_argument("--temperature", default=0.0, help="temperature to use for sampling", type=float)
    parser.add_argument("--best_of", type=optional_int, default=None if USE_EFFICIENT_BY_DEFAULT else 5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=None if USE_EFFICIENT_BY_DEFAULT else 5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations", type=str)
    parser.add_argument("--initial_prompt", default=None, help="optional text to provide as a prompt for the first window.", type=str)
    parser.add_argument("--condition_on_previous_text", default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop", type=str2bool)
    parser.add_argument("--fp16", default=None, help="whether to perform inference in fp16; Automatic by default (True if GPU available, False otherwise)", type=str2bool)

    parser.add_argument("--temperature_increment_on_fallback", default=0.0 if USE_EFFICIENT_BY_DEFAULT else 0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below", type=optional_float)
    parser.add_argument("--compression_ratio_threshold", default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed", type=optional_float)
    parser.add_argument("--logprob_threshold", default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed", type=optional_float)
    parser.add_argument("--no_speech_threshold", default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence", type=optional_float)
    parser.add_argument("--threads", default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS", type=optional_int)

    parser.add_argument("--compute_confidence", default=True, help="whether to compute confidence scores for words", type=str2bool)
    parser.add_argument("--verbose", type=str2bool, default=False, help="whether to print out the progress and debug messages of Whisper")
    parser.add_argument('--plot', help="plot word alignments (save the figures if an --output_dir is specified, otherwhise just show figures that have to be closed to continue)", default=False, action="store_true")
    parser.add_argument('--debug', help="print some debug information about word alignement", default=False, action="store_true")

    parser.add_argument('--simple_transcribe', default=False, action="store_true")

    class ActionSetAccurate(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            assert nargs is None
            super().__init__(option_strings, dest, nargs=0, **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, "best_of", 5)
            setattr(namespace, "beam_size", 5)
            setattr(namespace, "temperature_increment_on_fallback", 0.2)
    parser.add_argument('--accurate', help="Shortcut to use the same default option as in Whisper (best_of=5, beam_search=5, temperature_increment_on_fallback=0.2)", action=ActionSetAccurate)

    class ActionSetEfficient(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            assert nargs is None
            super().__init__(option_strings, dest, nargs=0, **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, "best_of", None)
            setattr(namespace, "beam_size", None)
            setattr(namespace, "temperature_increment_on_fallback", None)
    parser.add_argument('--efficient', help="Shortcut to disable beam size and options that requires to sample several times, for an efficient decoding", action=ActionSetEfficient)

    parser.add_argument('--naive', help="use naive approach, doing inference twice (once to get the transcription, once to get word timestamps and confidence scores).", default=False, action="store_true")

    args = parser.parse_args().__dict__
    args.pop("accurate")
    args.pop("efficient")

    temperature = args.pop("temperature")
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    threads = args.pop("threads")
    if threads:
        torch.set_num_threads(threads)

    audio_files = args.pop("audio")
    text_files = args.pop("text")
    
    model = args.pop("model")
    device = args.pop("device")
    model_dir = args.pop("model_dir")

    if device.lower().startswith("cuda"):
        force_cudnn_initialization(device)

    output_format = args.pop("output_format")

    model = load_model(model, device=device)

    plot_word_alignment = args.pop("plot")

    debug = args.pop("debug")
    logging.basicConfig()
    if debug:
        logger.setLevel(logging.DEBUG)
        # This supposes to plug a logger with name "WHISPER" into Whisper source code (no harm if it's not set)
        logging.getLogger("WHISPER").setLevel(logging.DEBUG)

    output_dir = args.pop("output_dir")
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    args["naive_approach"] = args.pop("naive")
    args["remove_punctuation_from_words"] = not args.pop("punctuations_with_words")
    args["compute_word_confidence"] = args.pop("compute_confidence")
    args["trust_whisper_timestamps"] = not args.pop("recompute_all_timestamps")

    if Path(audio_files[0]).is_dir() and Path(text_files[0]).is_dir():
      audio_files = list(Path(audio_files[0]).rglob('*.wav'))
      text_files = list(Path(text_files[0]).rglob('*.txt'))
      
      id_set = set([x.stem for x in audio_files]) & set([x.stem for x in text_files])
      audio_files = [x for x in audio_files if x.stem in id_set]
      text_files = [x for x in text_files if x.stem in id_set]
      audio_files.sort()
      text_files.sort()
      assert len(audio_files) == len(text_files), f"Number of audio files ({len(audio_files)}) and text files ({len(txt_files)}) do not match"
      for audio_path, txt_path in zip(audio_files, text_files):
        assert audio_path.stem == txt_path.stem, f"File {audio_path} and {txt_path} do not have the same name"
      audio_files = [str(x) for x in audio_files]
      text_files = [str(x) for x in text_files]    
  
    # Quick early check
    for audio_path in audio_files:
        assert os.path.isfile(audio_path), f"File {audio_path} does not exist"
    for txt_path in text_files:
        assert os.path.isfile(txt_path), f"File {txt_path} does not exist"
    
    use_simple_transcribe = args.pop("simple_transcribe")
    if use_simple_transcribe:
        whisper_options = dict(
            language=args["language"],
            task=args["task"],
            fp16=args["fp16"],
            temperature=temperature,
            best_of=args["best_of"],
            beam_size=args["beam_size"],
            patience=args["patience"],
            length_penalty=args["length_penalty"],
            condition_on_previous_text=args["condition_on_previous_text"],
            initial_prompt=args["initial_prompt"],
            suppress_tokens=args["suppress_tokens"],
            sample_len=None,
            verbose=args["verbose"] if (not args["vad"] or args["verbose"] is not True) else False,
    )
    # audio_files, text_files = reversed(audio_files), reversed(text_files)
    for audio_path, txt_path in tqdm(zip(audio_files, text_files)):

        outname = os.path.join(output_dir, os.path.basename(audio_path)) if output_dir else None
        # if Path(outname + ".words.json").exists():
        #     continue
        
        if use_simple_transcribe:
          result = model.transcribe(audio_path, txt_path, **whisper_options)
          if "json" in output_format:
              # save JSON
              with open(outname + ".json", "w", encoding="utf-8") as js:
                  json.dump(result, js, indent=2, ensure_ascii=False)
          continue
        
        
        try:
            result = transcribe_timestamped(
                model, audio_path, txt_path,
                temperature=temperature,
                plot_word_alignment=outname if (outname and plot_word_alignment) else plot_word_alignment,
                **args
            )
        except Exception as e:
          print(f"Error while processing {audio_path}: {e}")
          continue

        if output_dir:
            if "json" in output_format:
                # save JSON
                with open(outname + ".words.json", "w", encoding="utf-8") as js:
                    json.dump(result, js, indent=2, ensure_ascii=False)

            # save TXT
            if "txt" in output_format:
                with open(outname + ".txt", "w", encoding="utf-8") as txt:
                    write_txt(result["segments"], file=txt)

            # save VTT
            if "vtt" in output_format:
                with open(outname + ".vtt", "w", encoding="utf-8") as vtt:
                    write_vtt(remove_keys(result["segments"], "words"), file=vtt)
                with open(outname + ".words.vtt", "w", encoding="utf-8") as vtt:
                    write_vtt(flatten(result["segments"], "words"), file=vtt)

            # save SRT
            if "srt" in output_format:
                with open(outname + ".srt", "w", encoding="utf-8") as srt:
                    write_srt(remove_keys(result["segments"], "words"), file=srt)
                with open(outname + ".words.srt", "w", encoding="utf-8") as srt:
                    write_srt(flatten(result["segments"], "words"), file=srt)

            # save CSV
            if "csv" in output_format:
                with open(outname + ".csv", "w", encoding="utf-8") as csv:
                    write_csv(result["segments"], file=csv)
                with open(outname + ".words.csv", "w", encoding="utf-8") as csv:
                    write_csv(flatten(result["segments"], "words"), file=csv)

            # save TSV
            if "tsv" in output_format:
                with open(outname + ".tsv", "w", encoding="utf-8") as csv:
                    write_tsv(result["segments"], file=csv)
                with open(outname + ".words.tsv", "w", encoding="utf-8") as csv:
                    write_tsv(flatten(result["segments"], "words"), file=csv)

        elif not args["verbose"]:

            json.dump(filtered_keys(result), sys.stdout, indent=2, ensure_ascii=False)


def filtered_keys(result, keys = [
    "text",
    "segments", "words",
    "language",
    "start",
    "end",
    "confidence"
]):
    if isinstance(result, dict):
        return {k: filtered_keys(v, keys) for k, v in result.items() if k in keys}
    if isinstance(result, list):
        return [filtered_keys(v, keys) for v in result]
    if isinstance(result, float):
        return round(result, 2)
    return result


if __name__ == "__main__":
    cli()