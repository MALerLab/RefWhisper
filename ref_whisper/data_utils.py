import torch
import torchaudio
import random
from pathlib import Path
import nlptutti as metrics
from torch.nn.utils.rnn import pad_sequence
from whisper.whisper.tokenizer import RefProcessor

class TextProcessor():
  def __init__(self, id_list, lyric_path, each_lyric, random_ratio=0.8, max_len=256):
    self.lyric_path = Path(lyric_path)
    self.filtered_id = id_list
    self.lyrics_list = [lyric_path+i+'.txt' for i in self.filtered_id]
    self.each_lyric_list = self.read_text(each_lyric).split('\n')
    self.random_ratio = random_ratio
    self.max_len = max_len
    # self.teach_forcing_prob = teach_forcing_prob
  
  def read_text(self, txt_path):
    with open(txt_path, 'r') as f:
      text = f.read()
    return text
  
  def get_all_sentence_list(self):
    all_sentence = []
    for i in self.lyrics_list:
      all_sentence.append(self.read_text(i))
    return list(set(all_sentence))
  
  def __len__(self):
    return len(self.lyrics_list)
  
  def sample_including_target(self, target, lyrics, max_len=256):
    min_len = min([len(x) for x in lyrics])
    selected_line_idx = []
    for idx, line in enumerate(lyrics):
      if line[:min_len] == target[:min_len]:
        selected_line_idx.append(idx)
        target = target[len(line)+1:]
        break
    if idx == len(lyrics)-1:
      for idx, line in enumerate(lyrics):
        if line[:min_len] == target[:min_len]:
          selected_line_idx.append(idx)
    else:
      selected_line_idx.append(idx+1)

    selected_lyrics = [lyrics[i] for i in selected_line_idx]
    cur_len = sum([len(i) for i in selected_lyrics])

    max_line_len = max([len(i) for i in lyrics])
    
    # print("num_adding_lines", num_adding_lines, "cur_len", cur_len, "mean_len", mean_len, "max_len", max_len)
    remaining_lines = [i for i in range(len(lyrics)) if i not in selected_line_idx]
    num_adding_lines = min((max_len - cur_len) // max_line_len - 2, self.max_len//12 , len(remaining_lines))

    sample_idx = random.sample(remaining_lines, max(num_adding_lines, 0) ) 
    
    return selected_lyrics + [lyrics[i] for i in sample_idx]

  

  
  def get_random_sentence(self, target, lyric_name, max_char=256):
    # lyric_name = 'lyric0-1.txt'
    lyric_index = int(lyric_name.split('-')[0][5:])
    if random.random() < self.random_ratio:
      # return random.sample(self.get_all_sentence_list(), 30) # return randomly selected 30 sentences
      lyrics = random.choice(self.each_lyric_list)
    else:
      lyrics = self.each_lyric_list[lyric_index]
    num_char = len(lyrics)
    lyrics = lyrics.split(', ')
    
    if num_char > max_char: # slice
      lyrics = self.sample_including_target(target, lyrics, self.max_len)
      # target_lyrics = lyrics[:2]
      # other_lyrics = lyrics[2:]
      # random.shuffle(other_lyrics)
    random.shuffle(lyrics)
    return lyrics
    
    # if len(lyrics) > 30:
    #   lyrics = random.sample(lyrics, 30)
    # random.shuffle(lyrics)
    return lyrics
  
  def __getitem__(self, idx):
    txt_path = self.lyrics_list[idx]
    lyric_name = txt_path.split('/')[-1][:-4]
    # print(lyric_name)
    input_text = self.read_text(txt_path)
    output_sentence = self.get_random_sentence(input_text, lyric_name, self.max_len)
    output_sentence = '\n'.join(output_sentence)
    return input_text, output_sentence, lyric_name
  


class MinyoDataset():
  def __init__(self, result_path, lyric_path, processor, filtered_id_list, each_lyric, max_len, random_ratio):#filtered_audio_paths, processor, filtered_txt_paths, each_lyric, sorted_txt_paths, max_len = 1024):
    self.encoder_result_paths = Path(result_path)
    self.filtered_id_list = filtered_id_list
    self.text_process = TextProcessor(filtered_id_list, lyric_path, each_lyric, random_ratio, max_len=max_len)
    self.processor = processor
    self.ref_processor = RefProcessor(length=max_len)
    self.max_len = max_len
  
  
  def truncation(self, text):
    return
  # def trunaction(self, text, token_text):
  #   words = text.split()
    
  #   while len(token_text['input_ids'][0]) > self.max_len:
  #     words = words[:-1]
  #     text = '\n '.join(words) + '\n'
  #     token_text = self.processor.tokenizer(text, return_tensors="pt")
  #   return token_text

  def around_pad_seqence(self, token_text, pad_idx = 50257): # pad_idx = token_embedding + 1
    
    if len(token_text['input_ids'][0]) < self.max_len:
      pad_length = self.max_len - len(token_text['input_ids'][0])
      
      padding = torch.full((pad_length,), pad_idx, dtype=torch.long)
      attn_padding = torch.full((pad_length,), 0, dtype=torch.long)
      
      around_text_ids = torch.cat([token_text['input_ids'][0], padding])
      around_text_mask = torch.cat([token_text['attention_mask'][0], attn_padding])    
      return around_text_ids, around_text_mask

    else:
      # print("Warning: input text is longer than max_len")
      return token_text['input_ids'][0, :self.max_len], token_text['attention_mask'][0, :self.max_len]

  def __len__(self):
    return len(self.text_process)
  
  def __getitem__(self, idx):
    input_text, around_text, lyric_name = self.text_process[idx]
    audio_encoder_value = torch.load(self.encoder_result_paths / (lyric_name+'.pt'))
    # audio, sr = torchaudio.load(self.audio_paths / (lyric_name+'.wav'))
    # audio = self.processor.feature_extractor(audio[0] , sampling_rate=sr, padding_value = 0.0, return_tensors="pt", return_attention_mask = True)
    input_text = self.processor.tokenizer(input_text, return_tensors="pt")

    around_text = self.ref_processor.encode_text(around_text).unsqueeze(0)
    around_text = {'input_ids': around_text, 'attention_mask': torch.ones_like(around_text)}
    # around_text = self.processor.tokenizer(around_text, return_tensors="pt")
    # num_tokens = len(around_text['input_ids'][0])
    # if num_tokens > self.max_len:
    #   around_text = around_text['input_ids'][0][:self.max_len]
    # around_text['input_ids'] = around_text['input_ids'][:,self.num_prefix_tokens:]
    # around_text['attention_mask'] = around_text['attention_mask'][:, self.num_prefix_tokens:]
    #   around_text = self.trunaction(around_text_org, around_text)
    around_text_ids, around_text_mask = self.around_pad_seqence(around_text)

    # around_text_ids, around_text_mask = self.around_pad_seqence(input_text) # 임시로 바꿔놨음.

    return audio_encoder_value, input_text.input_ids[0], input_text.attention_mask[0], around_text_ids, around_text_mask


def custom_collate_fn(batch):
  audio, input_text, input_txt_attn, around_text, around_txt_attn = zip(*batch)
  audio = torch.stack(audio, dim = 0)
  # audio_attn = torch.stack(audio_attn, dim = 0)
  input_text = pad_sequence(input_text, batch_first=True, padding_value=50257)
  input_txt_attn = pad_sequence(input_txt_attn, batch_first=True, padding_value=0) 
  around_text = pad_sequence(around_text, batch_first=True, padding_value=50257)
  around_txt_attn = pad_sequence(around_txt_attn, batch_first=True, padding_value=0)
  return audio, input_text, input_txt_attn, around_text, around_txt_attn

def get_wer(pred, target, processor):
  cers_n = 0
  wers_n = 0
  wers_s = 0
  wers_d = 0
  wers_i = 0
  cers_s = 0 
  cers_d = 0
  cers_i = 0
  
  pred = pred.argmax(-1)
  pred_list = processor.batch_decode(pred, skip_special_tokens=True)
  target_list = processor.batch_decode(target, skip_special_tokens=True)

  for pred, target in zip(pred_list, target_list):

    cers_n += len(target.replace(" ", ""))
    wers_n += len(target.split())

    result_wer = metrics.get_wer(target, pred)
    result_cer = metrics.get_cer(target, pred)

    wers_s += result_wer['substitutions']
    wers_d += result_wer['deletions']
    wers_i += result_wer['insertions']

    cers_s += result_cer['substitutions']
    cers_d += result_cer['deletions']
    cers_i += result_cer['insertions']
        
    # result_crr = result_crr['crr']
    # result_cer = result_cer['cer']
    # result_wer = result_wer['wer']

  return pred_list, target_list, wers_n, wers_s, wers_d, wers_i, cers_n, cers_s, cers_d, cers_i 

def get_wer_from_string_pairs(pred_list, target_list):
  total_wer = 0
  total_cer = 0
  for pred, target in zip(pred_list, target_list):
    total_wer += metrics.get_wer(target, pred)['wer']
    total_cer += metrics.get_cer(target, pred)['cer']
  return total_wer/len(pred_list), total_cer/len(pred_list)


class MinyoDataset_for_test():
  def __init__(self, result_path, lyric_path, processor, filtered_id_list, each_lyric, max_len, random_ratio):#filtered_audio_paths, processor, filtered_txt_paths, each_lyric, sorted_txt_paths, max_len = 1024):
    self.encoder_result_paths = Path(result_path)
    self.filtered_id_list = filtered_id_list
    self.text_process = TextProcessor(filtered_id_list, lyric_path, each_lyric, random_ratio)
    self.processor = processor
    self.max_len = max_len
  
  def trunaction(self, text, token_text):
    words = text.split()
    
    while len(token_text['input_ids'][0]) > self.max_len:
      words = words[:-1]
      text = '\n '.join(words) + '\n'
      token_text = self.processor.tokenizer(text, return_tensors="pt")
    return token_text

  def around_pad_seqence(self, token_text):
    
    if len(token_text['input_ids'][0]) < self.max_len:
      pad_length = self.max_len - len(token_text['input_ids'][0])
      
      padding = torch.full((pad_length,), 50257, dtype=torch.long)
      attn_padding = torch.full((pad_length,), 0, dtype=torch.long)
      
      around_text_ids = torch.cat([token_text['input_ids'][0], padding])
      around_text_mask = torch.cat([token_text['attention_mask'][0], attn_padding])    
      return around_text_ids, around_text_mask

    else: 
      return token_text['input_ids'][0], token_text['attention_mask'][0]

  def __len__(self):
    return len(self.text_process)
  
  def __getitem__(self, idx):
    input_text, around_text, lyric_name = self.text_process[idx]
    around_text_org = around_text
    # audio_encoder_value = torch.load(self.encoder_result_paths / (lyric_name+'.pt'))
    # audio, sr = torchaudio.load(self.audio_paths / (lyric_name+'.wav'))
    # audio = self.processor.feature_extractor(audio[0] , sampling_rate=sr, padding_value = 0.0, return_tensors="pt", return_attention_mask = True)
    # input_text = self.processor.tokenizer(input_text, return_tensors="pt")
    around_text = self.processor.tokenizer(around_text, return_tensors="pt")
    
    num_tokens = len(around_text['input_ids'][0])
    
    if num_tokens > self.max_len:
      around_text = self.trunaction(around_text_org, around_text)

    around_text_ids, around_text_mask = self.around_pad_seqence(around_text)
    return num_tokens, around_text_org, input_text, lyric_name, around_text
