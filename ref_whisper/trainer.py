import torch 
import torch.nn 
import ast

from tqdm.auto import tqdm
from tqdm.auto import tqdm 
import wandb
import argparse
import datetime

import whisper.whisper as whisper

def get_argument_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument('--batch_size', type=int, default=8)
  parser.add_argument('--pre_trained_lr', type=float, default=1e-5)
  parser.add_argument('--new_lr', type=float, default=1e-4)
  parser.add_argument('--num_epochs', type=int, default=9)
  parser.add_argument('--warmup_steps', type=int, default=1000)
  parser.add_argument('--total_planend_steps', type=int, default=50000)
  parser.add_argument('--n_ref_encoder_layer', type=int, default=4)
  parser.add_argument('--n_ref_decoder_layer', type=int, default=4)
  parser.add_argument('--n_ref_text_ctx', type=int, default=256)
  parser.add_argument('--n_ref_text_state', type=int, default=1280)
  parser.add_argument('--ref_encoder_type', type=str, default='cnn')
  
  
  parser.add_argument('--random_ratio', type=float, default=0.0)

  parser.add_argument('--num_workers', type=int, default=8)
  parser.add_argument('--device', type=str, default='cuda')
  parser.add_argument('--num_test_samples', type=int, default=10)
  
  parser.add_argument('--finetune_token', action='store_true', default=False)
  parser.add_argument('--use_self_atn_in_ref_dec', default=True, type=lambda x: (str(x).lower() == 'true'))
  parser.add_argument('--use_mlp_in_ref_dec', default=True, type=lambda x: (str(x).lower() == 'true'))
  parser.add_argument('--use_double_cross_atn_block', default=True, type=lambda x: (str(x).lower() == 'true'))
  parser.add_argument('--use_audio_first', default=True, type=lambda x: (str(x).lower() == 'true'))
  parser.add_argument('--finetune_audio_cross_attn', default=False, type=lambda x: (str(x).lower() == 'true'))

  
  parser.add_argument('--debug', action='store_true', default=False)
  return parser

def make_experiment_name_with_date(args):
  current_time_in_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  return f'{current_time_in_str}-{args.ref_encoder_type}_tune_tk:{args.finetune_token}_dc:{args.use_double_cross_atn_block}_layers:{args.n_ref_encoder_layer}_{args.n_ref_decoder_layer}'


def train(model, train_dataloader, criterion, optimizer, scheduler, device, save_log, step_i):
  train_loss_record = []
  model.train() 
  for idx, batch in (enumerate(tqdm(train_dataloader, leave=False))):
    
    audio, input_text, input_txt_attn, around_text, around_txt_attn = batch
    x_input_text = input_text[:,:-1] #[batch, seq_len-1]
    true_input_text = input_text[:,4:] #[batch, seq_len-1]
    
    pred = model(audio.to(device), around_text.to(device), tokens=x_input_text.to(device), ref_mask=around_txt_attn.to(device))
    pred = pred[:, 3:]
    # pred = model(audio.to(device), input_text.to(device), tokens=x_input_text.to(device))
    
    true_input_text_with_mask = true_input_text.masked_fill(input_txt_attn[:, 4:].ne(1) , -100)
    
    #pred [batch, seq_len, ]
    loss = criterion(pred.reshape(-1, pred.size(-1)), true_input_text_with_mask.reshape(-1).to(device))
    if save_log:
      wandb.log({"train_loss": loss.item()}, step=step_i)
    # pred_list, target_list, wers_n, wers_s, wers_d, wers_i, cers_n, cers_s, cers_d, cers_i = get_wer(pred, true_input_text, processor)

    train_loss_record.append(loss.item())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    step_i += 1
  return train_loss_record, step_i
    


def inference(model, processor, audio_path_list, test_dataset_mid, idx):
  ref_text_str = '\n'.join(ast.literal_eval(test_dataset_mid.iloc[idx]['total_lyric']))
  # around_text_token = processor.tokenizer(around_text, return_tensors="pt")
  # if len(around_text_token['input_ids'][0]) > model.dims.n_ref_text_ctx:
  #   around_text_token = truncation(around_text, around_text_token, model.dims.n_ref_text_ctx)

  # around_text_token, _ = around_pad_seqence(around_text_token, max_len = model.dims.n_ref_text_ctx)
  # around_text_token = around_text_token.unsqueeze(0)
  audio_fn = audio_path_list[idx]

  pred = model.transcribe(audio = str(audio_fn), reference_text = ref_text_str, temperature=0, language='ko')
  target = ' '.join(ast.literal_eval(test_dataset_mid['selected_text'].iloc[idx]))
  return pred['text'].strip(), target

def get_accuracy(pred, true_input_text, eos_token = -100):
  total_num = 0
  correct_num = 0

  for i in range(true_input_text.size(0)):
    filtered_input_text = true_input_text[i][true_input_text[i] != eos_token]
    total_num += len(filtered_input_text)
    correct_num += int((filtered_input_text == pred[i].argmax(-1)[:len(filtered_input_text)]).sum())
  return correct_num/total_num

def validate(model, processor, valid_dataloader, criterion, device):
  
  model.eval()
  total_valid_acc = 0
  with torch.inference_mode():
    total_valid_loss = 0
    total_target_list = []
    total_pred_list = []
    for batch in tqdm(valid_dataloader, leave=False):
      audio, input_text, input_txt_attn, around_text, around_txt_attn = batch
      x_input_text = input_text[:,:-1] #[batch, seq_len-1]
      # train_batch += input_text.size(0)
      true_input_text = input_text[:,4:] #[batch, seq_len-1]
      
      pred = model(audio.to(device), around_text.to(device), tokens=x_input_text.to(device))
      pred = pred[:, 3:]
      # pred = model(audio.to(device), input_text.to(device), tokens=x_input_text.to(device))
      true_input_text_with_mask = true_input_text.masked_fill(input_txt_attn[:, 4:].ne(1) , -100)
      
      loss = criterion(pred.reshape(-1, pred.size(-1)), true_input_text_with_mask.reshape(-1).to(device))
      valid_loss = loss.item() * input_text.size(0)
      total_valid_loss += valid_loss
      valid_accuracy = get_accuracy(pred.to('cpu'), true_input_text_with_mask.to('cpu'))  
      batch_valid_acc = valid_accuracy * input_text.size(0)
      total_valid_acc += batch_valid_acc
    
   
      
      # pred_list, target_list, wers_n, wers_s, wers_d, wers_i, cers_n, cers_s, cers_d, cers_i = get_wer(pred, true_input_text, processor)
      # total_target_list += target_list
      # total_pred_list += pred_list
      
      # val_batch_wer_n += wers_n
      # val_batch_wer_s += wers_s
      # val_batch_wer_d += wers_d
      # val_batch_wer_i += wers_i
      # val_batch_cer_n += cers_n
      # val_batch_cer_s += cers_s
      # val_batch_cer_d += cers_d
      # val_batch_cer_i += cers_i

    mean_valid_acc = total_valid_acc / len(valid_dataloader.dataset) 
    average_valid_loss = total_valid_loss / len(valid_dataloader.dataset)
    return average_valid_loss, mean_valid_acc
    # valid_wer_record.append((val_batch_wer_s + val_batch_wer_d + val_batch_wer_i) / val_batch_wer_n)
    # valid_cer_record.append((val_batch_cer_s + val_batch_cer_d + val_batch_cer_i) / val_batch_cer_n)
    
    #Inference
    # audio_fn = 
    # inference_pred = model.transcribe(audio = str(audio_fn), around_text = around_text_token.to(model.device), temperature=0, language='ko')

    # valid_wer_record = (val_batch_wer_s + val_batch_wer_d + val_batch_wer_i) / val_batch_wer_n
    # valid_cer_record = (val_batch_cer_s + val_batch_cer_d + val_batch_cer_i) / val_batch_cer_n
    # result_table = wandb.Table(columns=["target", "pred"], data=list(zip(total_target_list[:100], total_pred_list[:100])))
    
    # return average_valid_loss, mean_valid_acc, valid_wer_record, valid_cer_record, result_table
    
def testset_inference(model, processor, audio_path_list, test_dataset_mid, num_test_samples):
    # Inference
  with torch.inference_mode():
    # selected_ids = random.sample(range(len(audio_path_list)), num_test_samples)
    # selected_ids = range(num_test_samples)
    selected_ids = range(len(audio_path_list))
    inf_texts = []
    inf_targets = []
    for s_idx in selected_ids:
      predicted_text, target_text = inference(model, processor, audio_path_list, test_dataset_mid, s_idx)
      inf_targets.append(target_text)
      inf_texts.append(predicted_text)
  return inf_texts, inf_targets