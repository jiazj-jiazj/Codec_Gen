import argparse
import os

import os  
import sys
# 获取当前工作目录  
current_working_directory = os.getcwd()  

# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory) 
import yaml
import math
import traceback
import numpy as np
from pathlib import Path
from tfnet_semantic_token.semantic_token_tfcodec.tfnet_vqvae_lmencoder2 import TFNet_lmencoder as TFCodec_lmencoder
from tfnet_semantic_token.audlib.audiolib import audioread, audiowrite
from lhotse import CutSet
import torch
from tqdm import tqdm
parser = argparse.ArgumentParser(description="Process command line parameters.")
parser.add_argument("--input_path", type=str, default="/home/v-zhijunjia/data/accent_iclr/ac_baseline_20cases/", help="Path of input audios.")
parser.add_argument("--model_path", type=str, default="semantic_token_resynt/tfnetv2_vqvae_lm2-val-1-loss-4.646021-vq-0.199693-iter-716000.ckpt")
parser.add_argument("-o", "--output_path", default="/home/v-zhijunjia/CodecGen/output_test", help="Output bitstream/audio path")  
parser.add_argument('--config', default="tfnet_semantic_token/semantic_token_tfcodec/config_inference_1.yaml")
parser.add_argument('--pretrained_model_root', type=str, default="semantic_token_resynt")  
parser.add_argument("-log", "--log_file", default=None, help="Output log file path")
parser.add_argument("--wav_dir", type=str, default="/scratch/data/Libritts/raw_files/LibriTTS", help="Output log file path")
parser.add_argument("--input_file", type=str, default="/scratch/data/Libritts/tokenized/cuts_train.jsonl.gz", help="Output log file path")    
parser.add_argument("--output_dir", type=str, default="/scratch/data/Libritts/tokenized_sem_tfnet", help="Output log file path") 
config = {}
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
wav_dir=args.wav_dir


cut_set = CutSet.from_file(args.input_file)
# cut_set = cut_set.to_eager()

if args.log_file is not None:
    logfile = Path(args.log_file).open("wt")

os.makedirs(args.output_path, exist_ok=True)   
print("Using config from ", args.config)
with open(args.config) as f:
    config = yaml.safe_load(f)

tfcodec_lmencoder = TFCodec_lmencoder(config)
tfcodec_lmencoder = tfcodec_lmencoder.to("cuda:0")

device = torch.device("cuda", 0)
ckpt_path = "/home/v-zhijunjia/data/valle-tensorboard-models/other_models/tfnet_semantic_tokens/semantic_token_resynt/tfcodec_256bps_disen/tfnetv2_vqvae_lm2-val-1-loss-4.646021-vq-0.199693-iter-716000.ckpt"

def load_pretrained_TFCodec(model, ckpt_path, device):
    model = model.to(device)
    new_dict = {}
    if ckpt_path is not None:
        tmp_dict = torch.load(ckpt_path, map_location=device)
        # print(tmp_dict.keys())
        # quit()
        tmp_dict2 = tmp_dict["gen"] if 'gen' in tmp_dict.keys() else tmp_dict
        # print('keys to load:{}'.format(len(tmp_dict2.keys())))
        for key in tmp_dict2.keys():
            if 'tfcodec_lmencoder' not in key:
                continue
            new_key = key.split('module.')[-1]
            new_key = key.split('tfcodec_lmencoder.')[-1]
            if 'generator.' in new_key:
                new_key = new_key.split('generator.')[-1]
            new_dict[new_key] = tmp_dict2[key]

            # model.load_state_dict(new_dict, strict=True)
    new_model_dict = model.state_dict()
    
    print('current model keys:{}'.format(len(new_model_dict.keys())))
    # print(new_model_dict.keys())
    print(f'current checkpoint keys:{len(new_dict.keys())}')
    # filter out unnecessary keys
    new_dict_opt = {k: v for k, v in new_dict.items() if k in new_model_dict}
    print('keys loaded :{}'.format(len(new_dict_opt.keys())))
    new_model_dict.update(new_dict_opt)
    model.load_state_dict(new_model_dict, strict=True)  # , strict=False)
    model.eval()

load_pretrained_TFCodec(tfcodec_lmencoder, ckpt_path, device)

#todo: put on GPU
#todo: restore model with args.model_path
# for cut_ in tqdm(cut_set):
# print(cut_)
wav_file_path = "/home/v-zhijunjia/zhijundata_small/data_local/others/000.wav"
# wav_file_path_1 = wav_file_path.split('LibriTTS')[1][1:]
audio_file =wav_file_path
frame_size = 320
shift = 160
input_wav, sr = audioread(audio_file)
input_length = len(input_wav)
# prepare the output wav file
num_frames = math.ceil(input_length * 1.0 / shift)  # use centered stft
output_wav = np.zeros([int(num_frames * shift)])
input_wav_padded = np.concatenate((input_wav, np.zeros(len(output_wav)-input_length)), axis=None)
# input_wav_padded:B,T
input_tensor = torch.from_numpy(input_wav_padded).unsqueeze(0)  
input_tensor = input_tensor.to(device)
result = tfcodec_lmencoder(input_tensor)
inds = result["quantization_inds"]#B,T,2
inds = inds.tolist()
print(inds)
