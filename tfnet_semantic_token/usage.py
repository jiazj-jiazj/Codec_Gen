import argparse
import os
import yaml
import math
import traceback
import numpy as np
from pathlib import Path
from semantic_token_tfcodec.tfnet_vqvae_lmencoder2 import TFNet_lmencoder as TFCodec_lmencoder
from audlib.audiolib import audioread, audiowrite
import torch
parser = argparse.ArgumentParser(description="Process command line parameters.")
parser.add_argument("--input_path", type=str, default="/home/v-zhijunjia/data/accent_iclr/ac_baseline_20cases/", help="Path of input audios.")
parser.add_argument("--model_path", type=str, default="semantic_token_resynt/tfnetv2_vqvae_lm2-val-1-loss-4.646021-vq-0.199693-iter-716000.ckpt")
parser.add_argument("-o", "--output_path", default="/home/v-zhijunjia/CodecGen/output_test", help="Output bitstream/audio path")  
parser.add_argument('--config', default="semantic_token_tfcodec/config_inference_1.yaml")
parser.add_argument('--pretrained_model_root', type=str, default="semantic_token_resynt")  
parser.add_argument("-log", "--log_file", default=None, help="Output log file path")    

config = {}
args = parser.parse_args()

if args.log_file is not None:
    logfile = Path(args.log_file).open("wt")

os.makedirs(args.output_path, exist_ok=True)   
print("Using config from ", args.config)
with open(args.config) as f:
    config = yaml.safe_load(f)

tfcodec_lmencoder = TFCodec_lmencoder(config)
#todo: put on GPU
#todo: restore model with args.model_path
audio_list = Path(args.input_path).rglob('*.wav')
for k, audio_file in enumerate(sorted(audio_list)):
    try:
        frame_size = 320
        shift = 160
        input_wav, sr = audioread(audio_file)
        input_length = len(input_wav)
        print(f"input_length:{input_length}")
        # prepare the output wav file
        num_frames = math.ceil(input_length * 1.0 / shift)  # use centered stft
        output_wav = np.zeros([int(num_frames * shift)])
        input_wav_padded = np.concatenate((input_wav, np.zeros(len(output_wav)-input_length)), axis=None)
        # input_wav_padded:B,T
        print(input_wav_padded.shape)
        input_tensor = torch.from_numpy(input_wav_padded).unsqueeze(0)  

        result = tfcodec_lmencoder(input_tensor)
        inds = result["quantization_inds"]#B,T,2
        print(inds.shape)
        quit()
        # print(inds)
    except Exception:
            traceback.print_exc()
            print('file exception: {}'.format(os.path.basename(audio_file))) 
            if args.log_file is not None:            
                logfile.write(os.path.basename(audio_file) + "    \n")
            #break
            continue 