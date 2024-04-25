import torch
from encodec import EncodecModel
import librosa
import fire
import json
import numpy as np
import os  
import sys
import yaml
import math
# 获取当前工作目录  
current_working_directory = os.getcwd()  
  
# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory) 
from tfnet_semantic_token.semantic_token_tfcodec.tfnet_vqvae_lmencoder2 import TFNet_lmencoder as TFCodec_lmencoder
from tfnet_semantic_token.audlib.audiolib import audioread, audiowrite, audioread_resample
from valle.data import (
    AudioTokenConfig,
    AudioTokenExtractor,
    AudioTokenConfig_16k,
    AudioTokenExtractor_16k,
    AudioTokenExtractor_16k_tfcodec,
    TextTokenizer,
    tokenize_text,
    ApplyKmeans,
    HubertFeatureReader
)
semantic_dic = {}

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)
# hubert_int():
ckpt_path = "/home/v-zhijunjia/data/valle-tensorboard-models/other_models/hubert/hubert_base_ls960.pt"
layer = 9
km_path = "/home/v-zhijunjia/data/valle-tensorboard-models/other_models/hubert/hubert_base_ls960_L9_km500.bin"
reader = HubertFeatureReader(ckpt_path, layer)
apply_kmeans = ApplyKmeans(km_path)

#tfnet_sem_init
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

sem_tfcodec_config = "tfnet_semantic_token/semantic_token_tfcodec/config_inference_1.yaml"
with open(sem_tfcodec_config) as f:
    config = yaml.safe_load(f)
    tfcodec_lmencoder = TFCodec_lmencoder(config)
    tfcodec_lmencoder = tfcodec_lmencoder.to(device)
    ckpt_path = "/home/v-zhijunjia/zhijundata_small_v2/data_local/data/valle-tensorboard-models/other_models/tfnet_semantic_tokens/semantic_token_resynt/tfcodec_256bps_disen/tfnetv2_vqvae_lm2-val-1-loss-4.646021-vq-0.199693-iter-716000.ckpt"
    load_pretrained_TFCodec(tfcodec_lmencoder, ckpt_path, device)

def computer_semantic_tfnet(audio_file, depup=False):
    frame_size = 320
    shift = 160
    input_wav, sr = audioread_resample(audio_file)
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
    inds = inds.tolist()[0]
    return inds

def computer_semantic(wav_path, depup=False):
    feat = reader.get_feats(wav_path)
    lab = apply_kmeans(feat).tolist()

    if depup is True:
        unique_tokens = []  
        for token in lab:  
            if token not in unique_tokens:  
                unique_tokens.append(token)
        lab = unique_tokens
    print(lab)
    return lab

def get_dic(folder_path="/scratch/data/l1_l2_arctic/combine_L1_L2/train_india/total", semantic_type=0, depup=False):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".wav"):
                file_path = os.path.join(root, file_name)
                spk = root.split('/')[-2]
                print(file_path)

                if file_name not in semantic_dic.keys():
                    print(f"{file_name} not in semantic_dic")
                    
                    semantic_dic[file_name] = {}
                if semantic_type==0:
                    semantic_dic[file_name][spk] = computer_semantic(file_path, depup)
                elif semantic_type==1:
                    semantic_dic[file_name][spk] = computer_semantic_tfnet(file_path, depup)

    if depup is False:
        with open("/home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/data/Accents/combine_L1_L2/semantic_tokens_dict/hubert_native_more_arctic_semantic_dic.json", "w") as json_file:
            json.dump(semantic_dic, json_file)
    else:
        with open("/home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/data/Accents/combine_L1_L2/semantic_tokens_dict/hubert_native_more_arctic_semantic_dic_depup.json", "w") as json_file_1:
            json.dump(semantic_dic, json_file_1)       

if __name__=="__main__":
    fire.Fire(get_dic)
    # with open("./l1_l2_arctic_semantic_dic.json", "r") as json_file:  
    #     loaded_dict = json.load(json_file) 
    #     for key, values in loaded_dict.items():
    #         print(key)
    #         for value in values:
    #             values_array = np.array(list(value))  

    #             print(values_array.shape)


