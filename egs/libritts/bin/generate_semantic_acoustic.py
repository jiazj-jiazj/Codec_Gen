import torch
from encodec import EncodecModel
import librosa
import os
import fire
import json
import numpy as np
from valle.data import AudioTokenizer_encodec_16k_tfcodec, tokenize_audio

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

ckpt_path = "/dev_huaying/zhijun/models/hubert/hubert_base_ls960.pt"
layer = 9
km_path = "/dev_huaying/zhijun/models/hubert/hubert_base_ls960_L9_km500.bin"
reader = HubertFeatureReader(ckpt_path, layer)
apply_kmeans = ApplyKmeans(km_path)

acoustics_dic = {}

def computer_semantic(wav_path, depup=False):
    feat = reader.get_feats(wav_path)
    lab = apply_kmeans(feat).tolist()

    if depup is True:
        unique_tokens = []  
        for token in lab:  
            if token not in unique_tokens:  
                unique_tokens.append(token)
        lab = unique_tokens

    return lab

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)


model_tfcodec = AudioTokenizer_encodec_16k_tfcodec()

acoustics_dic = {}
semantic_dic = {}
two_dic = {}




def computer_encodec(wav_path):

    codes_raw = tokenize_audio(model, wav_path)
    codes_raw = codes_raw[0][0]
    # print(codes_raw.shape) b x quant x t

    codes=codes_raw.cpu().squeeze(0).permute(1, 0).numpy()
    codes=codes.tolist()

    return codes

def computer_tfcodec(wav_path, depup=False):
    
    codes_raw = tokenize_audio(model_tfcodec, wav_path)
    # print(codes_raw.shape)  # b x t x quant
    codes=codes_raw.cpu().squeeze(0).numpy()
    codes=codes.tolist()

    if depup is True:

        unique_tokens = []  
        for token in codes:  
            if token not in unique_tokens:  
                unique_tokens.append(token)
        codes = unique_tokens
    return codes

def check_audio_duration(audio_path):  
    y, sr = librosa.load(audio_path, sr=None)  
    duration = librosa.get_duration(y=y, sr=sr)  
    if 3 < duration < 5:  
        return True
    return False

def get_dic(folder_path, codec_type="encodec", depup=False):
    for entry in os.listdir(folder_path):
        folder_path_1 = os.path.join(folder_path, entry)
        print(folder_path_1)
        yes = 0
        for root, dirs, files in os.walk(folder_path_1):
            for file_name in files:
                if file_name.endswith(".wav"):
                    file_path = os.path.join(root, file_name)
                    if check_audio_duration(file_path) is True:
     
                        acoustics_dic[file_name]=computer_tfcodec(file_path, depup)
                        semantic_dic[file_name] =computer_semantic(file_path)

                        yes=1
                        break
            if yes==1:
                break
    
    two_dic["acoustic_tokens"]=acoustics_dic
    two_dic["semantic_tokens"]=semantic_dic
                
    # if depup is True:                    
    #     with open("/mnt/zhijun/Accents/combine_L1_L2/acoustic_tokens_dic/native_l1_l2_arctic_tfcodec_acoustics_dic_v2.json", "w") as json_file:
    #         json.dump(acoustics_dic, json_file)
    # else:
    #     with open("/mnt/zhijun/Accents/combine_L1_L2/acoustic_tokens_dic/native_l1_l2_arctic_tfcodec_acoustics_dic_v2_depup.json", "w") as json_file_1:
    #         json.dump(acoustics_dic, json_file_1)        

    with open("/mnt/zhijun/Accents/combine_L1_L2/acoustic_tokens_dic/libritts_clean_100_acoustic_semantic.json", "w") as json_file_1:
        json.dump(two_dic, json_file_1)        
if __name__=="__main__":
    fire.Fire(get_dic)
    # with open("./l1_l2_arctic_acoustics_dic.json", "r") as json_file:  
    #     loaded_dict = json.load(json_file) 
    #     for key, values in loaded_dict.items():
    #         print(key)
    #         for value in values:
    #             values_array = np.array(list(value))  

    #             print(values_array.shape)

