#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Phonemize Text and EnCodec Audio.

Usage example:
    python3 bin/tokenizer.py \
        --src_dir ./data/manifests --output_dir ./data/tokenized

"""
import sys
import os
current_working_directory = os.getcwd()  
  
# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory) 

import argparse
import logging
from pathlib import Path
import torch
torch.backends.cudnn.enabled = False  
import soundfile as sf
import torch
import torch.multiprocessing
from icefall.utils import get_executor
from lhotse import CutSet, NumpyHdf5Writer
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm.auto import tqdm
import librosa
from icefall.utils import str2bool

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
from valle.data.fbank import get_fbank_extractor
from valle.utils import SymbolTable
from encodec import EncodecModel

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from academicodec.models.encodec.net3 import SoundStream
import torch.nn.functional as F

from collections import OrderedDict
import soundfile as sf
import joblib
from joblib import load  
import numpy as np

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")

# sys.path.insert(0, "/dev_huaying/zhijun/fairseq")



def main():

    dir_wav_path = ""
    ckpt_path = "/home/v-zhijunjia/data/valle-tensorboard-models/other_models/hubert/hubert_base_ls960.pt"
    layer = 9
    km_path = "/home/v-zhijunjia/data/valle-tensorboard-models/other_models/hubert/hubert_base_ls960_L9_km500.bin"
    # reader = HubertFeatureReader(ckpt_path, layer)
    # apply_kmeans = ApplyKmeans(km_path)    

    cut_set = cutset.from_file("/home/v-zhijunjia/zhijundata_small_v2/data/mls/mls_train_lhotse_dataset_huxue/mls-english_cuts_train_3.jsonl.gz")

    for c in tqdm(cut_set):

        if args.wav_path == "":
            wav_path = c.recording.sources[0].source
        else:
            wav_path = c.recording.sources[0].source
            wav_path = dir_wav_path + '/'.join(wav_path.split('/')[-4:])

            feat = reader.get_feats(wav_path)
            lab = apply_kmeans(feat).tolist()

            c.supervisions[0].custom["tokens"]["semantic_tokens"] =lab


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    # hubert chinese
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
    # model_path="TencentGameMate/chinese-hubert-base"
    # wav_path="/mnt/users/jiazhijun/data/wav_enhance_24k/D1220/ID1220W0003.wav"

    # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    # model = HubertModel.from_pretrained(model_path)

    # # for pretrain: Wav2Vec2ForPreTraining
    # # model = Wav2Vec2ForPreTraining.from_pretrained(model_path)
    # device = "cuda"
    # # print(model)
    # model = model.to(device)
    # model = model.half()
    # model.eval()

    # kmeans_model = ApplyKmeans("/mnt/users/jiazhijun/chinese_speech_pretrain/hubert_kmeans/hubert_base_iter2_32gpu_l9/model.mdl")
    
    # wav, sr = sf.read(wav_path)
    # target_sr = 16000 
    # if sr != target_sr:
    #     wav = librosa.resample(wav, sr, target_sr)  

    # lenn = 31999
    # # wav = wav[:lenn]
    # print(len(wav))
    # input_values = feature_extractor(wav, return_tensors="pt").input_values
    # input_values = input_values.half()
    # input_values = input_values.to(device)
    # with torch.no_grad():
    #     outputs = model(input_values, output_hidden_states=True)
    #     last_hidden_state = outputs.hidden_states[9]
    #     last_hidden_state = torch.squeeze(last_hidden_state, dim=0)  
    #     last_hidden_state = last_hidden_state.to(torch.float32)  

    #     lab = kmeans_model(last_hidden_state).tolist()  
    #     print(len(lab))
    #     print(lab)
    
    # # audio_extractor = AudioTokenExtractor_16k(AudioTokenConfig_16k())

    # # encodec 16k chinese
    # model1 = SoundStream(
    #         n_filters=32,
    #         D=512,
    #         ratios=[8, 5, 4, 2],
    #         sample_rate=16000,
    #         target_bandwidths=[1, 1.5, 2, 4, 6, 12])

    # parameter_dict = torch.load("/mnt/users/jiazhijun/data/encodec_16k_320d.pth")
    # new_state_dict = OrderedDict()
    # # k 为 module.xxx.weight, v 为权重
    # for k, v in parameter_dict.items():
    #     # 截取`module.`后面的xxx.weight
    #     name = k[7:]
    #     new_state_dict[name] = v
    # model1.load_state_dict(new_state_dict)  # load model
    # # remove_encodec_weight_norm(model1)
    # import librosa
    # wav, sr = librosa.load(wav_path, sr=16000)
    # wav = torch.tensor(wav).unsqueeze(0)
    # wav = wav.unsqueeze(1)
    # # wav = wav[:, :, :lenn]
    # print(wav.shape)
    # # print(wav.shape)
    # # codes_raw = model.encode(wav)
    # # print(codes_raw[0][0])
    # codes_raw1 = model1.encode(wav)
    # codes_raw1 = codes_raw1[:8]
    # print(codes_raw1.shape)
    # # out = model1.decode(codes_raw1)

    # # out = out.detach().cpu().squeeze(0)
    # # save_audio(wav=out, path="/dev_huaying/zhijun/valle_23_4_22/valle/data/test1.wav", sample_rate=16000, rescale=True)
    # print('finish decompressing')
    
