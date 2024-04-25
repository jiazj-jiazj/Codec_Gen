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
    python3 bin/infer.py \
        --decoder-dim 128 --nhead 4 --num-decoder-layers 4 --model-name valle \
        --text-prompts "Go to her." \
        --audio-prompts ./prompts/61_70970_000007_000001.wav \
        --output-dir infer/demo_valle_epoch20 \
        --checkpoint exp/valle_nano_v2/epoch-20.pt

"""
import os  
import time  
import librosa  
import soundfile as sf
import sys
import os
from datetime import datetime
current_time = datetime.now()
current_working_directory = os.getcwd()  
print("Current working directory:", current_working_directory)  
sys.path.append(current_working_directory)
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
import librosa
import argparse
import logging
import os
from pathlib import Path
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import soundfile as sf

import torch
import torchaudio
torch.backends.cudnn.enabled = False  

from icefall.utils import str2bool
torch.backends.cudnn.enabled = False  
from valle.data import (
    AudioTokenConfig,
    AudioTokenExtractor,
    AudioTokenConfig_16k,
    AudioTokenExtractor_16k, 
    TextTokenizer,
    tokenize_text,
    ApplyKmeans,
    HubertFeatureReader
)
from valle.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
    AudioTokenizer_encodec_16k_tfcodec
)
from valle.data.collation import get_text_token_collater
from valle.models import add_model_arguments, get_model
import shutil

def process_audio(semantic_prompts_file):  
    # Load the audio file  
    audio, sr = librosa.load(semantic_prompts_file, sr=None)  
  
    # Check if the duration is greater than or equal to 3 seconds  
    duration = librosa.get_duration(audio, sr)  
    if duration < 3:  
        print("Audio length is less than 3 seconds. Skipping.")  
        return semantic_prompts_file  
  
    # Trim the audio to the first 3 seconds  
    audio_trimmed = audio[:int(3 * sr)]  
  
    # Save the trimmed audio to the same folder as the input file with a timestamp  
    original_filename, original_extension = os.path.splitext(os.path.basename(semantic_prompts_file))  
    output_filename = f"{original_filename}_{time.strftime('%Y%m%d_%H%M%S')}{original_extension}"  
    output_path = os.path.join(os.path.dirname(semantic_prompts_file), output_filename)  
    sf.write(output_path, audio_trimmed, sr)  
  
    print(f"Trimmed audio saved to: {output_path}")  
    return output_path  

def longest_common_subsequence(seq1, seq2):  
    m, n = len(seq1), len(seq2)  
    dp = [[0] * (n + 1) for _ in range(m + 1)]  
  
    for i in range(1, m + 1):  
        for j in range(1, n + 1):  
            if seq1[i - 1] == seq2[j - 1]:  
                dp[i][j] = dp[i - 1][j - 1] + 1  
            else:  
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  
  
    lcs = []  
    i, j = m, n  
    while i > 0 and j > 0:  
        if seq1[i - 1] == seq2[j - 1]:  
            lcs.append(seq1[i - 1])  
            i -= 1  
            j -= 1  
        elif dp[i - 1][j] > dp[i][j - 1]:  
            i -= 1  
        else:  
            j -= 1  
  
    return lcs[::-1]  

def depup(semantic_token):
    unique_tokens = []  
    for token in semantic_token:  
        if unique_tokens==[] or token != unique_tokens[-1]:  
            unique_tokens.append(token)
    return unique_tokens

hubert_km_path = "/home/v-zhijunjia/data/valle-tensorboard-models/other_models/hubert/hubert_base_ls960_L9_km500.bin"
hubert_path = "/home/v-zhijunjia/data/valle-tensorboard-models/other_models/hubert/hubert_base_ls960.pt"


def best_max_common_in_group(group):
    best_common = 0  
    gt_text = group[0][1]
    best_path = ""
    for gen_ in group[1:]: 

        gen_file_path = gen_[0]
        gen_text = gen_[1]
        common_len = len(longest_common_subsequence(gen_text, gt_text))/min(len(gen_text), len(gt_text))
       
        if common_len > best_common: 
            best_path =  gen_file_path
            best_common = common_len  
  
    return best_common, best_path

import re
def main():

    checkpoint = 80
    ckpt_path = hubert_path
    layer = 9
    km_path = hubert_km_path
    reader = HubertFeatureReader(ckpt_path, layer)
    apply_kmeans = ApplyKmeans(km_path)    
    while True:
        print(f"checkpoint :{checkpoint}")
        input_wav_path = f"/home/v-zhijunjia/data/test_accent_baseline_paper/vctk20cases_ac_encodec"
        gt_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_ac100cases_chongchongchong/bdl"
        best_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_ac100cases_chongchongchong/ac_encodec_best"

        os.makedirs(best_dir, exist_ok=True)
        grouped_files = {}  
        
        def get_semantic_tokens(input_file_path):
            semantic_sys_feats = reader.get_feats(input_file_path)
            semantic_sys = apply_kmeans(semantic_sys_feats).tolist()
            return semantic_sys


        for root, dirs, files in os.walk(input_wav_path):  
            for file in files:  
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    semantic = get_semantic_tokens(file_path)
                    depup_semantic = depup(semantic)
                    l1 = len(depup_semantic)
                    pattern = r"([A-Z]+_arctic_b\d{4})"  
                    speaker_name = re.search(pattern, file).group(1)

                    pattern1 = r"(arctic_b\d{4})"  
                    gt_file_name = re.search(pattern1, speaker_name).group(1)  
                    gt_file = gt_file_name+ ".wav"
                    gt_file_path = os.path.join(gt_dir, gt_file)
                    gt_semantic = get_semantic_tokens(gt_file_path)
                    depup_gt_semantic = depup(gt_semantic)

                    l2 = len(depup_gt_semantic)

                    base_name = speaker_name+".wav"


                    if base_name not in grouped_files:  
                        grouped_files[base_name] = []
                        grouped_files[base_name].append([gt_file_path, depup_gt_semantic])
                    grouped_files[base_name].append([file_path, depup_semantic]) 

        total_ratio = 0
        case_num = 0

        for group in grouped_files.values():  
            best_result, best_file_path = best_max_common_in_group(group) 
            # print(best_file_path)
            # print(best_dir)
            # quit()
            shutil.copy(best_file_path, best_dir)
            total_ratio +=best_result
            case_num+=1

        print(case_num)
        print(f"average is {total_ratio/case_num}")
        checkpoint+=20

        break




if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
