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
import yaml
import math
import torch
import numpy as np
import os  
import time  
import librosa  
import soundfile as sf
import sys
import json
import os
from datetime import datetime
current_time = datetime.now()
current_working_directory = os.getcwd()
from tqdm import tqdm
print("Current working directory:", current_working_directory)  
sys.path.append(current_working_directory)
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
from torch.nn.utils.rnn import pad_sequence
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
from tfnet_semantic_token.semantic_token_tfcodec.tfnet_vqvae_lmencoder2 import TFNet_lmencoder as TFCodec_lmencoder
from tfnet_semantic_token.audlib.audiolib import audioread, audiowrite, audioread_resample
from icefall.utils import str2bool
# torch.backends.cudnn.enabled = False  
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
import torch.nn as nn  
from thop import profile

class ModelWrapper(nn.Module):  
    def __init__(self, model):  
        super(ModelWrapper, self).__init__()  
        self.model = model  
  
    def forward(self, semantic_tokens, semantic_tokens_lens, audio_prompts, enroll_x_lens, top_k, temperature, mode):  
        return self.model.inference_only_ar(  
            semantic_tokens,  
            semantic_tokens_lens,  
            audio_prompts,  
            enroll_x_lens=enroll_x_lens,  
            top_k=top_k,  
            top_p=top_p, 
            temperature=temperature,  
            mode=mode  
        )  

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-prompts-dir",
        type=str,
        help="Audio prompts which are separated by | and should be aligned with --text-prompts.",
    )
    parser.add_argument(
        "--nums",
        type=int,
        default=1,
        help="Number of converted ",
    )
    parser.add_argument(
        "--nums-stage2",
        type=int,
        default=1,
        help="Number of converted ",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="To get up and running quickly just follow the steps below.",
        help="Text to be synthesized.",
    )
    # model
    add_model_arguments(parser)

    parser.add_argument(
        "--semantic-tokens",
        type=str,
        default="data/tokenized/unique_text_tokens.k2symbols",
        help="Path to the unique text tokens file.",
    )
    parser.add_argument(
        "--txt-sem-path",
        type=str,
        default="",
        help="Path to the sem of txt.",
    )
    parser.add_argument(
        "--text-tokens",
        type=str,
        default="data/tokenized/unique_text_tokens.k2symbols",
        help="Path to the unique text tokens file.",
    )
    parser.add_argument(
        "--outputdir-name",
        type=str,
        default="converted_vc",
        help="Path of outputdir-name.",
    )
    parser.add_argument(
        "--input-semantic",
        type=str2bool,
        help="input-semantic.",
    )
    parser.add_argument(
        "--known-token-update",
        type=str2bool,
        default="false",
        help="input-semantic.",
    )
    parser.add_argument(
        "--sem-read",
        type=str2bool,
        default="True",
    )
    parser.add_argument(
        "--accent-remove",
        type=str2bool,
        default="False",
        help="accent-remove.",
    )
    parser.add_argument(
        "--pair-infer",
        type=str2bool,
        default="False",
        help="pair-infer.",
    )
    parser.add_argument(
        "--semantic-sys-dir",
        type=str,
        help="semantic-read",
    )
    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )

    parser.add_argument(
        "--checkpoint1",
        type=str,
        default="exp/vallf_nano_full/checkpoint-100000.pt",
        help="Path to the saved checkpoint.",
    )
    parser.add_argument(
        "--checkpoint2",
        type=str,
        default="exp/vallf_nano_full/checkpoint-100000.pt",
        help="Path to the saved checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-NAR",
        type=str,
        default="exp/vallf_nano_full/checkpoint-100000.pt",
        help="Path to the saved checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infer/demo"),
        help="Path to the tokenized files.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )
    parser.add_argument(
        "--top-k-know-token",
        type=int,
        default=None,
        help="topk of known token update.",
    )
    parser.add_argument(
        "--top-k-know-token-stage2",
        type=int,
        default=None,
        help="topk of known token update.",
    )
    parser.add_argument(
        "--top-k-stage2",
        type=int,
        default=-100,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )
    parser.add_argument(
        "--top-p-stage2",
        type=float,
        default=1.0,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature of AR Decoder top_k sampling.",
    )
    parser.add_argument(
        "--temperature-stage2",
        type=float,
        default=1.0,
        help="The temperature of AR Decoder top_k sampling.",
    )
    parser.add_argument(
        "--only-autoregressive",
        type=str2bool,
        default=False,
        help="ongly autoregressive, means tfcodec",
    )
    parser.add_argument(
        "--continual",
        type=str2bool,
        default=False,
        help="Do continual task.",
    )
    parser.add_argument(
        "--shared-linear",
        type=str2bool,
        default=False,
        help="ongly autoregressive, means tfcodec",
    )
    parser.add_argument(
        "--shared-linear-stage2",
        type=str2bool,
        default=False,
        help="ongly autoregressive, means tfcodec",
    )
    parser.add_argument(
        "--semantic-depup",
        type=str2bool,
        default=False,
        help="ongly autoregressive, means tfcodec",
    )

    parser.add_argument(
        "--input-language",
        type=int,
        default="0",
        help="0->english, 1->chinese",
    )
    parser.add_argument(
        "--target-mode",
        type=int,
        default="0",
        help="0 one-stage 2 two-stage",
    )
    parser.add_argument(
        "--input-codec",
        type=int,
        default="0",
        help="0->encodec, 1->tfcodec",
    )
    parser.add_argument(
        "--semantic-layer",
        type=int,
        default=9,
        help="suffix of the manifest file",
    )
    parser.add_argument(
        "--num-decoder-layers-stage2",
        type=int,
        default=12,
        help="suffix of the manifest file",
    )
    parser.add_argument(
        "--decoder-dim-stage2",
        type=int,
        default=1024,
        help="suffix of the manifest file",
    )
    parser.add_argument(
        "--nhead-stage2",
        type=int,
        default=16,
        help="suffix of the manifest file",
    )
    
    parser.add_argument(
        "--num-quantizers-stage2",
        type=int,
        default=16,
        help="suffix of the manifest file",
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="0->top-infer, 1->arg_max",
    )
    parser.add_argument(
        "--soundstorm-type",
        type=int,
        default=0,
        help="0->baseline, 1->group_ar",
    )
    parser.add_argument(
        "--mode-stage2",
        type=int,
        default=0,
        help="0->top-infer, 1->arg_max",
    )
    parser.add_argument(
        "--prepend-bos-stage2",
        type=str2bool,
        default='false',
    )
    parser.add_argument(
        "--txt2semantic-need-prompt",
        type=str2bool,
        default='false',
    )
    parser.add_argument(
        "--is-pretrain",
        type=str2bool,
        default='false',
        help="input-semantic.",
    )
    parser.add_argument(
        "--test-benchmark",
        type=str2bool,
        default='false',
        help="whether test-benchmark.",
    )
    parser.add_argument(
        "--encoder-num-layers",
        type=int,
        default=6
    )
    parser.add_argument(
        "--decoder-num-layers",
        type=int,
        default=6
    )
    parser.add_argument(
        "--pret-mode",
        type=int,
        default=0,
        help="0,1,2,3,4,5,6",
    )
    parser.add_argument(
        "--soundstorm-steps",
        type=int,
        default=16,
        help="0,1,2,3,4,5,6",
    )
    parser.add_argument(
        "--hubert-path",
        type=str,
        default="/home/v-zhijunjia/data/valle-tensorboard-models/other_models/hubert/hubert_base_ls960.pt",
        help="Path to the saved checkpoint.",
    )
    parser.add_argument(
        "--hubert-km-path",
        type=str,
        default="/home/v-zhijunjia/data/valle-tensorboard-models/other_models/hubert/hubert_base_ls960_L9_km500.bin",
        help="Path to the saved checkpoint.",
    )
    parser.add_argument(
        "--tfnet-ckpt",
        type=str,
        default="/home/v-zhijunjia/data/valle-tensorboard-models/other_models/tfcodec/890hrs16khz_tfnet_v2i_vqvae_20msVQ_hop5_combine4_rd1_6kbps/tfnet_v2i_vqvae_combineVQ-iter-514000-val1-0.348327-entropy-118.968-mel-0.067548-recon-0.012822.ckpt",
        help="Path to the saved checkpoint.",
    )
    parser.add_argument(
        "--dir-need2test",
        type=str,
        default="/home/v-zhijunjia/data/valle-tensorboard-models/other_models/tfcodec/890hrs16khz_tfnet_v2i_vqvae_20msVQ_hop5_combine4_rd1_6kbps/tfnet_v2i_vqvae_combineVQ-iter-514000-val1-0.348327-entropy-118.968-mel-0.067548-recon-0.012822.ckpt",
        help="tts test benchmark.",
    )
    parser.add_argument(
        "--prompt-pre-cut",
        type=str2bool,
        default="false",
        help="whether to trimm pre 3s",
    )
    parser.add_argument(
        "--prompt-cut-seconds",
        type=int,
        default=3,
        help = "3 ,5, 7"
    )
    parser.add_argument(
        "--semantic-type",
        type=int,
        default=0,
        help = "0->hubert, 1->tfnet_256bps"
    )
    parser.add_argument('--sem-tfcodec-config', default="tfnet_semantic_token/semantic_token_tfcodec/config_inference_1.yaml")
    parser.add_argument(
        "--task-id",
        type=int,
        default=0
    )
    parser.add_argument(
        "--parrallel-mode",
        type=int,
        default=0,
        help="path of semantic-tokens",
    )
    return parser.parse_args()



def pad_list_seq(tokens, padding_token=500):

    update_tokens =[torch.tensor(token) for token in tokens]
    original_lengths = torch.tensor([len(token) for token in tokens], dtype=torch.int32)
    padded_output = pad_sequence(update_tokens, batch_first=True, padding_value=padding_token)  

    return padded_output, original_lengths
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

            # model.load_state_dict(new_dict, strict=False)
    new_model_dict = model.state_dict()
    
    print('current model keys:{}'.format(len(new_model_dict.keys())))
    # print(new_model_dict.keys())
    print(f'current checkpoint keys:{len(new_dict.keys())}')
    # filter out unnecessary keys
    new_dict_opt = {k: v for k, v in new_dict.items() if k in new_model_dict}
    print('keys loaded :{}'.format(len(new_dict_opt.keys())))
    new_model_dict.update(new_dict_opt)
    model.load_state_dict(new_model_dict, strict=False)  # , strict=False)
    model.eval()

def sem_tfnet_load(tfcodec_lmencoder, audio_file, device):
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

def process_audio(acoustic_prompts_file, cut_seconds):  
    # Load the audio file  
    audio, sr = librosa.load(acoustic_prompts_file, sr=None)  
  
    # Check if the duration is greater than or equal to 3 seconds  
    duration = librosa.get_duration(audio, sr)
    if duration < cut_seconds:  
        print("Audio length is less than 3 seconds. Skipping.")  
        return acoustic_prompts_file  
  
    # Trim the audio to the first 3 seconds  
    audio_trimmed = audio[:int(cut_seconds * sr)]  
  
    # Save the trimmed audio to the same folder as the input file with a timestamp  
    original_filename, original_extension = os.path.splitext(os.path.basename(acoustic_prompts_file))  
    output_filename = f"{original_filename}_{time.strftime('%Y%m%d_%H%M%S')}{original_extension}"  
    temp_dir = os.path.dirname(acoustic_prompts_file) + "_temp"
    os.makedirs(temp_dir, exist_ok=True)
    output_path = os.path.join(temp_dir, output_filename)  
    sf.write(output_path, audio_trimmed, sr)  
  
    print(f"Trimmed audio saved to: {output_path}")  
    return output_path  


from collections import Counter  

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

def mark_and_count_lcs_tokens(a, lcs_ab):  
    lcs_index = 0  
    lcs_token_counts = [0 for xx in range(len(lcs_ab))]

    ll = len(a)
    i=0
    while True:
        if i==ll:
            break
        if lcs_index < len(lcs_ab) and a[i] == lcs_ab[lcs_index]:  
            lcs_token_counts[lcs_index] += 1  
            i+=1
            while i<ll and a[i]==lcs_ab[lcs_index] :
                lcs_token_counts[lcs_index] += 1 
                i+=1
            lcs_index += 1

            i-=1   
        i+=1
    return lcs_token_counts 


def del_lcs_update_token(a, lcs_ab, lcs_token_counts_a, lcs_token_count_b, update_nums):

    lcs_index = 0  
    lcs_token_counts = [0 for xx in range(len(lcs_ab))]
    updated_tokens = []
    ll = len(a)
    i=0
    while True:
        if i==ll:
            break
        if lcs_index < len(lcs_ab) and a[i] == lcs_ab[lcs_index]:  
            if lcs_token_counts_a[lcs_index] > lcs_token_count_b[lcs_index]:
                can_update_nums = lcs_token_counts_a[lcs_index]- lcs_token_count_b[lcs_index]
                if can_update_nums <= update_nums:
                    nums = lcs_token_count_b[lcs_index]
                    update_nums-=can_update_nums
                else:
                    nums = lcs_token_counts_a[lcs_index]
            else:
                nums = lcs_token_counts_a[lcs_index]

            updated_tokens+=[lcs_ab[lcs_index]]*nums
            i+=1
            while i<ll and a[i]==lcs_ab[lcs_index] :
                i+=1
            lcs_index += 1
            i-=1   
        else:
            updated_tokens+=[a[i]]
        i+=1
    return updated_tokens, update_nums


def del_non_lcs_update_token(a, lcs_ab, non_lcs_token_counts_a, update_nums):

    lcs_index = 0  
    lcs_token_counts = [0 for xx in range(len(lcs_ab))]
    non_lcs_token_counts_a = subtract_from_max_elements(non_lcs_token_counts_a[:], update_nums)  # 使用切片创建一个副本，以免修改原始序列

    updated_tokens = []
    ll = len(a)
    i=0
    while True:
        if i==ll:
            break
        if lcs_index < len(lcs_ab) and a[i] == lcs_ab[lcs_index]:  
            nums = non_lcs_token_counts_a[lcs_index]

            updated_tokens+=[lcs_ab[lcs_index]]*nums
            i+=1
            while i<ll and a[i]==lcs_ab[lcs_index] :
                i+=1
            lcs_index += 1
            i-=1   
        else:
            updated_tokens+=[a[i]]
        i+=1
    return updated_tokens
def get_non_lcs_tokens(a, lcs_ab):  
    lcs_index = 0  
    non_lcs_tokens = []
    updated_tokens = []
    ll = len(a)
    i=0
    while True:
        if i==ll:
            break
        if lcs_index < len(lcs_ab) and a[i] == lcs_ab[lcs_index]:  
            i+=1
            while i<ll and a[i]==lcs_ab[lcs_index]:
                i+=1
            lcs_index += 1

            i-=1   
        else:
            non_lcs_tokens+=[a[i]]
        i+=1
    
    lcs_index = 0  
    depup_non_lcs_tokens = depup(non_lcs_tokens)

    non_lcs_token_counts = [0 for xx in range(len(depup_non_lcs_tokens))]

    index = 0 
    ll = len(non_lcs_tokens)
    i=0
    while True:
        if i==ll:
            break
        if index < len(depup_non_lcs_tokens) and non_lcs_tokens[i] == depup_non_lcs_tokens[index]:
            i+=1
            non_lcs_token_counts[index]+=1

            while i<ll and non_lcs_tokens[i]==depup_non_lcs_tokens[index] :
                non_lcs_token_counts[index]+=1
                i+=1
            index += 1
            i-=1   
        i+=1


    return depup_non_lcs_tokens, non_lcs_token_counts

from heapq import heapify, heappop, heappush  
def subtract_from_max_elements(sequence, total_to_subtract):
    # 创建一个索引堆，以便知道哪个元素被减去
    index_heap = [(-val, i) for i, val in enumerate(sequence)]
    heapify(index_heap)  # 建立最大堆

    # 从最大的数字开始逐一减去1
    for _ in range(total_to_subtract):
        if not index_heap:
            break  # 如果堆为空，则停止
        # 弹出最大的数字
        max_val, max_index = heappop(index_heap)
        if sequence[max_index] > 0:  # 如果该数字已经是0，则不再减去
            sequence[max_index] -= 1  # 减去1
        if sequence[max_index] > 0:  # 如果减去1后大于0，则放回堆中
            heappush(index_heap, (-sequence[max_index], max_index))

    return sequence


def get_a_larger_b(a, b):

    lcs_ab = longest_common_subsequence(a, b)  
    lcs_ab_depup = depup(lcs_ab)
    
    # 标记序列a中属于最大公共子序列的token，并统计数量  
    lcs_token_counts_a = mark_and_count_lcs_tokens(a, lcs_ab_depup)  

    lcs_token_count_b = mark_and_count_lcs_tokens(b, lcs_ab_depup)  

    updated_tokens, update_nums = del_lcs_update_token(a, lcs_ab_depup, lcs_token_counts_a, lcs_token_count_b, len(a)-len(b))

    non_lcs_tokens_a, non_lcs_token_counts_a = get_non_lcs_tokens(a, lcs_ab_depup)

    final_tokens = del_non_lcs_update_token(updated_tokens, non_lcs_tokens_a, non_lcs_token_counts_a, update_nums)
    return final_tokens


def find_related_files(path, gen_txt_file):  
    file_without_ext, _ = os.path.splitext(gen_txt_file)  
    name = file_without_ext[4:]  
    
    prompt_txt_file = f"prompt_{name}.txt"  
    prompt_wav_file = f"prompt_{name}.wav"
    prompt_flac_file = f"prompt_{name}.flac"
    gt_wav_file = f"gt_{name}.wav"
    gt_flac_file = f"gt_{name}.flac"
    
    related_files = []  

    for root, _, files in os.walk(path):  
        if prompt_txt_file in files:  
            related_files.append(os.path.join(root, prompt_txt_file))  
        if prompt_flac_file in files:  
            related_files.append(os.path.join(root, prompt_flac_file))  
        if gt_flac_file in files:  
            related_files.append(os.path.join(root, gt_flac_file))  
        if prompt_wav_file in files:  
            related_files.append(os.path.join(root, prompt_wav_file))  
        if gt_wav_file in files:  
            related_files.append(os.path.join(root, gt_wav_file)) 
    return related_files  

def find_gen_files_and_related_files(path, prefix, ext):  
    all_files = []
    for root, dirs, files in os.walk(path):  
        if "temp" in str(root):
            print(f"root is ignored: {root}")
            continue
        for file in files:  
            if file.startswith(prefix) and file.endswith(ext):  
                gen_txt_file = os.path.join(root, file)  
                related_files = find_related_files(root, file)
                # print(related_files)  
                related_files.append(os.path.join(root,file))
                all_files.append(related_files)
    return all_files
def find_prompt_files(folder_prompt):
    all_files = []
    for root, dirs, files in os.walk(folder_prompt):  
        for file in files:  
            if file.endswith(".wav") or file.endswith(".flac"):  
                prompt_wav_file = os.path.join(root, file)
                prompt_txt_file = os.path.join(root, file.replace(".flac", ".txt").replace(".wav", ".txt"))
                related_files = []
                # print(related_files)
                if os.path.exists(prompt_txt_file):
                    related_files.append(prompt_txt_file)
                if os.path.exists(prompt_wav_file):
                    related_files.append(prompt_wav_file)
                all_files.append(related_files)
    return all_files
def find_txt_files(folder_txt):
    all_files = []
    for root, dirs, files in os.walk(folder_txt):  
        for file in files:  
            if file.endswith(".txt"):  
                gen_txt_file = os.path.join(root, file)  
                all_files.append(gen_txt_file)
    return all_files

def get_audio_files(folder_path):  
    audio_extensions = ('.flac', '.wav')  
    audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(audio_extensions)]  
    return audio_files

def init_Chinese_hubert(args, device):
    if args.semantic_layer==9:
        km_path = "chinese_speech_pretrain/hubert_kmeans/hubert_base_iter2_32gpu_l9/model.mdl"
    elif args.semantic_layer==6:
        km_path = "chinese_speech_pretrain/hubert_kmeans/hubert_base_iter1_32gpu_l6/model.mdl"

    model_path="TencentGameMate/chinese-hubert-base"

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    hubert_model = HubertModel.from_pretrained(model_path)
    # print(model)
    hubert_model = hubert_model.to(device)
    hubert_model = hubert_model.half()
    hubert_model.eval()
    apply_kmeans = ApplyKmeans(km_path)

    return feature_extractor, hubert_model, apply_kmeans

def init_English_HuBert(args):  
    ckpt_path = args.hubert_path
    layer = 9
    km_path = args.hubert_km_path
    reader = HubertFeatureReader(ckpt_path, layer)
    apply_kmeans = ApplyKmeans(km_path)
    return reader, apply_kmeans

def init_English_TF_Codec(args):
    with open(args.sem_tfcodec_config) as f:
        config = yaml.safe_load(f)
    tfcodec_lmencoder = TFCodec_lmencoder(config)
    tfcodec_lmencoder = tfcodec_lmencoder.to(device)
    ckpt_path = "/home/v-zhijunjia/data/valle-tensorboard-models/other_models/tfnet_semantic_tokens/semantic_token_resynt/tfcodec_256bps_disen/tfnetv2_vqvae_lm2-val-1-loss-4.646021-vq-0.199693-iter-716000.ckpt"
    load_pretrained_TFCodec(tfcodec_lmencoder, ckpt_path, device)
    return tfcodec_lmencoder

def extract_chinese_HuBert(args, file_path, hubert_model, apply_kmeans):
    wav, sr = sf.read(file_path)
    target_sr = 16000 
    if sr != target_sr:
        wav = librosa.resample(wav, sr, target_sr)  

    input_values = feature_extractor(wav, return_tensors="pt").input_values
    input_values = input_values.half()
    input_values = input_values.to(device)
    with torch.no_grad():
        outputs = hubert_model(input_values, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[args.semantic_layer]
        last_hidden_state = torch.squeeze(last_hidden_state, dim=0)  
        last_hidden_state = last_hidden_state.to(torch.float32)  
        semantic_sys = apply_kmeans(last_hidden_state).tolist()  
    return semantic_sys

@torch.no_grad()
def main():
    args = get_args()

    semantic_dict = {}
    if args.task_id ==1:
        text_tokenizer = TextTokenizer(backend=args.text_extractor)
        text_collater = get_text_token_collater(args.text_tokens)
    
    semantic_token_collater = get_text_token_collater(args.semantic_tokens)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    if args.semantic_type==1: # Tf-Codec semantic tokens
        args.semantic_num_quantizers=2

    if args.input_language==1:
        feature_extractor, hubert_model, apply_kmeans = init_Chinese_hubert(args, device)
    elif args.input_language==0:
        if args.semantic_type == 1: # input is tfnet_semantic tokens
            tfcodec_lmencoder = init_English_TF_Codec(args)
        elif args.semantic_type == 0: # input is semantic tokens
            reader, apply_kmeans = init_English_HuBert(args)

    if args.input_codec ==0:
        audio_tokenizer = AudioTokenizer()
    else:
        audio_tokenizer = AudioTokenizer_encodec_16k_tfcodec(tfnet_ckpt=args.tfnet_ckpt)
        
    if args.target_mode!=0:  # if correct phrase is true, only_autoregressive is True, first model is correct else: generative model
        bef_only = args.only_autoregressive
        args.only_autoregressive=True  

    model = get_model(args)   # correct phrase or ar phrase
    if args.target_mode!=0: 
        args.only_autoregressive = bef_only 
    if args.target_mode == 2:  # load_generative model correct + generative  

# Store original values in temporary variables  
        original_num_decoder_layers = args.num_decoder_layers  
        original_num_quantizers = args.num_quantizers  
        original_decoder_dim = args.decoder_dim  
        original_nhead = args.nhead  
        original_top_k = args.top_k  
        original_top_p = args.top_p 
        original_shared_linear = args.shared_linear  
        orginal_prepend_bos = args.prepend_bos       
        original_model_name = args.model_name

        # Set the values for stage2  
        args.num_decoder_layers = args.num_decoder_layers_stage2  # correct + generative, generative model: stage2 parameters
        args.num_quantizers = args.num_quantizers_stage2
        args.model_name = args.model_name_stage2
        args.decoder_dim = args.decoder_dim_stage2  
        args.nhead = args.nhead_stage2  
        args.top_k = args.top_k_stage2  
        args.top_p = args.top_p_stage2 
        args.shared_linear = args.shared_linear_stage2  
        args.prepend_bos = args.prepend_bos_stage2

        # Get the model_acoustic  
        model_acoustic = get_model(args)   # stage2 is generative
        
        # Restore the original values from the temporary variables  
        args.num_decoder_layers = original_num_decoder_layers  
        args.num_quantizers = original_num_quantizers  
        args.decoder_dim = original_decoder_dim  
        args.nhead = original_nhead  
        args.top_k = original_top_k
        args.top_p = original_top_p
        args.shared_linear = original_shared_linear   
        args.model_name = original_model_name
        args.prepend_bos = orginal_prepend_bos    

        if args.only_autoregressive:   # generative model is only checkpoint2 
            checkpoint2 = torch.load(args.checkpoint2, map_location=device)
            with open('model2_keys_output.txt', 'w') as f:  
                for key in model_acoustic.state_dict().keys():  
                    f.write(key + '\n') 
            with open('keys_output_checkpoint2.txt', 'w') as f:  
                for key in checkpoint2["model"].keys():  
                    f.write(key + '\n')  
            
            missing_keys1, unexpected_keys1 = model_acoustic.load_state_dict(
                checkpoint2["model"], strict=False
            )
            assert not missing_keys1

            model_acoustic.to(device)
            model_acoustic.eval()
        else:                            # generative model is checkpoint2 and nar
            checkpoint1 = torch.load(args.checkpoint2, map_location=device)
            checkpoint3 = torch.load(args.checkpoint_NAR, map_location=device)
            # # 将checkpoint2的键写入到另一个txt文件中  
            # with open('keys_output_checkpoint2.txt', 'w') as f:  
            #     for key in checkpoint2["model"].keys():  
            #         f.write(key + '\n')  
            missing_keys1, unexpected_keys1 = model_acoustic.load_state_dict(
                checkpoint1["model"], strict=False
            )
            assert not missing_keys1
            
            missing_keys3, unexpected_keys2 = model_acoustic.load_state_dict(
                checkpoint3["model"], strict=False
            )
            assert not missing_keys3
            for key in list(checkpoint1['model'].keys()):  
                if key.startswith('nar'):  
                    # 在 checkpoint2 中查找相应的以 'nar' 开头的属性  
                    if key in checkpoint3['model']:  
                        print(key)
                        # 将 checkpoint1 中的 'nar' 开头属性替换为 checkpoint2 中对应的 'nar' 开头属性  
                        checkpoint1['model'][key] = checkpoint3['model'][key]  
                    else:  
                        print(f"未找到与 {key} 对应的属性。")

            missing_keys1, unexpected_keys1 = model_acoustic.load_state_dict(
                checkpoint1["model"], strict=False
            )
            assert not missing_keys1
            model_acoustic.to(device)
            model_acoustic.eval()
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # load (1)convert, (2)txt->semantic module (3)one-stage only-ar generative module
    if args.target_mode!=0 or (args.target_mode==0 and args.only_autoregressive is True): # sec vc encodec 
        checkpoint1 = torch.load(args.checkpoint1, map_location=device)  # if target_mode!=0, checkpoint1 is correct model. second: encodec generative model
        # checkpoint2 = torch.load(args.checkpoint2, map_location=device)
        with open('model_keys_output.txt', 'w') as f:  
            for key in model.state_dict().keys():  
                f.write(key + '\n') 
        with open('keys_output_checkpoint1.txt', 'w') as f:  
            for key in checkpoint1["model"].keys():  
                f.write(key + '\n')  
        args.only_autoregressive= True
        # args.shared_linear= False
        # # 将checkpoint2的键写入到另一个txt文件中  
        # with open('keys_output_checkpoint2.txt', 'w') as f:  
        #     for key in checkpoint2["model"].keys():  
        #         f.write(key + '\n')  
        missing_keys1, unexpected_keys1 = model.load_state_dict(
            checkpoint1["model"], strict=False
        )
        print(missing_keys1)
        # print(missing_keys1)
        assert not missing_keys1
        model.to(device)
        model.eval()
        # args.only_autoregressive= False

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # one-stage ar+nar generative module
    elif args.target_mode==0 and not args.only_autoregressive:

        checkpoint1 = torch.load(args.checkpoint1, map_location=device)
        checkpoint3 = torch.load(args.checkpoint_NAR, map_location=device)
        # # 将checkpoint2的键写入到另一个txt文件中  
        # with open('keys_output_checkpoint2.txt', 'w') as f:  
        #     for key in checkpoint2["model"].keys():  
        #         f.write(key + '\n')  
        missing_keys1, unexpected_keys1 = model.load_state_dict(
            checkpoint1["model"], strict=False
        )
        assert not missing_keys1
        
        missing_keys3, unexpected_keys2 = model.load_state_dict(
            checkpoint3["model"], strict=False
        )
        assert not missing_keys3
        for key in list(checkpoint1['model'].keys()):  
            if key.startswith('nar'):  
                # 在 checkpoint2 中查找相应的以 'nar' 开头的属性
                if key in checkpoint3['model']:  
                    print(key)
                    # 将 checkpoint1 中的 'nar' 开头属性替换为 checkpoint2 中对应的 'nar' 开头属性  
                    checkpoint1['model'][key] = checkpoint3['model'][key]  
                else:  
                    print(f"未找到与 {key} 对应的属性。")  

        missing_keys1, unexpected_keys1 = model.load_state_dict(
            checkpoint1["model"], strict=False
        )
        assert not missing_keys1
        model.to(device)
        model.eval()
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
 
    if args.task_id == 0: # vc ac
        if args.accent_remove is False:
            audio_files_A = get_audio_files(args.semantic_sys_dir)  
            audio_files_B = get_audio_files(args.audio_prompts_dir)

            if args.pair_infer is True:
                assert len(audio_files_A) ==len(audio_files_B)
                audio_files_A = [[b, a] for a, b in zip(audio_files_A, audio_files_B)] 
                audio_files_B = audio_files_A[:1]

        else:
            audio_files_A = get_audio_files(args.semantic_sys_dir)
            audio_files_B = audio_files_A[:1]


    elif args.task_id == 1: # tts 

        if args.test_benchmark is True:  #tts benchmark
            folder_A = args.dir_need2test  # 请将此路径替换为实际文件夹A的路径  
            all_files = find_gen_files_and_related_files(folder_A, "gen_", ".txt")
        elif args.test_benchmark is False:
            all_files = []
            folder_prompt = args.audio_prompts_dir
            folder_txt = args.semantic_sys_dir  # txt
            all_files_prompt = find_prompt_files(folder_prompt)
            all_files_txt = find_txt_files(folder_txt)

            print(f"prompt_files_len:{len(all_files_prompt)}")
            print(f"txt_files_len:{len(all_files_txt)}")
            i=0
            for file_txt in all_files_txt:
                for file_prompt_pair in all_files_prompt:
                    
                    promp_pairs = file_prompt_pair.copy()
                    promp_pairs.append(file_txt)
                    all_files.append(promp_pairs)

        # task-id 1 no need for audio_files_B
        audio_files_A = all_files
        audio_files_B = all_files[:1]

    iii = 0 
    cases_len = len(audio_files_A)
    for i, file_pair_or_single in tqdm(enumerate(audio_files_A), total=cases_len):  # tts: text file vc: semantic file
        print(f"processing {i}th semantic_sys file")
        if args.task_id==1:
            if len(file_pair_or_single) == 4:
                semantic_prompt_file, acoustic_prompts_file, gt_flac_file, semantic_sys_file = file_pair_or_single
            elif len(file_pair_or_single) == 3:  # tts text&text_prompt&speech_prompt 
                semantic_prompt_file, acoustic_prompts_file, semantic_sys_file = file_pair_or_single
            elif len(file_pair_or_single) ==2 :
                acoustic_prompts_file, semantic_sys_file = file_pair_or_single
        elif args.task_id==0: # vc
            print(f"file_pair_or_single:{file_pair_or_single}")
            if len(file_pair_or_single) ==2 :
                acoustic_prompts_file, semantic_sys_file = file_pair_or_single
            else:
                semantic_sys_file = file_pair_or_single
        
        for file_B in audio_files_B:  # tts, vc: acoustic prompt tts-stage2: semantic && acoustic prompt
            
            print(iii)
            iii+=1
            if args.task_id==0: # vc ac
                if args.accent_remove is True:
                    acoustic_prompts_file = semantic_sys_file # 自己做prompt
                else:
                    if args.pair_infer is False:
                        acoustic_prompts_file = file_B

            # if args.accent_remove is True:
            #     acoustic_prompts_file = semantic_sys_file
            bef_acoustic_prompts_file = acoustic_prompts_file

            if args.prompt_pre_cut  is True:
                acoustic_prompts_file = process_audio(acoustic_prompts_file, args.prompt_cut_seconds)

            if args.task_id == 0:  # vc 
                if args.input_language==1:
                    print(f"input_language:{args.input_language}")
                    semantic_prompts = extract_chinese_HuBert(args, acoustic_prompts_file, hubert_model, apply_kmeans)
                    semantic_sys = extract_chinese_HuBert(args, semantic_sys_file, hubert_model, apply_kmeans)

                elif args.input_language==0:
                    if args.semantic_type==1:
                        semantic_prompts = sem_tfnet_load(tfcodec_lmencoder, acoustic_prompts_file, device) # [[[x, y], [a, b]]]
                        semantic_sys = sem_tfnet_load(tfcodec_lmencoder, semantic_sys_file, device)

                    elif args.semantic_type==0:
                        semantic_prompts_feats = reader.get_feats(acoustic_prompts_file)
                        semantic_prompts = apply_kmeans(semantic_prompts_feats).tolist()
                        semantic_sys_feats = reader.get_feats(semantic_sys_file)
                        semantic_sys = apply_kmeans(semantic_sys_feats).tolist()

                        # semantic_sys = [17, 17, 17, 17, 296, 296, 392, 392, 184, 184, 392, 392, 127, 114, 0, 0, 0, 330, 94, 479, 331, 331, 284, 284, 460, 240, 175, 175, 81, 41, 324, 324, 324, 256, 325, 256, 256, 284, 319, 203, 53, 70, 65, 65, 242, 348, 94, 94, 199, 255, 255, 255, 99, 99, 338, 338, 395, 395, 106, 106, 153, 153, 153, 387, 387, 372, 396, 313, 24, 131, 404, 225, 414, 414, 80, 80, 80, 80, 82, 127, 119, 351, 278, 278, 203, 399, 70, 65, 256, 256, 256, 256, 285, 285, 256, 256, 256, 274, 186, 162, 54, 54, 224, 256, 256, 256, 256, 375, 375, 98, 98, 98, 13, 229, 82, 140, 187, 391, 82, 289, 73, 289, 140, 412, 287, 287, 111, 111, 111, 438, 438, 378, 43, 345, 389, 389, 389, 389, 389, 389, 122, 32, 32, 239, 445, 485, 278, 278, 278, 173, 173, 280, 280, 106, 113, 113, 113, 113, 113, 274, 413, 122, 314, 314, 198, 198, 22, 283, 283, 455, 455, 32, 32, 32, 32, 354, 470, 443, 443, 443, 443, 169, 169, 150, 86, 238, 6, 82, 131, 472, 472, 66, 66, 172, 172, 115, 273, 494, 494, 278, 251, 251, 219, 485, 485, 374, 374, 132, 132, 99, 99, 99, 436, 60, 60, 298, 298, 298, 195, 195, 195, 117, 76, 76, 36, 36, 377, 123, 123, 123, 216, 32, 198, 114, 258, 258, 258, 31, 54, 9, 142, 221, 336, 82, 74, 190, 380, 488, 488, 488, 481, 215, 215, 8, 29, 359, 359, 81, 459, 134, 203, 203, 381, 381, 381, 117, 404, 229, 82, 312, 187, 187, 391, 80, 140, 289, 82, 188, 188, 340, 340, 340, 67, 212, 212, 22, 283, 455, 455, 4, 4, 280, 280, 278, 278, 278, 310, 310, 395, 395, 180, 256, 256, 256, 330, 116, 10, 479, 331, 256, 256, 416, 416, 416, 192, 256, 256, 256, 256, 313, 314, 401, 401, 82, 108, 377, 494, 256, 399, 70, 65, 65, 65, 284, 481, 206, 206, 285, 467, 84, 84, 375, 375, 98, 98, 98, 421, 392, 392, 392, 392, 392, 392, 392, 392, 128, 193, 193, 17]
                        # semantic_sys = [17, 17, 127, 258, 258, 258, 258, 31, 39, 54, 232, 390, 390, 390, 390, 390, 390, 390, 18, 97, 483, 226, 226, 226, 20, 197, 373, 66, 66, 482, 482, 482, 482, 482, 482, 105, 105, 336, 144, 180, 106, 106, 284, 284, 306, 306, 306, 306, 306, 372, 372, 59, 59, 59, 59, 452, 452, 452, 263, 263, 225, 225, 226, 226, 226, 20, 20, 209, 125, 125, 125, 125, 348, 348, 250, 250, 70, 46, 46, 46, 46, 438, 422, 349, 205, 205, 261, 25, 343, 343, 343, 343, 343, 343, 343, 358, 358, 358, 39, 39, 39, 433, 433, 160, 160, 160, 160, 18, 112, 427, 56, 56, 56, 20, 312, 187, 187, 187, 187, 12, 12, 12, 12, 408, 408, 408, 408, 391, 391, 228, 289, 20, 209, 188, 356, 356, 281, 281, 453, 198, 198, 22, 22, 283, 455, 42, 456, 456, 456, 456, 456, 368, 453, 453, 168, 106, 106, 481, 481, 293, 122, 122, 36, 36, 449, 69, 69, 223, 130, 130, 198, 198, 127, 114, 92, 92, 92, 92, 167, 167, 457, 393, 393, 205, 205, 261, 25, 106, 481, 481, 481, 481, 182, 182, 182, 375, 375, 98, 98, 98, 13, 13, 13]
                        # semantic_sys = [17, 17, 345, 333, 220, 220, 35, 259, 127, 5, 455, 455, 458, 208, 208, 190, 487, 487, 499, 315, 315, 315, 450, 450, 413, 413, 348, 394, 90, 465, 144, 27, 27, 437, 437, 319, 319, 319, 319, 319, 319, 203, 203, 53, 10, 309, 479, 331, 307, 307, 307, 61, 285, 285, 350, 350, 350, 350, 350, 359, 81, 166, 324, 324, 422, 349, 164, 164, 164, 164, 106, 106, 153, 153, 153, 387, 372, 372, 372, 396, 396, 396, 388, 195, 195, 471, 49, 269, 142, 221, 336, 159, 159, 159, 159, 285, 14, 14, 411, 297, 297, 297, 297, 293, 293, 186, 162, 54, 224, 224, 494, 496, 274, 186, 162, 162, 232, 232, 172, 115, 273, 278, 330, 116, 199, 469, 469, 469, 31, 54, 86, 238, 6, 272, 300, 334, 334, 334, 59, 59, 59, 59, 452, 263, 263, 225, 225, 225, 80, 80, 20, 20, 74, 74, 425, 425, 425, 425, 425, 386, 386, 386, 431, 405, 405, 405, 405, 405, 206, 206, 167, 167, 233, 233, 270, 270, 270, 433, 390, 160, 160, 160, 97, 97, 483, 226, 20, 209, 89, 89, 446, 446, 33, 33, 394, 76, 310, 161, 161, 161, 161, 487, 487, 487, 288, 213, 213, 213, 318, 318, 368, 453, 342, 168, 275, 275, 275, 275, 303, 303, 303, 117, 404, 404, 13, 78, 20, 20]

# indictts_phase2_rajasthani_male_speaker1_english-train_rajasthanimale_04450-35178_0: [17, 296, 261, 25, 106, 481, 481, 481, 481, 293, 175, 81, 84, 84, 84, 16, 274, 274, 216, 216, 22, 283, 455, 455, 129, 259, 74, 437, 351, 311, 311, 311, 311, 460, 460, 169, 169, 164, 164, 164, 69, 462, 462, 130, 402, 239, 239, 384, 371, 371, 485, 374, 374, 374, 374, 325, 325, 41, 41, 41, 41, 246, 19, 19, 19, 454, 454, 229, 229, 82, 312, 187, 187, 408, 408, 391, 391, 228, 140, 140, 83, 83, 55, 55, 322, 67, 250, 250, 147, 147, 380, 380, 499, 153, 153, 424, 497, 497, 122, 219, 219, 219, 222, 222, 222, 222, 313, 313, 186, 162, 162, 232, 172, 172, 115, 273, 273, 151, 151, 215, 215, 215, 96, 310, 395, 395, 469, 469, 178, 178, 96, 270, 270, 86, 142, 393, 155, 155, 332, 332, 332, 332, 467, 44, 44, 44, 58, 72, 72, 72, 437, 319, 319, 319, 348, 64, 212, 161, 300, 382, 382, 313, 313, 314, 239, 219, 219, 219, 219, 286, 286, 286, 286, 286, 286, 286, 468, 59, 59, 304, 304, 304, 304, 185, 185, 185, 185, 269, 390, 390, 18, 112, 427, 56, 56, 56, 312, 312, 187, 292, 292, 23, 408, 408, 391, 391, 228, 140, 140, 320, 7, 473, 473, 476, 476, 476, 171, 301, 416, 239, 144, 27, 180, 91, 91, 91, 91, 91, 206, 206, 240, 240, 314, 32, 32, 32, 259, 354, 425, 386, 431, 431, 151, 151, 169, 150, 150, 54, 54, 86, 224, 219, 477, 477, 477, 477, 477, 477, 477, 132, 132, 132, 98, 98, 13, 13]
# indictts_phase2_rajasthani_male_speaker1_english-train_rajasthanimale_04473-35377_0: [17, 17, 127, 258, 258, 258, 258, 31, 39, 54, 232, 390, 390, 390, 390, 390, 390, 390, 18, 97, 483, 226, 226, 226, 20, 197, 373, 66, 66, 482, 482, 482, 482, 482, 482, 105, 105, 336, 144, 180, 106, 106, 284, 284, 306, 306, 306, 306, 306, 372, 372, 59, 59, 59, 59, 452, 452, 452, 263, 263, 225, 225, 226, 226, 226, 20, 20, 209, 125, 125, 125, 125, 348, 348, 250, 250, 70, 46, 46, 46, 46, 438, 422, 349, 205, 205, 261, 25, 343, 343, 343, 343, 343, 343, 343, 358, 358, 358, 39, 39, 39, 433, 433, 160, 160, 160, 160, 18, 112, 427, 56, 56, 56, 20, 312, 187, 187, 187, 187, 12, 12, 12, 12, 408, 408, 408, 408, 391, 391, 228, 289, 20, 209, 188, 356, 356, 281, 281, 453, 198, 198, 22, 22, 283, 455, 42, 456, 456, 456, 456, 456, 368, 453, 453, 168, 106, 106, 481, 481, 293, 122, 122, 36, 36, 449, 69, 69, 223, 130, 130, 198, 198, 127, 114, 92, 92, 92, 92, 167, 167, 457, 393, 393, 205, 205, 261, 25, 106, 481, 481, 481, 481, 182, 182, 182, 375, 375, 98, 98, 98, 13, 13, 13]
# indictts_phase2_rajasthani_male_speaker1_english-train_rajasthanimale_04558-36258_0: [17, 17, 345, 333, 220, 220, 35, 259, 127, 5, 455, 455, 458, 208, 208, 190, 487, 487, 499, 315, 315, 315, 450, 450, 413, 413, 348, 394, 90, 465, 144, 27, 27, 437, 437, 319, 319, 319, 319, 319, 319, 203, 203, 53, 10, 309, 479, 331, 307, 307, 307, 61, 285, 285, 350, 350, 350, 350, 350, 359, 81, 166, 324, 324, 422, 349, 164, 164, 164, 164, 106, 106, 153, 153, 153, 387, 372, 372, 372, 396, 396, 396, 388, 195, 195, 471, 49, 269, 142, 221, 336, 159, 159, 159, 159, 285, 14, 14, 411, 297, 297, 297, 297, 293, 293, 186, 162, 54, 224, 224, 494, 496, 274, 186, 162, 162, 232, 232, 172, 115, 273, 278, 330, 116, 199, 469, 469, 469, 31, 54, 86, 238, 6, 272, 300, 334, 334, 334, 59, 59, 59, 59, 452, 263, 263, 225, 225, 225, 80, 80, 20, 20, 74, 74, 425, 425, 425, 425, 425, 386, 386, 386, 431, 405, 405, 405, 405, 405, 206, 206, 167, 167, 233, 233, 270, 270, 270, 433, 390, 160, 160, 160, 97, 97, 483, 226, 20, 209, 89, 89, 446, 446, 33, 33, 394, 76, 310, 161, 161, 161, 161, 487, 487, 487, 288, 213, 213, 213, 318, 318, 368, 453, 342, 168, 275, 275, 275, 275, 303, 303, 303, 117, 404, 404, 13, 78, 20, 20]
                        # print(f"semantic_sys:{semantic_sys}")
                        # quit()

            if args.task_id == 1:
                with open(semantic_sys_file, 'r') as f:
                    text_sys = f.read() 

                with open(semantic_prompt_file, 'r') as f:
                    text_prompts = f.read() 

                if args.input_language==1:
                    semantic_prompts = extract_chinese_HuBert(args, acoustic_prompts_file, hubert_model, apply_kmeans)
                elif args.input_language==0:
                    if args.semantic_type==1:
                        semantic_prompts = sem_tfnet_load(tfcodec_lmencoder, acoustic_prompts_file, device)
                    elif args.semantic_type==0:
                        semantic_prompts_feats = reader.get_feats(acoustic_prompts_file)
                        semantic_prompts = apply_kmeans(semantic_prompts_feats).tolist()

            if args.semantic_depup is True:
                semantic_prompts = depup(semantic_prompts)
                if args.task_id ==0:
                    semantic_sys = depup(semantic_sys)
            
            if args.target_mode==0: # ac or vc or tts-onestage
                
                audio_prompts = []
                encoded_frames = tokenize_audio(audio_tokenizer, acoustic_prompts_file)
                
                if args.input_codec ==0:
                    audio_prompts.append(encoded_frames[0][0])
                else:
                    audio_prompts.append(encoded_frames)

                audio_prompts = torch.concat(audio_prompts, dim=-1)
                audio_prompts = audio_prompts.to(device)

                print(f"audio_propts shape:{audio_prompts.shape}")

                if args.model_name.lower()=="soundstorm":
                    pad_token=500
                    semantic_total = [torch.tensor(seq) for seq in [semantic_prompts + semantic_sys]]
                    semantic_tokens = pad_sequence(semantic_total,batch_first=True, padding_value=pad_token)
                    semantic_tokens_lens = torch.tensor([len(seq) for seq in [semantic_prompts + semantic_sys]])
                else: # valle vallfe
                    if args.task_id ==0:
                        try:
                            semantic_tokens,  semantic_tokens_lens = semantic_token_collater(
                                [semantic_prompts + semantic_sys]
                            )
                            print(f"semantic_tokens_lens:{semantic_tokens_lens}")
                        except Exception as e:
                            print(f"An exception occurred: {e}")

                        enroll_x_lens = None
                        _, enroll_x_lens = semantic_token_collater(
                            [semantic_prompts]
                        )
                    elif args.task_id ==1:
                        print(f"synthesize text: {text_sys}") # no need for this 
                        print(f"synthesize text_prompts: {text_prompts}") # no need for this 
                        try:
                            # print(f"{text_prompts} {text_sys}")
                            text_tokens, text_tokens_lens = text_collater(  # (1, seq_len)
                                [
                                    tokenize_text(
                                        text_tokenizer, text=f"{text_prompts} {text_sys}".strip()
                                    )
                                ]
                            )
                        
                        except Exception as e:
                            print(f"An exception occurred: {e}") 
                            continue 
                        try:
                            _, enroll_x_lens = text_collater( # For NAR
                                    [
                                        tokenize_text(
                                            text_tokenizer, text=f"{text_prompts}".strip()
                                        )
                                    ]
                                )
                            print(f"enroll_x_lens:{enroll_x_lens}")
                        except Exception as e:
                            print(f"An exception occurred: {e}")
                            continue 
                        semantic_tokens = text_tokens
                        semantic_tokens_lens = text_tokens_lens
              
                wrapped_model = ModelWrapper(model).to(device) 
                total_parameters = 0  
                # for name, p in model.named_parameters(): 
                #     if name.startswith("ar"):
                #         if p.requires_grad:  
                #             param_size = p.numel()  
                #             print(f"Parameter name: {name}, Size: {param_size}")  
                #             total_parameters += param_size  
                
                # print(f"Total parameters: {total_parameters}")  
                # total_weight_parameters = 0  
                # for name, p in model.named_parameters():  
                #     if p.requires_grad and "weight" in name and name.startswith("ar"):  
                #         total_weight_parameters += p.numel()  
                
                # print(f"Total weight parameters: {total_weight_parameters}")  

                # print("模型的参数数量：", total_parameters/1e6)
                for i in range(args.nums):
                    # 使用随机张量作为输入以计算FLOPs和参数  
                    # flops, params = profile(  
                    #     wrapped_model,  
                    #     inputs=(  
                    #         semantic_tokens.to(device),  
                    #         semantic_tokens_lens.to(device),  
                    #         audio_prompts,  
                    #         enroll_x_lens,  
                    #         args.top_k,  
                    #         args.temperature,  
                    #         args.mode_stage2  
                    #     ),  
                    #     verbose=False,  
                    # )  
                    # print("模型的参数数量：", params/1e6)  
                    # print("模型的FLOPs：", flops/1e9) 
                    # quit()
                    # print(semantic_tokens)
                    # quit()
                    if args.model_name.lower()=="soundstorm":

                        if args.soundstorm_type ==0:
                            encoded_frames = model.inference(
                                semantic_tokens.to(device),
                                semantic_tokens_lens.to(device),
                                audio_prompts,
                                initial_temp=4.5,
                                T=args.soundstorm_steps,
                                top_k=args.top_k,
                                top_p=args.top_p,
                                top_k_know_token=args.top_k_know_token,
                                known_token_update=args.known_token_update,
                                temperature=args.temperature,
                            )
                        elif args.soundstorm_type ==1:
                            encoded_frames = model.inference_group_ar(
                                semantic_tokens.to(device),
                                semantic_tokens_lens.to(device),
                                audio_prompts,
                                initial_temp=4.5,
                                T=args.soundstorm_steps,
                                top_k=args.top_k,
                                top_p=args.top_p,
                                temperature=args.temperature,
                            )
                    elif args.only_autoregressive is True:
                        encoded_frames = model.inference_only_ar(
                            semantic_tokens.to(device),
                            semantic_tokens_lens.to(device),
                            audio_prompts,
                            enroll_x_lens=enroll_x_lens,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            temperature=args.temperature,
                            mode = args.mode_stage2
                        )
                        if len(encoded_frames)==1:
                            encoded_frames =encoded_frames[0]
                        encoded_frames = encoded_frames.squeeze(-1)
                    else:
                        if audio_prompts.shape[1]==8:
                            audio_prompts = torch.transpose(audio_prompts, 1, 2)
                        
                        encoded_frames = model.inference(
                            semantic_tokens.to(device),
                            semantic_tokens_lens.to(device),
                            audio_prompts,
                            enroll_x_lens=enroll_x_lens,
                            top_k=args.top_k,
                            temperature=args.temperature,
                        )
                    
                    if audio_prompts != []:

                        try:
                            if args.input_codec ==0:
                                samples = audio_tokenizer.decode(
                                    [(encoded_frames.transpose(2, 1), None)]
                                )
                            else:
                                samples = audio_tokenizer.decode(
                                encoded_frames
                                )
                            str1 = args.checkpoint1.split('/')[-1]
                            str2 = args.checkpoint2.split('/')[-1]
                            model_str = f"ar_{str1}_nar_{str2}"
                            audio_prompts_str = acoustic_prompts_file
                            audio_prompts_str= audio_prompts_str.split('/')[-1][:-4]

                            semantic_sys_str = semantic_sys_file
                            semantic_sys_str= semantic_sys_str.split('/')[-1][:-4]
                            
                            if args.test_benchmark is True:
                                args.output_dir = os.path.join('/'.join(args.dir_need2test.split('/')[:-1]), args.outputdir_name)
                            else:
                                args.output_dir = '/'.join(args.semantic_sys_dir.split('/')[:-1]) + '/'+ args.outputdir_name
                            os.makedirs(args.output_dir, exist_ok=True)
                            # store
                            timestamp = current_time.strftime("%Y_%m_%d_%H_%M_%S")
                            
                            print(f"sys_file:{semantic_sys_str}")
                            wav_file_name = f"prompt1_{audio_prompts_str}_sys2_{semantic_sys_str}_model3_{model_str}_{timestamp}_{i}"
                            wav_file_name = wav_file_name.replace('.', "")
                            
                            print(f"{args.output_dir}/{wav_file_name}.wav")

                            if args.input_codec ==0:
                                    torchaudio.save(
                                f"{args.output_dir}/{wav_file_name}.wav", samples[0].cpu(), 24000
                            )
                            else:
                                torchaudio.save(
                                    f"{args.output_dir}/{wav_file_name}.wav", samples.cpu(), 16000
                                )

                                torch.cuda.empty_cache() 
                            
                            # quit()
   
                        except Exception as e:
                            print(e)
            elif args.target_mode==1 or args.target_mode==2:  # correct + generative or txt->semantic -> generative 

                print(f"args.target_mode==1 or args.target_mode==2")
                semantic_prompts_test = torch.tensor(semantic_prompts)

                if args.semantic_type==0:
                    semantic_prompts_test = semantic_prompts_test.unsqueeze(0).unsqueeze(2) # [3] [[3]] # one codebook -> [[[3]]] -> batchsize=1
                elif args.semantic_type==1:
                    semantic_prompts_test = semantic_prompts_test.unsqueeze(0) #[[30, 34]] -> [[[30, 34]]]  batch_size=1

                semantic_prompts_no_work = semantic_prompts[:1]

                # for ac
                audio_prompts_no_work = torch.tensor(semantic_prompts_no_work)

                if args.semantic_type==0:
                    audio_prompts_no_work = audio_prompts_no_work.unsqueeze(0).unsqueeze(2) # [3] [[3]] # one codebook -> [[[3]]] -> batchsize=1
                elif args.semantic_type==1:
                    audio_prompts_no_work = audio_prompts_no_work.unsqueeze(0) #[[30, 34]] -> [[[30, 34]]]  batch_size=1

                # # print(audio_prompts.shape)
                if args.task_id ==0:  # ac correct 
                    if args.model_name == "valle_nar":
                        semantic_tokens_stage1,  semantic_tokens_lens_stage1 = pad_list_seq([semantic_sys])
                    
                    else:
                        try:
                            # print([semantic_sys + semantic_prompts])
                            semantic_tokens_stage1,  semantic_tokens_lens_stage1 = semantic_token_collater(
                                [semantic_prompts_no_work + semantic_sys]
                            )
                        except Exception as e:
                                print(f"An exception occurred: {e}")  
                            
                        enroll_x_lens = None
                        _, enroll_x_lens = semantic_token_collater(
                            [semantic_prompts_no_work]
                        )
                print(f"semantic nums is {args.nums}")

                if args.task_id ==1:
                    print(f"synthesize text: {text_sys}") # no need for this 
                    try:
                        if args.txt2semantic_need_prompt is True:
                            text_tokens, text_tokens_lens = text_collater(  # (1, seq_len)
                                [
                                    tokenize_text(
                                        text_tokenizer, text=f"{text_prompts} {text_sys}".strip()
                                    )
                                ]
                            )
                        else:
                            text_tokens, text_tokens_lens = text_collater(  # (1, seq_len)
                                [
                                    tokenize_text(
                                        text_tokenizer, text=f"{text_sys}".strip()
                                    )
                                ]
                            )
                    except Exception as e:
                        print(f"An exception occurred: {e}")  
                        continue 
                    try:
                        _, enroll_x_lens = text_collater( # For NAR
                                [
                                    tokenize_text(
                                        text_tokenizer, text=f"{text_prompts}".strip()
                                    )
                                ]
                            )
                        print(f"enroll_x_lens:{enroll_x_lens}")
                    except Exception as e:
                        print(f"An exception occurred: {e}")  
                        continue

                for i in range(args.nums):
                    
                    # task_id == 0 original tts
                    if args.task_id ==1: # text -> semantic
                        infer_task_id = 2
                        prompt_ = semantic_prompts_test
                        text_semantic_stage1 = text_tokens
                        text_semantic_lens_stage1 = text_tokens_lens

                    elif args.task_id ==0: # correct
                        infer_task_id = 1
                        prompt_ = audio_prompts_no_work
                        text_semantic_stage1 = semantic_tokens_stage1
                        text_semantic_lens_stage1 = semantic_tokens_lens_stage1
                
                    if args.txt2semantic_need_prompt is False and args.task_id==1:
                        print(f"txt2semantic need not prompt")
                        prompt_=None
                    else:
                        print(f"txt2semantic need prompt")
                        prompt_ = prompt_.to(device)
                    
                    if args.task_id ==1 and args.txt_sem_path!="" and args.sem_read is True:

                        sem_file_name = "sem_" + semantic_sys_file.split('/')[-1].replace('txt', 'json')

                        print(os.path.join(args.txt_sem_path, sem_file_name))
                        with open(os.path.join(args.txt_sem_path, sem_file_name), 'r') as f:
                            native_semantic = json.load(f)

                    else:
                        if args.model_name == "valle_nar":
                            native_semantic = model.inference(
                            text_semantic_stage1.to(device),
                            text_semantic_lens_stage1.to(device),
                            top_k=args.top_k,
                            top_p=args.top_p,
                            temperature=args.temperature
                        )
                        else:
                            native_semantic = model.inference_only_ar(
                                text_semantic_stage1.to(device),
                                text_semantic_lens_stage1.to(device),
                                prompt_,
                                enroll_x_lens=enroll_x_lens,
                                top_k=args.top_k,
                                top_p=args.top_p,
                                temperature=args.temperature,
                                mode = args.mode,
                                task_id=infer_task_id,
                                use_silence_token=True
                            )

                        if len(native_semantic)==1: #batch_size==1
                            native_semantic = native_semantic[0]
                        # print(f"source :{semantic_sys.shape}")
                        print("before_semantic:")
                        if args.task_id ==0:
                            print(len(semantic_sys))
                            print(semantic_sys)
                        # print(encoded_frames)
                        print("after is :")
                        if args.semantic_type==0:   
                            native_semantic = native_semantic.squeeze(0).squeeze(1).squeeze(0)  #need remove codebook dim
                            native_semantic = native_semantic.tolist() 
                        elif args.semantic_type==1:
                            native_semantic = native_semantic.squeeze(-1).squeeze(0)
                            native_semantic = native_semantic.tolist()

                        if args.sem_read is False:

                            # longest part sequence
                            print(len(native_semantic))
                            print(native_semantic)


                            sem_file_name = "sem_" + semantic_sys_file.split('/')[-1].replace('txt', 'json')

                            os.makedirs(args.txt_sem_path, exist_ok=True)
                            with open(os.path.join(args.txt_sem_path, sem_file_name), 'w') as f:
                                json.dump(native_semantic, f)
                            
                            continue
                    # print(f"longest_common_subsequence len is {len(longest_common_subsequence(semantic_sys, native_semantic))}")
                    # depup_before_semantic = depup(semantic_sys)
                    # depup_native_semantic = depup(native_semantic)

                    # print(f"depup before_semantic is {depup_before_semantic}")
                    # print(f"len is {len(depup_before_semantic)}")
                    # print(f"depup depup_native_semantic is {depup_native_semantic}")
                    # print(f"len is {len(depup_native_semantic)}")  
                    # print(f"depup longest_common_subsequence len is {len(longest_common_subsequence(depup_before_semantic, depup_native_semantic))}")
                    if args.target_mode==2:

                        if args.model_name_stage2.lower()=="soundstorm":
                            pad_token=500
                            semantic_total = [torch.tensor(seq) for seq in [semantic_prompts + native_semantic]]
                            semantic_tokens = pad_sequence(semantic_total,batch_first=True, padding_value=pad_token)
                            semantic_tokens_lens = torch.tensor([len(seq) for seq in [semantic_prompts + native_semantic]])
                        else:
                            semantic_tokens,  semantic_tokens_lens = semantic_token_collater(
                                [semantic_prompts + native_semantic]
                            )

                            enroll_x_lens = None
                            _, enroll_x_lens = semantic_token_collater(
                                [semantic_prompts]
                            )

                        audio_prompts = []
                        encoded_frames = tokenize_audio(audio_tokenizer, acoustic_prompts_file)
                        
                        if args.input_codec ==0:
                            audio_prompts.append(encoded_frames[0][0])
                        else:
                            audio_prompts.append(encoded_frames)

                        audio_prompts = torch.concat(audio_prompts, dim=-1)
                        audio_prompts = audio_prompts.to(device)

                        # for j in range(args.nums_stage2):
                        for j in range(args.nums_stage2):

                            if args.only_autoregressive is True:
                                if args.model_name_stage2.lower()=="soundstorm":
                                    if args.soundstorm_type ==0:

                                        encoded_frames = model_acoustic.inference(
                                            semantic_tokens.to(device),
                                            semantic_tokens_lens.to(device),
                                            audio_prompts,
                                            initial_temp=4.5,
                                            T=args.soundstorm_steps,
                                            top_k=args.top_k_stage2,
                                            top_p=args.top_p_stage2,
                                            top_k_know_token=args.top_k_know_token_stage2,
                                            known_token_update=args.known_token_update,
                                            temperature=args.temperature_stage2,
                                        )
                                    elif args.soundstorm_type ==1:

                                        encoded_frames = model_acoustic.inference_group_ar(
                                            semantic_tokens.to(device),
                                            semantic_tokens_lens.to(device),
                                            audio_prompts,
                                            initial_temp=4.5,
                                            T=args.soundstorm_steps,
                                            top_k=args.top_k_stage2,
                                            top_p=args.top_p_stage2,
                                            temperature=args.temperature_stage2,
                                        )

                                else:
                                    encoded_frames = model_acoustic.inference_only_ar(
                                        semantic_tokens.to(device),
                                        semantic_tokens_lens.to(device),
                                        audio_prompts,
                                        enroll_x_lens=enroll_x_lens,
                                        top_k=args.top_k_stage2,
                                        top_p=args.top_p_stage2,
                                        temperature=args.temperature_stage2,
                                        mode = args.mode_stage2,
                                        task_id=2
                                    )
                                    if len(encoded_frames)==1: #if batch_size==1
                                        encoded_frames = encoded_frames[0]
                                    encoded_frames = encoded_frames.squeeze(-1)
                                print(encoded_frames.shape)
                            else:           
                                if audio_prompts.shape[1]==8:
                                    audio_prompts = torch.transpose(audio_prompts, 1, 2)                
                                encoded_frames = model_acoustic.inference(
                                    semantic_tokens.to(device),
                                    semantic_tokens_lens.to(device),
                                    audio_prompts,
                                    enroll_x_lens=enroll_x_lens,
                                    top_k=args.top_k_stage2,
                                    top_p=args.top_p_stage2,
                                    temperature=args.temperature_stage2,
                                )

                            if audio_prompts != []:
                                try:
                                    if args.input_codec ==0:
                                        samples = audio_tokenizer.decode(
                                            [(encoded_frames.transpose(2, 1), None)]
                                        )
                                    else:
                                        samples = audio_tokenizer.decode(
                                        encoded_frames
                                        )
                                except Exception as e:
                                    continue
                                    print(f"An exception occurred in codec decode: {e}")
                                str1 = args.checkpoint1.split('/')[-1]
                                str2 = args.checkpoint2.split('/')[-1]
                                model_str = f"ar_{str1}_nar_{str2}"
                                audio_prompts_str = acoustic_prompts_file
                                audio_prompts_str= audio_prompts_str.split('/')[-1][:-4]

                                semantic_sys_str = semantic_sys_file
                                semantic_sys_str= semantic_sys_str.split('/')[-1][:-4]
                                
                                if args.task_id ==1 and args.test_benchmark is True:

                                    semantic_sys_dir = args.dir_need2test
                                else:
                                    semantic_sys_dir = args.semantic_sys_dir

                                if args.test_benchmark is True:
                                    args.output_dir = os.path.join('/'.join(args.dir_need2test.split('/')[:-1]), args.outputdir_name)
                                else:
                                    args.output_dir = '/'.join(args.semantic_sys_dir.split('/')[:-1]) + '/'+ args.outputdir_name
                                print(f"output_dir is {args.output_dir}")
                                os.makedirs(args.output_dir, exist_ok=True)
                                # store
                                timestamp = current_time.strftime("%Y_%m_%d_%H_%M_%S")
                                print(f"sys_file:{semantic_sys_str}")
                                # torchaudio.save(
                                #     f"{args.output_dir}/{audio_prompts_str}_{semantic_sys_str}_{model_str}_{timestamp}_{i}.wav", samples[0].cpu(), 24000
                                # )
                                wav_file_name = f"prompt1_{audio_prompts_str}_sys2_{semantic_sys_str}_model3_{model_str}_{timestamp}_{i}_{j}"
                                wav_file_name = wav_file_name.replace('.', "")
                                if args.input_codec ==0:
                                       torchaudio.save(
                                    f"{args.output_dir}/{wav_file_name}.wav", samples[0].cpu(), 24000
                                )
                                else:
                                    torchaudio.save(
                                        f"{args.output_dir}/{wav_file_name}.wav", samples.cpu(), 16000
                                    )
                                    print("generate")
                                torch.cuda.empty_cache() 
            if acoustic_prompts_file != bef_acoustic_prompts_file:
                try:
                    os.remove(acoustic_prompts_file)
                except Exception as e:
                        continue





torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
