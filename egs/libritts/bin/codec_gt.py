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
import sys
# 获取当前工作目录  
current_working_directory = os.getcwd()  
# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory) 

from datetime import datetime
current_time = datetime.now()

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
import fire
import torch
import torchaudio
torch.backends.cudnn.enabled = False  

from encodec import EncodecModel
from encodec.utils import convert_audio

from icefall.utils import str2bool
torch.backends.cudnn.enabled = False  
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
from valle.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from valle.data.collation import get_text_token_collater
from valle.models import add_model_arguments, get_model

extensions = ('.wav', '.flac', '.mp3')  # You can add other audio extensions if needed  

def find_audio_files(directory):  
    audio_files = []  
  
    for foldername, subfolders, filenames in os.walk(directory):  
        for filename in filenames:  
            if filename.lower().endswith(extensions):  
                # if ("prompt" in foldername) or ("source" in foldername):
                audio_files.append(os.path.join(foldername, filename))  
  
    return audio_files

@torch.no_grad()
def main(wav_dir, codec="encodec"):

    # Instantiate a pretrained EnCodec model
    audio_files = find_audio_files(wav_dir)
    print(audio_files)
    if codec == "encodec":
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
    elif codec =="tfcodec":
        model = AudioTokenExtractor_16k_tfcodec(AudioTokenConfig_16k(), tfnet_ckpt="/home/v-zhijunjia/zhijundata_small_v2/data_local/data/valle-tensorboard-models/other_models/tfcodec/890hrs16khz_tfnet_v2i_vqvae_20msVQ_hop5_combine4_rd1_6kbps/tfnet_v2i_vqvae_combineVQ-iter-514000-val1-0.348327-entropy-118.968-mel-0.067548-recon-0.012822.ckpt")
    # The number of codebooks used will be determined bythe bandwidth selected.
    # E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
    # Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
    # For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
    # of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.

    for audio_file in audio_files:
        print(audio_file)
        # Load and pre-process the audio waveform
        wav, sr = torchaudio.load(audio_file)
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        if codec == "encodec":
            wav = wav.unsqueeze(0) # b x t x codebook

            # Extract discrete codes from EnCodec
            with torch.no_grad():
                encoded_frames = model.encode(wav)
            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
            print(codes)
            wav = model.decode([(codes, None)])
            for extension in extensions:
                audio_file = audio_file.replace(extension, f"_codec{extension}")
                print(audio_file)
            torchaudio.save(audio_file, wav[0].cpu(), 24000)
        elif codec == "tfcodec":
            code = model.extract(wav, model.sample_rate)
            # wav: b x t  code: t x codebook
            print(code.shape)
            # codes: t x codebook
            code = torch.tensor(code)
            code = code.unsqueeze(dim=0)
            wav = model.tokenizer.decode(code)
            wav = wav.detach().cpu().numpy()[0]
            for extension in extensions:
                audio_file = audio_file.replace(extension, f"_tfcodec{extension}")
                print(audio_file)
            import soundfile as sf
            sf.write(audio_file, wav, 16000)



if __name__ == "__main__":
    fire.Fire(main)
