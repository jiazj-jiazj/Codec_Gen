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
        "--accent-remove",
        type=str2bool,
        default="False",
        help="accent-remove.",
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
        "--top-k-stage2",
        type=int,
        default=-100,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )

    parser.add_argument(
        "--temperature",
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
        help="0->acoustic, 1->semantic",
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
        "--mode-stage2",
        type=int,
        default=0,
        help="0->top-infer, 1->arg_max",
    )
    parser.add_argument(
        "--is-pretrain",
        type=str2bool,
        default='false',
        help="input-semantic.",
    )
    parser.add_argument(
        "--pret-mode",
        type=int,
        default=0,
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
        "--prompt_pre_3s",
        type=str2bool,
        default="false",
        help="whether to trimm pre 3s",
    )
    return parser.parse_args()

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

@torch.no_grad()
def main():
    args = get_args()
    semantic_token_collater = get_text_token_collater(args.semantic_tokens)


    if args.input_language==1:
        if args.semantic_layer==9:
            km_path = "chinese_speech_pretrain/hubert_kmeans/hubert_base_iter2_32gpu_l9/model.mdl"
        elif args.semantic_layer==6:
            km_path = "chinese_speech_pretrain/hubert_kmeans/hubert_base_iter1_32gpu_l6/model.mdl"

        model_path="TencentGameMate/chinese-hubert-base"

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        hubert_model = HubertModel.from_pretrained(model_path)
        device = "cuda"
        # print(model)
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.half()
        hubert_model.eval()
        apply_kmeans = ApplyKmeans(km_path)  
    
    
    elif args.input_language==0:
        ckpt_path = args.hubert_path
        layer = 9
        km_path = args.hubert_km_path
        reader = HubertFeatureReader(ckpt_path, layer)
        apply_kmeans = ApplyKmeans(km_path)    

    # text_tokenizer = TextTokenizer(backend=args.text_extractor)
    if args.input_codec ==0:
        audio_tokenizer = AudioTokenizer()
    else:
        audio_tokenizer = AudioTokenizer_encodec_16k_tfcodec(tfnet_ckpt=args.tfnet_ckpt)
        
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    model = get_model(args)

    
    if args.target_mode == 2:

# Store original values in temporary variables  
        original_num_decoder_layers = args.num_decoder_layers  
        original_num_quantizers = args.num_quantizers  
        original_decoder_dim = args.decoder_dim  
        original_nhead = args.nhead  
        original_top_k = args.top_k  
        original_shared_linear = args.shared_linear  
        
        # Set the values for stage2  
        args.num_decoder_layers = args.num_decoder_layers_stage2  
        args.num_quantizers = args.num_quantizers_stage2  
        args.decoder_dim = args.decoder_dim_stage2  
        args.nhead = args.nhead_stage2  
        args.top_k = args.top_k_stage2  
        args.shared_linear = args.shared_linear_stage2  
        
        # Get the model_acoustic  
        model_acoustic = get_model(args)  
        
        # Restore the original values from the temporary variables  
        args.num_decoder_layers = original_num_decoder_layers  
        args.num_quantizers = original_num_quantizers  
        args.decoder_dim = original_decoder_dim  
        args.nhead = original_nhead  
        args.top_k = original_top_k  
        args.shared_linear = original_shared_linear          

        checkpoint2 = torch.load(args.checkpoint2, map_location=device)
        with open('model2_keys_output.txt', 'w') as f:  
            for key in model_acoustic.state_dict().keys():  
                f.write(key + '\n') 
        with open('keys_output_checkpoint2.txt', 'w') as f:  
            for key in checkpoint2["model"].keys():  
                f.write(key + '\n')  
        
        # # 将checkpoint2的键写入到另一个txt文件中  
        # with open('keys_output_checkpoint2.txt', 'w') as f:  
        #     for key in checkpoint2["model"].keys():  
        #         f.write(key + '\n')  
        missing_keys1, unexpected_keys1 = model_acoustic.load_state_dict(
            checkpoint2["model"], strict=True
        )
        assert not missing_keys1

        model_acoustic.to(device)
        model_acoustic.eval()



    checkpoint1 = torch.load(args.checkpoint1, map_location=device)
    # checkpoint2 = torch.load(args.checkpoint2, map_location=device)
    with open('model_keys_output.txt', 'w') as f:  
        for key in model.state_dict().keys():  
            f.write(key + '\n') 
    with open('keys_output_checkpoint1.txt', 'w') as f:  
        for key in checkpoint1["model"].keys():  
            f.write(key + '\n')  
    
    # # 将checkpoint2的键写入到另一个txt文件中  
    # with open('keys_output_checkpoint2.txt', 'w') as f:  
    #     for key in checkpoint2["model"].keys():  
    #         f.write(key + '\n')  
    missing_keys1, unexpected_keys1 = model.load_state_dict(
        checkpoint1["model"], strict=True
    )
    assert not missing_keys1

    model.to(device)
    model.eval()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    import os  
    
    def get_audio_files(folder_path):  
        audio_extensions = ('.flac', '.wav')  
        audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(audio_extensions)]  
        return audio_files  
 
    if args.accent_remove is False:
        audio_files_A = get_audio_files(args.semantic_sys_dir)  
        audio_files_B = get_audio_files(args.audio_prompts_dir)
    else:
        audio_files_A = get_audio_files(args.semantic_sys_dir)
        audio_files_B = [1]

    audio_files_A = get_audio_files(args.semantic_sys_dir)

    if args.accent_remove is False:
        audio_files_B = get_audio_files(args.audio_prompts_dir)
    else:
        audio_files_B = audio_files_A[:1]

    for semantic_sys_file in audio_files_A:  
        for semantic_prompts_file in audio_files_B: 

            if args.accent_remove is True:
                semantic_prompts_file = semantic_sys_file
            # if args.accent_remove is True:
            #     semantic_prompts_file = semantic_sys_file
            bef_semantic_prompts_file = semantic_prompts_file

            if args.prompt_pre_3s  is True or args.accent_remove is True:
                semantic_prompts_file = process_audio(semantic_prompts_file)

            if args.input_language==1:

                print(f"input_language:{args.input_language}")
                def remove_spaces(text: str) -> str:  
                    return text.replace(" ", "")  
                    
                wav, sr = sf.read(semantic_prompts_file)
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
                    semantic_prompts = apply_kmeans(last_hidden_state).tolist()  
                
                wav, sr = sf.read(semantic_sys_file)
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

            elif args.input_language==0:
                semantic_prompts_feats = reader.get_feats(semantic_prompts_file)
                semantic_prompts = apply_kmeans(semantic_prompts_feats).tolist()

            # semantic_prompts_feats = reader.get_feats(semantic_prompts_file)
            # semantic_prompts = apply_kmeans(semantic_prompts_feats).tolist()

                semantic_sys_feats = reader.get_feats(semantic_sys_file)
                semantic_sys = apply_kmeans(semantic_sys_feats).tolist()
            
            # print(len(semantic_prompts))
            if args.semantic_depup:

                unique_tokens = []  
                for token in semantic_sys:  
                    if token not in unique_tokens:  
                        unique_tokens.append(token)
                semantic_sys = unique_tokens

                unique_tokens_prompts = []  
                for token in semantic_prompts:  
                    if token not in unique_tokens_prompts:  
                        unique_tokens_prompts.append(token)
                semantic_prompts = unique_tokens_prompts
            
            # # [17, 17, 20, 20, 296, 276, 276, 276, 387, 387, 240, 285, 34, 34, 242, 242, 116, 94, 335, 440, 154, 154, 154, 154, 154, 96, 96, 232, 172, 172, 115, 273, 470, 428, 428, 428, 146, 146, 146, 252, 143, 36, 192, 191, 191, 191, 313, 24, 314, 133, 345, 333, 333, 220, 38, 31, 54, 232, 482, 105, 105, 336, 336, 354, 106, 106, 387, 387, 387, 406, 406, 176, 176, 176, 328, 328, 200, 200, 335, 440, 440, 83, 89, 446, 446, 322, 67, 394, 76, 36, 144, 27, 27, 370, 319, 446, 348, 33, 90, 393, 393, 155, 261, 25, 148, 148, 148, 148, 387, 387, 387, 406, 406, 406, 176, 176, 176, 328, 200, 200, 248, 76, 401, 401, 82, 82, 377, 377, 494, 132, 236, 129, 129, 401, 259, 20, 74, 425, 386, 343, 343, 343, 343, 343, 171, 246, 358, 358, 358, 185, 39, 39, 323, 390, 390, 323, 390, 390, 18, 18, 18, 18, 112, 112, 439, 193, 193, 193]
            # [17, 17, 296, 296, 296, 276, 276, 276, 387, 387, 240, 285, 34, 242, 242, 242, 116, 94, 335, 440, 440, 154, 154, 154, 154, 96, 96, 232, 172, 172, 115, 273, 470, 428, 428, 428, 146, 146, 146, 252, 143, 36, 192, 191, 191, 191, 313, 24, 314, 133, 345, 333, 333, 220, 38, 31, 54, 232, 482, 105, 105, 336, 336, 354, 106, 106, 387, 387, 387, 406, 406, 176, 176, 176, 328, 328, 200, 200, 335, 440, 440, 83, 89, 446, 446, 322, 67, 394, 76, 36, 144, 27, 27, 370, 319, 446, 348, 33, 90, 393, 393, 155, 261, 25, 148, 148, 148, 148, 387, 387, 387, 406, 406, 406, 176, 176, 176, 328, 200, 200, 248, 76, 401, 401, 82, 82, 377, 377, 494, 132, 236, 129, 129, 401, 259, 20, 74, 425, 386, 343, 343, 343, 343, 343, 171, 246, 358, 358, 358, 185, 39, 39, 323, 390, 390, 390, 390, 390, 390, 18, 18, 18, 112, 112, 193, 193, 193, 193]
            # [17, 17, 296, 7, 364, 276, 276, 153, 387, 387, 240, 285, 34, 34, 242, 242, 116, 94, 335, 440, 440, 154, 154, 154, 154, 96, 96, 232, 172, 172, 115, 273, 470, 428, 428, 428, 146, 146, 146, 252, 143, 36, 192, 191, 191, 191, 313, 24, 314, 133, 345, 333, 333, 220, 38, 31, 54, 232, 482, 105, 105, 336, 336, 354, 106, 106, 387, 387, 387, 406, 406, 176, 176, 176, 328, 328, 200, 200, 335, 440, 440, 83, 89, 446, 446, 322, 67, 394, 76, 36, 144, 27, 27, 370, 319, 446, 348, 33, 90, 393, 393, 155, 261, 25, 148, 148, 148, 148, 387, 387, 387, 406, 406, 406, 176, 176, 176, 328, 200, 200, 248, 76, 401, 401, 82, 82, 377, 377, 494, 132, 236, 129, 129, 401, 259, 20, 74, 425, 386, 343, 343, 343, 343, 343, 171, 246, 358, 358, 358, 39, 39, 323, 323, 390, 390, 390, 390, 390, 390, 18, 18, 18, 18, 112, 112, 193, 193, 193]
            # print(f"semantic_sys: {semantic_sys}")
            # print(f'len:{len(semantic_sys)}')
            # # semantic_sys = [17, 296, 152, 152, 152, 152, 3, 464, 14, 14, 411, 297, 297, 297, 297, 293, 293, 122, 122, 77, 77, 342, 224, 494, 494, 494, 494, 236, 196, 10, 309, 479, 444, 213, 213, 213, 213, 213, 252, 339, 325, 335, 440, 440, 44, 44, 44, 44, 38, 31, 342, 142, 105, 196, 196, 70, 70, 70, 65, 65, 481, 481, 481, 481, 293, 293, 497, 497, 122, 143, 401, 401, 401, 491, 74, 425, 386, 386, 486, 486, 376, 376, 460, 460, 169, 150, 39, 86, 238, 6, 6, 272, 470, 469, 469, 178, 96, 96, 270, 68, 68, 238, 221, 196, 196, 479, 331, 331, 290, 171, 171, 171, 171, 171, 252, 143, 143, 491, 192, 192, 483, 226, 226, 226, 491, 209, 83, 55, 55, 322, 322, 199, 44, 44, 44, 8, 32, 32, 32, 354, 354, 278, 278, 278, 278, 252, 143, 458, 96, 401, 401, 401, 401, 401, 108, 119, 119, 106, 499, 499, 265, 265, 85, 85, 146, 146, 438, 349, 349, 234, 234, 234, 261, 190, 190, 380, 380, 499, 499, 91, 91, 405, 405, 206, 206, 206, 274, 416, 233, 82, 208, 208, 393, 393, 155, 332, 332, 332, 332, 216, 22, 283, 455, 143, 401, 401, 491, 144, 445, 445, 278, 278, 278, 37, 37, 314, 233, 233, 491, 491, 270, 270, 433, 160, 160, 160, 193, 193, 193]

            # semantic_sys = [17, 50, 152, 152, 30, 30, 30, 465, 50, 50, 500, 500, 500, 297, 297, 293, 293, 122, 122, 77, 77, 342, 224, 494, 494, 494, 494, 236, 196, 10, 309, 479, 444, 213, 213, 213, 213, 213, 252, 339, 325, 335, 440, 440, 44, 44, 44, 44, 38, 31, 342, 142, 105, 196, 196, 70, 70, 70, 65, 65, 481, 481, 481, 481, 293, 293, 497, 497, 122, 143, 401, 401, 401, 491, 74, 425, 386, 386, 486, 486, 376, 376, 460, 460, 169, 150, 39, 86, 238, 6, 6, 272, 470, 469, 469, 178, 96, 96, 270, 68, 68, 238, 221, 196, 196, 479, 331, 331, 290, 171, 171, 171, 171, 171, 252, 143, 143, 491, 192, 192, 483, 226, 226, 226, 491, 209, 83, 55, 55, 322, 322, 199, 44, 44, 44, 8, 32, 32, 32, 354, 354, 278, 278, 278, 278, 252, 143, 458, 96, 401, 401, 401, 401, 401, 108, 119, 119, 106, 499, 499, 265, 265, 85, 85, 146, 146, 438, 349, 349, 234, 234, 234, 261, 190, 190, 380, 380, 499, 499, 91, 91, 405, 405, 206, 206, 206, 274, 416, 233, 82, 208, 208, 393, 393, 155, 332, 332, 332, 332, 216, 22, 283, 455, 143, 401, 401, 491, 144, 445, 445, 278, 278, 278, 37, 37, 314, 233, 233, 491, 491, 270, 270, 433, 160, 160, 160, 193, 193, 193]
            # print("after substitude:")
            # print(f"semantic_sys: {semantic_sys}")
            # print(f'len:{len(semantic_sys)}')
            import random
            def mask_transform(semantic_token, prob=0.5, mask_token=500):

                # 设置替换为501的概率（0到1之间）  
                replacement_probability = 0.5  
                # 遍历列表并根据概率替换元素  
                for i, token in enumerate(semantic_token):  
                    if random.random() < replacement_probability:  
                        semantic_token[i] = mask_token  
                return semantic_token
            
            # semantic_sys = mask_transform(semantic_sys, prob=0.8)
            # quit()
            if args.target_mode==0:
                # semantic_sys = [184, 448, 448, 236, 36, 108, 119, 278, 139, 139, 27, 121, 121, 33, 394, 478, 172, 115, 273, 432, 53, 76, 99, 436, 436, 60, 298, 253, 253, 168, 44, 44, 94, 199, 145, 145, 443, 173, 280, 469, 325, 11, 11, 11, 379, 77, 342, 224, 462, 401, 139, 139, 293, 356, 356, 6, 87, 87, 386, 376, 376, 376, 169, 150, 86, 453, 168, 44, 107, 395, 401, 496, 193, 193, 17]
                # semantic_sys =[17, 17, 296, 287, 284, 284, 405, 405, 206, 206, 169, 164, 164, 164, 300, 300, 382, 467, 467, 223, 223, 130, 198, 198, 22, 283, 455, 236, 239, 384, 371, 93, 290, 290, 290, 290, 434, 339, 339, 64, 107, 382, 382, 313, 236, 36, 161, 161, 487, 487, 487, 288, 288, 290, 139, 139, 375, 375, 375, 98, 349, 393, 234, 234, 261, 261, 25, 470, 139, 139, 175, 175, 81, 81, 215, 215, 96, 66, 6, 371, 444, 213, 213, 213, 286, 464, 139, 139, 302, 497, 185, 49, 269, 168, 483, 440, 236, 478, 66, 172, 273, 470, 151, 151, 240, 285, 495, 495, 406, 467, 423, 423, 423, 423, 423, 263, 263]
                audio_prompts = []
                encoded_frames = tokenize_audio(audio_tokenizer, semantic_prompts_file)
                
                if args.input_codec ==0:
                    audio_prompts.append(encoded_frames[0][0])
                else:
                    audio_prompts.append(encoded_frames)

                audio_prompts = torch.concat(audio_prompts, dim=-1)
                audio_prompts = audio_prompts.to(device)

                print(f"audio_propts shape:{audio_prompts.shape}")
                print(audio_prompts)

                try:

                    print([semantic_prompts + semantic_sys])
                    semantic_tokens,  semantic_tokens_lens = semantic_token_collater(
                        [semantic_prompts + semantic_sys]
                    )

                except Exception as e:
                    print(f"An exception occurred: {e}")  


                enroll_x_lens = None
                _, enroll_x_lens = semantic_token_collater(
                    [semantic_prompts]
                )
    
                for i in range(args.nums):
                    encoded_frames = model.inference_only_ar(
                        semantic_tokens.to(device),
                        semantic_tokens_lens.to(device),
                        audio_prompts,
                        enroll_x_lens=enroll_x_lens,
                        top_k=args.top_k,
                        temperature=args.temperature,
                        mode = args.mode_stage2
                    )
                    encoded_frames = encoded_frames.squeeze(-1)
                    print(encoded_frames.shape)
                    if audio_prompts != []:
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
                        audio_prompts_str = semantic_prompts_file
                        audio_prompts_str= audio_prompts_str.split('/')[-1][:-4]

                        semantic_sys_str = semantic_sys_file
                        semantic_sys_str= semantic_sys_str.split('/')[-1][:-4]
                        
                        args.output_dir = '/'.join(args.semantic_sys_dir.split('/')[:-1]) + '/'+ args.outputdir_name
                        os.makedirs(args.output_dir, exist_ok=True)
                        # store
                        timestamp = current_time.strftime("%Y_%m_%d_%H_%M_%S")

                        # torchaudio.save(
                        #     f"{args.output_dir}/{audio_prompts_str}_{semantic_sys_str}_{model_str}_{timestamp}_{i}.wav", samples[0].cpu(), 24000
                        # )
                        wav_file_name = f"{audio_prompts_str}_{semantic_sys_str}_{model_str}_{timestamp}_{i}"
                        wav_file_name = wav_file_name.replace('.', "")

                        torchaudio.save(
                            f"{args.output_dir}/{wav_file_name}.wav", samples.cpu(), 16000
                        )
                        print("generate")
                        torch.cuda.empty_cache() 
                    else:  # Transformer
                        pass
            elif args.target_mode==1 or args.target_mode==2:
                semantic_prompts_no_work = semantic_prompts[:1]
                audio_prompts_no_work = torch.tensor(semantic_prompts_no_work)
                audio_prompts_no_work = audio_prompts_no_work.unsqueeze(0).unsqueeze(2)
                # print(audio_prompts.shape)
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
                for i in range(args.nums):
                    native_semantic = model.inference_only_ar(
                        semantic_tokens_stage1.to(device),
                        semantic_tokens_lens_stage1.to(device),
                        audio_prompts_no_work.to(device),
                        enroll_x_lens=enroll_x_lens,
                        top_k=args.top_k,
                        temperature=args.temperature,
                        mode = args.mode
                    )
                    # print(f"source :{semantic_sys.shape}")
                    print("before_semantic:")
                    print(len(semantic_sys))
                    print(semantic_sys)
                    # print(encoded_frames)
                    native_semantic = native_semantic.squeeze(0).squeeze(1).squeeze(1)
                    native_semantic = native_semantic.tolist()

                    # longest part sequence
                    print("after is :")
                    print(len(native_semantic))
                    print(native_semantic)

                    # print(f"longest_common_subsequence len is {len(longest_common_subsequence(semantic_sys, native_semantic))}")
                    # depup_before_semantic = depup(semantic_sys)
                    # depup_native_semantic = depup(native_semantic)

                    # print(f"depup before_semantic is {depup_before_semantic}")
                    # print(f"len is {len(depup_before_semantic)}")
                    # print(f"depup depup_native_semantic is {depup_native_semantic}")
                    # print(f"len is {len(depup_native_semantic)}")  
                    # print(f"depup longest_common_subsequence len is {len(longest_common_subsequence(depup_before_semantic, depup_native_semantic))}")


                    if args.target_mode==2:
                        semantic_tokens,  semantic_tokens_lens = semantic_token_collater(
                            [semantic_prompts + native_semantic]
                        )

                        enroll_x_lens = None
                        _, enroll_x_lens = semantic_token_collater(
                            [semantic_prompts]
                        )

                        audio_prompts = []
                        encoded_frames = tokenize_audio(audio_tokenizer, semantic_prompts_file)
                        
                        if args.input_codec ==0:
                            audio_prompts.append(encoded_frames[0][0])
                        else:
                            audio_prompts.append(encoded_frames)

                        audio_prompts = torch.concat(audio_prompts, dim=-1)
                        audio_prompts = audio_prompts.to(device)

                        for j in range(args.nums_stage2):
                            encoded_frames = model_acoustic.inference_only_ar(
                                semantic_tokens.to(device),
                                semantic_tokens_lens.to(device),
                                audio_prompts,
                                enroll_x_lens=enroll_x_lens,
                                top_k=args.top_k_stage2,
                                temperature=args.temperature,
                                mode = args.mode_stage2
                            )
                            encoded_frames = encoded_frames.squeeze(-1)
                            print(encoded_frames.shape)

                            if audio_prompts != []:
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
                                audio_prompts_str = semantic_prompts_file
                                audio_prompts_str= audio_prompts_str.split('/')[-1][:-4]

                                semantic_sys_str = semantic_sys_file
                                semantic_sys_str= semantic_sys_str.split('/')[-1][:-4]
                                
                                args.output_dir = '/'.join(args.semantic_sys_dir.split('/')[:-1]) + '/'+ args.outputdir_name
                                print(f"output_dir is {args.output_dir}")
                                os.makedirs(args.output_dir, exist_ok=True)
                                # store
                                timestamp = current_time.strftime("%Y_%m_%d_%H_%M_%S")

                                # torchaudio.save(
                                #     f"{args.output_dir}/{audio_prompts_str}_{semantic_sys_str}_{model_str}_{timestamp}_{i}.wav", samples[0].cpu(), 24000
                                # )
                                wav_file_name = f"{audio_prompts_str}_{semantic_sys_str}_{model_str}_{timestamp}_{i}_{j}"
                                wav_file_name = wav_file_name.replace('.', "")
                                torchaudio.save(
                                    f"{args.output_dir}/{wav_file_name}.wav", samples.cpu(), 16000
                                )
                                print("generate")
                                torch.cuda.empty_cache() 
            if semantic_prompts_file != bef_semantic_prompts_file:
                os.remove(semantic_prompts_file)





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
