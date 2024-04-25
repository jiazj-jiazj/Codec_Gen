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

import sys

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
        "--shared-linear",
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
        "--input-language",
        type=int,
        default="0",
        help="0->english, 1->chinese",
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

    return parser.parse_args()


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
        ckpt_path = "/dev_huaying/zhijun/models/hubert/hubert_base_ls960.pt"
        layer = 9
        km_path = "/dev_huaying/zhijun/models/hubert/hubert_base_ls960_L9_km500.bin"
        reader = HubertFeatureReader(ckpt_path, layer)
        apply_kmeans = ApplyKmeans(km_path)    


    # text_tokenizer = TextTokenizer(backend=args.text_extractor)
    if args.input_codec ==0:
        audio_tokenizer = AudioTokenizer()
    else:
        audio_tokenizer = AudioTokenizer_encodec_16k_tfcodec()
        
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    model = get_model(args)

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
 
    
    audio_files_A = get_audio_files(args.semantic_sys_dir)  
    audio_files_B = get_audio_files(args.audio_prompts_dir)  
    
    for semantic_sys_file in audio_files_A:  
        for semantic_prompts_file in audio_files_B:  
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
                # quit()
    # semantic_sys =[17, 17, 296, 287, 284, 284, 405, 405, 206, 206, 169, 164, 164, 164, 300, 300, 382, 467, 467, 223, 223, 130, 198, 198, 22, 283, 455, 236, 239, 384, 371, 93, 290, 290, 290, 290, 434, 339, 339, 64, 107, 382, 382, 313, 236, 36, 161, 161, 487, 487, 487, 288, 288, 290, 139, 139, 375, 375, 375, 98, 349, 393, 234, 234, 261, 261, 25, 470, 139, 139, 175, 175, 81, 81, 215, 215, 96, 66, 6, 371, 444, 213, 213, 213, 286, 464, 139, 139, 302, 497, 185, 49, 269, 168, 483, 440, 236, 478, 66, 172, 273, 470, 151, 151, 240, 285, 495, 495, 406, 467, 423, 423, 423, 423, 423, 263, 263]

            semantic_prompts = semantic_prompts[:1]
            # print(semantic_prompts)
            # quit()
            audio_prompts = torch.tensor(semantic_prompts)

            audio_prompts = audio_prompts.unsqueeze(0).unsqueeze(2)
            # print(audio_prompts.shape)


            try:

                # print([semantic_sys + semantic_prompts])
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
                    audio_prompts.to(device),
                    enroll_x_lens=enroll_x_lens,
                    top_k=args.top_k,
                    temperature=args.temperature,
                )
                print(f"source :{semantic_tokens.shape}")
                print(semantic_sys)
                # print(encoded_frames)
                encoded_frames = encoded_frames.squeeze(0).squeeze(1).squeeze(1)
                print(encoded_frames.shape)
                print(encoded_frames)
                # encoded_frames = encoded_frames.squeeze(-1)
                # print(encoded_frames.shape)
                # if audio_prompts != []:
                #     if args.input_codec ==0:
                #         samples = audio_tokenizer.decode(
                #             [(encoded_frames.transpose(2, 1), None)]
                #         )
                #     else:
                #         samples = audio_tokenizer.decode(
                #         encoded_frames
                #         )
                #     str1 = args.checkpoint1.split('/')[-1]
                #     str2 = args.checkpoint2.split('/')[-1]
                #     model_str = f"ar_{str1}_nar_{str2}"
                #     audio_prompts_str = semantic_prompts_file
                #     audio_prompts_str= audio_prompts_str.split('/')[-1][:-4]

                #     semantic_sys_str = semantic_sys_file
                #     semantic_sys_str= semantic_sys_str.split('/')[-1][:-4]   
                #         f"{args.output_dir}/{audio_prompts_str}_{semantic_sys_str}_{model_str}_{timestamp}_{i}.wav", samples.cpu(), 16000
                #     )
                #     print("generate")
                #     torch.cuda.empty_cache() 
                # else:  # Transformer
                #     pass


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
