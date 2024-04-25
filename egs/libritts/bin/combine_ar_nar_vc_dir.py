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
import os
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
import torch.nn as nn  
from thop import profile

class ModelWrapper(nn.Module):  
    def __init__(self, model):  
        super(ModelWrapper, self).__init__()  
        self.model = model  
  
    def forward(self, semantic_tokens, semantic_tokens_lens, audio_prompts, enroll_x_lens, top_k, temperature):  
        return self.model.inference(  
            semantic_tokens,  
            semantic_tokens_lens,  
            audio_prompts,  
            enroll_x_lens=enroll_x_lens,  
            top_k=top_k,  
            temperature=temperature,  
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
        ckpt_path = "/home/v-zhijunjia/data/valle-tensorboard-models/other_models/hubert/hubert_base_ls960.pt"
        layer = 9
        km_path = "/home/v-zhijunjia/data/valle-tensorboard-models/other_models/hubert/hubert_base_ls960_L9_km500.bin"
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

    args.shared_linear = False
    args.is_pretrain = False
    args.pret_mode = 0
    model = get_model(args)

    checkpoint1 = torch.load(args.checkpoint1, map_location=device)
    checkpoint2 = torch.load(args.checkpoint2, map_location=device)  
    
    # # 将checkpoint2的键写入到另一个txt文件中  
    # with open('keys_output_checkpoint2.txt', 'w') as f:  
    #     for key in checkpoint2["model"].keys():  
    #         f.write(key + '\n')  
    missing_keys1, unexpected_keys1 = model.load_state_dict(
        checkpoint1["model"], strict=True
    )
    assert not missing_keys1
    
    missing_keys2, unexpected_keys2 = model.load_state_dict(
        checkpoint2["model"], strict=True
    )
    assert not missing_keys2

    
    for key in list(checkpoint1['model'].keys()):  
        if key.startswith('nar'):  
            # 在 checkpoint2 中查找相应的以 'nar' 开头的属性  
            if key in checkpoint2['model']:  
                print(key)
                # 将 checkpoint1 中的 'nar' 开头属性替换为 checkpoint2 中对应的 'nar' 开头属性  
                checkpoint1['model'][key] = checkpoint2['model'][key]  
            else:  
                print(f"未找到与 {key} 对应的属性。")  

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

            audio_prompts = []
            encoded_frames = tokenize_audio(audio_tokenizer, semantic_prompts_file)
     
            if args.input_codec ==0:
                audio_prompts.append(encoded_frames[0][0])
            else:
                audio_prompts.append(encoded_frames)

            if args.input_codec ==0:
                audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
            else:
                audio_prompts = torch.concat(audio_prompts, dim=-1)
            # batch t codebook
            audio_prompts = audio_prompts.to(device)


            try:

                print([semantic_sys + semantic_prompts])
                semantic_tokens,  semantic_tokens_lens = semantic_token_collater(
                    [semantic_prompts + semantic_sys]
                )

            except Exception as e:
                print(f"An exception occurred: {e}")  

            wrapped_model = ModelWrapper(model).to(device) 
            enroll_x_lens = None
            _, enroll_x_lens = semantic_token_collater(
                [semantic_prompts]
            )
            for i in range(args.nums):
                flops, params = profile(  
                        wrapped_model,  
                        inputs=(  
                            semantic_tokens.to(device),  
                            semantic_tokens_lens.to(device),  
                            audio_prompts,  
                            enroll_x_lens,  
                            args.top_k,  
                            args.temperature,  
                        ),  
                        verbose=False,  
                    )
                # print("模型的参数数量：", params/1e6)  
                # print("模型的FLOPs：", flops/1e9) 
                encoded_frames = model.inference(
                    semantic_tokens.to(device),
                    semantic_tokens_lens.to(device),
                    audio_prompts,
                    enroll_x_lens=enroll_x_lens,
                    top_k=args.top_k,
                    temperature=args.temperature,
                )
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
                    if args.input_codec ==0:
                        torchaudio.save(
                            f"{args.output_dir}/{audio_prompts_str}_{semantic_sys_str}_{model_str}_{timestamp}_{i}.wav", samples[0].cpu(), 24000
                        )
                    else:
                        torchaudio.save(
                            f"{args.output_dir}/{audio_prompts_str}_{semantic_sys_str}_{model_str}_{timestamp}_{i}.wav", samples.cpu(), 16000
                        )
                    print("generate")
                    torch.cuda.empty_cache() 
                else:  # Transformer
                    pass


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
