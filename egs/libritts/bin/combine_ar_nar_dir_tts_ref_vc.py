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
from datetime import datetime
current_time = datetime.now()
timestamp = current_time.strftime("%Y_%m_%d_%H_%M_%S")

from datetime import datetime, timedelta  
import time
import argparse
import logging
import os
from pathlib import Path
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

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
    HubertFeatureReader)

from valle.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from valle.data.collation import get_text_token_collater
from valle.models import add_model_arguments, get_model


def get_args():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--prompts-dir",
        type=str,
        help="Audio prompts which are separated by | and should be aligned with --text-prompts.",
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
        "--text-tokens",
        type=str,
        default="data/tokenized/unique_text_tokens.k2symbols",
        help="Path to the unique text tokens file.",
    )
    parser.add_argument(
        "--semantic-tokens",
        type=str,
        default="data/tokenized/unique_text_tokens.k2symbols",
        help="Path to the unique text tokens file.",
    )
    parser.add_argument(
        "--input-semantic",
        type=str2bool,
        default="False", 
        help="input-semantic.",
    )
    parser.add_argument(
        "--txt-sys-dir",
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
        "--continual",
        type=str2bool,
        default=False,
        help="Do continual task.",
    )
    parser.add_argument(
        "--dir-need2test",
        type=str,
        default="/dev_huaying/zhijun/data/LibriSpeech/filtered_4s_10s-test-spk-wer",
        help="Path to the saved checkpoint.",
    )
    parser.add_argument(
        "--repeat-nums",
        type=int,
        default=1,
        help="synthesis nums",
    )

    return parser.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    print(args)
    text_tokenizer = TextTokenizer(backend=args.text_extractor)
    text_collater = get_text_token_collater(args.text_tokens)
    audio_tokenizer = AudioTokenizer()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    model = get_model(args)

    checkpoint1 = torch.load(args.checkpoint1, map_location=device)
    checkpoint2 = torch.load(args.checkpoint2, map_location=device)
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

    dir = args.dir_need2test

    def get_beijing_time():  
        # 获取当前 UTC 时间  
        current_utc_time = datetime.utcnow()  
        
        # 计算本地时间与 UTC 时间的偏移量  
        local_time = datetime.now()  
        utc_time = datetime.utcnow()  
        time_offset = local_time - utc_time  
        
        # 计算北京时间与 UTC 时间的偏移量  
        beijing_offset = timedelta(hours=8)  
        
        # 计算当前北京时间  
        current_beijing_time = current_utc_time + beijing_offset + time_offset  
        
        # 将 datetime 对象格式化为字符串  
        formatted_beijing_time = current_beijing_time.strftime('%Y_%m_%d_%H_%M_%S')  
        
        return formatted_beijing_time  
      
    def find_related_files(path, gen_txt_file):  
        file_without_ext, _ = os.path.splitext(gen_txt_file)  
        name = file_without_ext[4:]  
        
        prompt_txt_file = f"prompt_{name}.txt"  
        prompt_flac_file = f"prompt_{name}.flac"  
        gt_flac_file = f"gt_{name}.flac"  
        
        related_files = []  

        for root, _, files in os.walk(path):  
            if prompt_txt_file in files:  
                related_files.append(os.path.join(root, prompt_txt_file))  
            if prompt_flac_file in files:  
                related_files.append(os.path.join(root, prompt_flac_file))  
            if gt_flac_file in files:  
                related_files.append(os.path.join(root, gt_flac_file))  
        return related_files  

    def find_gen_files_and_related_files(path, prefix, ext):  
        all_files = []
        for root, dirs, files in os.walk(path):  
            for file in files:  
                if file.startswith(prefix) and file.endswith(ext):  
                    gen_txt_file = os.path.join(root, file)  
                    related_files = find_related_files(root, file)
                    # print(related_files)  
                    related_files.append(os.path.join(root,file))
                    all_files.append(related_files)
        return all_files
        


    def get_audio_files(folder_path):  
        audio_extensions = ('.flac', '.wav')  
        audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(audio_extensions)]  
        return audio_files  
    
    def get_txt_files(folder_path):  
        audio_extensions = '.txt' 
        audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(audio_extensions)]  
        return audio_files  
    
    txt_sys_files = get_txt_files(args.txt_sys_dir)  
    audio_prompts_files = get_audio_files(args.prompts_dir)  


    for prompt_flac_file in audio_prompts_files:

        for txt_sys_file in txt_sys_files:
            prompt_txt_file = prompt_flac_file.replace('.wav', '.txt').replace('.flac', '.txt')

            with open(prompt_txt_file, 'r', encoding='utf-8') as f:  
                txt_prompt = f.read()  
            with open(txt_sys_file, 'r', encoding='utf-8') as f:  
                gen_content = f.read()  
            audio_prompts = []

            print(f"sys {gen_content}")


            encoded_frames = tokenize_audio(audio_tokenizer, prompt_flac_file)
            audio_prompts.append(encoded_frames[0][0])
            audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
            audio_prompts = audio_prompts.to(device)

            text_prompts = txt_prompt
            text = gen_content
            print(f"prompt file :{prompt_flac_file}")
            print(f"prompt txt: {txt_prompt}")
            print(f"sys txt: {text}")
            logging.info(f"synthesize text: {text}")
            # try:
            text_tokens, text_tokens_lens = text_collater(
                [
                    tokenize_text(
                        text_tokenizer, text=f"{text_prompts} {text}".strip()
                    )
                ]
            )
            # except Exception as e:
            #     print(f"An exception occurred: {e}")  
            #     continue 
            _, enroll_x_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=f"{text_prompts}".strip()
                        )
                    ]
                )
            for i in range(args.repeat_nums):
                encoded_frames = model.inference(
                    text_tokens.to(device),
                    text_tokens_lens.to(device),
                    audio_prompts,
                    enroll_x_lens=enroll_x_lens,
                    top_k=args.top_k,
                    temperature=args.temperature,
                )
                samples = audio_tokenizer.decode(
                    [(encoded_frames.transpose(2, 1), None)]
                )
                str1 = args.checkpoint1.split('/')[-1]
                str2 = args.checkpoint2.split('/')[-1]
                model_str = f"ar_{str1}_nar_{str2}"
                audio_prompts_str = prompt_flac_file
                audio_prompts_str= audio_prompts_str.split('/')[-1][:-4]

                semantic_sys_str = txt_sys_file
                semantic_sys_str= semantic_sys_str.split('/')[-1][:-4]
                
                args.output_dir = '/'.join(args.txt_sys_dir.split('/')[:-1]) + '/tts'
                os.makedirs(args.output_dir, exist_ok=True)

                print(args.output_dir)
                # store
                timestamp = current_time.strftime("%Y_%m_%d_%H_%M_%S")

                torchaudio.save(
                    f"{args.output_dir}/{audio_prompts_str}_{semantic_sys_str}_{model_str}_{timestamp}_{i}.wav", samples[0].cpu(), 24000
                )
                torch.cuda.empty_cache()  


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
