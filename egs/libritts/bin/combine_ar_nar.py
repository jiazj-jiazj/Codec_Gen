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
        "--text-prompts",
        type=str,
        default="",
        help="Text prompts which are separated by |.",
    )

    parser.add_argument(
        "--audio-prompts",
        type=str,
        default="",
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
    
    # 保存修改后的 checkpoint1  
    # torch.save(checkpoint1, 'modified_checkpoint1.pth')  
    

    missing_keys1, unexpected_keys1 = model.load_state_dict(
        checkpoint1["model"], strict=True
    )
    assert not missing_keys1

    model.to(device)
    model.eval()


    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    text_prompts = " ".join(args.text_prompts.split("|"))

    audio_prompts = []
    if args.audio_prompts:
        for n, audio_file in enumerate(args.audio_prompts.split("|")):
            encoded_frames = tokenize_audio(audio_tokenizer, audio_file)
            if False:
                samples = audio_tokenizer.decode(encoded_frames)
                torchaudio.save(
                    f"{args.output_dir}/p{n}.wav", samples[0], 24000
                )

            audio_prompts.append(encoded_frames[0][0])

        assert len(args.text_prompts.split("|")) == len(audio_prompts)
        audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
        audio_prompts = audio_prompts.to(device)

    if os.path.isfile(args.text):  # for demos
        # https://github.com/lifeiteng/lifeiteng.github.com/blob/main/valle/prepare.py
        with open(args.text) as f:
            for line in f:
                fields = line.strip().split("\t")
                assert len(fields) == 4
                prompt_text, prompt_audio, text, audio_path = fields
                logging.info(f"synthesize text: {text}")
                text_tokens, text_tokens_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=f"{prompt_text} {text}".strip()
                        )
                    ]
                )
                _, enroll_x_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=f"{prompt_text}".strip()
                        )
                    ]
                )

                audio_prompts = tokenize_audio(audio_tokenizer, prompt_audio)
                audio_prompts = audio_prompts[0][0].transpose(2, 1).to(device)

                # synthesis
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
                # store
                torchaudio.save(audio_path, samples[0].cpu(), 24000)
        return

    for n, text in enumerate(args.text.split("|")):
        logging.info(f"synthesize text: {text}")
        # print(text_tokenizer)
        try:
            text_tokens, text_tokens_lens = text_collater(
                [
                    tokenize_text(
                        text_tokenizer, text=f"{text_prompts} {text}".strip()
                    )
                ]
            )
        except Exception as e:
            print(f"An exception occurred: {e}")  
            continue 

        # synthesis
        if args.continual:
            assert text == ""
            encoded_frames = model.continual(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                audio_prompts,
            )
        else:
            enroll_x_lens = None
            if text_prompts:
                _, enroll_x_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=f"{text_prompts}".strip()
                        )
                    ]
                )
            encoded_frames = model.inference(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                audio_prompts,
                enroll_x_lens=enroll_x_lens,
                top_k=args.top_k,
                temperature=args.temperature,
            )

        if audio_prompts != []:
            samples = audio_tokenizer.decode(
                [(encoded_frames.transpose(2, 1), None)]
            )
            str1 = args.checkpoint1.split('/')[-1]
            str2 = args.checkpoint2.split('/')[-1]
            model_str = f"ar_{str1}_nar_{str2}"
            text_prompts_str = args.text_prompts
            audio_prompts_str = args.audio_prompts
            audio_prompts_str= audio_prompts_str.split('/')[-1][:-4]
            text = args.text
            
            # store
            torchaudio.save(
                f"{args.output_dir}/{text_prompts_str[:5]}_{audio_prompts_str[-5:]}_{text[:5]}_{model_str}_{timestamp}.wav", samples[0].cpu(), 24000
            )
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
