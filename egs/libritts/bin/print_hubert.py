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
        "--continual",
        type=str2bool,
        default=False,
        help="Do continual task.",
    )

    return parser.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    semantic_token_collater = get_text_token_collater(args.semantic_tokens)

    ckpt_path = "/dev_huaying/zhijun/fairseq/examples/hubert/tests/hubert_base_ls960.pt"
    layer = 9
    km_path = "/dev_huaying/zhijun/fairseq/examples/hubert/tests/hubert_base_ls960_L9_km500.bin"
    reader = HubertFeatureReader(ckpt_path, layer)
    apply_kmeans = ApplyKmeans(km_path)
    # text_tokenizer = TextTokenizer(backend=args.text_extractor)
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

    import os  
    
    def get_audio_files(folder_path):  
        audio_extensions = ('.flac', '.wav')  
        audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(audio_extensions)]  
        return audio_files  
 
    
    audio_files_A = get_audio_files(args.semantic_sys_dir)  
    # audio_files_B = get_audio_files(args.audio_prompts_dir)  
    
    for semantic_prompts_file in audio_files_A:  

        semantic_prompts_feats = reader.get_feats(semantic_prompts_file)
        semantic_prompts = apply_kmeans(semantic_prompts_feats).tolist()

        output_dir = '/'.join(args.semantic_sys_dir.split('/')[:-1]) + '/hubert_values_source'
        output_filename = os.path.splitext(semantic_prompts_file.split('/')[-1])[0] + ".txt"  
        output_path = os.path.join(output_dir, str(output_filename))  
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as output_file:  
            output_file.write(" ".join(str(x) for x in semantic_prompts)) 


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
