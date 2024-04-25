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

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Path to the manifest files",
    )

    parser.add_argument(
        "--wav-path",
        type=str,
        default="",
        help="Path to each wav",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="Path to the tokenized files",
    )
    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )
    parser.add_argument(
        "--audio-extractor",
        type=str,
        default="Encodec",
        help="Encodec or Fbank or Tfcodec",
    )
    parser.add_argument(
        "--dataset-parts",
        type=str,
        default="dev-clean test-clean",
        help="Space separated dataset parts",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="libritts",
        help="prefix of the manifest file",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="jsonl.gz",
        help="suffix of the manifest file",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=400.0,
        help="The maximum number of audio seconds in a batch."
        "Determines batch size dynamically.",
    )
    parser.add_argument(
        "--acoustic-sample",
        type=int,
        default=16000,
        help="Sample of input speech.",
    )
    parser.add_argument(
        "--input-language",
        type=int,
        default="1",
        help="0->english, 1->chinese",
    )
    parser.add_argument(
        "--semantic-layer",
        type=int,
        default=9,
        help="suffix of the manifest file",
    )
    parser.add_argument(
        "--tfnet-ckpt",
        type=str,
        default="../data/valle-tensorboard-models/other_models/tfcodec/890hrs16khz_tfnet_v2i_vqvae_20msVQ_hop5_combine4_rd1_6kbps/tfnet_v2i_vqvae_combineVQ-iter-514000-val1-0.348327-entropy-118.968-mel-0.067548-recon-0.012822.ckpt",
        help="suffix of the manifest file",
    )
    parser.add_argument(
        "--part-th",
        type=int,
        default=0,
        help="th file",
    )
    parser.add_argument(
        "--add-semantic",
        type=str2bool,
        default='False')
    return parser.parse_args()

def main():
    args = get_args()

    dataset_parts = args.dataset_parts.replace("--dataset-parts", "").strip()
    if dataset_parts == "all":  # LibriTTS
        dataset_parts = [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]
    else:
        dataset_parts = dataset_parts.replace("-p", "").strip().split(" ")

    assert len(dataset_parts) >= 1

    # print(dataset_parts)

    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=args.src_dir,
        prefix=args.prefix,
        suffix=args.suffix,
        types=["recordings", "supervisions", "cuts"],
    )
    text_tokenizer = None
    if args.text_extractor:
        text_tokenizer = TextTokenizer(backend=args.text_extractor)

    audio_extractor = None
    if args.audio_extractor:
        if args.audio_extractor == "Encodec":
            if args.acoustic_sample == 24000:
                audio_extractor = AudioTokenExtractor(AudioTokenConfig())
            elif args.acoustic_sample == 16000:
                audio_extractor = AudioTokenExtractor_16k(AudioTokenConfig_16k())

        elif args.audio_extractor == "Tfcodec":
            if args.acoustic_sample == 16000:
                audio_extractor = AudioTokenExtractor_16k_tfcodec(AudioTokenConfig_16k(), tfnet_ckpt=args.tfnet_ckpt)

        else:

            assert args.audio_extractor == "Fbank"
            audio_extractor = get_fbank_extractor()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    unique_symbols = set()
    # num_jobs = min(32, os.cpu_count()//2)
    num_jobs = 3

    print(f"num_jobs:{num_jobs}")
    logging.info(f"dataset_parts: {dataset_parts} manifests {len(manifests)}")

    prefix = args.prefix
    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"

    if args.input_language==1:
        if args.semantic_layer==9:
            km_path = "/mnt/users/jiazhijun/chinese_speech_pretrain/hubert_kmeans/hubert_base_iter2_32gpu_l9/model.mdl"
        elif args.semantic_layer==6:
            km_path = "/mnt/users/jiazhijun/chinese_speech_pretrain/hubert_kmeans/hubert_base_iter1_32gpu_l6/model.mdl"

        model_path="TencentGameMate/chinese-hubert-base"

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        model = HubertModel.from_pretrained(model_path)
        device = "cuda"
        # print(model)
        model = model.to(device)
        model = model.half()
        model.eval()

        apply_kmeans = ApplyKmeans(km_path)  
    
        
    elif args.input_language==0:
        ckpt_path = "/dev_huaying/zhijun/models/hubert/hubert_base_ls960.pt"
        layer = 9
        km_path = "/dev_huaying/zhijun/models/hubert/hubert_base_ls960_L9_km500.bin"
        # reader = HubertFeatureReader(ckpt_path, layer)
        # apply_kmeans = ApplyKmeans(km_path)    


    with get_executor() as ex:
        # for partition, m in manifests.items():

            try:
                i=args.part_th
                cut_set = CutSet.from_file(f'/raid/dataset/lhotse_dataset/mls_train_lhotse_dataset/mls-english_cuts_train_{i}.jsonl.gz')
                cut_set.describe()
            except Exception:
                cut_set = m["cuts"]

            # AudioTokenizer
            if args.audio_extractor:
                if args.audio_extractor == "Encodec":
                    storage_path = (
                        f"{args.output_dir}/{args.prefix}_encodec_{partition}_{i}"
                    )
                    # print(storage_path)
                    # quit()
                elif args.audio_extractor == "Tfcodec":
                    storage_path = (
                        f"{args.output_dir}/{args.prefix}_tfcodec_{partition}_{i}"
                    )
                
                else:
                    storage_path = (
                        f"{args.output_dir}/{args.prefix}_fbank_{partition}_{i}"
                    )
                # if args.prefix.lower() in ["ljspeech", "aishell", "baker"]:

                #!!whether need to resample
                # cut_set = cut_set.resample(args.acoustic_sample)
                    # https://github.com/lifeiteng/vall-e/issues/90
                    # if args.prefix == "aishell":
                    #     # NOTE: the loudness of aishell audio files is around -33
                    #     # The best way is datamodule --on-the-fly-feats --enable-audio-aug
                    #     cut_set = cut_set.normalize_loudness(
                    #         target=-20.0, affix_id=True
                #     #     )
                with torch.no_grad():
                    if (
                        torch.cuda.is_available()
                        and args.audio_extractor == "Encodec"
                    ):
                        cut_set = cut_set.compute_and_store_features_batch(
                            extractor=audio_extractor,
                            storage_path=storage_path,
                            num_workers=num_jobs,
                            batch_duration=args.batch_duration,
                            collate=False,
                            overwrite=True,
                            storage_type=NumpyHdf5Writer,
                        )
                    elif (
                        torch.cuda.is_available()
                        and args.audio_extractor == "Tfcodec"
                    ):
                        cut_set = cut_set.compute_and_store_features_batch(
                            extractor=audio_extractor,
                            storage_path=storage_path,
                            num_workers=num_jobs,
                            batch_duration=args.batch_duration,
                            collate=False,
                            overwrite=True,
                            storage_type=NumpyHdf5Writer,
                        )
                    else:
                        cut_set = cut_set.compute_and_store_features(
                            extractor=audio_extractor,
                            storage_path=storage_path,
                            num_jobs=num_jobs if ex is None else 64,
                            executor=ex,
                            storage_type=NumpyHdf5Writer,
                        )
                    # cuts_filename = f"{prefix}cuts_{partition}_{i}.{args.suffix}"
                    # cut_set.to_file(f"{args.output_dir}/{cuts_filename}")
            # TextTokenizer
            if args.prefix == "l1_l2_arctic":
                import json
                with open("/mnt/zhijun/Accents/combine_L1_L2/acoustic_tokens_dic/native_l1_l2_arctic_tfcodec_acoustics_dic_v2.json", "r") as json_file:  
                    loaded_dict = json.load(json_file) 
                with open("/mnt/zhijun/Accents/combine_L1_L2/acoustic_tokens_dic/native_l1_l2_arctic_semantic_dic_v2.json", "r") as json_file_1:  
                    semantic_loaded_dict = json.load(json_file_1)              
            iteration=0
            if args.text_extractor:
                if (
                    args.prefix == "baker"
                    and args.text_extractor == "labeled_pinyin"
                ):
                    for c in tqdm(cut_set):
                        phonemes = c.supervisions[0].custom["tokens"]["text"]
                        unique_symbols.update(phonemes)
                else:
                    for c in tqdm(cut_set): 
                        if c.supervisions[0].custom ==None:
                            c.supervisions[0].custom = {}
                        #also need to change in this :/lhotse/audio.py 
                        if args.wav_path == "":
                            wav_path = c.recording.sources[0].source
                        else:
                            wav_path = c.recording.sources[0].source
                            wav_path = args.wav_path + '/'.join(wav_path.split('/')[-4:])

                        if args.input_language==1:
                            def remove_spaces(text: str) -> str:  
                                return text.replace(" ", "")  
                                
                            wav, sr = sf.read(wav_path)
                            target_sr = 16000 
                            if sr != target_sr:
                                wav = librosa.resample(wav, sr, target_sr)  

                            input_values = feature_extractor(wav, return_tensors="pt").input_values
                            input_values = input_values.half()
                            input_values = input_values.to(device)
                            with torch.no_grad():
                                outputs = model(input_values, output_hidden_states=True)
                                last_hidden_state = outputs.hidden_states[args.semantic_layer]
                                last_hidden_state = torch.squeeze(last_hidden_state, dim=0)  
                                last_hidden_state = last_hidden_state.to(torch.float32)  
                                lab = apply_kmeans(last_hidden_state).tolist()  
    
                        elif args.input_language==0:
                            pass
                            # feat = reader.get_feats(wav_path)
                            # lab = apply_kmeans(feat).tolist()

                        if args.prefix == "ljspeech":
                            text = c.supervisions[0].custom["normalized_text"]
                            text = text.replace("”", '"').replace("“", '"')
                            phonemes = tokenize_text(text_tokenizer, text=text)
                        elif args.prefix == "aishell":
                            text = c.supervisions[0].text
                            text = remove_spaces(text) 
                            phonemes = tokenize_text(
                                text_tokenizer, text=c.supervisions[0].text
                            )
                            c.supervisions[0].custom = {}
                        elif args.prefix == "libritts":
                            assert args.prefix == "libritts"
                            phonemes = tokenize_text(
                                text_tokenizer, text=c.supervisions[0].text
                            )
                        elif args.prefix == "aishell2":
                            phonemes = tokenize_text(
                                text_tokenizer, text=c.supervisions[0].text
                            )
                            c.supervisions[0].custom = {}    
                        elif args.prefix == "aishell3":
                            phonemes = tokenize_text(
                                text_tokenizer, text=c.supervisions[0].text
                            )
                            c.supervisions[0].custom = {}    
                        elif args.prefix == "l1_l2_arctic":
                            phonemes = tokenize_text(
                                text_tokenizer, text=c.supervisions[0].text
                            )
                            file_name = wav_path.split('/')[-1]
                            acoustic_tokens=loaded_dict[file_name]
                            native_semantic_tokens=semantic_loaded_dict[file_name]
                            assert acoustic_tokens, "Error: acoustic_tokens list is empty"
                            assert native_semantic_tokens, "Error: semantic_tokens list is empty"
                            accent = c.supervisions[0].custom["accent"]
                            c.supervisions[0].custom = {'accent':accent, 'acoustic_tokens':acoustic_tokens, 'acoustic_tokens_nums':len(acoustic_tokens), 'native_semantic_tokens': native_semantic_tokens}  
                        else:
                            phonemes = tokenize_text(
                                text_tokenizer, text=c.supervisions[0].text
                            )
                        if args.add_semantic is True:
                            c.supervisions[0].custom["tokens"] = {"text": phonemes, "semantic_tokens": lab}
                        else:
                            c.supervisions[0].custom["tokens"] = {"text": phonemes}    # print(f"text: {phonemes}, lab: {lab}")
                        unique_symbols.update(phonemes)

            cut_set.to_file(f"{args.output_dir}/mls-english_cuts_train_{i}.jsonl.gz")

            unique_phonemes = SymbolTable()
            for s in sorted(list(unique_symbols)):
                unique_phonemes.add(s)
            logging.info(f"{len(unique_symbols)} unique phonemes: {unique_symbols}")

            unique_phonemes_file = f"{args.output_dir}/unique_text_tokens_{i}.k2symbols"
            unique_phonemes.to_file(unique_phonemes_file)


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
    
