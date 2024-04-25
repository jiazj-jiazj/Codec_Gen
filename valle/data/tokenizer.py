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

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Pattern, Union

import numpy as np
import torch
torch.backends.cudnn.enabled = False

import sys

import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from lhotse.features import FeatureExtractor
from lhotse.utils import Seconds, compute_num_frames
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
from academicodec.models.encodec.net3 import SoundStream
from collections import OrderedDict

try:
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials
except Exception:
    pass
#tfnet
from functools import partial, wraps
import math
import torch
from torch import nn, einsum
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from beartype.typing import Optional, Union, List
from beartype import beartype
from einops import rearrange, repeat, reduce
from tqdm import tqdm
from tfnet.tfnet_models_mp_lm.utils import default, exists, maybe, ceil_div, prob_mask_like, get_embeds, eval_decorator, batch_unique_consecutive, append_eos_id, generate_mask_with_prob
from tfnet.tfnet_models_mp.tfnet_v2i_vqvae import TFNet as TFNetV2_interl_VQVAE
from einops import rearrange, repeat, reduce
import yaml

import sys
sys.path.insert(0, "/dev_huaying/zhijun/fairseq")

class PypinyinBackend:
    """PypinyinBackend for Chinese. Most codes is referenced from espnet.
    There are two types pinyin or initials_finals, one is
    just like "ni1 hao3", the other is like "n i1 h ao3".
    """

    def __init__(
        self,
        backend="initials_finals",
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
    ) -> None:
        self.backend = backend
        self.punctuation_marks = punctuation_marks

    def phonemize(
        self, text: List[str], separator: Separator, strip=True, njobs=1
    ) -> List[str]:
        assert isinstance(text, List)
        phonemized = []
        for _text in text:
            _text = re.sub(" +", " ", _text.strip())
            _text = _text.replace(" ", separator.word)
            phones = []
            if self.backend == "pypinyin":
                for n, py in enumerate(
                    pinyin(
                        _text, style=Style.TONE3, neutral_tone_with_five=True
                    )
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)

                        phones.extend(list(py[0]))
                    else:
                        phones.extend([py[0], separator.syllable])
            elif self.backend == "pypinyin_initials_finals":
                for n, py in enumerate(
                    pinyin(
                        _text, style=Style.TONE3, neutral_tone_with_five=True
                    )
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)
                        phones.extend(list(py[0]))
                    else:
                        if py[0][-1].isalnum():
                            initial = get_initials(py[0], strict=False)
                            if py[0][-1].isdigit():
                                final = (
                                    get_finals(py[0][:-1], strict=False)
                                    + py[0][-1]
                                )
                            else:
                                final = get_finals(py[0], strict=False)
                            phones.extend(
                                [
                                    initial,
                                    separator.phone,
                                    final,
                                    separator.syllable,
                                ]
                            )
                        else:
                            assert ValueError
            else:
                raise NotImplementedError
            phonemized.append(
                "".join(phones).rstrip(f"{separator.word}{separator.syllable}")
            )
        return phonemized


class TextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        if backend == "espeak":
            phonemizer = EspeakBackend(
                language,
                punctuation_marks=punctuation_marks,
                preserve_punctuation=preserve_punctuation,
                with_stress=with_stress,
                tie=tie,
                language_switch=language_switch,
                words_mismatch=words_mismatch,
            )
        elif backend in ["pypinyin", "pypinyin_initials_finals"]:
            phonemizer = PypinyinBackend(
                backend=backend,
                punctuation_marks=punctuation_marks + separator.word,
            )
        else:
            raise NotImplementedError(f"{backend}")

        self.backend = phonemizer
        self.separator = separator

    def to_list(self, phonemized: str) -> List[str]:
        fields = []
        for word in phonemized.split(self.separator.word):
            # "ɐ    m|iː|n?"    ɹ|ɪ|z|ɜː|v; h|ɪ|z.
            pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
            fields.extend(
                [p for p in pp if p != self.separator.phone]
                + [self.separator.word]
            )
        assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
            self.separator.phone
        )
        return fields[:-1]

    def __call__(self, text, strip=True) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]

        phonemized = self.backend.phonemize(
            text, separator=self.separator, strip=strip, njobs=1
        )
        # print(phonemized)
        return [self.to_list(p) for p in phonemized]


def tokenize_text(tokenizer: TextTokenizer, text: str) -> List[str]:
    phonemes = tokenizer([text.strip()])
    return phonemes[0]  # k2symbols
def remove_encodec_weight_norm(model):
    from encodec.modules import SConv1d
    from encodec.modules.seanet import SConvTranspose1d, SEANetResnetBlock
    from torch.nn.utils import remove_weight_norm

    encoder = model.encoder.model
    for key in encoder._modules:
        if isinstance(encoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(encoder._modules[key].shortcut.conv.conv)
            block_modules = encoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(encoder._modules[key], SConv1d):
            remove_weight_norm(encoder._modules[key].conv.conv)

    decoder = model.decoder.model
    for key in decoder._modules:
        if isinstance(decoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(decoder._modules[key].shortcut.conv.conv)
            block_modules = decoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(decoder._modules[key], SConvTranspose1d):
            remove_weight_norm(decoder._modules[key].convtr.convtr)
        elif isinstance(decoder._modules[key], SConv1d):
            remove_weight_norm(decoder._modules[key].conv.conv)
class AudioTokenizer:
    """EnCodec audio."""

    def __init__(
        self,
        device: Any = None,
    ) -> None:
        # Instantiate a pretrained EnCodec model
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        remove_encodec_weight_norm(model)

        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")

        self._device = device

        self.codec = model.to(device)
        self.sample_rate = model.sample_rate
        self.channels = model.channels

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(wav.to(self.device))

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(frames)
class AudioTokenizer_encodec_16k:
    """EnCodec audio."""

    def __init__(
        self,
        device: Any = None,
    ) -> None:

        model = SoundStream(
        n_filters=32,
        D=512,
        ratios=[8, 5, 4, 2],
        sample_rate=16000,
        target_bandwidths=[1, 1.5, 2, 4, 6, 12])
        parameter_dict = torch.load("/mnt/users/jiazhijun/models/encodec/encodec_16k_320d.pth")
        new_state_dict = OrderedDict()
        # k 为 module.xxx.weight, v 为权重
        for k, v in parameter_dict.items():
            # 截取`module.`后面的xxx.weight
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)  # load model
        remove_encodec_weight_norm(model)

        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")

        self._device = device

        self.codec = model.to(device)
        self.sample_rate = 16000
        self.channels = 1

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(wav.to(self.device))

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(frames)
class AudioTokenizer_encodec_16k_tfcodec:
    """EnCodec audio."""
    def __init__(
        self,
        device: Any = None,
        config_path = "tfnet/config_6k_tfnetv2_20msvq_hop5_combine4_rd_multi_lingual.yaml",
        tfnet_ckpt = "v-zhijunjia/models/tfcodec/890hrs16khz_tfnet_v2i_vqvae_20msVQ_hop5_combine4_rd1_6kbps/tfnet_v2i_vqvae_combineVQ-iter-514000-val1-0.348327-entropy-118.968-mel-0.067548-recon-0.012822.ckpt"
    ) -> None:
        with open(config_path, "r") as config_file:  
            config = yaml.safe_load(config_file) 
        tfnet_ckpt = tfnet_ckpt
        config_tfnet = config
        if config_tfnet['model_type'] =='tfnetv2_interleave_vqvae':
            tfnet = TFNetV2_interl_VQVAE(config=config_tfnet, )
        # load with tfnet_ckpt
        tfnet.load(tfnet_ckpt)
        tfnet.freeze_encoder()
        tfnet.freeze_codebook()
        tfnet.eval()
        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")

        self._device = device
        self.codec = tfnet.to(device)
        self.sample_rate = 16000
        self.num_quantizers = tfnet.codebook_num
        self.combine_frames = tfnet.combine_frames
        self.channels = 1

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        if len(wav.shape)==3:
            wav = wav[0,:,:]
        _, indices, _ = self.codec(wav.to(self.device))
        clean_token_ids = indices[..., :16]
        return clean_token_ids

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.to(self.device)
        wav, _ = self.codec.decode_from_codebook_indices(frames)
        return wav


def tokenize_audio(tokenizer, audio_path: str):
    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_path)

    # wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
    wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)

    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
    return encoded_frames


@dataclass
class AudioTokenConfig:
    frame_shift: Seconds = 320.0 / 24000
    num_quantizers: int = 8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AudioTokenConfig":
        return AudioTokenConfig(**data)
@dataclass
class AudioTokenConfig_16k:
    frame_shift: Seconds = 320.0 / 16000
    num_quantizers: int = 8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AudioTokenConfig_16k":
        return AudioTokenConfig_16k(**data)

class AudioTokenExtractor(FeatureExtractor):
    name = "encodec"
    config_type = AudioTokenConfig

    def __init__(self, config: Optional[Any] = None):
        super(AudioTokenExtractor, self).__init__(config)
        self.tokenizer = AudioTokenizer()

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> np.ndarray:
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if sampling_rate != self.tokenizer.sample_rate:
            samples = convert_audio(
                samples,
                sampling_rate,
                self.tokenizer.sample_rate,
                self.tokenizer.channels,
            )
        if len(samples.shape) == 2:
            samples = samples.unsqueeze(0)
        else:
            raise ValueError()

        device = self.tokenizer.device
        encoded_frames = self.tokenizer.encode(samples.detach().to(device))
        codes = encoded_frames[0][0]  # [B, n_q, T]
        if True:
            duration = round(samples.shape[-1] / sampling_rate, ndigits=12)
            expected_num_frames = compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=sampling_rate,
            )
            assert abs(codes.shape[-1] - expected_num_frames) <= 1
            codes = codes[..., :expected_num_frames]
        return codes.cpu().squeeze(0).permute(1, 0).numpy()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_quantizers

    def pad_tensor_list(self, tensor_list, device, padding_value=0):
        # 计算每个张量的长度
        lengths = [tensor.shape[0] for tensor in tensor_list]
        # 使用pad_sequence函数进行填充
        tensor_list = [torch.Tensor(t).to(device) for t in tensor_list]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            tensor_list, batch_first=True, padding_value=padding_value
        )
        return padded_tensor, lengths

    def extract_batch(self, samples, sampling_rate, lengths) -> np.ndarray:
        samples = [wav.squeeze() for wav in samples]
        device = self.tokenizer.device
        samples, lengths = self.pad_tensor_list(samples, device)
        samples = samples.unsqueeze(1)

        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if len(samples.shape) != 3:
            raise ValueError()
        if sampling_rate != self.tokenizer.sample_rate:
            samples = [
                convert_audio(
                    wav,
                    sampling_rate,
                    self.tokenizer.sample_rate,
                    self.tokenizer.channels,
                )
                for wav in samples
            ]
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.tokenizer.encode(samples.detach().to(device))
        encoded_frames = encoded_frames[0][0]  # [B, n_q, T]
        batch_codes = []
        for b, length in enumerate(lengths):
            codes = encoded_frames[b]
            duration = round(length / sampling_rate, ndigits=12)
            expected_num_frames = compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=sampling_rate,
            )
            batch_codes.append(codes[..., :expected_num_frames])
        return [codes.cpu().permute(1, 0).numpy() for codes in batch_codes]
class AudioTokenExtractor_16k(FeatureExtractor):
    name = "encodec_16k"
    config_type = AudioTokenConfig_16k

    def __init__(self, config: Optional[Any] = None):
        super(AudioTokenExtractor_16k, self).__init__(config)
        self.tokenizer = AudioTokenizer_encodec_16k()

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> np.ndarray:
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if sampling_rate != self.tokenizer.sample_rate:
            samples = convert_audio(
                samples,
                sampling_rate,
                self.tokenizer.sample_rate,
                self.tokenizer.channels,
            )
        if len(samples.shape) == 2:
            samples = samples.unsqueeze(0)
        else:
            raise ValueError()

        device = self.tokenizer.device
        encoded_frames = self.tokenizer.encode(samples.detach().to(device))

        encoded_frames = torch.transpose(encoded_frames[:8], 0, 1)  # [B, n_q, T]
        codes = encoded_frames[0][0]  # [B, n_q, T]
        if True:
            duration = round(samples.shape[-1] / sampling_rate, ndigits=12)
            expected_num_frames = compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=sampling_rate,
            )
            assert abs(codes.shape[-1] - expected_num_frames) <= 1
            codes = codes[..., :expected_num_frames]
        return codes.cpu().squeeze(0).permute(1, 0).numpy()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_quantizers

    def pad_tensor_list(self, tensor_list, device, padding_value=0):
        # 计算每个张量的长度
        lengths = [tensor.shape[0] for tensor in tensor_list]
        # 使用pad_sequence函数进行填充
        tensor_list = [torch.Tensor(t).to(device) for t in tensor_list]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            tensor_list, batch_first=True, padding_value=padding_value
        )
        return padded_tensor, lengths

    def extract_batch(self, samples, sampling_rate, lengths) -> np.ndarray:

        if sampling_rate != self.tokenizer.sample_rate:
            samples = [
                convert_audio(
                    wav,
                    sampling_rate,
                    self.tokenizer.sample_rate,
                    self.tokenizer.channels,
                )
                for wav in samples
            ]
        samples = [wav.squeeze() for wav in samples]
        device = self.tokenizer.device

        samples, lengths = self.pad_tensor_list(samples, device)
        samples = samples.unsqueeze(1)

        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if len(samples.shape) != 3:
            raise ValueError()
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.tokenizer.encode(samples.detach().to(device))
        encoded_frames = torch.transpose(encoded_frames[:8], 0, 1)  # [B, n_q, T]
        batch_codes = []
        for b, length in enumerate(lengths):
            codes = encoded_frames[b]
            duration = round(length / sampling_rate, ndigits=12)
            expected_num_frames = compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=sampling_rate,
            )
            batch_codes.append(codes[..., :expected_num_frames])
        return [codes.cpu().permute(1, 0).numpy() for codes in batch_codes]


class AudioTokenExtractor_16k_tfcodec(FeatureExtractor):
    name = "tfcodec_16k"
    config_type = AudioTokenConfig_16k

    def __init__(self, config: Optional[Any] = None, tfnet_ckpt=None):
        super(AudioTokenExtractor_16k_tfcodec, self).__init__(config)
        self.tokenizer = AudioTokenizer_encodec_16k_tfcodec(tfnet_ckpt=tfnet_ckpt)
        self.num_quantizers = self.tokenizer.num_quantizers
        self.sample_rate=16000
        self.channels=1

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> np.ndarray:
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if sampling_rate != self.tokenizer.sample_rate:
            samples = convert_audio(
                samples,
                sampling_rate,
                self.tokenizer.sample_rate,
                self.tokenizer.channels,
            )
        if len(samples.shape) == 2:
            samples = samples.unsqueeze(0)
        else:
            raise ValueError()

        device = self.tokenizer.device
        encoded_frames = self.tokenizer.encode(samples.detach().to(device))
        
        encoded_frames = torch.transpose(encoded_frames, 1, 2)  # [B, n_q, T]
        codes = encoded_frames  # [B, n_q, T]
        return codes.cpu().squeeze(0).permute(1, 0).numpy()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_quantizers

    def pad_tensor_list(self, tensor_list, device, padding_value=0):
        # 计算每个张量的长度
        lengths = [tensor.shape[0] for tensor in tensor_list]
        # 使用pad_sequence函数进行填充
        tensor_list = [torch.Tensor(t).to(device) for t in tensor_list]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            tensor_list, batch_first=True, padding_value=padding_value
        )
        return padded_tensor, lengths

    def extract_batch(self, samples, sampling_rate=16000, lengths=None) -> np.ndarray:
        if sampling_rate != self.tokenizer.sample_rate:
            samples = [
                convert_audio(
                    wav,
                    sampling_rate,
                    self.tokenizer.sample_rate,
                    self.tokenizer.channels,
                )
                for wav in samples
            ]
        samples = [wav.squeeze() for wav in samples]
        device = self.tokenizer.device

        samples, lengths = self.pad_tensor_list(samples, device)
        # samples = samples.unsqueeze(1)
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if len(samples.shape) != 2:
            raise ValueError()
        # Extract discrete codes from EnCodec
        # with open(f'clean_samples_tokenizer_0.txt', 'w') as f:  
        #     for row in samples:  
        #         f.write(' '.join([str(elem) for elem in row]) + '\n')  
        with torch.no_grad():
            encoded_frames = self.tokenizer.encode(samples.detach().to(device))
        encoded_frames = torch.transpose(encoded_frames, 1, 2)  # [B, n_q, T]
        # encoded_frames = torch.transpose(encoded_frames[:8], 0, 1)  # [B, n_q, T]
        batch_codes = []
        for b, length in enumerate(lengths):
            codes = encoded_frames[b]
            expected_num_frames = math.ceil((int(length/80)+1)//self.tokenizer.combine_frames)
            batch_codes.append(codes[..., :expected_num_frames])
        return [codes.cpu().permute(1, 0).numpy() for codes in batch_codes]
import fairseq

import logging 
import os
import sys
from fairseq.data.audio.audio_utils import get_features_or_waveform
import torch.nn.functional as F
import joblib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")
class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)

class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk


    def read_audio(self, path, ref_len=None):
        wav = get_features_or_waveform(path, need_waveform=True, use_sample_rate=self.task.cfg.sample_rate)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        # print(path)
        x = self.read_audio(path, ref_len=ref_len)
        # x = torch.cat((x, x), dim=0)
        # print(x.shape)
        # quit()
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)
            # x = torch.cat((x, x), dim=0)        
            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                # print(f"x_chunk_{x_chunk.shape}")
                # quit()
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )

                feat.append(feat_chunk)
        # quit()
        return torch.cat(feat, 1).squeeze(0)

import soundfile as sf

def save_audio(wav: torch.Tensor,
               path,
               sample_rate: int,
               rescale: bool=False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    wav = wav.squeeze().cpu().numpy()
    sf.write(path, wav, sample_rate)


if __name__ == "__main__":
    # #tfcodec
    # model = AudioTokenExtractor_16k_tfcodec()
    # import librosa
    # wav1, sr = librosa.load("/mnt/zhijun/Accents/combine_L1_L2/train_india_split/train/SVBI/wav/arctic_a0235.wav", sr=16000)
    # wav2, sr = librosa.load("/mnt/zhijun/Accents/combine_L1_L2/train_native/total/cmu_us_bdl_arctic/wav/arctic_a0235.wav", sr=16000)
    # wav =[wav1, wav2]

    # codes = model.extract_batch(wav)
    # idx=0
    # for code in codes:
    #     print(code)

    #     # with open(f'clean_token_id_tokenizer_{idx}.txt', 'w') as f:  
    #     #     for row in code:  
    #     #         f.write(' '.join([str(elem) for elem in row]) + '\n')  

    #     # code = torch.tensor(code)
    #     # code = code.unsqueeze(dim=0)
    #     # print(code)
    #     # wav = model.tokenizer.decode(code)
    #     # wav = wav.detach().cpu().numpy()[0]
    #     # import soundfile as sf
    #     # sf.write(f"/mnt/users/jiazhijun/valle_23_4_22/tfnet/testv2_arctic_{idx}.wav", wav, 16000)
    #     # idx+=1

    ckpt_path = "/dev_huaying/zhijun/fairseq/examples/hubert/tests/hubert_base_ls960.pt"
    layer = 9
    km_path = "/dev_huaying/zhijun/fairseq/examples/hubert/tests/hubert_base_ls960_L9_km500.bin"
    wav_path = "/mnt/zhijun/Accents/combine_L1_L2/train_native/total/cmu_us_bdl_arctic/wav/arctic_a0354.wav"
    reader = HubertFeatureReader(ckpt_path, layer)
    apply_kmeans = ApplyKmeans(km_path)
    

    feats = reader.get_feats(wav_path)


    lab = apply_kmeans(feats).tolist()
    print(lab)
    # model = EncodecModel.encodec_model_24khz()
    # model.set_target_bandwidth(6.0)

    # model1 = SoundStream(
    #         n_filters=32,
    #         D=512,
    #         ratios=[8, 5, 4, 2],
    #         sample_rate=16000,
    #         target_bandwidths=[1, 1.5, 2, 4, 6, 12])

    # parameter_dict = torch.load("/dev_huaying/zhijun/data/encodec_16k_320d.pth")
    # new_state_dict = OrderedDict()
    # # k 为 module.xxx.weight, v 为权重
    # for k, v in parameter_dict.items():
    #     # 截取`module.`后面的xxx.weight
    #     name = k[7:]
    #     new_state_dict[name] = v
    # model1.load_state_dict(new_state_dict)  # load model
    # # remove_encodec_weight_norm(model1)
    # import librosa
    # wav, sr = librosa.load("/dev_huaying/zhijun/fairseq/examples/hubert/tests/6313-76958-0021.flac", sr=16000)
    # wav = torch.tensor(wav).unsqueeze(0)
    # wav = wav.unsqueeze(1)

    # # print(wav.shape)
    # codes_raw = model.encode(wav)
    # print(codes_raw[0][0])
    # codes_raw1 = model1.encode(wav)
    # codes_raw1 = codes_raw1[:8]
    # out = model1.decode(codes_raw1)

    # out = out.detach().cpu().squeeze(0)
    # save_audio(wav=out, path="/dev_huaying/zhijun/valle_23_4_22/valle/data/test1.wav", sample_rate=16000, rescale=True)
    # print('finish decompressing')

    # codes_raw1 = codes_raw1[:8]
    # codes_raw1 = torch.transpose(codes_raw1, 0, 1)    


    # print(codes_raw1)
    # remove_encodec_weight_norm(model)
    # codes_norm = model.encode(wav)
    # print(codes_norm.shape)
