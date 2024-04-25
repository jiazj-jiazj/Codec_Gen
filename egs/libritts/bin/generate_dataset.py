
#!/usr/bin/env python3
# Copyright    2021-2022  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo)
# Copyright    2023                           (authors: Feiteng Li)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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
Usage:
python3 bin/trainer.py \
    --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
    --max-duration 40 --model-name valle \
    --exp-dir exp/valle
    --dtype "bfloat16" \
"""
import os  
import sys
# 获取当前工作目录  
# current_working_directory = os.getcwd()  
  
# # 将当前工作目录添加到 sys.path 的首位  
# sys.path.insert(0, current_working_directory) 
import random  
from collections import defaultdict  
from lhotse import CutSet, load_manifest_lazy
import argparse
import copy
import logging
import sys  
from pathlib import Path 
import os
from contextlib import nullcontext
from lhotse import CutSet, Features, MonoCut, Recording, AudioSource
from typing import Union, Optional, List  
from tqdm import tqdm
import torch.distributed as dist
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import random
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union
import json
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import AttributeDict, MetricsTracker, setup_logger, str2bool
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.enabled = False
from valle.data import TtsDataModule
from valle.models import add_model_arguments, get_model
from valle.modules.optim import Eden, Eve, ScaledAdam
from valle.modules.scheduler import get_scheduler
from valle.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
    AudioTokenizer_encodec_16k_tfcodec
)
from valle.data.collation import get_text_token_collater
LRSchedulerType = torch.optim.lr_scheduler._LRScheduler
run_now = None
import re
import torchaudio
torch.backends.cudnn.enabled = False  

# if global_variable== True:
#     from azureml.core.run import Run
#     run_now = Run.get_context()

def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for module in model.modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )
    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp/valle_dev",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )
    parser.add_argument(
        "--optimizer-name",
        type=str,
        default="ScaledAdam",
        help="The optimizer.",
    )
    parser.add_argument(
        "--scheduler-name",
        type=str,
        default="Eden",
        help="The scheduler.",
    )
    parser.add_argument(
        "--base-lr", type=float, default=0.05, help="The base learning rate."
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )
    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )
    parser.add_argument(
        "--save-every-n",
        type=int,
        default=10000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
        """,
    )
    parser.add_argument(
        "--valid-interval",
        type=int,
        default=10000,
        help="""Run validation if batch_idx % valid_interval is 0.""",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="""Run validation if batch_idx % valid_interval is 0.""",
    )
    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=40,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )
    parser.add_argument(
        "--average-period",
        type=int,
        default=0,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )
    parser.add_argument(
        "--accumulate-grad-steps",
        type=int,
        default=1,
        help="""update gradient when batch_idx_train % accumulate_grad_steps == 0.
        """,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Training dtype: float32 bfloat16 float16.",
    )
    parser.add_argument(
        "--newfile-suffix",
        type=str,
        default="test1",
        help="cuts_train.jsonl.gz->cuts_train_{suffix}.jsonl.gz. cuts_dev.jsonl.gz->cuts_dev_{suffix}.jsonl.gz.",
    )
    parser.add_argument(
        "--filter-min-duration",
        type=float,
        default=0.0,
        help="Keep only utterances with duration > this.",
    )
    parser.add_argument(
        "--filter-max-duration",
        type=float,
        default=20.0,
        help="Keep only utterances with duration < this.",
    )
    parser.add_argument(
        "--train-stage",
        type=int,
        default=0,
        help="""0: train all modules, For VALL-E, support 1: AR Decoder 2: NAR Decoder(s)
        """,
    )
    parser.add_argument(
        "--sheduler-steps",
        type=int,
        default=5000,
        help="""0: train all modules, For VALL-E, support 1: AR Decoder 2: NAR Decoder(s)
        """,
    )
    parser.add_argument(
        "--sheduler-epochs",
        type=int,
        default=4,
        help="""0: train all modules, For VALL-E, support 1: AR Decoder 2: NAR Decoder(s)
        """,
    )
    parser.add_argument(
        "--visualize",
        type=str2bool,
        default=False,
        help="visualize model results in eval step.",
    )
    parser.add_argument(
        "--is-local",
        type=str2bool,
        default=False,
        help="visualize model results in eval step.",
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
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )
    parser.add_argument(
        "--checkpoint-ar",
        type=str,
        default="/dev_huaying/zhijun/data/valle-tensorboard-models/ar/Name_VALLE_max-duration_80_dtype_float32_base-lr_0.01_world-size_8_train-stage_1_echo_150_start_echo_1_2023_05_29_03_00_16/best-valid-loss.pt",
        help="path of ar_model checkpoint",
    )
    parser.add_argument(
        "--checkpoint-nar",
        type=str,
        default="/dev_huaying/zhijun/data/valle-tensorboard-models/nar/Name_VALLE_max-duration_70_dtype_float32_base-lr_0.01_world-size_8_train-stage_2_echo_150_start_echo_1_2023_05_29_01_26_40/best-valid-loss.pt",
        help="path of nar_model checkpoint",
    )
    parser.add_argument(
        "--text-prompts",
        nargs="+",
        type=str,
        default=["looked out and tens the fives.", "Windows。The woman shout across over that."],
        help="Text prompts which are separated by |.",
    )
    parser.add_argument(
        "--audio-prompts",
        nargs="+",
        type=str,
        default=["/dev_huaying/zhijun/data/test_valle_naturalspeech2_yourtts_styleTTS/test1/reference_LibriSpeech_1st_txt_looktown.wav", "/dev_huaying/zhijun/data/test_valle_naturalspeech2_yourtts_styleTTS/test1/reference_LibriSpeech_2nd_txt_windows.wav"],
        help="Audio prompts which are separated by | and should be aligned with --text-prompts.",
    )
    parser.add_argument(
        "--text",
        nargs="+",
        type=str,
        default=["To get up and running quickly just follow the steps below.", "say i believe in you, and you also believe in me"],
        help="Text to be synthesized.",
    )
    parser.add_argument(
        "--min-test-epoch",
        type=int,
        default=10,
        help = "test from this epoch"
    )
    parser.add_argument('--test-demo', type=str2bool, default=False)  
    
    parser.add_argument(
        "--restore-file-name",
        type=str,
        default="/dev_huaying/zhijun/data/valle-tensorboard-models/nar/Name_VALLE_max-duration_70_dtype_float32_base-lr_0.01_world-size_8_train-stage_2_echo_150_start_echo_1_2023_05_29_01_26_40/best-valid-loss.pt",
        help="path of restore model checkpoint",
    )
    parser.add_argument(
        "--restore",
        type=str2bool,
        default=False,
        help="restore.",
    )
    parser.add_argument(
        "--input-semantic",
        type=str2bool,
        default=False,
        help="input-semantic.",
    )
    parser.add_argument(
        "--is-pretrain",
        type=str2bool,
        default=False,
        help="input-semantic.",
    )
    parser.add_argument(
        "--semantic-remove",
        type=str2bool,
        default=False,
        help="semantic-change.",
    )
    parser.add_argument(
        "--only-autoregressive",
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
        "--shared-linear",
        type=str2bool,
        default=False,
        help="only one linear no matter how num-quantizers",
    )
    parser.add_argument(
        "--random-tgt-spk",
        type=str2bool,
        default=False,
        help="ac semantic target spks is random ",
    )
    parser.add_argument(
        "--semantic-tokens",
        type=str,
        help="path of semantic-tokens",
    )
    parser.add_argument(
        "--tgt-spk-name",
        type=str,
        default="cmu_us_bdl_arctic",
        help="path of semantic-tokens",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default="/mnt/zhijun/data/LibriTTS",
        help="audio-source",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/mnt/zhijun/data/LibriTTS",
        help="audio-source",
    )   
    parser.add_argument(
        "--ref-tokens-path",
        type=str,
        default="/mnt/zhijun/data/LibriTTS",
        help="audio-source",
    )  
    parser.add_argument(
        "--config-path",
        type=str,
        default="tfnet/config_6k_tfnetv2_20msvq_hop5_combine4_rd_multi_lingual.yaml",
        help="audio-source",
    )  
    parser.add_argument(
        "--tfnet-ckpt",
        type=str,
        default="/dev_huaying/zhijun/models/tfcodec/890hrs16khz_tfnet_v2i_vqvae_20msVQ_hop5_combine4_rd1_6kbps/tfnet_v2i_vqvae_combineVQ-iter-514000-val1-0.348327-entropy-118.968-mel-0.067548-recon-0.012822.ckpt",
        help="audio-source",
    )  
    parser.add_argument(
        "--input-codec",
        type=int,
        default="1",
        help="0->encodec, 1->tfcodec",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infer/demo"),
        help="Path to the tokenized files.",
    )
    parser.add_argument("--local_rank", type=int)  
    parser.add_argument("--nnodes", type=int)  
    parser.add_argument("--nproc_per_node", type=int)


    add_model_arguments(parser)
    return parser

def get_params() -> AttributeDict:
    """Return a dict containing training parameters.
    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.
    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.
    Explanation of options saved in `params`:
        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.
        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.
        - best_train_epoch: It is the epoch that has the best training loss.
        - best_valid_epoch: It is the epoch that has the best validation loss.
        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.
        - log_interval:  Print training loss if batch_idx % log_interval` is 0
        - reset_interval: Reset statistics if batch_idx % reset_interval is 0
        - valid_interval:  Run validation if batch_idx % valid_interval is 0
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 100,  # 10: debug 100: train
            "reset_interval": 200,
            "valid_interval": 10000,
            # parameters for TTS
            "env_info": get_env_info(),
        }
    )
    return params
def find_max_epoch_file(exp_dir):  
    max_epoch = -1  
    max_epoch_file = None  
  
    for file in os.listdir(exp_dir):  
        match = re.match(r'epoch-(\d+).pt', file)  
        if match:  
            epoch_num = int(match.group(1))  
            if epoch_num > max_epoch:  
                max_epoch = epoch_num  
                max_epoch_file = file  
                  
    if max_epoch == -1:  
        return None, 0  
    else:  
        return max_epoch_file, max_epoch 
    
def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.
    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.
    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.
    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    max_epoch_file, max_epoch = find_max_epoch_file(params.exp_dir)
    if max_epoch+1 > params.start_epoch:
        print(f"Continue from epoch:{max_epoch}")
        params.start_epoch = max_epoch + 1
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    elif params.restore is True:
        print(f"Restore from {str(params.exp_dir / params.restore_file_name)} ")
        filename = params.exp_dir / params.restore_file_name
    elif params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"        
    else:
        return None
    assert filename.is_file(), f"{filename} does not exist!"
    if isinstance(model, DDP):
        raise ValueError("load_checkpoint before DDP")
    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    saved_stage = saved_params.get("train_stage", 0)
    if params.train_stage != saved_stage:
    # not impleted if restore is True
        # switch training stage
        if params.train_stage and saved_stage:  # switch between 1 and 2
            params.start_epoch = 1
            params.start_batch = 0
        else:
            # switch between 0 and 1/2
            assert params.num_epochs >= params.start_epoch
            params.batch_idx_train = saved_params["batch_idx_train"]
        for key in ["optimizer", "grad_scaler", "sampler"]:
            if key in saved_params:
                saved_params.pop(key)
        # when base on stage 0, we keep scheduler
        if saved_stage != 0:
            for key in ["scheduler"]:
                if key in saved_params:
                    saved_params.pop(key)
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        if best_train_filename.is_file():
            copyfile(
                src=best_train_filename,
                dst=params.exp_dir / f"best-train-loss-stage{saved_stage}.pt",
            )
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        if best_valid_filename.is_file():
            copyfile(
                src=best_valid_filename,
                dst=params.exp_dir / f"best-valid-loss-stage{saved_stage}.pt",
            )
    else:
        if params.restore is True and params.start_epoch == 1:
            keys = [
                "best_train_loss",
                "best_valid_loss",
            ]
            for k in keys:
                params[k] = saved_params[k]
                print(f" {k}:{saved_params[k]}")
            for key in ["optimizer", "grad_scaler", "sampler", "scheduler"]:
                if key in saved_params:
                    saved_params.pop(key)
        else:
            keys = [
                "best_train_epoch",
                "best_valid_epoch",
                "batch_idx_train",
                "best_train_loss",
                "best_valid_loss",
            ]
            for k in keys:
                params[k] = saved_params[k]
                print(f" {k}:{saved_params[k]}")
            if params.start_batch > 0:
                if "cur_epoch" in saved_params:
                    params["start_epoch"] = saved_params["cur_epoch"]
    return saved_params
def test_demos(params,
               model,
            rank,
            tb_writer,):
    print("begin to test_demos")
    if rank!=0 or params.cur_epoch<params.min_test_epoch:
        return
    demo_model = copy.deepcopy(model)
    # demo_model = model
    if params.world_size > 1:
        demo_model = demo_model.module
    demo_model.eval()
    
    text_tokenizer = TextTokenizer(backend=params.text_extractor)
    text_collater = get_text_token_collater(params.text_tokens)
    audio_tokenizer = AudioTokenizer()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    if params.checkpoint_ar and params.train_stage==2:
        print(f"test demo: ar model is loading")
        checkpoint_ar = torch.load(params.checkpoint_ar, map_location=device)
        # 提取以 AR 开头的属性  
        ar_attributes = {k: v for k, v in checkpoint_ar['model'].items() if k.startswith('ar')}  
        missing_keys1, unexpected_keys1 = demo_model.load_state_dict(
            ar_attributes, strict=False
        )
        # print(f"ar missing_keys1: {missing_keys1}, unexpected_keys1: {unexpected_keys1}")
  
    if params.checkpoint_nar and params.train_stage==1:
        print(f"test demo: nar model is loading")
        checkpoint_nar = torch.load(params.checkpoint_nar, map_location=device)
        nar_attributes = {k: v for k, v in checkpoint_nar['model'].items() if k.startswith('nar')}
    
        missing_keys2, unexpected_keys2 = demo_model.load_state_dict(
            nar_attributes, strict=False
        )    
        # print(f"nar missing_keys2: {missing_keys2}, unexpected_keys2: {unexpected_keys2}")
    
    for params_text in params.text:
        text = params_text
        print(f"text:{params_text}")
        for paras_text_prompts, paras_audio_prompts in zip(params.text_prompts, params.audio_prompts):
            text_prompts = " ".join(paras_text_prompts.split("|"))
            print(f"text_prompts :{text_prompts}")
            text_tokens, text_tokens_lens = text_collater(
                [
                    tokenize_text(
                        text_tokenizer, text=f"{text_prompts} {text}".strip()
                    )
                ]
            )
            enroll_x_lens = None
            if text_prompts:
                _, enroll_x_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=f"{text_prompts}".strip()
                        )
                    ]
                )
            audio_prompts = []
            audiofile_name = None
            for n, audio_file in enumerate(paras_audio_prompts.split("|")):
                print(f"n：{n}, audio_file: {audio_file}")
                audiofile_name = audio_file.split('/')[-1]
                with torch.no_grad():
                    encoded_frames = tokenize_audio(audio_tokenizer, audio_file)
                audio_prompts.append(encoded_frames[0][0])
            assert len(paras_text_prompts.split("|")) == len(audio_prompts)
            audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
            audio_prompts = audio_prompts.to(device)
        
            with torch.no_grad():
                encoded_frames = demo_model.inference(
                    text_tokens.to(device),
                    text_tokens_lens.to(device),
                    audio_prompts,
                    enroll_x_lens=enroll_x_lens,
                    top_k=params.top_k,
                    temperature=params.temperature,
                )
                if audio_prompts != []:
                    samples = audio_tokenizer.decode(
                        [(encoded_frames.transpose(2, 1), None)]
                    )
                
            tb_writer.add_audio(f'text: {text} reference_speech: {audiofile_name}', samples[0].cpu(), global_step=params.cur_epoch, sample_rate=24000)  
    del text_tokenizer
    del text_collater
    del audio_tokenizer
    del demo_model
    torch.cuda.empty_cache()
    
    
def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.
    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler, 
        scaler=scaler,
        rank=rank,
    )
    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)
    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)

def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute transducer loss given the model and its inputs.
    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Zipformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
ganjyou        values >= 1.0 are fully warmed up and have all modules present.
    """
    device = (
        model.device
        if isinstance(model, DDP)
        else next(model.parameters()).device
    )
    # at entry, TextTokens is (N, P)
    try:
        text_tokens = batch["text_tokens"].to(device)
        text_tokens_lens = batch["text_tokens_lens"].to(device)
    except:
        pass
    # assert text_tokens.ndim == 2
    audio_features = batch["audio_features"].to(device)
    audio_features_lens = batch["audio_features_lens"].to(device)
    semantic_tokens = batch["semantic_tokens"].to(device)
    semantic_tokens_lens = batch["semantic_tokens_lens"].to(device)
    assert audio_features.ndim == 3

    if params.input_semantic is True:
        x=semantic_tokens
        x_lens=semantic_tokens_lens    
    else:
        x=text_tokens
        x_lens=text_tokens_lens   
    with torch.set_grad_enabled(is_training):
        predicts, loss, metrics = model(
            x=x,
            x_lens=x_lens,
            y=audio_features,
            y_lens=audio_features_lens,
            train_stage=params.train_stage,
        )
    assert loss.requires_grad == is_training
    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (audio_features_lens).sum().item()
        info["utterances"] = x.size(0)
    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    for metric in metrics:
        if isinstance(metrics[metric], int):
            info[metric] = metrics[metric]
        else:
            info[metric] = metrics[metric].detach().cpu().item()
    del metrics
    return predicts, loss, info

def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    global run_now
    
    model.eval()
    tot_loss = MetricsTracker()
    
    for batch_idx, batch in enumerate(valid_dl):
        predicts, loss, loss_info = compute_loss(
            params=params,
            model=model,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info
    # print(f"device:{loss.device} loss is {tot_loss['loss']}")
        # print(f"loss.item() is {loss.item()}")
    if world_size > 1:
        # print(f"begin to reduce. loss.device is {loss.device}")
        
        # val_loss_tensor = torch.tensor([tot_loss], device=loss.device)  
        # print(f"device:{loss.device}, now_device_loss : {val_loss_tensor}")
        # dist.barrier()  
        # dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)  
        # print(f"device:{loss.device}, total_valid_loss : {val_loss_tensor}")
        tot_loss.reduce(loss.device)
        # print("reduce over")
    
    loss_value = tot_loss["loss"] / tot_loss["frames"]

    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value
        
        if loss.device==0 and params.is_local is False:
            run_now.log(f"val_best_epoch", params.cur_epoch)  
            run_now.log(f"val_best_epoch_loss", loss_value)                     

    if params.visualize:
        output_dir = Path(
            f"{params.exp_dir}/eval/step-{params.batch_idx_train:06d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        model.visualize(predicts, batch, output_dir=output_dir)
    return tot_loss


def infer_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    rng: random.Random,
    world_size: int = 1,
    rank: int = 0,
    semantic_token_collater=None,
    two_dict = []
) -> None:
    acoustic_prompts_dic = two_dict["acoustic_tokens"]
    semantic_prompts_dic = two_dict["semantic_tokens"]
    
    random_key = random.choice(list(acoustic_prompts_dic.keys())) 
    audio_prompts = acoustic_prompts_dic[random_key]
    semantic_prompts = semantic_prompts_dic[random_key]

    print(f"audio_prompts:{audio_prompts}")
    print(f"semantic_prompts:{semantic_prompts}")

    model.eval()
    iter_dl = iter(train_dl)
    dtype, enabled = torch.float32, False
    if params.dtype in ["bfloat16", "bf16"]:
        dtype, enabled = torch.bfloat16, True
    elif params.dtype in ["float16", "fp16"]:
        dtype, enabled = torch.float16, True

    batch_idx = 0
    while True:
        try:
            batch = next(iter_dl)
            
        except StopIteration:
            logging.info("Reaches end of dataloader.")
            print(f"curren_rank: {rank} total_batch is {batch_idx}")
            break
        batch_idx += 1
        params.batch_idx_train += 1
        batch_size = len(batch["text"])
        # try:
    # print(f"Rank_{rank} params.batch_idx_train_{params.batch_idx_train} before computer_loss {torch.cuda.memory_allocated() / 1000000:.4f}MB"  )
        with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):

            device = (
                model.device
                if isinstance(model, DDP)
                else next(model.parameters()).device
            )
            # at entry, TextTokens is (N, P)
            try:
                text_tokens = batch["text_tokens"].to(device)
                text_tokens_lens = batch["text_tokens_lens"].to(device)
            except:
                pass
            # assert text_tokens.ndim == 2
            audio_features = batch["audio_features"].to(device)
            audio_features_lens = batch["audio_features_lens"].to(device)
            semantic_tokens = batch["semantic_tokens"].to(device)
            semantic_tokens_lens = batch["semantic_tokens_lens"].to(device)
            assert audio_features.ndim == 3

            enroll_x_lens = None
            _, enroll_x_lens = semantic_token_collater(
                [semantic_prompts]
            )

            if params.input_semantic is True:
                x=semantic_tokens
                x_lens=semantic_tokens_lens    
            else:
                x=text_tokens
                x_lens=text_tokens_lens   
            with torch.set_grad_enabled(False):
                encoded_frames = model.inference_only_ar(
                    semantic_tokens.to(device),
                    semantic_tokens_lens.to(device),
                    audio_prompts,
                    enroll_x_lens=enroll_x_lens,
                    top_k=params.top_k,
                    temperature=params.temperature,
                )
                print(encoded_frames.shape)


        if batch_idx % 100 == 0 and params.dtype in ["float16", "fp16"]:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            print(f"has converted {batch_idx}  cases")
         


def filter_short_and_long_utterances(
    cuts: CutSet, min_duration: float, max_duration: float
) -> CutSet:
    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 0.6 second and 20 seconds
        if c.duration < min_duration or c.duration > max_duration:
            # logging.warning(
            #     f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            # )
            return False
        return True
    cuts = cuts.filter(remove_short_and_long_utt)
    return cuts

def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    # print(args.train_dir_name)
    # quit()
    # pre_process
    params = get_params()

    cpu_count = os.cpu_count()  
    num_workers = cpu_count // 2 
    
    args.num_workers = num_workers

    print(f"num_workers:{num_workers}")
    params.update(vars(args))
    
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)
        
    train_dir_name, val_dir_name = change_file_path(args, rank)
    args.train_dir_name = train_dir_name
    args.val_dir_name = val_dir_name
    
    # params = get_params()
    params.update(vars(args))
    print(f"params : {params}")

    if args.is_local is False:
        global run_now
        from azureml.core.run import Run
        run_now = Run.get_context()
    fix_random_seed(params.seed)
    rng = random.Random(params.seed)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")
    if args.tensorboard and rank == 0:
        if params.train_stage:
            tb_writer = SummaryWriter(
                log_dir=f"{params.exp_dir}/tensorboard_stage{params.train_stage}"
            )
        else:
            tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
        # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    logging.info(f"Device: {device}")
    logging.info(params)
    logging.info("About to create model")
    model = get_model(params)
    # with open(f"{params.exp_dir}/model.txt", "w") as f:
    #     print(model)
    #     print(model, file=f)
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")
    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0 and params.average_period > 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)
    assert params.start_epoch > 0, params.start_epoch

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # # 将checkpoint2的键写入到另一个txt文件中  
    # with open('keys_output_checkpoint2.txt', 'w') as f:  
    #     for key in checkpoint2["model"].keys():  
    #         f.write(key + '\n')  
    missing_keys, unexpected_keys1 = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys

    model.to(device)
    model.eval()

    # print(f"Rank_{rank} before put model to device {torch.cuda.memory_allocated() / 1000000:.4f}MB")
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        # print(f"Rank_{rank} after put model to ddp {torch.cuda.memory_allocated() / 1000000:.4f}MB")

    # print(f"Rank_{rank} before optimizer zero_grad {torch.cuda.memory_allocated() / 1000000:.4f}MB")
    # print(f"Rank_{rank} after optimizer zero_grad {torch.cuda.memory_allocated() / 1000000:.4f}MB")

    if params.inf_check:
        register_inf_check_hooks(model)

    sampler_state_dict = None
    dataset = TtsDataModule(args)
    train_cuts = dataset.train_cuts()
    valid_cuts = dataset.dev_cuts()
    for cut in train_cuts:  
        features = cut.features  
        path = cut.features.storage_path
        print(f"final updated cut.features.storage_path :{path}")
        break
    
    train_cuts = filter_short_and_long_utterances(
        train_cuts, params.filter_min_duration, params.filter_max_duration
    )
    valid_cuts = filter_short_and_long_utterances(
        valid_cuts, params.filter_min_duration, params.filter_max_duration
    )
    train_dl = dataset.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )
    valid_dl = dataset.valid_dataloaders(valid_cuts)

    semantic_token_collater = get_text_token_collater(args.semantic_tokens)

    device = (
        model.device
        if isinstance(model, DDP)
        else next(model.parameters()).device
    )
    if args.input_codec ==0:
        audio_tokenizer = AudioTokenizer()
    else:
        audio_tokenizer = AudioTokenizer_encodec_16k_tfcodec(device=device, config_path=params.config_path, tfnet_ckpt=params.tfnet_ckpt)
    with open(params.ref_tokens_path, "r") as json_file:  
        two_dict = json.load(json_file) 
    acoustic_prompts_dic = two_dict["acoustic_tokens"]
    semantic_prompts_dic = two_dict["semantic_tokens"]

    targets = acoustic_prompts_dic.keys()
    targets = list(targets)


    speakers = ["ASI", "KSP", "RRBI", "SVBI", "TNI"]
    speakers2id = {}
    for i , speaker in enumerate(speakers):
        speakers2id[speaker] = i


    def divide_targets_by_speakers(speakers, targets):  
        # 计算每份的大小  
        size = len(speakers)  
        # 分割targets  
        chunks = [targets[i:i+size] for i in range(0, len(targets), size)]  
        
        # 删除不足speaker数量的部分  
        chunks = [chunk for chunk in chunks if len(chunk) == size]  
        
        return chunks  
    chunks = divide_targets_by_speakers(speakers, targets)


    for epoch in range(params.start_epoch, params.num_epochs + 1):
        # print(f"Rank_{rank} memory before train epoch_{epoch} {torch.cuda.memory_allocated() / 1000000:.4f}MB")

        iter_dl = iter(train_dl)
        dtype, enabled = torch.float32, False
        if params.dtype in ["bfloat16", "bf16"]:
            dtype, enabled = torch.bfloat16, True
        elif params.dtype in ["float16", "fp16"]:
            dtype, enabled = torch.float16, True

        batch_idx = 0
        while True:
            try:
                batch = next(iter_dl)
                
                
            except StopIteration:

                logging.info("Reaches end of dataloader.")
                print(f"curren_rank: {rank} total_batch is {batch_idx}")
                break
            batch_idx += 1
            params.batch_idx_train += 1
            batch_size = len(batch["text"])
            # try:
        # print(f"Rank_{rank} params.batch_idx_train_{params.batch_idx_train} before computer_loss {torch.cuda.memory_allocated() / 1000000:.4f}MB"  )
            with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):

                device = (
                    model.device
                    if isinstance(model, DDP)
                    else next(model.parameters()).device
                )
                # audio_prompts = audio_prompts.to(device)
                semantic_syss = batch["bef_semantic_tokens"]
                sys_paths = batch["storage_path"]

                idx = 0
                for semantic_sys, sys_path in zip(semantic_syss, sys_paths):

                    txt_spker = sys_path.split('/')[-3]
                    try :
                        assigned_target = chunks[epoch][speakers2id[txt_spker]]
                    except StopIteration:
                        print("speaker is over")
                        quit()

                    audio_prompts = acoustic_prompts_dic[assigned_target]
                    
                    audio_prompts = torch.tensor(audio_prompts)
                    audio_prompts = audio_prompts.unsqueeze(0)
                    audio_prompts = audio_prompts.to(device)

                    semantic_prompts = semantic_prompts_dic[assigned_target]

                    # print(f"audio_propmts shape is {audio_prompts.shape}")
                    # print(f"audio_prompts:{audio_prompts}")
                    # print(f"semantic_prompts:{semantic_prompts}")

                    prompt_spker = assigned_target.split('_')[0]
                    file_name = sys_path.split('/')[-1]

                    tgt_spker = txt_spker + '_' + prompt_spker
                    # print(semantic_sys)
                    # print(sys_path)

                    print(f"txt_spker:{txt_spker}, prompt_spker:{prompt_spker}")
                    # print(f"semantic_propts+semantic_sys: {semantic_prompts + semantic_sys}")
                    semantic_tokens,  semantic_tokens_lens = semantic_token_collater(
                        [semantic_prompts + semantic_sys]
                    )
                    
                    enroll_x_lens = None
                    _, enroll_x_lens = semantic_token_collater(
                        [semantic_prompts]
                    )

                    with torch.set_grad_enabled(False):
                        encoded_frames = model.module.inference_only_ar(
                            semantic_tokens.to(device),
                            semantic_tokens_lens.to(device),
                            audio_prompts,
                            enroll_x_lens=enroll_x_lens,
                            top_k=params.top_k,
                            temperature=params.temperature,
                        )
                        print(encoded_frames.shape)
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
                        os.makedirs(f"{args.output_dir}", exist_ok=True)            
                        os.makedirs(f"{args.output_dir}/{tgt_spker}", exist_ok=True)
                        torchaudio.save(
                            f"{args.output_dir}/{tgt_spker}/{file_name}", samples.cpu(), 16000
                        )
                        print("generate")

        
        # print(f"Rank_{rank} memory after train epoch_{epoch} {torch.cuda.memory_allocated() / 1000000:.4f}MB"  )
    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()

def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
) -> None:
    """Display the batch statistics and save the batch into disk.
    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
    """
    from lhotse.utils import uuid4
    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)



def update_mono_cut_features(cut: MonoCut, new_features: Features) -> MonoCut:  
        return MonoCut(  
            id=cut.id,  
            start=cut.start,  
            duration=cut.duration,  
            channel=cut.channel,  
            supervisions=cut.supervisions,  
            features=new_features,  
            recording=cut.recording,
            custom = cut.custom
        )    
def update_mono_cut_recording(cut: MonoCut, new_recording) -> MonoCut:  
        return MonoCut(  
            id=cut.id,  
            start=cut.start,  
            duration=cut.duration,  
            channel=cut.channel,  
            supervisions=cut.supervisions,  
            features=cut.features,  
            recording=new_recording,
            custom = cut.custom
        )  
def change_file_path(args, rank=0):
    Prefix = args.manifest_dir
    # print(f"prefix_{Prefix}")
    # Prefix = '/'.join(str(Prefix).split('/')[:-2])
    
    newfile_train = f"cuts_train_{args.newfile_suffix}_rank{rank}.jsonl.gz"
    newfile_valid = f"cuts_valid_{args.newfile_suffix}_rank{rank}.jsonl.gz"
    # print(f"Prefix: {Prefix}")
    # update_train_file = False
    update_train_file = False
    update_valid_file = False
    
    cut_set_train = CutSet.from_file(args.manifest_dir / "cuts_train.jsonl.gz")
    train_dir_name = "cuts_train.jsonl.gz"
    val_dir_name = "cuts_dev.jsonl.gz"
    for cut_id, cut in cut_set_train.cuts.items():
        old_path = cut.features.storage_path
        # quit()
        dir_path = '/'.join(old_path.split('/')[:-1])
        if dir_path != str(Prefix):
            update_train_file = True
            print(f"Need to update_train_dev, Input_Prefix:{Prefix} Read_Prefix:{dir_path}, Read_Path:{old_path}")
        else:
            print(f"No Need to update_train_dev, Prefix:{Prefix}, Read_Path:{old_path}")
        break
    
    if update_train_file is True:
        # 遍历 CutSet 中的所有 Cut 并修改它们的 features.storage_path 属性  
        i = 0
        new_cuts = []
        for cut_id, cut in cut_set_train.cuts.items():  
            old_features = cut.features  
            old_path = cut.features.storage_path
            rela_path = old_path.split('/')[-1]
            new_storage_path = str(Prefix/rela_path) 
            new_features = Features(  
                type=old_features.type,  
                num_frames=old_features.num_frames,  
                num_features=old_features.num_features,  
                frame_shift=old_features.frame_shift,  
                sampling_rate=old_features.sampling_rate,  
                start=old_features.start,  
                duration=old_features.duration,  
                storage_type=old_features.storage_type,  
                storage_path=new_storage_path,  
                storage_key=old_features.storage_key,  
                recording_id=old_features.recording_id,  
                channels=old_features.channels  
            )
            new_cut = update_mono_cut_features(cut, new_features) 
            new_cuts.append(new_cut)
            
        updated_cuts_train = CutSet.from_cuts(new_cuts) 
        updated_cuts_train.to_file(args.manifest_dir / newfile_train)  
        print(f"update train file : {args.manifest_dir}/{newfile_train}")
        train_dir_name = newfile_train
    cut_set_valid = CutSet.from_file(args.manifest_dir / "cuts_dev.jsonl.gz")  
    for cut_id, cut in cut_set_valid.cuts.items():
        old_path = cut.features.storage_path
        # quit()
        dir_path = '/'.join(old_path.split('/')[:-1])
        if dir_path != str(Prefix):
            update_valid_file = True
            print(f"Need to update_valid_dev, Input_Prefix:{Prefix} Read_Prefix:{dir_path}, Read_Path:{old_path}")
        else:
            print(f"No Need to update_valid_dev, Prefix:{Prefix}, Read_Path:{old_path}")
        break
    
    if update_valid_file is True:
        new_cuts = []
        for cut_id, cut in cut_set_valid.cuts.items():  
            old_features = cut.features  
            old_path = cut.features.storage_path
            rela_path = old_path.split('/')[-1]
            # here is str not posix
            new_storage_path = str(Prefix/rela_path) 
            new_features = Features(  
                type=old_features.type,  
                num_frames=old_features.num_frames,  
                num_features=old_features.num_features,  
                frame_shift=old_features.frame_shift,  
                sampling_rate=old_features.sampling_rate,  
                start=old_features.start,  
                duration=old_features.duration,  
                storage_type=old_features.storage_type,  
                storage_path=new_storage_path,  
                storage_key=old_features.storage_key,  
                recording_id=old_features.recording_id,  
                channels=old_features.channels  
            )
            new_cut = update_mono_cut_features(cut, new_features) 
            # old_recording = cut.recording
            # old_audioSource = cut.recording.sources[0]
            # # print(old_audioSource)
            # old_source = old_audioSource.source
            # new_source = args.audio_path + '/' +'/'.join(old_source.split('/')[-4:])
            # new_audiosourcce = AudioSource(
            #     type = old_audioSource.type,
            #     channels = old_audioSource.channels,
            #     source= new_source
            # )
            # new_Recording = Recording(
            #     id = old_recording.id,
            #     sources = [new_audiosourcce],
            #     sampling_rate=old_recording.sampling_rate, 
            #     num_samples=old_recording.num_samples, 
            #     duration=old_recording.duration, 
            #     channel_ids=old_recording.channel_ids, 
            #     transforms=old_recording.transforms,
            # )
            
            # new_cut = update_mono_cut_recording(new_cut, new_Recording)
            new_cuts.append(new_cut)
        updated_cuts_valid = CutSet.from_cuts(new_cuts) 
        updated_cuts_valid.to_file(args.manifest_dir / newfile_valid)
        print(f"update valid file : {args.manifest_dir}/{newfile_valid}")
 
        # cut_set_valid.to_file(args.manifest_dir / "cuts_dev_jia6.jsonl.gz")
        print(f"keep to {args.manifest_dir}/{newfile_valid}")
        val_dir_name = newfile_valid
    return train_dir_name, val_dir_name

# def change_file_path(args, rank=0):
    # Prefix = args.manifest_dir
    # Prefix = '/'.join(str(Prefix).split('/')[:-2])
    
    # newfile_train = f"cuts_train_{args.newfile_suffix}_rank{rank}.jsonl.gz"
    # newfile_valid = f"cuts_valid_{args.newfile_suffix}_rank{rank}.jsonl.gz"
    # print(f"Prefix: {Prefix}")
    # update_train_file = False
    # update_valid_file = False
    
    # cut_set_train = CutSet.from_file(args.manifest_dir / "cuts_train.jsonl.gz")
    # train_dir_name = "cuts_train.jsonl.gz"
    # val_dir_name = "cuts_dev.jsonl.gz"
    # for cut_id, cut in cut_set_train.cuts.items():
    #     old_path = cut.features.storage_path
    #     dir_path = '/'.join(old_path.split('/')[:-3])
    #     if dir_path != Prefix:
    #         update_train_file = True
    #         print(f"Need to update_train_dev, Input_Prefix:{Prefix} Read_Prefix:{dir_path}, Read_Path:{old_path}")
    #     else:
    #         print(f"No Need to update_train_dev, Prefix:{Prefix}, Read_Path:{old_path}")
    #     break
                
    # if update_train_file is True:
    #     # 遍历 CutSet 中的所有 Cut 并修改它们的 features.storage_path 属性  
    #     i = 0
    #     new_cuts = []
    #     for cut_id, cut in cut_set_train.cuts.items():  
    #         old_features = cut.features  
    #         old_path = cut.features.storage_path
    #         rela_path = '/'.join(old_path.split('/')[-3:])
    #         new_storage_path = Prefix+ '/' +rela_path 
    #         new_features = Features(  
    #             type=old_features.type,  
    #             num_frames=old_features.num_frames,  
    #             num_features=old_features.num_features,  
    #             frame_shift=old_features.frame_shift,  
    #             sampling_rate=old_features.sampling_rate,  
    #             start=old_features.start,  
    #             duration=old_features.duration,  
    #             storage_type=old_features.storage_type,  
    #             storage_path=new_storage_path,  
    #             storage_key=old_features.storage_key,  
    #             recording_id=old_features.recording_id,  
    #             channels=old_features.channels  
    #         )
    #         new_cut = update_mono_cut_features(cut, new_features) 
    #         new_cuts.append(new_cut)
    #     updated_cuts_train = CutSet.from_cuts(new_cuts) 
    #     updated_cuts_train.to_file(args.manifest_dir / newfile_train)  
    #     print(f"update train file : {args.manifest_dir}/{newfile_train}")
    #     train_dir_name = newfile_train
    # cut_set_valid = CutSet.from_file(args.manifest_dir / "cuts_dev.jsonl.gz")  
    # for cut_id, cut in cut_set_valid.cuts.items():
    #     old_path = cut.features.storage_path
    #     dir_path = '/'.join(old_path.split('/')[:-3])
    #     if dir_path != Prefix:
    #         update_valid_file = True
    #         print(f"Need to update_valid_dev, Input_Prefix:{Prefix} Read_Prefix:{dir_path}, Read_Path:{old_path}")
    #     else:
    #         print(f"No Need to update_valid_dev, Prefix:{Prefix}, Read_Path:{old_path}")
    #     break
    
    # if update_valid_file is True:
    #     new_cuts = []
    #     for cut_id, cut in cut_set_valid.cuts.items():  
    #         old_features = cut.features  
    #         old_path = cut.features.storage_path
    #         rela_path = '/'.join(old_path.split('/')[-3:])
    #         new_storage_path = Prefix+ '/' +rela_path 
    #         new_features = Features(  
    #             type=old_features.type,  
    #             num_frames=old_features.num_frames,  
    #             num_features=old_features.num_features,  
    #             frame_shift=old_features.frame_shift,  
    #             sampling_rate=old_features.sampling_rate,  
    #             start=old_features.start,  
    #             duration=old_features.duration,  
    #             storage_type=old_features.storage_type,  
    #             storage_path=new_storage_path,  
    #             storage_key=old_features.storage_key,  
    #             recording_id=old_features.recording_id,  
    #             channels=old_features.channels  
    #         )
    #         new_cut = update_mono_cut_features(cut, new_features) 
    #         new_cuts.append(new_cut)
    #     updated_cuts_valid = CutSet.from_cuts(new_cuts) 
    #     updated_cuts_valid.to_file(args.manifest_dir / newfile_valid)
    #     print(f"update valid file : {args.manifest_dir}/{newfile_valid}")
 
    #     # cut_set_valid.to_file(args.manifest_dir / "cuts_dev_jia6.jsonl.gz")
    #     print(f"keep to {args.manifest_dir}/{newfile_valid}")
    #     val_dir_name = newfile_valid
    # return train_dir_name, val_dir_name
def main():
    parser = get_parser()
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    world_size = args.world_size
    assert world_size >= 1

    print(args.is_local)
    if args.is_local is True:
        local_rank = 0
    else:
        local_rank = int(os.environ["LOCAL_RANK"])

    os.environ["NCCL_ASYNC_ERROR_HANDING"] = "1"
    
 
    # dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size) 
    # import os    
    run(rank=local_rank, world_size=world_size, args=args)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if __name__ == "__main__":
    import sys   
    main()
# import argparse  
# import os  
# import torch  
# import torch.distributed as dist  
# from pathlib import Path  
# from valle.data import TtsDataModule
# def main(local_rank, world_size, args):  
#     os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  
#     print(f"nnodes:{args.nnodes}")  
#     print(f"nproc_per_node:{args.nproc_per_node}")  
  
#     print(f"current_rank: {local_rank}")  
#     print(f"current_world_size: {world_size}")  
  
#     dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)  
#     # Your training code  
  
# if __name__ == "__main__":  
  
#     parser = get_parser()  # Assuming you have a function called get_parser() that returns an ArgumentParser  
#     TtsDataModule.add_arguments(parser)  
#     local_rank = int(os.environ["LOCAL_RANK"])
#     os.environ["NCCL_ASYNC_ERROR_HANDING"] = "1"    
#     args = parser.parse_args()  
#     args.exp_dir = Path(args.exp_dir)  
  
#     world_size = 2
#     assert world_size >= 1  
  
#     main(local_rank, world_size, args)  
