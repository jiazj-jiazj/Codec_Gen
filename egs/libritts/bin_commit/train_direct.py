
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
current_working_directory = os.getcwd()  
  
# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory) 
from lhotse import CutSet, load_manifest_lazy, manipulation
# from icefall.utils import get_executor
# import k2
# print("k2 success")
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
)
from valle.data.collation import get_text_token_collater
LRSchedulerType = torch.optim.lr_scheduler._LRScheduler
run_now = None
import re

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
    parser.add_argument(
        "--nar-mask-type",
        type=int,
        default=0,
        help = "0-> soundstorm random 1-> group time generate"
    )    
    parser.add_argument(
        "--nar-mask-ratio",
        type=float,
        default=0.5,
        help = "ratito between mask nar: baseline and group_ar"
    )
    parser.add_argument(
        "--group-in-mask-replace-prob",
        type=float,
        default=0.15,
        help = "ratito group-in-mask-replace"
    )
    parser.add_argument(
        "--group-in-mask-replace-all-prob",
        type=float,
        default=0.05,
        help = "ratito group-in-mask-replace"
    )
    parser.add_argument(
        "--group-in-mask",
        type=str2bool,
        default=False,
        help="restore.",
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
        "--only-comp-mask-loss",
        type=str2bool,
        default=False,
        help="only one linear no matter how num-quantizers",
    )
    parser.add_argument(
        "--is-pretrain",
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
        "--tgt-spk-names",
        type=str,
        help="path of semantic-tokens",
    )
    parser.add_argument(
        "--random-tgt-spkers",
        type=int,
        default=1,
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
        "--parrallel-mode",
        type=int,
        default=0,
        help="path of semantic-tokens",
    )
    parser.add_argument(
        "--group-in-mask-replace-all-varible",
        type=str2bool,
        default=False,
        help="path of semantic-tokens",
    )
    parser.add_argument(
        "--ac-tune-mode",
        type=int,
        default=0,
        help="path of semantic-tokens",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default="/mnt/zhijun/data/LibriTTS",
        help="audio-source",
    )
    parser.add_argument(
        "--train-dir-name",
        type=str,
        default="cuts_train.jsonl.gz",
    )
    parser.add_argument(
        "--val-dir-name",
        type=str,
        default="cuts_dev.jsonl.gz",
    )

    parser.add_argument(
        "--pret-mode",
        type=int,
        default=5,
        help = "test from this epoch"
    )
    parser.add_argument(
        "--pret-prob",
        type=float,
        default=0.15,
        help = "test from this epoch"
    )
    parser.add_argument(
        "--pret-lam",
        type=int,
        default=5,
        help = "test from this epoch"
    )
    parser.add_argument(
        "--pret-token",
        type=int,
        default=500,
        help = "test from this epoch"
    )
    parser.add_argument(
        "--semantic-type",
        type=int,
        default=0,
        help = "0->hubert, 1->tfnet_256bps"
    )
    parser.add_argument(
        "--encoder-num-layers",
        type=int,
        default=6
    )
    parser.add_argument(
        "--decoder-num-layers",
        type=int,
        default=6
    )
    parser.add_argument(
        "--sec-dataset",
        type=str2bool,
        default=False,
        help="only one linear no matter how num-quantizers",
    )
    parser.add_argument(
        "--manifest-dir-sec",
        type=Path,
        default=Path('/scratch/data/LibriTTS/vc_tokenized_16k_tfcodec_16codes/cuts_train.jsonl.gz'),
        help="audio-source",
    )
    parser.add_argument(
        "--ac-native-mask",
        type=str2bool,
        default="false",
        help="whether mask native input tokens",
    )
    parser.add_argument(
        "--parts-req-gra",
        type=int,
        default=0,
        help="0->all, 1->encoder, 2->first 3 encoder layers",
    )
    parser.add_argument('--num-workers', type=int, default=None)  
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
    # try:
    #     text_tokens = batch["text_tokens"].to(device)
    #     text_tokens_lens = batch["text_tokens_lens"].to(device)
    # except:
    #     pass
    # assert text_tokens.ndim == 2

    if batch["audio_features_correct"]!=None:
        audio_features_correct = batch["audio_features_correct"].to(device)
    else:
        audio_features_correct = None

    audio_features = batch["audio_features"].to(device)
    audio_features_lens = batch["audio_features_lens"].to(device)
    maskd_indices_batch = batch["maskd_indices_batch"].to(device)
    only_comp_mask_loss = batch["only_comp_mask_loss"]

    assert audio_features.ndim == 3

    if params.input_semantic is True:
        semantic_tokens = batch["semantic_tokens"].to(device)
        semantic_tokens_lens = batch["semantic_tokens_lens"].to(device)
        x=semantic_tokens
        x_lens=semantic_tokens_lens    
    else:
        text_tokens = batch["text_tokens"].to(device)
        text_tokens_lens = batch["text_tokens_lens"].to(device)
        x=text_tokens
        x_lens=text_tokens_lens   
    with torch.set_grad_enabled(is_training):
        if audio_features_correct ==None:
            predicts, loss, metrics = model(
                x=x,
                x_lens=x_lens,
                y=audio_features,
                y_lens=audio_features_lens,
                train_stage=params.train_stage,
                maskd_indices_batch = maskd_indices_batch,
                only_comp_mask_loss=only_comp_mask_loss
            )
        else:
            predicts, loss, metrics = model(
                x=x,
                x_lens=x_lens,
                y=audio_features,
                y_lens=audio_features_lens,
                train_stage=params.train_stage,
                maskd_indices_batch = maskd_indices_batch,
                y_correct= audio_features_correct,
                only_comp_mask_loss=only_comp_mask_loss
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

    if world_size > 1:
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

def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    rng: random.Random,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.
    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.
    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      rng:
        Random for selecting.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    global run_now
    model.train()
    tot_loss = MetricsTracker()
    # each_tot_codec_loss = MetricsTracker()
    iter_dl = iter(train_dl)
    dtype, enabled = torch.float32, False
    if params.dtype in ["bfloat16", "bf16"]:
        dtype, enabled = torch.bfloat16, True
    elif params.dtype in ["float16", "fp16"]:
        dtype, enabled = torch.float16, True
    model_context = model.join if isinstance(model, DDP) else nullcontext
    with model_context(): # error need to Modification
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
                _, loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                )
            torch.cuda.empty_cache()
            # print(loss_info)
            # summary stats
            # print(f"Rank_{rank} params.batch_idx_train_{params.batch_idx_train} after computer_loss {torch.cuda.memory_allocated() / 1000000:.4f}MB"  )
            # loss_info.print_metrics()
            # tot_loss.print_metrics()
            # tot_loss.update_with_all_metrics(loss_info, params.reset_interval)  
            tot_loss = (
                tot_loss * (1 - 1 / params.reset_interval)
            ) + loss_info * (1 / params.reset_interval)
            # tot_loss.print_metrics()
            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()
            # print(f"Rank_{rank} params.batch_idx_train_{params.batch_idx_train} after loss backward {torch.cuda.memory_allocated() / 1000000:.4f}MB"  )
            if params.batch_idx_train >= params.accumulate_grad_steps:
                if (
                    params.batch_idx_train % params.accumulate_grad_steps
                    == 0
                ):
                    if params.optimizer_name not in ["ScaledAdam", "Eve"]:
                        # Unscales the gradients of optimizer's assigned params in-place
                        scaler.unscale_(optimizer)
                        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    for k in range(params.accumulate_grad_steps):
                        if isinstance(scheduler, Eden):
                            scheduler.step_batch(params.batch_idx_train)
                        else:
                            scheduler.step()
            set_batch_count(model, params.batch_idx_train)
            # except:  # noqa
            #     # Save the broken batch
            #     logging.warning(f"Hit a broken batch of training data. Cut ID: {batch['utt_id']} Text: {batch['text']} - Skipping...")
            #     display_and_save_batch(batch, params=params)
            #     # Clean up batch data from Memory and GPU
            #     del batch["text_tokens"]
            #     del batch["text_tokens_lens"]
            #     del batch["audio_features"]
            #     del batch["audio_features_lens"]
            #     del batch
            #     try:
            #         del loss
            #         del loss_info
            #     except UnboundLocalError:
            #             pass
            #     torch.cuda.empty_cache()
            #     # Continue training
            #     continue
            if params.average_period > 0:
                if (
                    params.batch_idx_train > 0
                    and params.batch_idx_train % params.average_period == 0
                ):
                    # Perform Operation in rank 0
                    if rank == 0:
                        update_averaged_model(
                            params=params,
                            model_cur=model,
                            model_avg=model_avg,
                        )
                    
            if (
                params.batch_idx_train > 0
                and params.batch_idx_train % params.save_every_n == 0
            ):       
                out_dir = Path(params.exp_dir) 
                if not out_dir.is_dir():  
                    print("The specified directory does not exist.")  
                    sys.exit(1)       
                # Perform Operation in rank 0
                if rank == 0:
                    out_dir = Path(params.exp_dir) 
                    if not out_dir.is_dir():  
                        print("The specified directory does not exist.")  
                        sys.exit(1)  

                    # print(f"Rank_{rank} before save_checkpoint_with_global_batch_idx", f"{torch.cuda.memory_allocated() / 1000000:.4f}MB"  )
                    save_checkpoint_with_global_batch_idx(
                        out_dir=params.exp_dir,
                        global_batch_idx=params.batch_idx_train,
                        model=model,
                        model_avg=model_avg,
                        params=params,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        sampler=train_dl.sampler,
                        scaler=scaler,
                        rank=rank,
                    )
                    # remove_checkpoints(
                    #     out_dir=params.exp_dir,
                    #     topk=params.keep_last_k,
                    #     rank=rank,
                    # )
                    # print(f"Rank_{rank} after save_checkpoint_with_global_batch_idx", f"{torch.cuda.memory_allocated() / 1000000:.4f}MB"  )
                    
            if batch_idx % 100 == 0 and params.dtype in ["float16", "fp16"]:
                # If the grad scale was less than 1, try increasing it.    The _growth_interval
                # of the grad scaler is configurable, but we can't configure it to have different
                # behavior depending on the current grad scale.
                cur_grad_scale = scaler._scale.item()
                if cur_grad_scale < 1.0 or (
                    cur_grad_scale < 8.0 and batch_idx % 400 == 0
                ):
                    scaler.update(cur_grad_scale * 2.0)
                if cur_grad_scale < 0.01:
                    logging.warning(f"Grad scale is small: {cur_grad_scale}")
                if cur_grad_scale < 1.0e-05:
                    raise RuntimeError(
                        f"grad_scale is too small, exiting: {cur_grad_scale}"
                    )
            if batch_idx % params.log_interval == 0:
                
                cur_lr = scheduler.get_last_lr()[0]                                   
            
                cur_grad_scale = (
                    scaler._scale.item()
                    if params.dtype in ["float16", "fp16"]
                    else 1.0
                )
                logging.info(
                    f"Epoch {params.cur_epoch}, "
                    f"batch {batch_idx}, train_loss[{loss_info}], "
                    f"tot_loss[{tot_loss}], "
                    f"batch size: {batch_size}, "
                    f"lr: {cur_lr:.2e}"
                    + (
                        f", grad_scale: {cur_grad_scale}"
                        if params.dtype in ["float16", "fp16"]
                        else ""
                    )
                )
                if tb_writer is not None:
                    tb_writer.add_scalar(
                        "train/learning_rate", cur_lr, params.batch_idx_train
                    )
                    loss_info.write_summary(
                        tb_writer,
                        "train/current_",
                        params.batch_idx_train,
                    )
                    tot_loss.write_summary(
                        tb_writer, "train/tot_", params.batch_idx_train
                    )
                    if params.dtype in ["float16", "fp16"]:
                        tb_writer.add_scalar(
                            "train/grad_scale",
                            cur_grad_scale,
                            params.batch_idx_train,
                        )
                    
                if rank==0 and params.is_local is False:
                    run_now.log(f"train_learning_rate", cur_lr)
                    # run.log(f"train_{params.batch_idx_train}_current_loss", loss_info)
                    # run.log(f"train_{params.batch_idx_train}_total_loss", tot_loss)
                    # run_now.log(f"train_batch_size", batch_size)
                    # for k, v in loss_info.norm_items():
                    #     run_now.log(f"train_current_{params.batch_idx_train}_{k}", v)
                    for k, v in tot_loss.norm_items():
                        run_now.log(f"train_total_{k}", v)                       
                    
                        
            if params.batch_idx_train % params.valid_interval == 0:
                logging.info("Computing validation loss")
                # print(f"Rank_{rank} memory before valid batch_idx_train_{params.batch_idx_train}", f"{torch.cuda.memory_allocated() / 1000000:.4f}MB"  )
                with torch.cuda.amp.autocast(dtype=dtype):
                    valid_info = compute_validation_loss(
                        params=params,
                        model=model,
                        valid_dl=valid_dl,
                        world_size=world_size,
                    )                

                model.train()
                logging.info(
                    f"Epoch {params.cur_epoch}, validation: {valid_info}"
                )

                logging.info(
                    f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated() / 1000000:.4f}MB"
                )
                if tb_writer is not None:
                    valid_info.write_summary(
                        tb_writer, "valid/valid_", params.batch_idx_train
                    )
                if rank==0 and params.is_local is False:
                    for k, v in valid_info.norm_items():
                        run_now.log(f"valid_{k}", v)
                    run_now.log(f"valid_epoch", params.cur_epoch)
                
                # print(f"Rank_{rank} memory after valid batch_idx_train_{params.batch_idx_train}", f"{torch.cuda.memory_allocated() / 1000000:.4f}MB")

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss
        if rank==0 and params.is_local is False:
            run_now.log(f"train_best_epoch", params.cur_epoch)
            run_now.log(f"train_best_epoch_loss", params.train_loss)
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

def combine_v1(traincuts, train_cuts_sec_native, val_cuts, val_cuts_sec_native):
    # select source spker
    # traincuts = traincuts.filter(lambda r: 'tni' in r.id)
    # traincuts = traincuts.subset(first=550)
    # print(f"train_first 550")
    # for traincut in traincuts:
    #     print(traincut.id)
    # quit()
    # traincuts = traincuts.to_eager()

    # clean100_filter_train_cuts_sec_native = train_cuts_sec_native.filter(lambda r: 'clean-100' in r.recording.sources[0].source)
    clean100_filter_train_cuts_sec_native = train_cuts_sec_native.shuffle()
    clean100_filter_train_cuts_sec_native.describe()
    # clean100_filter_train_cuts_sec_native = clean100_filter_train_cuts_sec_native.subset(first=3200)

    # first_100_clean100_filter_train_cuts_sec_native = clean100_filter_train_cuts_sec_native.subset(first=1)
    # first_100_clean100_filter_train_cuts_sec_native = clean100_filter_train_cuts_sec_native
    combine_native_non_nat = manipulation.combine([clean100_filter_train_cuts_sec_native, traincuts])
    combine_native_non_nat = combine_native_non_nat.shuffle()

    # combine_native_non_nat.to_file("/scratch/data/libritts_clean_100_5000_l1l2/tokenized/cuts_train.jsonl.gz")

    # #149736
    # #33236 clean-100
    # #4900
    # clean100_filter_val_cuts_sec_native = val_cuts_sec_native.filter(lambda r: 'clean' in r.recording.sources[0].source)
    # clean100_filter_val_cuts_sec_native = clean100_filter_val_cuts_sec_native.subset(first=1)

    # val_combine_native_non_nat = manipulation.combine([clean100_filter_val_cuts_sec_native, val_cuts])
    # val_combine_native_non_nat = val_combine_native_non_nat.shuffle()
    #10349 all
    #500
    return combine_native_non_nat, val_cuts

def split_source(traincuts, train_cuts_sec_native, val_cuts, val_cuts_sec_native):

    clean100_filter_train_cuts_sec_native = train_cuts_sec_native.filter(lambda r: 'clean-100' in r.recording.sources[0].source)
    first_100_clean100_filter_train_cuts_sec_native = clean100_filter_train_cuts_sec_native.subset(first=5000)
    # first_100_clean100_filter_train_cuts_sec_native = clean100_filter_train_cuts_sec_native
    combine_native_non_nat = manipulation.combine([first_100_clean100_filter_train_cuts_sec_native, traincuts])
    combine_native_non_nat = combine_native_non_nat.shuffle()

    # combine_native_non_nat.to_file("/scratch/data/libritts_clean_100_5000_l1l2/tokenized/cuts_train.jsonl.gz")

    # #149736
    # #33236 clean-100
    # #4900
    clean100_filter_val_cuts_sec_native = val_cuts_sec_native.filter(lambda r: 'clean' in r.recording.sources[0].source)
    first_500_clean100_filter_val_cuts_sec_native = clean100_filter_val_cuts_sec_native.subset(first=2)

    val_combine_native_non_nat = manipulation.combine([first_500_clean100_filter_val_cuts_sec_native, val_cuts])
    val_combine_native_non_nat = val_combine_native_non_nat.shuffle()

    #10349 all

    #500

    return combine_native_non_nat, val_combine_native_non_nat

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

    # here is 1 for debug
    if args.num_workers == None:
        cpu_count = os.cpu_count()  
        num_workers = cpu_count // 2 
        args.num_workers = num_workers
    
    print(f"num_workers:{args.num_workers}")
    params.update(vars(args))
    
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)
    
    # # attention please 
    # train_dir_name, val_dir_name = change_file_path(args, rank)

    if args.sec_dataset:
        bef_manifest_dir = args.manifest_dir
        args.manifest_dir = args.manifest_dir_sec
        # train_dir_name_v2, val_dir_name_v2 = change_file_path(args, rank)
        args.manifest_dir = bef_manifest_dir
        args.train_dir_name_sec = args.train_dir_name
        args.val_dir_name_sec = args.val_dir_name

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

    # i=0
    # for param in model.parameters():  
    #     i+=1
    #     param.requires_grad = True  

    # 冻结所有层的权重  
    for param in model.parameters():  
        param.requires_grad = False  

    if params.parts_req_gra==0:
        for param in model.parameters():  
            param.requires_grad = True
    elif params.parts_req_gra==1:
        for idx, layer in enumerate(model.encoder.layers):  
            for param in layer.parameters():  
                param.requires_grad = True
    elif params.parts_req_gra==2:
        num_layers = len(model.encoder.layers)
        for idx, layer in enumerate(model.encoder.layers):  
            if idx < num_layers - 3:  
                for param in layer.parameters():  
                    param.requires_grad = True      
    elif params.parts_req_gra==4:
        num_layers = len(model.ar_decoder.layers)
        for idx, layer in enumerate(model.ar_decoder.layers):
            for param in layer.parameters():  
                param.requires_grad = True  
    # # 解冻TransformerEncoder的后三层的权重  
    # num_layers = len(model.ar_decoder.layers)  
    # for idx, layer in enumerate(model.ar_decoder.layers):  
    #     if idx >= num_layers - 3:  
    #         for param in layer.parameters():  
    #             param.requires_grad = True  
    
    # # 解冻线性层的权重  
    # for param in model.ar_predict_layers.parameters():  
    #     param.requires_grad = True  
    
    # 打印模型以查看哪些层可训练  
    # # fine_tune_linear
    # # 假设您的模型实例名为valle_model  
    # # 为ar_predict_layers启用梯度更新，其他部分保持不变  
    # for param in model.parameters():  
    #     param.requires_grad = False  
    # # for param in model.ar_predict_layers.parameters():  
    # #     param.requires_grad = True  
    
    # # fine_tune_3layers_bef
    # num_layers_to_finetune = 3  
    
    # for i, layer in enumerate(model.ar_decoder.layers):  
    #     if i < num_layers_to_finetune:  
    #         for param in layer.parameters():  
    #             param.requires_grad = True  


    # print(model)
    # quit()

    
    # with open(f"{params.exp_dir}/model.txt", "w") as f:
    #     print(model)
    #     print(model, file=f)
    num_trainable_param = sum([1 for p in model.parameters() if p.requires_grad])  
    
    logging.info(f"Number of trainable model names: {num_trainable_param}")  
    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0 and params.average_period > 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)
    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )
    # print(f"Rank_{rank} before put model to device {torch.cuda.memory_allocated() / 1000000:.4f}MB")
    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        # print(f"Rank_{rank} after put model to ddp {torch.cuda.memory_allocated() / 1000000:.4f}MB")
    if params.train_stage:
        _model = model.module if isinstance(model, DDP) else model
        model_parameters = _model.stage_parameters(params.train_stage)
    else:
        model_parameters = (param for param in model.parameters() if param.requires_grad)  
    # model_parameters = (param for param in model.parameters() if param.requires_grad)  
    # model_parameters = model.ar_predict_layers.parameters()
    # print("traned params is: ")
    # for name, param in model.named_parameters():  
    #     if param.requires_grad:  
    #         print(name)  
    if params.optimizer_name == "ScaledAdam":
        parameters_names = []
        # if params.train_stage:  # != 0
        #     _model = model.module if isinstance(model, DDP) else model
        #     parameters_names.append(
        #         [
        #             name_param_pair[0]
        #             for name_param_pair in _model.stage_named_parameters(
        #                 params.train_stage
        #             )
        #         ]
        #     )
        # else:
        #     parameters_names.append(
        #         [
        #             name_param_pair[0]
        #             for name_param_pair in model.named_parameters()
        #         ]
        #     )
        if params.train_stage:  # != 0  
            _model = model.module if isinstance(model, DDP) else model  
            parameters_names.append(  
                [  
                    name_param_pair[0]  
                    for name_param_pair in _model.stage_named_parameters(  
                        params.train_stage  
                    )  
                    if name_param_pair[1].requires_grad  # Check if parameter is trainable  
                ]  
            )  

        else:  
            parameters_names.append(  
                [  
                    name_param_pair[0]  
                    for name_param_pair in model.named_parameters()  
                    if name_param_pair[1].requires_grad  # Check if parameter is trainable  
                ]  
            )  
        optimizer = ScaledAdam(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=parameters_names,
            show_dominant_parameters=False,
            clipping_update_period=1000,
        )
    elif params.optimizer_name == "Eve":
        optimizer = Eve(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.98),
            target_rms=0.1,
        )
    elif params.optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.95),
            weight_decay=1e-2,
            eps=1e-8,
        )
    elif params.optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    else:
        raise NotImplementedError()
    scheduler = get_scheduler(params, optimizer)
    # print(f"Rank_{rank} before optimizer zero_grad {torch.cuda.memory_allocated() / 1000000:.4f}MB")
    optimizer.zero_grad()
    # print(f"Rank_{rank} after optimizer zero_grad {torch.cuda.memory_allocated() / 1000000:.4f}MB")
    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])
    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])
    if params.inf_check:
        register_inf_check_hooks(model)
    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None
    dataset = TtsDataModule(args)
    train_cuts = dataset.train_cuts()
    valid_cuts = dataset.dev_cuts()

    # l1_l2 finetune
        # train_cuts = train_cuts.filter(lambda r: 'tni' in r.id)
        # train_cuts = train_cuts.subset(first=550)
    # print(f"train_first 550")
    if params.sec_dataset:

        bef_manifest_dir = args.manifest_dir
        args.manifest_dir = args.manifest_dir_sec
        # train_dir_name_v2, val_dir_name_v2 = change_file_path(args, rank)
        train_cuts_sec = dataset.train_cuts_sec()
        valid_cuts_sec = dataset.dev_cuts_sec()
        args.manifest_dir = bef_manifest_dir
        train_cuts, valid_cuts = combine_v1(train_cuts, train_cuts_sec, valid_cuts, valid_cuts_sec)
    
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
    scaler = GradScaler(
        enabled=(params.dtype in ["fp16", "float16"]), init_scale=1.0
    )
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])
    for epoch in range(params.start_epoch, params.num_epochs + 1):
        # print(f"Rank_{rank} memory before train epoch_{epoch} {torch.cuda.memory_allocated() / 1000000:.4f}MB")
        if isinstance(scheduler, Eden):
            scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)
        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)
        params.cur_epoch = epoch
        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            rng=rng,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )
        if params.test_demo is True:
            test_demos(
                params=params,
                model=model,
                rank=rank,
                tb_writer=tb_writer,
            )
        
        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
            
        )
        # print(f"Rank_{rank} memory after train epoch_{epoch} {torch.cuda.memory_allocated() / 1000000:.4f}MB"  )
    logging.info("Done!")
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

def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches
    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    dtype = torch.float32
    if params.dtype in ["bfloat16", "bf16"]:
        dtype = torch.bfloat16
    elif params.dtype in ["float16", "fp16"]:
        dtype = torch.float16
    for criterion, cuts in batches.items():
        # print(cuts)
        batch = train_dl.dataset[cuts]
        try:                
            # print(f"Rank_{model.device} scan before computer_loss {torch.cuda.memory_allocated() / 1000000:.4f}MB"  )
            with torch.cuda.amp.autocast(dtype=dtype):
                _, loss, _ = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                )
            # print(f"Rank_{model.device} scan after computer_loss {torch.cuda.memory_allocated() / 1000000:.4f}MB"  )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params)
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated() / 1000000:.4f}MB"
        )

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

    print(f"here:{args.manifest_dir}")
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
            new_cuts.append(new_cut)
        updated_cuts_valid = CutSet.from_cuts(new_cuts) 
        updated_cuts_valid.to_file(args.manifest_dir / newfile_valid)
        print(f"update valid file : {args.manifest_dir}/{newfile_valid}")
 
        # cut_set_valid.to_file(args.manifest_dir / "cuts_dev_jia6.jsonl.gz")
        print(f"keep to {args.manifest_dir}/{newfile_valid}")
        val_dir_name = newfile_valid
    return train_dir_name, val_dir_name


def main():
    parser = get_parser()
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()

    if args.semantic_type==1:
        args.semantic_num_quantizers=2
    print(f"args.semantic_num_quantizers:{args.semantic_num_quantizers}")
    args.exp_dir = Path(args.exp_dir)
    world_size = args.world_size
    assert world_size >= 1

    print(f"args.is_local is {args.is_local}")
    if args.is_local is True:
        local_rank = 0
    else:
        local_rank = int(os.environ["LOCAL_RANK"])

    os.environ["NCCL_ASYNC_ERROR_HANDING"] = "1"
    
    # train.py  
    # import re  
    # train_dir = os.path.dirname(os.path.abspath(__file__))  
    # path_file = os.path.join(train_dir, "PATH.py")  

    # print(f"path_file:{path_file}")
    # # 新的 manifest_dir 值  
    # new_manifest_dir = args.manifest_dir
    # # 打开并读取 PATH.py  
    # with open(path_file, "r") as file:  
    #     content = file.read()
    
    # # 使用正则表达式找到并替换 manifest_dir 变量  
    # content = re.sub(r'manifest_dir\s*=\s*".*"', f'manifest_dir = "{new_manifest_dir}"', content)  
    
    # # 将修改后的内容写回 PATH.py  
    # with open(path_file, "w") as file:  
    #     file.write(content)  

    
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
