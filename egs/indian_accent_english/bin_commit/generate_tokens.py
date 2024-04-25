
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
import json
import sys
# 获取当前工作目录  
current_working_directory = os.getcwd()  
import time  

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
import h5py  
import numpy as np
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
from valle.data import TtsDataModule_infer
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
from datetime import datetime  

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
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=0
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="""Run validation if batch_idx % valid_interval is 0.""",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Training dtype: float32 bfloat16 float16.",
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
        "--checkpoint",
        type=str,
        default="/dev_huaying/zhijun/data/valle-tensorboard-models/ar/Name_VALLE_max-duration_80_dtype_float32_base-lr_0.01_world-size_8_train-stage_1_echo_150_start_echo_1_2023_05_29_03_00_16/best-valid-loss.pt",
        help="path of ar_model checkpoint",
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
    parser.add_argument('--test-demo', type=str2bool, default=False)  
    
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
        "--parts-req-gra",
        type=int,
        default=0,
        help="0->all, 1->encoder, 2->first 3 encoder layers",
    )
    parser.add_argument('--num-workers', type=int, default=None)  
    parser.add_argument("--local_rank", type=int)  
    parser.add_argument("--nnodes", type=int)  
    parser.add_argument("--nproc_per_node", type=int)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/dev_huaying/zhijun/data/valle-tensorboard-models/nar/Name_VALLE_max-duration_70_dtype_float32_base-lr_0.01_world-size_8_train-stage_2_echo_150_start_echo_1_2023_05_29_01_26_40/best-valid-loss.pt",
        help="path of nar_model checkpoint",
    )

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


def generate_batch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
) -> Tuple[Tensor, MetricsTracker]:

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

    text_tokens = batch["text_tokens"].to(device)
    text_tokens_lens = batch["text_tokens_lens"].to(device)
    semantic_tokens = batch["semantic_tokens"].to(device)
    semantic_tokens_lens = batch["semantic_tokens_lens"].to(device)
    utt_ids = batch["utt_id"]

    batch_codes = model.inference_only_ar(
        x=text_tokens,
        x_lens=text_tokens_lens,
        y=None,
        top_k=2,
        task_id=2,
        use_silence_token=True
    )

    return batch_codes, utt_ids


def generate_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    args,
    world_size: int = 1,
    rank: int = 0,
) -> None:

    global run_now
    model.eval()
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
        dic = {}
        while True:
            try:
                batch = next(iter_dl)
            except StopIteration:
                logging.info("Reaches end of dataloader.")
                print(f"curren_rank: {rank} total_batch is {batch_idx}")
                current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_name = args.train_dir_name.split('/')[-1].split('.')[0]
                os.makedirs(args.output_dir, exist_ok=True)
                with h5py.File(os.path.join(args.output_dir, f'{file_name}_sem_dic_batch_idx_{batch_idx}_rank_{rank}_{current_time}.h5'), 'w') as h5f:  
                    for key, value in dic.items():  
                        h5f.create_dataset(key, data=np.array(value, dtype=np.int32)) 
                break
            batch_idx += 1
            params.batch_idx_train += 1
            batch_size = len(batch["text"])
            seed = int(time.time())  
            torch.manual_seed(seed)  

            with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
                for i in range(1):
                    batch_codes, cut_ids = generate_batch(
                        params=params,
                        model=model,
                        batch=batch,
                    )
                    for code, cut_id in zip(batch_codes, cut_ids):
                        code = code.squeeze(0).squeeze(1)  #need remove codebook dim
                        code = code.tolist() 
                        cut_id = cut_id +f"_{i}"
                        dic[cut_id] = code
            if batch_idx%10==0:
                print(batch_idx)
            # if batch_idx%1==0:
            #     # 将字典保存为.h5文件  
            #     # print(f"curren_rank: {rank} total_batch is {batch_idx}")
            #     current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            #     file_name = args.train_dir_name.split('/')[-1].split('.')[0]
            #     os.makedirs(args.output_dir, exist_ok=True)
            #     with h5py.File(os.path.join(args.output_dir, f'{file_name}_sem_dic_batch_idx_{batch_idx}_rank_{rank}_{current_time}.h5'), 'w') as h5f:  
            #         for key, value in dic.items():  
            #             h5f.create_dataset(key, data=np.array(value, dtype=np.int32)) 
            #     quit()
              
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


    checkpoint = torch.load(args.checkpoint, map_location=device)  # if target_mode!=0, checkpoint1 is correct model. second: encodec generative model
    # checkpoint2 = torch.load(args.checkpoint2, map_location=device)

    args.only_autoregressive= True
    # args.shared_linear= False
    # # 将checkpoint2的键写入到另一个txt文件中  
    # with open('keys_output_checkpoint2.txt', 'w') as f:  
    #     for key in checkpoint2["model"].keys():  
    #         f.write(key + '\n')  
    missing_keys1, unexpected_keys1 = model.load_state_dict(
        checkpoint["model"], strict=False
    )
    # print(missing_keys1)
    assert not missing_keys1
    model.to(device)
    model.eval()
    # args.only_autor
   
    if params.inf_check:
        register_inf_check_hooks(model)

    sampler_state_dict = None
    dataset = TtsDataModule_infer(args)
    train_cuts = dataset.train_cuts()

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

    train_dl = dataset.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )


        # print(f"Rank_{rank} memory before train epoch_{epoch} {torch.cuda.memory_allocated() / 1000000:.4f}MB")

    # fix_random_seed(params.seed + epoch - 1)
    # train_dl.sampler.set_epoch(epoch - 1)

    generate_one_epoch(
        params=params,
        model=model,
        train_dl=train_dl,
        world_size=world_size,
        rank=rank,
        args=args
    )

        
                    # print(f"Rank_{rank} memory after train epoch_{epoch} {torch.cuda.memory_allocated() / 1000000:.4f}MB"  )
    logging.info("Done!")
    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    TtsDataModule_infer.add_arguments(parser)
    args = parser.parse_args()

    if args.semantic_type==1:
        args.semantic_num_quantizers=2
    print(f"args.semantic_num_quantizers:{args.semantic_num_quantizers}")
    world_size = args.world_size
    assert world_size >= 1

    print(f"args.is_local is {args.is_local}")
    if args.is_local is True:
        local_rank = 0
    else:
        local_rank = int(os.environ["LOCAL_RANK"])

    os.environ["NCCL_ASYNC_ERROR_HANDING"] = "1"
    
    
    run(rank=local_rank, world_size=world_size, args=args)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if __name__ == "__main__":
    import sys   
    main()
