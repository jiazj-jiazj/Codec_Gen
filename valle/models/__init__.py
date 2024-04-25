import argparse

import torch.nn as nn
from icefall.utils import AttributeDict, str2bool

from .macros import (
    NUM_AUDIO_TOKENS,
    NUM_MEL_BINS,
    NUM_SPEAKER_CLASSES,
    NUM_TEXT_TOKENS,
    SPEAKER_EMBEDDING_DIM,
)
from .transformer import Transformer
from .valle import VALLE, VALLF, VALLF_E, Soundstorm, VALLE_NAR
from .visualizer import visualize


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model-name",
        type=str,
        default="VALL-E",
        help="VALL-E, VALL-F, Transformer.",
    )
    parser.add_argument(
        "--model-name-stage2",
        type=str,
        default="VALL-E",
        help="VALL-E, VALL-F, Transformer.",
    )
    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=1024,
        help="Embedding dimension in the decoder model.",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=16,
        help="Number of attention heads in the Decoder layers.",
    )
    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=12,
        help="Number of Decoder layers.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Model scale factor which will be assigned different meanings in different models.",
    )
    parser.add_argument(
        "--norm-first",
        type=str2bool,
        default=True,
        help="Pre or Post Normalization.",
    )
    parser.add_argument(
        "--add-prenet",
        type=str2bool,
        default=False,
        help="Whether add PreNet after Inputs.",
    )

    # VALL-E & F
    parser.add_argument(
        "--prefix-mode",
        type=int,
        default=0,
        help="The mode for how to prefix VALL-E NAR Decoder, "
        "0: no prefix, 1: 0 to random, 2: random to random, 4: chunk of pre or post utterance.",
    )
    parser.add_argument(
        "--share-embedding",
        type=str2bool,
        default=True,
        help="Share the parameters of the output projection layer with the parameters of the acoustic embedding.",
    )
    parser.add_argument(
        "--prepend-bos",
        type=str2bool,
        default=False,
        help="Whether prepend <BOS> to the acoustic tokens -> AR Decoder inputs.",
    )
    parser.add_argument(
        "--num-quantizers",
        type=int,
        default=8,
        help="Number of Audio/Semantic quantization layers.",
    )
    parser.add_argument(
        "--semantic-num-quantizers",
        type=int,
        default=1,
        help="Number of Semantic quantization layers.",
    )

    # Transformer
    parser.add_argument(
        "--scaling-xformers",
        type=str2bool,
        default=False,
        help="Apply Reworked Conformer scaling on Transformers.",
    )


def get_model(params: AttributeDict) -> nn.Module:


    if params.model_name.lower() in ["vall-f", "vallf"]:
        model = VALLF(
            params.decoder_dim,
            params.nhead,
            params.num_decoder_layers,
            norm_first=params.norm_first,
            add_prenet=params.add_prenet,
            prefix_mode=params.prefix_mode,
            share_embedding=params.share_embedding,
            nar_scale_factor=params.scale_factor,
            prepend_bos=params.prepend_bos,
            num_quantizers=params.num_quantizers,
            semantic_num_quantizers=params.semantic_num_quantizers,
            input_semantic=params.input_semantic,
            only_autoregressive=params.only_autoregressive,
            shared_linear=params.shared_linear,
            is_pretrain = params.is_pretrain,
            pret_mode = params.pret_mode

        )
    elif params.model_name.lower() in ["vall-e", "valle"]:
        if "only_autoregressive" not in params:
            params.only_autoregressive = False
        if "shared_linear" not in params:
            params.shared_linear = False
        if "is_pretrain" not in params:
            params.is_pretrain = False
        if "pret_mode" not in params:
            params.pret_mode = False
        model = VALLE(
            params.decoder_dim,
            params.nhead,
            params.num_decoder_layers,
            norm_first=params.norm_first,
            add_prenet=params.add_prenet,
            prefix_mode=params.prefix_mode,
            share_embedding=params.share_embedding,
            nar_scale_factor=params.scale_factor,
            prepend_bos=params.prepend_bos,
            num_quantizers=params.num_quantizers,
            semantic_num_quantizers=params.semantic_num_quantizers,
            input_semantic=params.input_semantic,
            only_autoregressive=params.only_autoregressive,
            shared_linear=params.shared_linear,
            is_pretrain = params.is_pretrain,
            pret_mode = params.pret_mode
        )
    elif params.model_name.lower() in ["vallfe", "vallf_e", "vallf-e"]:
        if "only_autoregressive" not in params:
            params.only_autoregressive = False
        if "shared_linear" not in params:
            params.shared_linear = False
        if "is_pretrain" not in params:
            params.is_pretrain = False
        if "pret_mode" not in params:
            params.pret_mode = False
        model = VALLF_E(
            params.decoder_dim,
            params.nhead,
            params.decoder_num_layers,
            norm_first=params.norm_first,
            add_prenet=params.add_prenet,
            prefix_mode=params.prefix_mode,
            share_embedding=params.share_embedding,
            nar_scale_factor=params.scale_factor,
            prepend_bos=params.prepend_bos,
            num_quantizers=params.num_quantizers,
            semantic_num_quantizers=params.semantic_num_quantizers,
            encoder_num_layers=params.encoder_num_layers,
            input_semantic=params.input_semantic,
            only_autoregressive=params.only_autoregressive,
            shared_linear=params.shared_linear,
            is_pretrain = params.is_pretrain,
            pret_mode = params.pret_mode
        )
    elif params.model_name.lower() in ["soundstorm"]:
        if "only_autoregressive" not in params:
            params.only_autoregressive = False
        if "shared_linear" not in params:
            params.shared_linear = False
        if "is_pretrain" not in params:
            params.is_pretrain = False
        if "pret_mode" not in params:
            params.pret_mode = False
        model = Soundstorm(
            params.decoder_dim,
            params.nhead,
            params.num_decoder_layers,
            norm_first=params.norm_first,
            add_prenet=params.add_prenet,
            prefix_mode=params.prefix_mode,
            share_embedding=params.share_embedding,
            nar_scale_factor=params.scale_factor,
            prepend_bos=params.prepend_bos,
            num_quantizers=params.num_quantizers,
            semantic_num_quantizers=params.semantic_num_quantizers,
            input_semantic=params.input_semantic,
            only_autoregressive=params.only_autoregressive,
            shared_linear=params.shared_linear,
            is_pretrain = params.is_pretrain,
            pret_mode = params.pret_mode,
            parrallel_mode = params.parrallel_mode
        )
    elif params.model_name.lower() in ["valle_nar"]:
        if "only_autoregressive" not in params:
            params.only_autoregressive = False
        if "shared_linear" not in params:
            params.shared_linear = False
        if "is_pretrain" not in params:
            params.is_pretrain = False
        if "pret_mode" not in params:
            params.pret_mode = False
        model = VALLE_NAR(
            params.decoder_dim,
            params.nhead,
            params.num_decoder_layers,
            norm_first=params.norm_first,
            add_prenet=params.add_prenet,
            prefix_mode=params.prefix_mode,
            share_embedding=params.share_embedding,
            nar_scale_factor=params.scale_factor,
            prepend_bos=params.prepend_bos,
            num_quantizers=params.num_quantizers,
            semantic_num_quantizers=params.semantic_num_quantizers,
            input_semantic=params.input_semantic,
            only_autoregressive=params.only_autoregressive,
            shared_linear=params.shared_linear,
            is_pretrain = params.is_pretrain,
            pret_mode = params.pret_mode
        )

    else:
        assert params.model_name in ["Transformer"]
        model = Transformer(
            params.decoder_dim,
            params.nhead,
            params.num_decoder_layers,
            norm_first=params.norm_first,
            add_prenet=params.add_prenet,
            scaling_xformers=params.scaling_xformers,
        )

    return model
