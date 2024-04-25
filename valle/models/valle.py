# Copyright    2023                             (authors: Feiteng Li)
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
import numpy
import random
from typing import Dict, Iterator, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from icefall.utils import make_pad_mask
from torchmetrics.classification import MulticlassAccuracy
from tfnet.tfnet_models_mp_lm.utils import default, exists, maybe, ceil_div, prob_mask_like, get_embeds, eval_decorator, batch_unique_consecutive, append_eos_id, generate_mask_with_prob
import math
from valle.data.input_strategies import PromptedFeatures
from valle.modules.embedding import SinePositionalEmbedding, TokenEmbedding
from einops import rearrange, repeat, reduce
from valle.modules.transformer import (
    AdaptiveLayerNorm,
    LayerNorm,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from .macros import NUM_AUDIO_TOKENS, NUM_TEXT_TOKENS, NUM_SEMANTIC_TOKENS, IGNORE_TOKENS, NUM_SEMANTIC_TOKENS_TFNET
from .visualizer import visualize
from torch import nn, einsum


class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)
        
def beam_search(logits, beam_size, temperature=1.0):  
    probs = torch.softmax(logits / temperature, dim=-1)  
    log_probs, indices = torch.topk(probs.log(), beam_size, dim=-1)  
    return log_probs, indices 

# NOTE: There are two ways to implement the model
#       1) [VALL-F] standard TransformerDecoder, use x as memory
#       2) [VALL-E] modified TransformerDecoder like GPT-x(e.g. causal TransformerEncoder),
#          use x as the prefix of decoder inputs
class VALLF(nn.Module):
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        decoder_cls: Union[
            nn.TransformerDecoder, nn.TransformerEncoder
        ] = nn.TransformerDecoder,
        decoder_layer_cls: Union[
            TransformerDecoderLayer, TransformerEncoderLayer
        ] = TransformerDecoderLayer,
        prefix_mode: int = 0,
        share_embedding: bool = True,
        nar_scale_factor: float = 1.0,
        prepend_bos: bool = False,
        semantic_num_quantizers: int = 1,
        num_quantizers: int = 8,
        input_semantic: bool = False,
        only_autoregressive: bool = False,
        shared_linear: bool = False,
        is_pretrain: bool = False,
        pret_mode: int = 0 ,
        parrallel_mode: int = 0
    ):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super().__init__()
        if only_autoregressive is True:
            print(f"only_antoregressive is True")
        self.num_quantizers = num_quantizers
        nar_d_model = int(d_model * nar_scale_factor)
        self.input_semantic = input_semantic

        # HERE
        self.semantic_num_quantizers = semantic_num_quantizers
        if semantic_num_quantizers==1:
            if input_semantic is True:
                self.ar_text_embedding = TokenEmbedding(d_model, NUM_SEMANTIC_TOKENS)  # W_x
                if only_autoregressive is False:
                    self.nar_text_embedding = TokenEmbedding(nar_d_model, NUM_SEMANTIC_TOKENS)
            else:
                self.ar_text_embedding = TokenEmbedding(d_model, NUM_TEXT_TOKENS)  # W_x
                if only_autoregressive is False:
                    self.nar_text_embedding = TokenEmbedding(nar_d_model, NUM_TEXT_TOKENS)
        elif semantic_num_quantizers==2:

            embedding_dim_sem = d_model // semantic_num_quantizers
            self.embedding_dim_sem = embedding_dim_sem
            assert embedding_dim_sem * semantic_num_quantizers == d_model, "The dimension of transformer must be divided by the number of quantizers!"
            if input_semantic is True:
                self.ar_text_embedding = TokenEmbedding(embedding_dim_sem, semantic_num_quantizers * (NUM_SEMANTIC_TOKENS_TFNET + 1))  # real is 256+3(pad, eos, bos)
                if only_autoregressive is False:
                    self.nar_text_embedding = TokenEmbedding(nar_d_model, NUM_SEMANTIC_TOKENS) # no need for nar process, because tfnet is only for ar
            else:
                # if input is text, semantic_num_quantizers==1
                self.ar_text_embedding = TokenEmbedding(d_model, NUM_TEXT_TOKENS)  # W_x
                if only_autoregressive is False:
                    self.nar_text_embedding = TokenEmbedding(nar_d_model, NUM_TEXT_TOKENS)
        # ID NUM_AUDIO_TOKENS     -> PAD
        # ID NUM_AUDIO_TOKENS + 1 -> BOS
        self.ar_audio_prepend_bos = prepend_bos
        self.only_autoregressive = only_autoregressive
        self.shared_linear = shared_linear

        self.is_pretrain = is_pretrain
        self.pret_mode = pret_mode
        self.parrallel_mode = parrallel_mode
        if only_autoregressive is False:
            self.ar_audio_embedding = TokenEmbedding(
                d_model, NUM_AUDIO_TOKENS + 1 + int(prepend_bos)
            )
        if only_autoregressive is True:
            self.only_autoregressive = only_autoregressive
            if self.parrallel_mode ==0:
                embedding_dim = d_model // num_quantizers
                self.ar_audio_embedding = TokenEmbedding(embedding_dim, num_quantizers * (NUM_AUDIO_TOKENS + 1 + int(prepend_bos)))
                assert embedding_dim * num_quantizers == d_model, "The dimension of transformer must be divided by the number of quantizers!"
            elif self.parrallel_mode ==1:
                embedding_dim = d_model
                self.ar_audio_embedding = TokenEmbedding(embedding_dim, num_quantizers * (NUM_AUDIO_TOKENS + 1 + int(prepend_bos)))
                self.codebook_embedding = TokenEmbedding(d_model, self.num_quantizers)
            elif self.parrallel_mode ==2:
                embedding_dim = d_model//(num_quantizers//2)
                print(f"embedding_dim:{embedding_dim}")
                self.ar_audio_embedding = TokenEmbedding(embedding_dim, num_quantizers * (NUM_AUDIO_TOKENS + 1 + int(prepend_bos)))
                self.codebook_embedding = TokenEmbedding(d_model, 2)    

            self.embedding_dim = embedding_dim
            
            if prepend_bos is True:
                self.start_token = nn.Parameter(torch.randn(d_model))

            if shared_linear is False:
                self.ar_predict_layers = nn.ModuleList(   # 这里不需要加上int(prepend_bos)  #有输入即可
                    [
                        nn.Linear(embedding_dim, NUM_AUDIO_TOKENS+1, bias=False)
                        for i in range(num_quantizers)
                    ]
                )
            else:
                self.ar_predict_layer = nn.Linear(embedding_dim, NUM_AUDIO_TOKENS+1, bias=False)
                    
            self.ar_accuracy_metric = MulticlassAccuracy(
                NUM_AUDIO_TOKENS + 1,
                top_k=10,
                average="micro",
                multidim_average="global",
                ignore_index=NUM_AUDIO_TOKENS,
            )
            self.ar_accuracy_metric_top1 = MulticlassAccuracy(
                NUM_AUDIO_TOKENS + 1,
                top_k=1,
                average="micro",
                multidim_average="global",
                ignore_index=NUM_AUDIO_TOKENS,
            )

            if is_pretrain is True and (self.pret_mode ==5 or self.pret_mode ==7):
                self.ar_accuracy_metric_ignore = MulticlassAccuracy(
                NUM_AUDIO_TOKENS + 1,
                top_k=10,
                average="micro",
                multidim_average="global",
                ignore_index=IGNORE_TOKENS,
                )
                self.ar_accuracy_metric_top1_ignore = MulticlassAccuracy(
                    NUM_AUDIO_TOKENS + 1,
                    top_k=1,
                    average="micro",
                    multidim_average="global",
                    ignore_index=IGNORE_TOKENS,
                )
        # PreNet
        # print(f"add_prenet :{add_prenet}")
        if add_prenet:
            print("add_prenet is True")
            self.ar_text_prenet = nn.Sequential(
                Transpose(),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                Transpose(),
                nn.Linear(d_model, d_model),
            )

            self.ar_audio_prenet = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, d_model),
            )
        else:
            print("add_prenet is False")
            self.ar_text_prenet = nn.Identity()
            self.ar_audio_prenet = nn.Identity()

        self.ar_text_position = SinePositionalEmbedding(
            d_model,
            dropout=0.1,
            scale=False,
            alpha=True,
        )

        self.ar_audio_position = SinePositionalEmbedding(
            d_model,
            dropout=0.1,
            scale=False,
            alpha=True,
        )

        self.ar_decoder = decoder_cls(
            decoder_layer_cls(
                d_model,
                nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=LayerNorm(d_model) if norm_first else None,
        )
        if only_autoregressive is False:
            self.ar_predict_layer = nn.Linear(
                d_model, NUM_AUDIO_TOKENS + 1, bias=False
            )

            self.ar_accuracy_metric = MulticlassAccuracy(
                NUM_AUDIO_TOKENS + 1,
                top_k=10,
                average="micro",
                multidim_average="global",
                ignore_index=NUM_AUDIO_TOKENS,
            )
            self.ar_accuracy_metric_top1 = MulticlassAccuracy(
                NUM_AUDIO_TOKENS + 1,
                top_k=1,
                average="micro",
                multidim_average="global",
                ignore_index=NUM_AUDIO_TOKENS,
            )

            if self.is_pretrain is True and (self.pret_mode ==5 or self.pret_mode ==7):
                self.ar_accuracy_metric_ignore = MulticlassAccuracy(
                    NUM_AUDIO_TOKENS + 1,
                    top_k=10,
                    average="micro",
                    multidim_average="global",
                    ignore_index=IGNORE_TOKENS,
                )
                self.ar_accuracy_metric_top1_ignore = MulticlassAccuracy(
                    NUM_AUDIO_TOKENS + 1,
                    top_k=1,
                    average="micro",
                    multidim_average="global",
                    ignore_index=IGNORE_TOKENS,
                )


        self.rng = random.Random(0)
        self.num_heads = nhead
        self.prefix_mode = prefix_mode

        assert num_quantizers >= 1
        if num_quantizers > 1 and only_autoregressive is False:
            self.nar_audio_embeddings = nn.ModuleList(
                [TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS + 1)]
                + [
                    TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS)
                    for i in range(num_quantizers - 1)
                ]
            )  # W_a

            # PreNet
            if add_prenet:
                self.nar_text_prenet = nn.Sequential(
                    Transpose(),
                    nn.Conv1d(
                        nar_d_model, nar_d_model, kernel_size=5, padding="same"
                    ),
                    nn.BatchNorm1d(nar_d_model),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Conv1d(
                        nar_d_model, nar_d_model, kernel_size=5, padding="same"
                    ),
                    nn.BatchNorm1d(nar_d_model),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Conv1d(
                        nar_d_model, nar_d_model, kernel_size=5, padding="same"
                    ),
                    nn.BatchNorm1d(nar_d_model),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    Transpose(),
                    nn.Linear(nar_d_model, nar_d_model),
                )
                self.nar_audio_prenet = nn.Sequential(
                    nn.Linear(nar_d_model, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, nar_d_model),
                )
            else:
                self.nar_text_prenet = nn.Identity()
                self.nar_audio_prenet = nn.Identity()

            self.nar_text_position = SinePositionalEmbedding(
                nar_d_model,
                dropout=0.0,
                scale=False,
                alpha=False,
            )
            self.nar_audio_position = SinePositionalEmbedding(
                nar_d_model,
                dropout=0.1,
                scale=False,
                alpha=False,
            )

            self.nar_decoder = decoder_cls(
                decoder_layer_cls(
                    nar_d_model,
                    int(nhead * nar_scale_factor),
                    dim_feedforward=nar_d_model * 4,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=norm_first,
                    adaptive_layer_norm=True,
                ),
                num_layers=int(num_layers * nar_scale_factor),
                norm=AdaptiveLayerNorm(
                    nar_d_model, norm=nn.LayerNorm(nar_d_model)
                )
                if norm_first
                else None,
            )
            self.nar_predict_layers = nn.ModuleList(
                [
                    nn.Linear(nar_d_model, NUM_AUDIO_TOKENS, bias=False)
                    for i in range(num_quantizers - 1)
                ]
            )
            self.nar_stage_embeddings = nn.ModuleList(
                [
                    TokenEmbedding(nar_d_model, 1)
                    for i in range(num_quantizers - 1)
                ]
            )

            if share_embedding:
                # We share the parameters of the output projection layer with the parameters of the acoustic embedding Wa
                # NOTE(Feiteng): In the experiment, this undermines accuracy
                # self.ar_predict_layer.weight = self.ar_audio_embedding.weight

                # We also share the parameters of the acoustic embedding layer and the output prediction layer,
                # which means the weights of the j-th prediction layer are the same as the (j + 1)-th acoustic embedding layer.
                for j in range(0, num_quantizers - 2):
                    self.nar_predict_layers[
                        j
                    ].weight = self.nar_audio_embeddings[j + 2].weight

            self.nar_accuracy_metric = MulticlassAccuracy(
                NUM_AUDIO_TOKENS + 1,
                top_k=10,
                average="micro",
                multidim_average="global",
                ignore_index=NUM_AUDIO_TOKENS,
            )
            self.nar_accuracy_metric_top1 = MulticlassAccuracy(
                NUM_AUDIO_TOKENS + 1,
                top_k=1,
                average="micro",
                multidim_average="global",
                ignore_index=NUM_AUDIO_TOKENS,
            )

    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, (nn.Linear)):
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=1.0)

    def stage_parameters(self, stage: int = 1) -> Iterator[nn.Parameter]:
        assert stage > 0
        if stage == 1:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if name.startswith("ar_") or not name.startswith("nar_"):
                        print(f" AR parameter: {name}")
                        yield param

        if stage == 2:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if name.startswith("nar_") or not name.startswith("ar_"):
                        print(f"NAR parameter: {name}")
                        yield param

    def stage_named_parameters(
        self, stage: int = 1
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        assert stage > 0
        if stage == 1:
            for pair in self.named_parameters():
                if pair[0].startswith("ar_") or not pair[0].startswith("nar_"):
                    yield pair

        if stage == 2:
            for pair in self.named_parameters():
                if pair[0].startswith("nar_") or not pair[0].startswith("ar_"):
                    yield pair

    def pad_y_eos(self, y, y_mask_int, eos_id):

        if len(y.shape)==3:
            y_mask_int = y_mask_int.unsqueeze(2)  
            y_mask_int = y_mask_int.repeat_interleave(self.num_quantizers, dim=-1)  
            targets = F.pad(y, (0, 0, 0, 1), value=0) + eos_id * F.pad(
                y_mask_int, (0, 0, 0, 1), value=1
            )
        else: 
            targets = F.pad(y, (0, 1), value=0) + eos_id * F.pad(
            y_mask_int, (0, 1), value=1
            )
        # inputs, targets
        #need to update if y.size==3
        if len(y.shape)==3:
            if self.ar_audio_prepend_bos:
                return F.pad(targets[:, :-1, :], (0, 0, 1, 0), value=NUM_AUDIO_TOKENS + 1), targets

            else:
                return targets[:, :-1, :], targets[:, 1:, :]
        else:
            if self.ar_audio_prepend_bos:
                return F.pad(targets[:, :-1], (1, 0), value=NUM_AUDIO_TOKENS + 1),targets
            else:
                return targets[:, :-1], targets[:, 1:]

    def _prepare_prompts(self, y, y_lens, codes, nar_stage, y_prompts_codes):
        # 5.1 For the NAR acoustic prompt tokens, we select a random segment waveform of 3 seconds
        # from the same utterance.
        # We implement this differently.
        if self.prefix_mode == 0:
            # no prefix
            prefix_len = 0
            y_emb = self.nar_audio_embeddings[0](y)
            for j in range(1, nar_stage):
                # Formula (4) (5)
                y_emb = y_emb + self.nar_audio_embeddings[j](codes[..., j])
        elif self.prefix_mode == 1:
            # prefix at begining
            int_low = (0.25 * y_lens.min()).type(torch.int64).item()
            prefix_len = torch.randint(int_low, int_low * 2, size=()).item()
            prefix_len = min(prefix_len, 225)  # 24000/320 * 3s = 225 frames

            y_prompts = self.nar_audio_embeddings[0](y[:, :prefix_len])
            y_emb = self.nar_audio_embeddings[0](y[:, prefix_len:])
            for j in range(1, self.num_quantizers):
                y_prompts += self.nar_audio_embeddings[j](
                    codes[:, :prefix_len, j]
                )
                if j < nar_stage:
                    y_emb += self.nar_audio_embeddings[j](
                        codes[:, prefix_len:, j]
                    )
            y_emb = torch.concat([y_prompts, y_emb], axis=1)
        elif self.prefix_mode in [2, 4]:
            if self.prefix_mode == 2:
                # random prefix
                prefix_len = min(225, int(0.25 * y_lens.min().item()))

                y_prompts_codes = []
                for b in range(codes.shape[0]):
                    start = self.rng.randint(0, y_lens[b].item() - prefix_len)
                    y_prompts_codes.append(
                        torch.clone(codes[b, start : start + prefix_len])
                    )
                    codes[
                        b, start : start + prefix_len, nar_stage
                    ] = NUM_AUDIO_TOKENS
                y_prompts_codes = torch.stack(y_prompts_codes, dim=0)
            else:
                prefix_len = y_prompts_codes.shape[1]

            y_prompts = self.nar_audio_embeddings[0](y_prompts_codes[..., 0])
            y_emb = self.nar_audio_embeddings[0](y)
            for j in range(1, self.num_quantizers):
                y_prompts += self.nar_audio_embeddings[j](
                    y_prompts_codes[..., j]
                )
                if j < nar_stage:
                    y_emb += self.nar_audio_embeddings[j](codes[..., j])
            y_emb = torch.concat([y_prompts, y_emb], axis=1)
        else:
            raise ValueError

        return y_emb, prefix_len

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        # semantic: torch.Tensor,
        # semantic_lens: torch.Tensor,
        y: Union[torch.Tensor, PromptedFeatures],
        y_lens: Union[torch.Tensor, PromptedFeatures],
        reduction: str = "sum",
        train_stage: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (N, T, 8).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          train_stage:
            0: AR & NAR modules, 1: AR modules, 2: NAR modules
        Returns:
          Return the predicted audio code matrix, cross-entropy loss and Top-10 accuracy.
        """

        # x = semantic
        # x_lens = semantic_lens

        if self.semantic_num_quantizers==1:
            assert x.ndim == 2, x.shape
            assert x_lens.ndim == 1, x_lens.shape
        elif self.semantic_num_quantizers==2:  # text is semantic tfnet
            assert x.ndim == 3, x.shape
            assert x_lens.ndim == 1, x_lens.shape

        y_prompts_codes = None
        if isinstance(y, PromptedFeatures):
            y_prompts_codes, y = y.data
            prompts_len, y_lens = y_lens.data
            assert prompts_len.min() == prompts_len.max()
            assert self.prefix_mode == 4
            y_prompts_codes = y_prompts_codes.type(torch.int64)

        assert y.ndim == 3, y.shape
        assert y_lens.ndim == 1, y_lens.shape

        # NOTE: x has been padded in TextTokenCollater
        x_mask = make_pad_mask(x_lens).to(x.device)

        text = x
        x = self.ar_text_embedding(text)
        # x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        total_loss, metrics = 0.0, {}

        y_mask = make_pad_mask(y_lens).to(y.device)
        y_mask_int = y_mask.type(torch.int64)


        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))

        # Training
        # AR Decoder
        # y, targets = self.pad_y_eos(
        #     codes[..., 0], y_mask_int, eos_id=NUM_AUDIO_TOKENS
        # )
        if self.only_autoregressive:   # no matter target token is acoustic or semantic token, eos_is is 1024
            y, targets = self.pad_y_eos(   ## attention BOS 
                codes, y_mask_int, eos_id=NUM_AUDIO_TOKENS
            )
        else:
            y, targets = self.pad_y_eos(
                codes[...,0], y_mask_int, eos_id=NUM_AUDIO_TOKENS
            )

        # print(y.shape)
        # print(targets.shape)
        # quit()

        if self.only_autoregressive is True:
            y = rearrange(y, 'b ... -> b (...)') # b x T x 16-> bx (T x 16)
            offsets = (NUM_AUDIO_TOKENS+1+int(self.ar_audio_prepend_bos)) * torch.arange(self.num_quantizers, device = y.device) # [0, 501, 1002, ...]
            offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(y.shape[-1], self.num_quantizers))  # n==T
            offsets = offsets[:, :y.shape[-1]]
            y = y + offsets

            y_emb = self.ar_audio_embedding(y)
            y_emb = rearrange(y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers)
        else:
            y_emb = self.ar_audio_embedding(y)

        if train_stage in [0, 1]:
            if self.only_autoregressive is True:
                y = rearrange(y, 'b ... -> b (...)') # b x T x 16-> bx (T x 16)
                offsets = (NUM_AUDIO_TOKENS+1+int(self.ar_audio_prepend_bos)) * torch.arange(self.num_quantizers, device = y.device) # [0, 501, 1002, ...]
                offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(y.shape[-1], self.num_quantizers))  # n==T
                offsets = offsets[:, :y.shape[-1]]
                y = y + offsets

                y_emb = self.ar_audio_embedding(y)
                y_emb = rearrange(y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers)
            else:
                y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)

            ar_y_mask = y_mask
            if self.ar_audio_prepend_bos:
                ar_y_mask = F.pad(y_mask, (1, 0), value=False)

            y_len = y_lens.max() + int(self.ar_audio_prepend_bos)
            tgt_mask = torch.triu(
                torch.ones(y_len, y_len, device=y.device, dtype=torch.bool),
                diagonal=1,
            )
            y_dec, _ = self.ar_decoder(   # b x t x 1024
                (y_pos, None),
                x,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=ar_y_mask,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )

            y_dec = rearrange(y_dec, 'b t (q d) -> b t q d', q = self.num_quantizers)
                # why not directly pass b t q d to nn.linear ? different codec layer has different linear
                # logits = einsum('q c d, b t q d -> b t q c', self.logit_weights, xy_dec)
                # # logits = rearrange(logits, 'b t q c -> b (t q) c')
                # # logits = logits.permute(0,3,1,2)
                # logits = logits.permute(0,3,1,2)
            logits_list = []
            if self.shared_linear is False:
                for j in range(self.num_quantizers):
                    logits = self.ar_predict_layers[j](y_dec[:, :, j, :]).permute(0, 2, 1) # b t d -> b t num_classes -> b num_classes t 
                    logits_list.append(logits)

                logits = torch.stack(logits_list, dim=-1) # b num_classes t -> b num_classes t num_quantizers
            
            else:
                logits = self.ar_predict_layer(y_dec).permute(0, 3, 1, 2) # b t q d -> b num_class t q  

            total_loss = F.cross_entropy(logits, targets, reduction=reduction)
            metrics["ArTop10Accuracy"] = self.ar_accuracy_metric(
                logits.detach(), targets
            ).item() * y_lens.sum().type(torch.float32)

        if self.num_quantizers == 1:
            return ((x, codes), total_loss, metrics)

        # Non-AR Decoders
        if self.ar_audio_prepend_bos:
            y = y[:, 1:]

        if train_stage in [0, 2]:
            num_nar_layers = self.num_quantizers - 1
            nar_stage = self.rng.choices(
                [_k for _k in range(1, self.num_quantizers)],
                weights=[1.0 / num_nar_layers] * num_nar_layers,
                k=1,
            )[0]

            x = self.nar_text_embedding(text)
            x = self.nar_text_prenet(x)
            x = self.nar_text_position(x)

            y_emb, prefix_len = self._prepare_prompts(
                y, y_lens, codes, nar_stage, y_prompts_codes
            )

            y_len = y_lens.max()
            targets = codes[..., nar_stage] + NUM_AUDIO_TOKENS * y_mask_int
            if self.prefix_mode in [2, 4]:
                targets = targets
                y_mask = F.pad(y_mask, (y_emb.shape[1] - y_len, 0), value=False)
            elif self.prefix_mode == 1:
                targets = targets[:, prefix_len:]
            else:
                assert prefix_len == 0

            y_pos = self.nar_audio_prenet(y_emb)
            y_pos = self.nar_audio_position(y_pos)

            y_dec, _ = self.nar_decoder(
                (y_pos, self.nar_stage_embeddings[nar_stage - 1].weight),
                x,
                tgt_mask=None,
                tgt_key_padding_mask=y_mask,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )
            if self.prefix_mode != 0:
                y_dec = y_dec[:, prefix_len:]
                if self.prefix_mode == 4:
                    prefix_len = 0  # reset for Top10Accuracy metric

            logits = self.nar_predict_layers[nar_stage - 1](y_dec).permute(
                0, 2, 1
            )
            # loss
            total_length = (y_lens).sum().type(torch.float32)
            total_loss += (
                F.cross_entropy(
                    logits,
                    targets,
                    ignore_index=NUM_AUDIO_TOKENS,
                    reduction=reduction,
                )
                * (total_length / (total_length - prefix_len * x.shape[0]))
            )
            metrics["NarTop10Accuracy"] = (
                self.nar_accuracy_metric(
                    F.pad(
                        logits.detach(),
                        (0, 0, 0, 1, 0, 0),
                        value=logits.min().cpu().item(),
                    ),
                    targets,
                ).item()
                * total_length
            )
            metrics[f"NarTop10Accuracy_{nar_stage}"] = (
                self.nar_accuracy_metric(
                    F.pad(
                        logits.detach(),
                        (0, 0, 0, 1, 0, 0),
                        value=logits.min().cpu().item(),
                    ),
                    targets,
                ).item()
                * total_length
            )
            # metrics[f"frames_{nar_stage}"] = (y_lens).sum().item()


        #             info["frames"] = (audio_features_lens).sum().item()
        # info["utterances"] = x.size(0)

        if train_stage == 0:
            total_loss = total_loss / 2.0

        return ((x, codes), total_loss, metrics)

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        enroll_x_lens: Union[torch.Tensor, None] = None,
        top_k: int = -100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, 8).
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
          Return the predicted audio code matrix and cross-entropy loss.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)
        # NOTE: x has been padded in TextTokenCollater
        x_mask = make_pad_mask(x_lens).to(x.device)

        prompts = y
        prefix_len = y.shape[1]

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        y = prompts[..., 0]
        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=NUM_AUDIO_TOKENS + 1)

        while True:
            y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)

            tgt_mask = torch.triu(
                torch.ones(
                    y.shape[1], y.shape[1], device=y.device, dtype=torch.bool
                ),
                diagonal=1,
            )

            y_dec, _ = self.ar_decoder(
                (y_pos, None),
                x,
                tgt_mask=tgt_mask,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )
            logits = self.ar_predict_layer(y_dec[:, -1])
            samples = topk_sampling(
                logits, top_k=top_k, top_p=1.0, temperature=temperature
            )

            if (
                torch.argmax(logits, dim=-1)[0] == NUM_AUDIO_TOKENS
                or samples[0, 0] == NUM_AUDIO_TOKENS
                or (y.shape[1] - prefix_len) > x_lens.max() * 16
            ):
                if prompts.shape[1] == y.shape[1]:
                    raise SyntaxError(
                        "well trained model shouldn't reach here."
                    )

                print(f"VALL-F EOS [{prefix_len} -> {y.shape[1]}]")
                break

            y = torch.concat([y, samples], dim=1)

        codes = [y[:, prefix_len + int(self.ar_audio_prepend_bos) :]]
        if self.num_quantizers == 1:
            return torch.stack(codes, dim=-1)

        # Non-AR Decoders
        y_emb = self.nar_audio_embeddings[0](
            y[:, int(self.ar_audio_prepend_bos) :]
        )
        if self.prefix_mode in [2, 4]:  # Exclude enrolled_phonemes
            enrolled_len = enroll_x_lens.max().item()
            # SOS + Synthesis Text + EOS
            text = torch.concat(
                [
                    text[:, :1],
                    text[:, enrolled_len - 1 :],
                ],
                dim=1,
            )
            assert text.shape[0] == 1

        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        if self.prefix_mode != 0:
            for j in range(1, self.num_quantizers):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](
                    prompts[..., j]
                )

        for i, (predict_layer, embedding_layer) in enumerate(
            zip(
                self.nar_predict_layers,
                self.nar_audio_embeddings[1:],
            )
        ):
            y_pos = self.nar_audio_prenet(y_emb)
            y_pos = self.nar_audio_position(y_pos)
            y_dec, _ = self.nar_decoder(
                (y_pos, self.nar_stage_embeddings[i].weight),
                x,
                tgt_mask=None,
                memory_mask=None,
                memory_key_padding_mask=None,
            )
            logits = predict_layer(y_dec[:, prefix_len:])
            samples = torch.argmax(logits, dim=-1)
            codes.append(samples)
            # Formula (4) (5)
            if i < 6:
                if self.prefix_mode == 0:
                    y_emb[:, :prefix_len] += embedding_layer(
                        prompts[..., i + 1]
                    )
                y_emb[:, prefix_len:] += embedding_layer(samples)

        assert len(codes) == self.num_quantizers
        return torch.stack(codes, dim=-1)

    def visualize(
        self,
        predicts: Tuple[torch.Tensor],
        batch: Dict[str, Union[List, torch.Tensor]],
        output_dir: str,
        limit: int = 4,
    ) -> None:
        visualize(predicts, batch, output_dir, limit=limit)


class VALLE(VALLF):

    
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        prefix_mode: int = 0,
        share_embedding: bool = True,
        nar_scale_factor: float = 1.0,
        only_autoregressive: bool = False,
        is_pretrain: bool = False,
        pret_mode: int = 0, 
        **kwargs,
    ):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super(VALLE, self).__init__(
            d_model,
            nhead,
            num_layers,
            norm_first=norm_first,
            add_prenet=add_prenet,
            decoder_cls=TransformerEncoder,
            decoder_layer_cls=TransformerEncoderLayer,
            prefix_mode=prefix_mode,
            share_embedding=share_embedding,
            nar_scale_factor=nar_scale_factor,
            only_autoregressive=only_autoregressive,
            is_pretrain=is_pretrain,
            pret_mode=pret_mode,
            **kwargs,
        )
        print(f"self.ar_text_prenet : {self.ar_text_prenet}")
        print(f"add_prenet：{add_prenet}")

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: Union[torch.Tensor, PromptedFeatures],
        y_lens: Union[torch.Tensor, PromptedFeatures],
        reduction: str = "sum",
        train_stage: int = 0,
        maskd_indices_batch: List =[],
        **kwargs,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
          x: if tts-> x is phoneme else: semantic tokens
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y: if pretrain and ac finetune -> y is semantic tokens else: acoustic tokens
            A 3-D tensor of shape (N, T, 8).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          train_stage:
            0: AR & NAR modules, 1: AR modules, 2: NAR modules
        Returns:
          Return the predicted audio code matrix, cross-entropy loss and Top-10 accuracy.
        """
        # print(f"x:{x}")

        # print(f"x.shape:{x.shape}")
        # print(f"y.shape:{y.shape}")
        if self.semantic_num_quantizers==1:
            assert x.ndim == 2, x.shape
            assert x_lens.ndim == 1, x_lens.shape
        elif self.semantic_num_quantizers==2:  # text is semantic tfnet
            assert x.ndim == 3, x.shape
            assert x_lens.ndim == 1, x_lens.shape

        y_prompts_codes = None
        if isinstance(y, PromptedFeatures): # 没用这个
            y_prompts_codes, y = y.data
            prompts_len, y_lens = y_lens.data
            assert prompts_len.min() == prompts_len.max()
            assert self.prefix_mode == 4
            y_prompts_codes = y_prompts_codes.type(torch.int64)

        assert y.ndim == 3, y.shape
        assert y_lens.ndim == 1, y_lens.shape

        # NOTE: x has been padded in TextTokenCollater
        x_mask = make_pad_mask(x_lens).to(x.device)    
        y_mask = make_pad_mask(y_lens).to(y.device)
        y_mask_int = y_mask.type(torch.int64)

        text = x
        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))

        # add eos token and all masked token changed to eos tokens
        if self.only_autoregressive:   # no matter target token is acoustic or semantic token, eos_is is 1024
            y, targets = self.pad_y_eos(   ## attention BOS 
                codes, y_mask_int, eos_id=NUM_AUDIO_TOKENS
            )
        else:
            y, targets = self.pad_y_eos(
                codes[...,0], y_mask_int, eos_id=NUM_AUDIO_TOKENS
            )

        x_len = x_lens.max()

        metrics = {}
        total_loss = 0.0
        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)

        if self.ar_audio_prepend_bos: # whether add bos token in y. used in ac and pretrain, because no y prompt we need a bos token. 
            ar_xy_padding_mask = torch.concat(
                [x_mask, F.pad(y_mask, (1, 0), value=False)], dim=1
            )
        else:
            ar_xy_padding_mask = xy_padding_mask
        # AR Decoder
        if train_stage in [0, 1]:
            if self.semantic_num_quantizers==1:  # input is phoneme or hubert token
                x = self.ar_text_embedding(text) 
            elif self.semantic_num_quantizers==2: # input is tfnet semantic tokens
                x = rearrange(x, 'b ... -> b (...)') # b x T x 2-> bx (T x 2)
                # 有待商榷
                offsets_x = (NUM_SEMANTIC_TOKENS_TFNET+1) * torch.arange(self.semantic_num_quantizers, device = x.device) # [0, 501, 1002, ...]
                offsets_x = repeat(offsets_x, 'q -> 1 (n q)', n = ceil_div(x.shape[-1], self.semantic_num_quantizers))  # n==T
                offsets_x = offsets_x[:, :x.shape[-1]]
                x = x + offsets_x
                x = self.ar_text_embedding(x)
                x = rearrange(x, 'b (t q) c -> b t (q c)', q = self.semantic_num_quantizers)
            x = self.ar_text_prenet(x)
            x = self.ar_text_position(x)
            
            y_len = y_lens.max() + int(self.ar_audio_prepend_bos)

            x_attn_mask = F.pad(
                torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(
                    torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                    diagonal=1,
                ),
                (x_len, 0),
                value=False,
            )
            xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
            # merge key padding and attention masks
            bsz, src_len = x.shape[0], x_len + y_len
            _xy_padding_mask = (
                ar_xy_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(bsz * self.num_heads, 1, src_len)
            )
            # _xy_padding_mask: 400 1 392 xy_attn_mask: 400 392 392 -> 400 392 392
            xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
            new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
            new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
            xy_attn_mask = new_attn_mask

            if self.only_autoregressive is True:
                y = rearrange(y, 'b ... -> b (...)') # b x T x 16-> bx (T x 16)
                offsets = (NUM_AUDIO_TOKENS+1+int(self.ar_audio_prepend_bos)) * torch.arange(self.num_quantizers, device = y.device) # [0, 501, 1002, ...]
                offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(y.shape[-1], self.num_quantizers))  # n==T
                offsets = offsets[:, :y.shape[-1]]
                y = y + offsets

                y_emb = self.ar_audio_embedding(y)
                y_emb = rearrange(y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers)
            else:
                y_emb = self.ar_audio_embedding(y)

            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1) #bsz * t * embedding_size

            xy_dec, _ = self.ar_decoder(
                (xy_pos, None),
                mask=xy_attn_mask,
                # src_key_padding_mask=xy_padding_mask,
                # is_causal=True,
            )
            xy_dec = xy_dec[:, x_len:]

            logits_list = []

            if self.only_autoregressive is True:
                xy_dec = rearrange(xy_dec, 'b t (q d) -> b t q d', q = self.num_quantizers)
                # 当时用einsum 有bug， loss会高一些
                # logits = einsum('q c d, b t q d -> b t q c', self.logit_weights, xy_dec)
                # # logits = rearrange(logits, 'b t q c -> b (t q) c')
                # # logits = logits.permute(0,3,1,2)
                # logits = logits.permute(0,3,1,2)
                if self.shared_linear is False:
                    for j in range(self.num_quantizers):
                        logits = self.ar_predict_layers[j](xy_dec[:, :, j, :]).permute(0, 2, 1) # b t d -> b t num_classes -> b num_classes t 
                        logits_list.append(logits)

                    logits = torch.stack(logits_list, dim=-1) # b num_classes t -> b num_classes t num_quantizers
                
                else:
                    logits = self.ar_predict_layer(xy_dec).permute(0, 3, 1, 2) # b t q d -> b num_class t q  

            else:
                logits = self.ar_predict_layer(xy_dec).permute(0, 2, 1)

            if maskd_indices_batch.numel() != 0: # only compute masked loss
                maskd_indices_batch = maskd_indices_batch.unsqueeze(-1).expand_as(targets)
                maskd_targets = targets * maskd_indices_batch.to(targets.dtype)  + ((1- maskd_indices_batch).to(targets.dtype)  * IGNORE_TOKENS)

                total_loss = F.cross_entropy(logits, maskd_targets, reduction=reduction, ignore_index=IGNORE_TOKENS)

            else:
                total_loss = F.cross_entropy(logits, targets, reduction=reduction)

            # total_loss = F.cross_entropy(logits, targets[:,:,0,:], reduction=reduction)
            if self.only_autoregressive is True:
                for j in range(self.num_quantizers):
                    metrics[f"ArTop10Accuracy_{j}"] = self.ar_accuracy_metric(
                    logits.detach()[...,j], targets[...,j]
                    ).item() * y_lens.sum().type(torch.float32)

                    metrics[f"ArTop10Accuracy_top1_{j}"] = self.ar_accuracy_metric_top1(
                    logits.detach()[...,j], targets[...,j]
                    ).item() * y_lens.sum().type(torch.float32)
 
                    if maskd_indices_batch.numel() != 0:
                        metrics[f"ArTop10Accuracyignore_{j}"] = self.ar_accuracy_metric_ignore(
                        logits.detach()[...,j], maskd_targets[...,j]
                        ).item() * y_lens.sum().type(torch.float32)

                        metrics[f"ArTop10Accuracyignore_top1_{j}"] = self.ar_accuracy_metric_top1_ignore(
                        logits.detach()[...,j], maskd_targets[...,j]
                        ).item() * y_lens.sum().type(torch.float32)        

            metrics["ArTop10Accuracy"] = self.ar_accuracy_metric(  # it maybe casuse error if batch_size if large.
            logits.detach(), targets
            ).item() * y_lens.sum().type(torch.float32)

            metrics["ArTop10Accuracy_top1"] = self.ar_accuracy_metric_top1(
            logits.detach(), targets
            ).item() * y_lens.sum().type(torch.float32)

            if maskd_indices_batch.numel() != 0:
                metrics["ArTop10Accuracyignore"] = self.ar_accuracy_metric_ignore(
                logits.detach(), maskd_targets
                ).item() * y_lens.sum().type(torch.float32)

                metrics["ArTop10Accuracyignore_top1"] = self.ar_accuracy_metric_top1_ignore(
                logits.detach(), maskd_targets
                ).item() * y_lens.sum().type(torch.float32)

        if self.num_quantizers == 1:
            return ((x, codes), total_loss, metrics)

        # 可以不用看了 直接看该段函数的最后一行
        # Non-AR Decoders
        if self.ar_audio_prepend_bos:
            y = y[:, 1:]
        if train_stage in [0, 2]:
            num_nar_layers = self.num_quantizers - 1
            nar_stage = self.rng.choices(
                [_k for _k in range(1, self.num_quantizers)],
                weights=[1.0 / num_nar_layers] * num_nar_layers,
                k=1,
            )[0]

            x = self.nar_text_embedding(text)
            x = self.nar_text_prenet(x)
            x = self.nar_text_position(x)

            y_emb, prefix_len = self._prepare_prompts(
                y, y_lens, codes, nar_stage, y_prompts_codes
            )

            y_len = y_lens.max()
            targets = codes[..., nar_stage] + NUM_AUDIO_TOKENS * y_mask_int
            if self.prefix_mode in [2, 4]:
                xy_padding_mask = torch.concat(
                    [
                        x_mask,
                        F.pad(y_mask, (y_emb.shape[1] - y_len, 0), value=False),
                    ],
                    dim=1,
                )
            elif self.prefix_mode == 1:
                targets = targets[:, prefix_len:]

            y_pos = self.nar_audio_prenet(y_emb)
            y_pos = self.nar_audio_position(y_pos)
            xy_pos = torch.concat([x, y_pos], dim=1)
            xy_dec, _ = self.nar_decoder(
                (xy_pos, self.nar_stage_embeddings[nar_stage - 1].weight),
                src_key_padding_mask=xy_padding_mask,
                # is_causal=False,
            )
            xy_dec = xy_dec[:, x_lens.max() + prefix_len :]
            if self.prefix_mode == 4:
                prefix_len = 0  # reset for Top10Accuracy metric
            logits = self.nar_predict_layers[nar_stage - 1](xy_dec).permute(
                0, 2, 1
            )

            # loss
            total_length = (y_lens).sum().type(torch.float32)
            total_loss += (
                F.cross_entropy(
                    logits,
                    targets,
                    ignore_index=NUM_AUDIO_TOKENS,
                    reduction=reduction,
                )
                * (total_length / (total_length - prefix_len * x.shape[0]))
            )
            metrics["NarTop10Accuracy"] = (
                self.nar_accuracy_metric(
                    F.pad(
                        logits.detach(),
                        (0, 0, 0, 1, 0, 0),
                        value=logits.min().cpu().item(),
                    ),
                    targets,
                ).item()
                * total_length
            )
            metrics["NarTop10Accuracytop1"] = (
                self.nar_accuracy_metric_top1(
                    F.pad(
                        logits.detach(),
                        (0, 0, 0, 1, 0, 0),
                        value=logits.min().cpu().item(),
                    ),
                    targets,
                ).item()
                * total_length
            )
            metrics[f"NarTop10Accuracy_top1_{nar_stage}"] = (
                self.nar_accuracy_metric_top1(
                    F.pad(
                        logits.detach(),
                        (0, 0, 0, 1, 0, 0),
                        value=logits.min().cpu().item(),
                    ),
                    targets,
                ).item()
                * total_length
            )
            metrics[f"NarTop10Accuracy_{nar_stage}"] = (
                self.nar_accuracy_metric(
                    F.pad(
                        logits.detach(),
                        (0, 0, 0, 1, 0, 0),
                        value=logits.min().cpu().item(),
                    ),
                    targets,
                ).item()
                * total_length
            )
            metrics[f"frames_{nar_stage}"] = (y_lens).sum().item()
        if train_stage == 0:
            total_loss = total_loss / 2.0

        return ((x, codes), total_loss, metrics)

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        enroll_x_lens: torch.Tensor,
        top_k: int = -100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, 8).
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
          Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        # NOTE: x has been padded in TextTokenCollater
        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        text_len = x_lens.max()
        prompts = y
        prefix_len = y.shape[1]

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        y = prompts[..., 0]
        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=NUM_AUDIO_TOKENS + 1)

        x_len = x_lens.max()
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)

        while True:
            y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)

            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(
                    torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1
                ),
                (x_len, 0),
                value=False,
            )
            xy_attn_mask = torch.concat(
                [x_attn_mask_pad, y_attn_mask], dim=0
            ).to(y.device)

            xy_dec, _ = self.ar_decoder(
                (xy_pos, None),
                mask=xy_attn_mask,
            )
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = topk_sampling(
                logits, top_k=top_k, top_p=1.0, temperature=temperature
            )

            if (
                torch.argmax(logits, dim=-1)[0] == NUM_AUDIO_TOKENS
                or samples[0, 0] == NUM_AUDIO_TOKENS
                or (y.shape[1] - prompts.shape[1]) > x_lens.max() * 16
            ):
                if prompts.shape[1] == y.shape[1]:
                    raise SyntaxError(
                        "well trained model shouldn't reach here."
                    )

                print(f"VALL-E EOS [{prompts.shape[1]} -> {y.shape[1]}]")
                break

            y = torch.concat([y, samples], dim=1)

        codes = [y[:, prefix_len + int(self.ar_audio_prepend_bos) :]]
        if self.num_quantizers == 1:
            return torch.stack(codes, dim=-1)

        # Non-AR Decoders
        y_emb = self.nar_audio_embeddings[0](
            y[:, int(self.ar_audio_prepend_bos) :]
        )

        if self.prefix_mode in [2, 4]:  # Exclude enrolled_phonemes
            enrolled_len = enroll_x_lens.max().item()
            # SOS + Synthesis Text + EOS
            text = torch.concat(
                [
                    text[:, :1],
                    text[:, enrolled_len - 1 :],
                ],
                dim=1,
            )
            text_len = text_len - (enrolled_len - 2)
            assert text.shape[0] == 1

        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        if self.prefix_mode == 0:
            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < self.num_quantizers - 2:
                    y_emb[:, :prefix_len] += embedding_layer(
                        prompts[..., i + 1]
                    )
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        else:
            for j in range(1, self.num_quantizers):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](
                    prompts[..., j]
                )

            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < self.num_quantizers - 2:
                    y_emb[:, prefix_len:] += embedding_layer(samples)

        assert len(codes) == self.num_quantizers
        return torch.stack(codes, dim=-1)
    def inference_only_ar(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        enroll_x_lens: torch.Tensor,
        top_k: int = -100,
        top_p: float = 1.0, 
        temperature: float = 1.0,
        mode=0,
        task_id=0, # task_id=0, tts, 1-> semantic token convert, 2-> generative model-input_sem
        use_silence_token=False
    ) -> torch.Tensor:
        """
        Args:
            x:
            A 2-D tensor of shape (1, S).
            x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
            y:
            A 3-D tensor of shape (1, T, 8).
            top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
            temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
            Return the predicted audio code matrix.
        """
        if self.semantic_num_quantizers==1:
            assert x.ndim == 2, x.shape
            assert x_lens.ndim == 1, x_lens.shape
        elif self.semantic_num_quantizers==2:  # text is semantic tfnet
            assert x.ndim == 3, x.shape
            assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape

        assert torch.all(x_lens > 0)

        text = x

        if self.semantic_num_quantizers==1:
            x = self.ar_text_embedding(text)
        elif self.semantic_num_quantizers==2:
            x = rearrange(x, 'b ... -> b (...)') # b x T x 2-> bx (T x 2)
            offsets_x = (NUM_SEMANTIC_TOKENS_TFNET+1) * torch.arange(self.semantic_num_quantizers, device = x.device) # [0, 501, 1002, ...]
            offsets_x = repeat(offsets_x, 'q -> 1 (n q)', n = ceil_div(x.shape[-1], self.semantic_num_quantizers))  # n==T
            offsets_x = offsets_x[:, :x.shape[-1]]
            x = x + offsets_x
            x = self.ar_text_embedding(x)
            x = rearrange(x, 'b (t q) c -> b t (q c)', q = self.semantic_num_quantizers)

        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        print(x_lens)
        text_len = x_lens.max()
        print(f"text_len:{text_len}")
        prompts = y

        prefix_len = y.shape[1]
        print(f"prefix_len:{prefix_len}")

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        # print(prompts.shape)
        # quit()
        # y = prompts[..., 0]
        

        # here 可能有问题
        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=NUM_AUDIO_TOKENS + 1)
        
        y_bef = y.clone()
        # print(y_bef)
        x_len = x_lens.max()
        if task_id==0: # tts or generative model
            generate_max_lens =  x_lens.max() *16
        elif task_id==1: # the same hz
            generate_max_lens = x_lens.max() +2
        elif task_id==2: # 25hz-> 50hz
            generate_max_lens = x_lens.max() *5
        x_mask = make_pad_mask(x_lens).to(x.device)
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)

        end=0
        bsz = x.shape[0]
        end_np = [0 for i in range(bsz)] # 表示一个batch 样本的终止
        end_loc = [0 for i in range(bsz)] # 截止

        # y_bef = y_bef[:, :100,]
        while True:
            y_len = y_bef.shape[1]
            y = y_bef.clone()
            y = rearrange(y, 'b ... -> b (...)')
            offsets = (NUM_AUDIO_TOKENS+1+int(self.ar_audio_prepend_bos)) * torch.arange(self.num_quantizers, device = y.device)
            offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(y.shape[-1], self.num_quantizers))
            offsets = offsets[:, :y.shape[-1]]
            y = y + offsets

            y_emb = self.ar_audio_embedding(y)
            y_emb = rearrange(y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers)

            # y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)
            ar_xy_padding_mask = F.pad(
                x_mask,
                (0, y_len),
                value=False,
            )
            # y_len = y.shape[1]
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(
                    torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1
                ),
                (x_len, 0),
                value=False,
            )
            xy_attn_mask = torch.concat(
                [x_attn_mask_pad, y_attn_mask], dim=0
            ).to(y.device)
            
            src_len = ar_xy_padding_mask.shape[1]
            _xy_padding_mask = (
                ar_xy_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(bsz * self.num_heads, 1, src_len)
            )

            xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
            new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
            new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
            xy_attn_mask = new_attn_mask

            
            xy_dec, _ = self.ar_decoder(
                (xy_pos, None),
                mask=xy_attn_mask,
            )
            # print(f"xy_dec shape {xy_dec.shape}")
            xy_dec = xy_dec[:,-1,:]

            xy_dec = xy_dec.unsqueeze(dim=1)
            xy_dec = rearrange(xy_dec, 'b t (q d) -> b t q d', q = self.num_quantizers)

            if self.shared_linear is False:
                logits_list = []
                for j in range(self.num_quantizers):
                    logits = self.ar_predict_layers[j](xy_dec[:, :, j, :]).permute(0, 2, 1)
                    logits_list.append(logits)

                logits = torch.stack(logits_list, dim=-1)
            else:
                logits = self.ar_predict_layer(xy_dec).permute(0, 3, 1, 2) # b t q d -> b t q num_class 

            samples = []

            logits = rearrange(logits, 'b d t q -> (b q) d t')
            logits = logits.squeeze(-1)

            if mode==0:
                samples = topk_sampling(
                    logits, top_k=top_k, top_p=top_p, temperature=temperature
                )
            elif mode==1:
                samples = arg_max_sampling(
                    logits
                )

            samples = samples.squeeze(1)
            samples = rearrange(samples, '(b q) -> b q', q= self.num_quantizers)

            SILEN_TOKENS = [193]
            # print(f"use_silence_token:{use_silence_token}")
            if use_silence_token is False:
                SILEN_TOKENS = []
            for i, one_batch_samles in enumerate(samples):
                for sample in one_batch_samles:
                    if (sample == NUM_AUDIO_TOKENS or (sample in SILEN_TOKENS) )or (task_id==1 and sample.item()>500):   # 注意要不要加个argmax
                        if sample == NUM_AUDIO_TOKENS or (sample in SILEN_TOKENS):
                            print(f"第{i}个序列meet end_token: {sample}")
                        else:
                            print(f"第{i}个序列meet oon")
                        sample = NUM_AUDIO_TOKENS
                        if end_np[i]==0:
                            end_np[i]=1
                            end_loc[i]=y_bef.shape[1]-prefix_len # 第多少个loc开始停的
                            print(f"第{i}个序列end， 终止长度为{y_bef.shape[1]+1-prefix_len}（包含终止符）")
                            break
            y_bef = torch.concat([y_bef, samples.unsqueeze(1)], dim=1)

            if (y_bef.shape[1] - prompts.shape[1]) > generate_max_lens:
                print(f"长度过长， 终止")
                break
            if all(end_np) ==1:
                print(f"所有序列终止")
                break

                # y_bef 1 T 1
                # print(f"len is {y_bef.shape[1] - prompts.shape[1]})")
                # if (
                # torch.argmax(logits_part, dim=-1)[0] == NUM_AUDIO_TOKENS
                # or sample == NUM_AUDIO_TOKENS
                # or (y_bef.shape[1] - prompts.shape[1]) > generate_max_lens
                # ):
                #     if (torch.argmax(logits_part, dim=-1)[0] == NUM_AUDIO_TOKENS or sample == NUM_AUDIO_TOKENS):
                #         print("now it is eos token")
                #     else:
                #         print(f"now is max_token nums")

                #     if prompts.shape[1] == y_bef.shape[1]:
                #         raise SyntaxError(
                #             "well trained model shouldn't reach here."
                #         )
                #     end=1
                #     print(f"VALL-E EOS [{prompts.shape[1]} -> {y_bef.shape[1]}]")
                #     break
                # if (
                # torch.argmax(logits_part, dim=-1)[0] == NUM_AUDIO_TOKENS
                # or sample == NUM_AUDIO_TOKENS
                # or (y_bef.shape[1] - prompts.shape[1]) > x_lens.max() * 16
                # ):
                #     if prompts.shape[1] == y_bef.shape[1]:
                #         raise SyntaxError(
                #             "well trained model shouldn't reach here."
                #         )
                #     end=1
                #     print(f"VALL-E EOS [{prompts.shape[1]} -> {y_bef.shape[1]}]")
                #     break
            # quit()

        codes = y_bef[:, prefix_len + int(self.ar_audio_prepend_bos) :]

        print(end_loc[i]-1)
        batch_codes = [codes[i, :int(end_loc[i])-1, :].unsqueeze(0) for i in range(bsz)]

        # if self.num_quantizers == 1: 
        #     return torch.stack(codes, dim=-1)
        # print(codes[0].shape)
        # print(f"batch_codes:{batch_codes}")
        return batch_codes

    def inference_only_ar_batch(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        enroll_x_lens: torch.Tensor,
        top_k: int = -100,
        temperature: float = 1.0,
        mode=0
    ) -> torch.Tensor:
        """
        Args:
            x:
            A 2-D tensor of shape (1, S).  (B, S)
            x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
            y:
            A 3-D tensor of shape (1, T, 8). (B, T, 16)
            top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
            temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
            Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        # assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        # assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        # NOTE: x has been padded in TextTokenCollater
        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        text_len = x_lens.max()
        prompts = y

        prefix_len = y.shape[1]
        print(f"prefix_len:{prefix_len}")

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        # print(prompts.shape)
        # quit()
        # y = prompts[..., 0]
        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=NUM_AUDIO_TOKENS + 1)
        
        y_bef = y.clone()

        x_len_max = x_lens.max()

        x_mask = make_pad_mask(x_lens).to(x.device)
        x_attn_mask = torch.zeros((x_len_max, x_len_max), dtype=torch.bool)
        # y_attn_mask = torch.zeros((prefix_len, prefix_len), dtype=torch.bool)
        end=0
        bsz = x.shape[0]
        end_np = [0 for i in range(bsz)] # 表示每个单batch的终止
        end_loc = [0 for i in range(bsz)]
        while True:
            y_len = y_bef.shape[1]
            y = y_bef.clone()

            y = rearrange(y, 'b ... -> b (...)')
            offsets = (NUM_AUDIO_TOKENS+1+int(self.ar_audio_prepend_bos)) * torch.arange(self.num_quantizers, device = y.device)
            offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(y.shape[-1], self.num_quantizers))
            offsets = offsets[:, :y.shape[-1]]
            y = y + offsets

            y_emb = self.ar_audio_embedding(y)
            y_emb = rearrange(y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers)

            # y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)
            ar_xy_padding_mask = F.pad(
                x_mask,
                (0, y_len),
                value=False,
            )
            # y_len = y.shape[1]
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(
                    torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1
                ),
                (x_len_max, 0),
                value=False,
            )
            xy_attn_mask = torch.concat(
                [x_attn_mask_pad, y_attn_mask], dim=0
            ).to(y.device)

            src_len = ar_xy_padding_mask.shape[1]
            _xy_padding_mask = (
                ar_xy_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(bsz * self.num_heads, 1, src_len)
            )

            xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
            new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
            new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
            xy_attn_mask = new_attn_mask

            xy_dec, _ = self.ar_decoder(
                (xy_pos, None),
                mask=xy_attn_mask,
            )

            # print(f"xy_dec shape {xy_dec.shape}")
            xy_dec = xy_dec[:,-1,:]
            xy_dec = xy_dec.unsqueeze(1)
            # xy_dec = xy_dec.unsqueeze(dim=1)
            xy_dec = rearrange(xy_dec, 'b t (q d) -> b t q d', q = self.num_quantizers)
            
            if self.shared_linear is False:
                logits_list = []
                for j in range(self.num_quantizers):
                    logits = self.ar_predict_layers[j](xy_dec[:, :, j, :]).permute(0, 2, 1)
                    logits_list.append(logits)

                logits = torch.stack(logits_list, dim=-1)
            else:
                logits = self.ar_predict_layer(xy_dec).permute(0, 3, 1, 2) # b t q d -> b d(p of each class) t q
            samples = []
            # print(logits.shape)
            # quit()
            logits = rearrange(logits, 'b d t q -> (b q) d t')
            # for j in range(self.num_quantizers):
                # logits_part = logits[...,j]
            logits = logits.squeeze(-1)
            if mode==0:
                samples = topk_sampling(
                    logits, top_k=top_k, top_p=1.0, temperature=temperature
                )
            elif mode==1:
                samples = arg_max_sampling(
                    logits
                )
            samples = samples.squeeze(1)
            samples = rearrange(samples, '(b q) -> b q', q= self.num_quantizers)

            for i, one_batch_samles in enumerate(samples):
                for sample in one_batch_samles:
                    if sample == NUM_AUDIO_TOKENS or (task_id==1 and sample.item()>500):   # 注意要不要加个argmax
                        if sample == NUM_AUDIO_TOKENS:
                            print(f"第{i}个序列meet end_token")
                        else:
                            print(f"第{i}个序列meet oon")
                        if end_np[i]==0:
                            end_np[i]=1
                            end_loc[i]=y_bef.shape[1]-prefix_len # 第多少个loc开始停的
                            print(f"第{i}个序列end， 终止长度为{y_bef.shape[1]+1-prefix_len}（包含终止符）")
                            break
            y_bef = torch.concat([y_bef, samples.unsqueeze(1)], dim=1)
            if (y_bef.shape[1] - prompts.shape[1]) > x_lens.max() * 15:
                print(f"长度过长， 终止")
                break
            if all(end_np) ==1:
                print(f"所有序列终止")
                break
            #y_bef 1 T 1
            # print(f"len is {y_bef.shape[1] - prompts.shape[1]})")
            # if (
            # torch.argmax(logits_part, dim=-1)[0] == NUM_AUDIO_TOKENS
            # or sample == NUM_AUDIO_TOKENS
            # or (y_bef.shape[1] - prompts.shape[1]) > (x_lens.max()+2)
            # ):
            #     if prompts.shape[1] == y_bef.shape[1]:
            #         raise SyntaxError(
            #             "well trained model shouldn't reach here."
            #         )
            #     end=1
            #     print(f"VALL-E EOS [{prompts.shape[1]} -> {y_bef.shape[1]}]")
            #     break


        codes = y_bef[:, prefix_len + int(self.ar_audio_prepend_bos) :]

        print(end_loc[i]-1)

        
        batch_codes = [codes[i, :int(end_loc[i])-1, :].unsqueeze(0) for i in range(bsz)]

        if self.num_quantizers == 1:
            return torch.stack(codes, dim=-1)
        # print(codes[0].shape)
        return batch_codes



    def inference_only_ar_beam_search(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        enroll_x_lens: torch.Tensor,
        top_k: int = -100,
        temperature: float = 1.0,
        mode=0
    ) -> torch.Tensor:
        """
        Args:
            x:
            A 2-D tensor of shape (1, S).
            x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
            y:
            A 3-D tensor of shape (1, T, 8).
            top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
            temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
            Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        # NOTE: x has been padded in TextTokenCollater
        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        text_len = x_lens.max()
        prompts = y

        prefix_len = y.shape[1]
        print(f"prefix_len:{prefix_len}")

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        # print(prompts.shape)
        # quit()
        # y = prompts[..., 0]
        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=NUM_AUDIO_TOKENS + 1)
        
        y_bef = y.clone()

        x_len = x_lens.max()
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)


        beam_size = 5
        beam_probs = torch.zeros((1, 1), device=x.device)  
        beam_candidates = [y_bef.clone()] 
        end=0

        while True:
            new_beam_probs = torch.zeros((1, beam_size * len(beam_candidates)), device=x.device)  
            new_beam_candidates = [] 
            
            # print(f"当前candidate is {beam_candidates}")

            for i, candidate in enumerate(beam_candidates):
                
                y_len = candidate.shape[1]
                y = candidate.clone()

                y = rearrange(y, 'b ... -> b (...)')
                offsets = (NUM_AUDIO_TOKENS+1+int(self.ar_audio_prepend_bos)) * torch.arange(self.num_quantizers, device = y.device)
                offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(y.shape[-1], self.num_quantizers))
                offsets = offsets[:, :y.shape[-1]]
                y = y + offsets

                y_emb = self.ar_audio_embedding(y)
                y_emb = rearrange(y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers)

                # y_emb = self.ar_audio_embedding(y)
                y_emb = self.ar_audio_prenet(y_emb)
                y_pos = self.ar_audio_position(y_emb)
                xy_pos = torch.concat([x, y_pos], dim=1)

                # y_len = y.shape[1]
                x_attn_mask_pad = F.pad(
                    x_attn_mask,
                    (0, y_len),
                    value=True,
                )
                y_attn_mask = F.pad(
                    torch.triu(
                        torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1
                    ),
                    (x_len, 0),
                    value=False,
                )
                xy_attn_mask = torch.concat(
                    [x_attn_mask_pad, y_attn_mask], dim=0
                ).to(y.device)
                
                xy_dec, _ = self.ar_decoder(
                    (xy_pos, None),
                    mask=xy_attn_mask,
                )
                # print(f"xy_dec shape {xy_dec.shape}")
                xy_dec = xy_dec[:,-1]


                xy_dec = xy_dec.unsqueeze(dim=1)
                xy_dec = rearrange(xy_dec, 'b t (q d) -> b t q d', q = self.num_quantizers)

                if self.shared_linear is False:
                    logits_list = []
                    for j in range(self.num_quantizers):
                        logits = self.ar_predict_layers[j](xy_dec[:, :, j, :]).permute(0, 2, 1)
                        logits_list.append(logits)

                    logits = torch.stack(logits_list, dim=-1)
                else:
                    logits = self.ar_predict_layer(xy_dec).permute(0, 3, 1, 2) # b t q d -> b t q num_class 


                # samples = []
                logits_part = logits[...,0]
                logits_part = logits_part.squeeze(-1)

                log_probs, sample = beam_search(logits_part, beam_size, temperature)

                new_beam_probs[:, i * beam_size:(i + 1) * beam_size] = beam_probs[0, i] + log_probs  
                for s in sample.squeeze(0):
                    if (
                     (candidate.shape[1] - prompts.shape[1]) > (x_lens.max()-4)
                    ):
                        if prompts.shape[1] == candidate.shape[1]:
                            raise SyntaxError(
                                "well trained model shouldn't reach here."
                            )
                        end=1
                        print(f"VALL-E EOS [{prompts.shape[1]} -> {candidate.shape[1]}]")
                        break

                    s = s.unsqueeze(0)
                    new_candidate = torch.cat([candidate, s.unsqueeze(0).unsqueeze(0)], dim=1)  
                    new_beam_candidates.append(new_candidate)
                
                
                if end==1:
                    break

            if end==1:
                # print("here")
                # print(beam_candidates)
                # print(beam_probs)
                break
            # print(beam_candidates)
            # print(beam_probs)
            new_beam_probs, new_beam_indices = torch.topk(new_beam_probs.view(-1), beam_size)
            # print(f"step over, {new_beam_probs},{new_beam_indices}")
            beam_candidates = [new_beam_candidates[i] for i in new_beam_indices]
            beam_probs = new_beam_probs.view(1, -1)

                # print(f"len is {y_bef.shape[1] - prompts.shape[1]})")
        best_index = torch.argmax(beam_probs)    
        y_bef = beam_candidates[best_index]


        codes = [y_bef[:, prefix_len + int(self.ar_audio_prepend_bos) :]]
        if self.num_quantizers == 1:
            return torch.stack(codes, dim=-1)
        # print(codes[0].shape)
        return torch.stack(codes, dim=-1)
        
    def inference_as_train(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        top_k: int = -100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            x:
            A 2-D tensor of shape (1, S).
            x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
            y:
            A 3-D tensor of shape (1, T, 8).
            top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
            temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
            Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        # NOTE: x has been padded in TextTokenCollater
        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        text_len = x_lens.max()
        prompts = y
        prefix_len = y.shape[1]

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        y = prompts[..., 0]
        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=NUM_AUDIO_TOKENS + 1)

        x_len = x_lens.max()
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)

        y_emb = self.ar_audio_embedding(y)
        y_emb = self.ar_audio_prenet(y_emb)
        y_pos = self.ar_audio_position(y_emb)
        xy_pos = torch.concat([x, y_pos], dim=1)

        y_len = y.shape[1]
        x_attn_mask_pad = F.pad(
            x_attn_mask,
            (0, y_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1
            ),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = torch.concat(
            [x_attn_mask_pad, y_attn_mask], dim=0
        ).to(y.device)

        # print(xy_pos.shape)

        xy_dec, _ = self.ar_decoder(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        # print(xy_dec.shape)
        # quit()

        xy_dec = xy_dec[:, x_len:]

        logits = self.ar_predict_layer(xy_dec).permute(0, 2, 1) # b,1024,t

        # logits = self.ar_predict_layer(xy_dec[:, -1])
        samples = topk_sampling(
            logits, top_k=top_k, top_p=1.0, temperature=temperature
        )

            # if (
            #     torch.argmax(logits, dim=-1)[0] == NUM_AUDIO_TOKENS
            #     or samples[0, 0] == NUM_AUDIO_TOKENS
            #     or (y.shape[1] - prompts.shape[1]) > x_lens.max() * 16
            # ):
            #     if prompts.shape[1] == y.shape[1]:
            #         raise SyntaxError(
            #             "well trained model shouldn't reach here."
            #         )

            #     print(f"VALL-E EOS [{prompts.shape[1]} -> {y.shape[1]}]")
            #     break

        # y = torch.concat([y, samples], dim=1)

        codes = [y[:, prefix_len + int(self.ar_audio_prepend_bos) :]]
        if self.num_quantizers == 1:
            return torch.stack(codes, dim=-1)


        # Non-AR Decoders
        y_emb = self.nar_audio_embeddings[0](
            y[:, int(self.ar_audio_prepend_bos) :]
        )

        if self.prefix_mode in [2, 4]:  # Exclude enrolled_phonemes
            enrolled_len = enroll_x_lens.max().item()
            # SOS + Synthesis Text + EOS
            text = torch.concat(
                [
                    text[:, :1],
                    text[:, enrolled_len - 1 :],
                ],
                dim=1,
            )
            text_len = text_len - (enrolled_len - 2)
            assert text.shape[0] == 1

        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        if self.prefix_mode == 0:
            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < self.num_quantizers - 2:
                    y_emb[:, :prefix_len] += embedding_layer(
                        prompts[..., i + 1]
                    )
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        else:
            for j in range(1, self.num_quantizers):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](
                    prompts[..., j]
                )

            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < self.num_quantizers - 2:
                    y_emb[:, prefix_len:] += embedding_layer(samples)

        assert len(codes) == self.num_quantizers
        return torch.stack(codes, dim=-1)

    def continual(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
            A 2-D tensor of shape (1, S).
            x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
            y:
            A 3-D tensor of shape (1, T, 8).
        Returns:
            Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)
        assert self.num_quantizers == 8

        # NOTE: x has been padded in TextTokenCollater
        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        text_len = x_lens.max()

        prefix_len = min(int(y.shape[1] * 0.5), 3 * 75)

        # AR Decoder
        prompts = y[:, :prefix_len]

        codes = [y[:, prefix_len:, 0]]
        # Non-AR Decoders
        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        y_emb = self.nar_audio_embeddings[0](y[..., 0])

        if self.prefix_mode == 0:
            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_position(y_emb)
                y_pos = self.nar_audio_prenet(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < 6:
                    y_emb[:, :prefix_len] += embedding_layer(
                        prompts[..., i + 1]
                    )
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        else:
            for j in range(1, 8):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](
                    prompts[..., j]
                )

            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < 6:
                    y_emb[:, prefix_len:] += embedding_layer(samples)

        assert len(codes) == 8
        return torch.stack(codes, dim=-1)

    # https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py
class VALLF_E(VALLF):
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        encoder_num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        prefix_mode: int = 0,
        share_embedding: bool = True,
        nar_scale_factor: float = 1.0,
        only_autoregressive: bool = False,
        is_pretrain: bool = False,
        pret_mode: int = 0, 
        **kwargs,
    ):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super(VALLF_E, self).__init__(
            d_model,
            nhead,
            num_layers,
            norm_first=norm_first,
            add_prenet=add_prenet,
            decoder_cls=nn.TransformerDecoder,
            decoder_layer_cls=TransformerDecoderLayer,
            prefix_mode=prefix_mode,
            share_embedding=share_embedding,
            nar_scale_factor=nar_scale_factor,
            only_autoregressive=only_autoregressive,
            is_pretrain=is_pretrain,
            pret_mode=pret_mode,
            **kwargs,
        )
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=encoder_num_layers,
            norm=LayerNorm(d_model) if norm_first else None,
        )
        print(f"self.ar_text_prenet : {self.ar_text_prenet}")
        print(f"add_prenet：{add_prenet}")
        print(f"self.encoder_layers:{encoder_num_layers}")
        print(f"self.decoder_layers：{num_layers}")

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        # semantic: torch.Tensor,
        # semantic_lens: torch.Tensor,
        y: Union[torch.Tensor, PromptedFeatures],
        y_lens: Union[torch.Tensor, PromptedFeatures],
        maskd_indices_batch: List =[],
        reduction: str = "sum",
        train_stage: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (N, T, 8).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          train_stage:
            0: AR & NAR modules, 1: AR modules, 2: NAR modules
        Returns:
          Return the predicted audio code matrix, cross-entropy loss and Top-10 accuracy.
        """
        # x = semantic
        # x_lens = semantic_lens
        
        if self.semantic_num_quantizers==1:
            assert x.ndim == 2, x.shape
            assert x_lens.ndim == 1, x_lens.shape
        elif self.semantic_num_quantizers==2:  # text is semantic tfnet
            assert x.ndim == 3, x.shape
            assert x_lens.ndim == 1, x_lens.shape

        y_prompts_codes = None
        if isinstance(y, PromptedFeatures):
            y_prompts_codes, y = y.data
            prompts_len, y_lens = y_lens.data
            assert prompts_len.min() == prompts_len.max()
            assert self.prefix_mode == 4
            y_prompts_codes = y_prompts_codes.type(torch.int64)

        assert y.ndim == 3, y.shape
        assert y_lens.ndim == 1, y_lens.shape

        # NOTE: x has been padded in TextTokenCollater
        x_mask = make_pad_mask(x_lens).to(x.device)

        text = x
        x = self.ar_text_embedding(text)
        # x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        bsz = x.shape[0]
        src_len = x_lens.max()
        x_attn_mask = torch.zeros(bsz * self.num_heads, x_lens.max(), x_lens.max(), dtype=x.dtype).to(x.device)
    
        x_padding_mask = (
        x_mask.view(bsz, 1, 1, src_len)
        .expand(-1, self.num_heads, -1, -1)
        .reshape(bsz * self.num_heads, 1, src_len)
        )
        # _xy_padding_mask: 400 1 392 xy_attn_mask: 400 392 392 -> 400 392 392
        final_x_attn_mask = x_attn_mask.logical_or(x_padding_mask)  #only_padding_mask
        new_attn_mask = torch.zeros_like(final_x_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(final_x_attn_mask, float("-inf"))
        final_x_attn_mask = new_attn_mask

        x, _ = self.encoder(
            (x, None),
            mask=final_x_attn_mask,
            # src_key_padding_mask=xy_padding_mask,
            # is_causal=True,
        )
        total_loss, metrics = 0.0, {}
        y_mask = make_pad_mask(y_lens).to(y.device) # no BOS
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))
        # Training
        # AR Decoder
        # y, targets = self.pad_y_eos(
        #     codes[..., 0], y_mask_int, eos_id=NUM_AUDIO_TOKENS
        # )
        if self.only_autoregressive:   # no matter target token is acoustic or semantic token, eos_is is 1024
            y, targets = self.pad_y_eos(   ## attention BOS 
                codes, y_mask_int, eos_id=NUM_AUDIO_TOKENS
            )
        else:
            y, targets = self.pad_y_eos(
                codes[...,0], y_mask_int, eos_id=NUM_AUDIO_TOKENS
            )
        # with open("test_txt_22_40.txt", "w") as f:
        #     f.write("codes[...,0" + str(codes[...,0]))
        #     f.write("y"+ str(y))
        #     f.write("targets:"+str(targets))
        # quit()

        if train_stage in [0, 1]:
            if self.only_autoregressive is True:
                y = rearrange(y, 'b ... -> b (...)') # b x T x 16-> bx (T x 16)
                # 注意这里有BOS 是有问题的
                offsets = (NUM_AUDIO_TOKENS+1+int(self.ar_audio_prepend_bos)) * torch.arange(self.num_quantizers, device = y.device) # [0, 501, 1002, ...]
                offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(y.shape[-1], self.num_quantizers))  # n==T
                offsets = offsets[:, :y.shape[-1]]
                y = y + offsets
                y_emb = self.ar_audio_embedding(y)
                y_emb = rearrange(y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers)
            else:
                y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)

            ar_y_mask = y_mask  # no BOS
            if self.ar_audio_prepend_bos:
                ar_y_mask = F.pad(y_mask, (1, 0), value=False)
            y_len = y_lens.max() + int(self.ar_audio_prepend_bos)  # no BOS
            tgt_mask = torch.triu(
                torch.ones(y_len, y_len, device=y.device, dtype=torch.bool),
                diagonal=1,
            )
            y_dec, _ = self.ar_decoder(   # b x t x 1024
                (y_pos, None),
                x,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=ar_y_mask,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )
            if self.only_autoregressive is True:
                y_dec = rearrange(y_dec, 'b t (q d) -> b t q d', q = self.num_quantizers)
                    # why not directly pass b t q d to nn.linear ? different codec layer has different linear
                    # logits = einsum('q c d, b t q d -> b t q c', self.logit_weights, xy_dec)
                    # # logits = rearrange(logits, 'b t q c -> b (t q) c')
                    # # logits = logits.permute(0,3,1,2)
                    # logits = logits.permute(0,3,1,2)
                logits_list = []
                if self.shared_linear is False:
                    for j in range(self.num_quantizers):
                        logits = self.ar_predict_layers[j](y_dec[:, :, j, :]).permute(0, 2, 1) # b t d -> b t num_classes -> b num_classes t 
                        logits_list.append(logits)

                    logits = torch.stack(logits_list, dim=-1) # b num_classes t -> b num_classes t num_quantizers
                
                else:
                    logits = self.ar_predict_layer(y_dec).permute(0, 3, 1, 2) # b t q d -> b num_class t q  
            else:
                 logits = self.ar_predict_layer(xy_dec).permute(0, 3, 1, 2) # b t q d -> b num_class t q  

            # total_loss = F.cross_entropy(logits, targets, reduction=reduction)

            if maskd_indices_batch.numel() != 0:
                maskd_indices_batch = maskd_indices_batch.unsqueeze(-1).expand_as(targets)
                maskd_targets = targets * maskd_indices_batch.to(targets.dtype)  + ((1- maskd_indices_batch).to(targets.dtype)  * IGNORE_TOKENS)

                # total_loss = F.cross_entropy(logits, targets, reduction=reduction)
                total_loss = F.cross_entropy(logits, maskd_targets, reduction=reduction, ignore_index=IGNORE_TOKENS)

            else:
                # max_indices = torch.argmax(logits, dim=1)
                # max_indices_list = max_indices.tolist()
                # print(max_indices_list)
                total_loss = F.cross_entropy(logits, targets, reduction=reduction)

            if self.only_autoregressive is True:
                for j in range(self.num_quantizers):
                    metrics[f"ArTop10Accuracy_{j}"] = self.ar_accuracy_metric(
                    logits.detach()[...,j], targets[...,j]
                    ).item() * y_lens.sum().type(torch.float32)

                    metrics[f"ArTop10Accuracy_top1_{j}"] = self.ar_accuracy_metric_top1(
                    logits.detach()[...,j], targets[...,j]
                    ).item() * y_lens.sum().type(torch.float32)
 
                    if maskd_indices_batch.numel() != 0: # only masked loss
                        metrics[f"ArTop10Accuracyignore_{j}"] = self.ar_accuracy_metric_ignore(
                        logits.detach()[...,j], maskd_targets[...,j]
                        ).item() * y_lens.sum().type(torch.float32)

                        metrics[f"ArTop10Accuracyignore_top1_{j}"] = self.ar_accuracy_metric_top1_ignore(
                        logits.detach()[...,j], maskd_targets[...,j]
                        ).item() * y_lens.sum().type(torch.float32)        

            metrics["ArTop10Accuracy"] = self.ar_accuracy_metric(  # it maybe casuse error if batch_size if large.
            logits.detach(), targets
            ).item() * y_lens.sum().type(torch.float32)

            metrics["ArTop10Accuracy_top1"] = self.ar_accuracy_metric_top1(
            logits.detach(), targets
            ).item() * y_lens.sum().type(torch.float32)

            if maskd_indices_batch.numel() != 0:
                metrics["ArTop10Accuracyignore"] = self.ar_accuracy_metric_ignore(
                logits.detach(), maskd_targets
                ).item() * y_lens.sum().type(torch.float32)

                metrics["ArTop10Accuracyignore_top1"] = self.ar_accuracy_metric_top1_ignore(
                logits.detach(), maskd_targets
                ).item() * y_lens.sum().type(torch.float32)


        if self.num_quantizers == 1:
            return ((x, codes), total_loss, metrics)

        # # Non-AR Decoders
        # if self.ar_audio_prepend_bos:
        #     y = y[:, 1:]

        # if train_stage in [0, 2]:
        #     num_nar_layers = self.num_quantizers - 1
        #     nar_stage = self.rng.choices(
        #         [_k for _k in range(1, self.num_quantizers)],
        #         weights=[1.0 / num_nar_layers] * num_nar_layers,
        #         k=1,
        #     )[0]

        #     x = self.nar_text_embedding(text)
        #     x = self.nar_text_prenet(x)
        #     x = self.nar_text_position(x)

        #     y_emb, prefix_len = self._prepare_prompts(
        #         y, y_lens, codes, nar_stage, y_prompts_codes
        #     )

        #     y_len = y_lens.max()
        #     targets = codes[..., nar_stage] + NUM_AUDIO_TOKENS * y_mask_int
        #     if self.prefix_mode in [2, 4]:
        #         targets = targets
        #         y_mask = F.pad(y_mask, (y_emb.shape[1] - y_len, 0), value=False)
        #     elif self.prefix_mode == 1:
        #         targets = targets[:, prefix_len:]
        #     else:
        #         assert prefix_len == 0

        #     y_pos = self.nar_audio_prenet(y_emb)
        #     y_pos = self.nar_audio_position(y_pos)

        #     y_dec, _ = self.nar_decoder(
        #         (y_pos, self.nar_stage_embeddings[nar_stage - 1].weight),
        #         x,
        #         tgt_mask=None,
        #         tgt_key_padding_mask=y_mask,
        #         memory_mask=None,
        #         memory_key_padding_mask=x_mask,
        #     )
        #     if self.prefix_mode != 0:
        #         y_dec = y_dec[:, prefix_len:]
        #         if self.prefix_mode == 4:
        #             prefix_len = 0  # reset for Top10Accuracy metric

        #     logits = self.nar_predict_layers[nar_stage - 1](y_dec).permute(
        #         0, 2, 1
        #     )
        #     # loss
        #     total_length = (y_lens).sum().type(torch.float32)
        #     total_loss += (
        #         F.cross_entropy(
        #             logits,
        #             targets,
        #             ignore_index=NUM_AUDIO_TOKENS,
        #             reduction=reduction,
        #         )
        #         * (total_length / (total_length - prefix_len * x.shape[0]))
        #     )
        #     metrics["NarTop10Accuracy"] = (
        #         self.nar_accuracy_metric(
        #             F.pad(
        #                 logits.detach(),
        #                 (0, 0, 0, 1, 0, 0),
        #                 value=logits.min().cpu().item(),
        #             ),
        #             targets,
        #         ).item()
        #         * total_length
        #     )
        #     metrics[f"NarTop10Accuracy_{nar_stage}"] = (
        #         self.nar_accuracy_metric(
        #             F.pad(
        #                 logits.detach(),
        #                 (0, 0, 0, 1, 0, 0),
        #                 value=logits.min().cpu().item(),
        #             ),
        #             targets,
        #         ).item()
        #         * total_length
        #     )
            # metrics[f"frames_{nar_stage}"] = (y_lens).sum().item()


        #             info["frames"] = (audio_features_lens).sum().item()
        # info["utterances"] = x.size(0)

        if train_stage == 0:
            total_loss = total_loss / 2.0

        return ((x, codes), total_loss, metrics)


    def inference_only_ar(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        top_k: int = -100,
        top_p: float=1.0, 
        temperature: float = 1.0,
        mode=0,
        task_id=0, # task_id=0, tts, 1-> semantic token convert, 2-> generative model-input_sem,
        use_silence_token=False,
        enroll_x_lens=None,
        y_lens=None
    ) -> torch.Tensor:
        """
        Args:
            x:
            A 2-D tensor of shape (N, S).
            x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
            y:
            A 3-D tensor of shape (B, T, 8).
            top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
            temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
            Return the predicted audio code matrix.
        """
        if self.semantic_num_quantizers==1:
            assert x.ndim == 2, x.shape
            assert x_lens.ndim == 1, x_lens.shape
        elif self.semantic_num_quantizers==2:  # text is semantic tfnet
            assert x.ndim == 3, x.shape
            assert x_lens.ndim == 1, x_lens.shape
        assert y==None or y.ndim == 3, y.shape
        assert torch.all(x_lens > 0)

        # NOTE: x has been padded in TextTokenCollater
        text = x
        bsz = x.shape[0]


        if self.ar_audio_prepend_bos and y!=None:
            prefix_len = y.shape[1] # not include bos
            # print(f"prefix_len:{prefix_len}")
            y = F.pad(y, (0, 0, 1, 0), value=NUM_AUDIO_TOKENS + 1)
        elif y==None:
            prefix_len = 0
            y = torch.full((bsz, 1, 1), NUM_AUDIO_TOKENS + 1, dtype=x.dtype, device=x.device)
            # print(f"prefix_len:0")

        if self.semantic_num_quantizers==1:
            x = self.ar_text_embedding(text)
        elif self.semantic_num_quantizers==2:
            x = rearrange(x, 'b ... -> b (...)') # b x T x 2-> bx (T x 2)
            offsets_x = (NUM_SEMANTIC_TOKENS_TFNET+1) * torch.arange(self.semantic_num_quantizers, device = x.device) # [0, 501, 1002, ...]
            offsets_x = repeat(offsets_x, 'q -> 1 (n q)', n = ceil_div(x.shape[-1], self.semantic_num_quantizers))  # n==T
            offsets_x = offsets_x[:, :x.shape[-1]]
            x = x + offsets_x
            x = self.ar_text_embedding(x)
            x = rearrange(x, 'b (t q) c -> b t (q c)', q = self.semantic_num_quantizers)

        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        x_mask = make_pad_mask(x_lens).to(x.device)
        x_attn_mask = torch.zeros(bsz * self.num_heads, x_lens.max(), x_lens.max(), dtype=x.dtype).to(x.device)

        src_len = x_lens.max()
        x_padding_mask = (
        x_mask.view(bsz, 1, 1, src_len)
        .expand(-1, self.num_heads, -1, -1)
        .reshape(bsz * self.num_heads, 1, src_len)
        )
        final_x_attn_mask = x_attn_mask.logical_or(x_padding_mask)  #only_padding_mask
        new_attn_mask = torch.zeros_like(final_x_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(final_x_attn_mask, float("-inf"))
        final_x_attn_mask = new_attn_mask

        x, _ = self.encoder(
            (x, None),
            mask=final_x_attn_mask,
            # src_key_padding_mask=xy_padding_mask,
            # is_causal=True,
        )

        prompts = y
            
        y_bef = y.clone()
        x_len = x_lens.max()
        if task_id==0: # tts
            generate_max_lens =  x_lens.max() *16
        elif task_id==1: # the same hz
            generate_max_lens = x_lens.max() +2
        elif task_id==2: # 25hz-> 50hz
            generate_max_lens = x_lens.max() *9

        end=0
        bsz = x.shape[0]
        # print(f"bsz:{bsz}")
        end_np = [0 for i in range(bsz)] # 表示一个batch 样本的终止
        end_loc = [0 for i in range(bsz)] # 截止

        while True:
            y_len = y_bef.shape[1]
            y = y_bef.clone()

            y = rearrange(y, 'b ... -> b (...)')
            offsets = (NUM_AUDIO_TOKENS+1+int(self.ar_audio_prepend_bos)) * torch.arange(self.num_quantizers, device = y.device)
            offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(y.shape[-1], self.num_quantizers))
            offsets = offsets[:, :y.shape[-1]]
            y = y + offsets

            y_emb = self.ar_audio_embedding(y)
            y_emb = rearrange(y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers)

            # y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)
            # ar_y_mask = y_mask  # no BOS
            # if self.ar_audio_prepend_bos:
            #     ar_y_mask = F.pad(y_mask, (1, 0), value=False)

            # y_len = y_len + int(self.ar_audio_prepend_bos)
            tgt_mask = torch.triu(
                torch.ones(y_len, y_len, device=y.device, dtype=torch.bool),
                diagonal=1,
            )
            y_dec, _ = self.ar_decoder(   # b x t x 1024
                (y_pos, None),
                x,
                tgt_mask=tgt_mask,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )
            # print(f"xy_dec shape {xy_dec.shape}")
            y_dec = y_dec[:,-1,:]

            y_dec = y_dec.unsqueeze(dim=1)
            y_dec = rearrange(y_dec, 'b t (q d) -> b t q d', q = self.num_quantizers)

            if self.shared_linear is False:
                logits_list = []
                for j in range(self.num_quantizers):
                    logits = self.ar_predict_layers[j](y_dec[:, :, j, :]).permute(0, 2, 1)
                    logits_list.append(logits)
                logits = torch.stack(logits_list, dim=-1)
            else:
                # print(self.ar_predict_layer)
                # quit()
                logits = self.ar_predict_layer(y_dec).permute(0, 3, 1, 2) # b t q d -> b t q num_class 

            samples = []

            logits = rearrange(logits, 'b d t q -> (b q) d t')
            logits = logits.squeeze(-1)  #(b x q, d)

            if mode==0:
                # token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                # # 应用 softmax 来计算概率  
                # probabilities = F.softmax(logits, dim=-1)  
                
                # # 找到最大概率的 token 及其概率  
                # max_prob, max_token = torch.max(probabilities, dim=-1)  
                
                # # 打印最大 token 和对应的概率  
                # print(f"Max token: {max_token.item()}, Probability: {max_prob.item()}")  
                
                # # 打印 token 1024 的概率  
                # token_1024_prob = probabilities[0, 1024]  
                # print(f"Probability of token 1024: {token_1024_prob.item()}") 
                samples = topk_sampling(
                    logits, top_k=top_k, top_p=top_p, y_bef=y_bef, temperature=temperature, num_quantizer=self.num_quantizers
                )
            elif mode==1:
                samples = arg_max_sampling(
                    logits
                )
            

            samples = samples.squeeze(1)
            samples = rearrange(samples, '(b q) -> b q', q= self.num_quantizers)

            # print(f"samples:{samples}")
            SILEN_TOKENS = [193]
            SILEN_TOKENS_appear = False
            if use_silence_token ==False:
                SILEN_TOKENS = []
            for i, one_batch_samles in enumerate(samples):
                for sample in one_batch_samles:
                    # if sample in SILEN_TOKENS:
                    #     SILEN_TOKENS_appear = True
                    if (sample == NUM_AUDIO_TOKENS or (sample in SILEN_TOKENS and (y_bef.shape[1]+1-prefix_len)>20 ) )or (task_id==1 and sample.item()>500):   # 注意要不要加个argmax
                    # if (sample == NUM_AUDIO_TOKENS or ((y_bef.shape[1]-1-prefix_len)== prefix_len+1 and SILEN_TOKENS_appear is True) )or (task_id==1 and sample.item()>500):   # 注意要不要加个argmax
                        if sample == NUM_AUDIO_TOKENS or (sample in SILEN_TOKENS):
                            # print(f"第{i}个序列meet end_token:{sample}")
                            pass
                        else:
                            # print(f"第{i}个序列meet oon")
                            pass
                        if end_np[i]==0: 
                            end_np[i]=1
                            end_loc[i]=y_bef.shape[1]-prefix_len # 第多少个loc开始停的
                            # print(f"第{i}个序列end， 终止长度为{y_bef.shape[1]+1-prefix_len}（包含终止符）")
                            break
            y_bef = torch.concat([y_bef, samples.unsqueeze(1)], dim=1)

            if (y_bef.shape[1] - prompts.shape[1]) > generate_max_lens:
                # print(f"长度过长， 终止")
                break
            if all(end_np) ==1:
                # print(f"所有序列终止")
                break

        codes = y_bef[:, prefix_len + int(self.ar_audio_prepend_bos) :]

        # print(codes.shape)
        # print(codes[0])
        # print(codes)
        # quit()

        batch_codes = [codes[i, :int(end_loc[i])-1, :].unsqueeze(0) for i in range(bsz)]


        # if self.num_quantizers == 1: 
        #     return torch.stack(codes, dim=-1)
        # print(codes[0].shape)
        return batch_codes
def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(
            max(top_k, min_tokens_to_keep), logits.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits

def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0, y_bef= None, penalty_coe=1.3, num_quantizer=1, penalty_type=0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    # y_bef (b x t x num_quantizer)        1 x 132 x 1 - 16 x 132 x 1
    # logits (b x num_quantizer) x dim  -> 1 x 1025 - 16 x 1025
    global ii
    if temperature != 1.0:
        logits = logits / temperature

    if penalty_type==1: # add penalty to all past tokens
        y_bef_v2 = torch.empty((0,))
        if y_bef!=None:
            y_bef_v2 = y_bef.clone()
            y_bef_v2 =y_bef_v2[:,1:,:]
        if y_bef_v2.numel() !=0:
            batch_size, t, num_quantizer = y_bef_v2.size()
            y_aft = torch.ones(batch_size, 1025, num_quantizer, device=y_bef_v2.device)
            indices = y_bef_v2.long()

            for i in range(indices.size(0)):  # 遍历每个batch  可以优化
                for j in range(indices.size(2)):  # 遍历每个q
                        y_aft[i, indices[i,t-1,j], j] = y_aft[i, indices[i,t-1,j], j] * penalty_coe

            y_aft = rearrange(y_aft, 'b d q -> (b q) d')

            logits = logits/y_aft
    elif penalty_type==2:
        y_bef_v2 = torch.empty((0,))
        if y_bef!=None:
            y_bef_v2 = y_bef.clone()
            y_bef_v2 =y_bef_v2[:,1:,:]
            single_num = y_bef_v2[0,1:,0].numel()
        if y_bef_v2.numel() !=0 and single_num >=2:
            batch_size, t, num_quantizer = y_bef_v2.size()
            y_aft = torch.ones(batch_size, 1025, num_quantizer, device=y_bef_v2.device)
            indices = y_bef_v2.long()
            for i in range(indices.size(0)):  # 遍历每个batch  可以优化
                for j in range(indices.size(2)):  # 遍历每个q
                    if indices[i,t-1,j]==indices[i,t-2,j]:
                        y_aft[i, indices[i,t-1,j], j] = y_aft[i, indices[i,t-1,j], j] * penalty_coe
            y_aft = rearrange(y_aft, 'b d q -> (b q) d')
            logits = logits/y_aft
    elif penalty_type==0:
        pass
    
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p) #(b x q, d)

  
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    return token

def arg_max_sampling(logits):
    token = torch.argmax(logits, dim=-1).unsqueeze(-1)

    return token

class Soundstorm(VALLF):

    
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        prefix_mode: int = 0,
        share_embedding: bool = True,
        nar_scale_factor: float = 1.0,
        only_autoregressive: bool = False,
        is_pretrain: bool = False,
        pret_mode: int = 0, 
        **kwargs,
    ):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super(Soundstorm, self).__init__(
            d_model,
            nhead,
            num_layers,
            norm_first=norm_first,
            add_prenet=add_prenet,
            decoder_cls=TransformerEncoder,
            decoder_layer_cls=TransformerEncoderLayer,
            prefix_mode=prefix_mode,
            share_embedding=share_embedding,
            nar_scale_factor=nar_scale_factor,
            only_autoregressive=only_autoregressive,
            is_pretrain=is_pretrain,
            pret_mode=pret_mode,
            **kwargs,
        )
        self.ar_text_prenet = None
        self.ar_audio_prenet = None
        self.ar_text_positio = None
        print(f"self.ar_text_prenet : {self.ar_text_prenet}")
        print(f"add_prenet：{add_prenet}")

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: Union[torch.Tensor, PromptedFeatures],
        y_lens: Union[torch.Tensor, PromptedFeatures],
        reduction: str = "sum",
        train_stage: int = 0,
        maskd_indices_batch: List =[],
        y_correct=None,
        only_comp_mask_loss=False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
          x: if tts-> x is phoneme else: semantic tokens
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y: if pretrain and ac finetune -> y is semantic tokens else: acoustic tokens
            A 3-D tensor of shape (N, T, 8).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          train_stage:
            0: AR & NAR modules, 1: AR modules, 2: NAR modules
        Returns:
          Return the predicted audio code matrix, cross-entropy loss and Top-10 accuracy.
        """
        # print(f"x:{x}")

        # maskd_indices_batch:输入会被mask
        # x_mask_int:padding mask
        if self.parrallel_mode==1:
            # sample one codebook
            random_codebook = random.randint(0, 15)  
            y = y[:, :, random_codebook:random_codebook+1]
            if y_correct!=None:
                y_correct =y_correct[:, :, random_codebook:random_codebook+1]
        elif self.parrallel_mode==2:
            random_codebook = random.randint(0, 1)  
            if random_codebook==0:
                y = y[:, :, 0:self.num_quantizers//2]
                if y_correct!=None:
                    y_correct =y_correct[:, :, 0:self.num_quantizers//2]
            elif random_codebook==1:
                y = y[:, :, self.num_quantizers//2:]
                if y_correct!=None:
                    y_correct =y_correct[:, :, self.num_quantizers//2:]     

        if self.semantic_num_quantizers==1:
            assert x.ndim == 2, x.shape
            assert x_lens.ndim == 1, x_lens.shape
        elif self.semantic_num_quantizers==2:  # text is semantic tfnet
            assert x.ndim == 3, x.shape
            assert x_lens.ndim == 1, x_lens.shape

        y_prompts_codes = None
        assert y.ndim == 3, y.shape
        assert y_lens.ndim == 1, y_lens.shape

        # NOTE: x has been padded in TextTokenCollater
        # for attention
        x_mask = make_pad_mask(x_lens).to(x.device)    
        x_mask_int = x_mask.type(torch.int64)
        x_len = x_lens.max()
        bsz, src_len = x.shape[0], x.shape[1]

        # with open("maskd_indices_batch_x_mask_int.txt", "w") as f:
        #     torch.set_printoptions(threshold=float('inf'))  
        #     f.write(str(maskd_indices_batch))
        #     f.write(str(x_mask_int))
        #     f.write('\n')
        # quit()
        text = x
        # torch.set_printoptions(threshold=float('inf'))  # threshold参数设置为大于你张量中元素总数的值 
        codes = y.type(torch.int64) * (1 - x_mask_int.unsqueeze(dim=-1)) + NUM_AUDIO_TOKENS*x_mask_int.unsqueeze(dim=-1)

        y_pad_mask_indices_mask = maskd_indices_batch | x_mask_int
        y = codes.clone()*((1 - y_pad_mask_indices_mask.unsqueeze(dim=-1))) + y_pad_mask_indices_mask.unsqueeze(dim=-1)*NUM_AUDIO_TOKENS
        
        if y_correct==None:
            targets = codes
        else:
            targets = y_correct.type(torch.int64) * (1 - x_mask_int.unsqueeze(dim=-1)) + NUM_AUDIO_TOKENS*x_mask_int.unsqueeze(dim=-1)
        metrics = {}
        total_loss = 0.0

        if self.semantic_num_quantizers==1:  # input is phoneme or hubert token
            x = self.ar_text_embedding(text) 
        elif self.semantic_num_quantizers==2: # input is tfnet semantic tokens
            x = rearrange(x, 'b ... -> b (...)') # b x T x 2-> bx (T x 2)
            # 有待商榷
            offsets_x = (NUM_SEMANTIC_TOKENS_TFNET+1) * torch.arange(self.semantic_num_quantizers, device = x.device) # [0, 501, 1002, ...]
            offsets_x = repeat(offsets_x, 'q -> 1 (n q)', n = ceil_div(x.shape[-1], self.semantic_num_quantizers))  # n==T
            offsets_x = offsets_x[:, :x.shape[-1]]
            x = x + offsets_x
            x = self.ar_text_embedding(x)
            x = rearrange(x, 'b (t q) c -> b t (q c)', q = self.semantic_num_quantizers)
        
        y = rearrange(y, 'b ... -> b (...)') # b x T x 16-> bx (T x 16)

        if self.parrallel_mode==0:
            offsets = (NUM_AUDIO_TOKENS+1) * torch.arange(self.num_quantizers, device = y.device) # [0, 501, 1002, ...]
            offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(y.shape[-1], self.num_quantizers))  # n==T
            offsets = offsets[:, :y.shape[-1]]
            y = y + offsets
        elif self.parrallel_mode==1:
            offsets = (NUM_AUDIO_TOKENS+1) *random_codebook
            y = y+offsets
        elif self.parrallel_mode==2:
            offsets = (NUM_AUDIO_TOKENS+1) * torch.arange(self.num_quantizers, device = y.device) # [0, 501, 1002, ...]
            if random_codebook==0:
                offsets = offsets[0:self.num_quantizers//2]
            elif random_codebook==1:
                offsets = offsets[self.num_quantizers//2:]
            offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(y.shape[-1], self.num_quantizers//2))  # n==T
            offsets = offsets[:, :y.shape[-1]]
            y = y + offsets

        y_emb = self.ar_audio_embedding(y)

        if self.parrallel_mode==0:
            y_emb = rearrange(y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers)
        elif self.parrallel_mode==1:
            y_emb = rearrange(y_emb, 'b (t q) c -> b t (q c)', q = 1)
        elif self.parrallel_mode==2:
            y_emb = rearrange(y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers//2)
        xy_pos = x + y_emb
        if self.parrallel_mode==1:
            random_codebook_tensor = torch.tensor(random_codebook).to(y_emb.device) 
            # Reshape the tensor to (b, 1, 1)  
            random_codebook_tensor = random_codebook_tensor.view(1, 1)  
            codebook_emb = self.codebook_embedding(random_codebook_tensor)
            xy_pos=xy_pos + codebook_emb
        elif self.parrallel_mode==2:
            random_codebook_tensor = torch.tensor(random_codebook).to(y_emb.device) 
            # Reshape the tensor to (b, 1, 1)  
            random_codebook_tensor = random_codebook_tensor.view(1, 1)  
            codebook_emb = self.codebook_embedding(random_codebook_tensor)
            xy_pos=xy_pos + codebook_emb        
            
        xy_emb = self.ar_audio_position(xy_pos)

        # attention mask
        x_attn_mask = torch.zeros(bsz * self.num_heads, x_lens.max(), x_lens.max(), dtype=x.dtype).to(x.device)
        x_padding_mask = (
        x_mask.view(bsz, 1, 1, src_len)
        .expand(-1, self.num_heads, -1, -1)
        .reshape(bsz * self.num_heads, 1, src_len)
        )
        # _xy_padding_mask: 400 1 392 xy_attn_mask: 400 392 392 -> 400 392 392
        final_x_attn_mask = x_attn_mask.logical_or(x_padding_mask)  #only_padding_mask
        new_attn_mask = torch.zeros_like(final_x_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(final_x_attn_mask, float("-inf"))
        final_x_attn_mask = new_attn_mask

        xy_dec, _ = self.ar_decoder(
            (xy_emb, None),
            mask=final_x_attn_mask,

        )
        logits_list = []

        if self.only_autoregressive is True:
            if self.parrallel_mode==0:
                xy_dec = rearrange(xy_dec, 'b t (q d) -> b t q d', q = self.num_quantizers)
            elif self.parrallel_mode==1:
                xy_dec = rearrange(xy_dec, 'b t (q d) -> b t q d', q = 1)
            elif self.parrallel_mode==2:
                xy_dec = rearrange(xy_dec, 'b t (q d) -> b t q d', q = self.num_quantizers//2)
            if self.shared_linear is False:
                if self.parrallel_mode==0:
                    for j in range(self.num_quantizers):
                        logits = self.ar_predict_layers[j](xy_dec[:, :, j, :]).permute(0, 2, 1) # b t d -> b t num_classes -> b num_classes t 
                        logits_list.append(logits)
                elif self.parrallel_mode==1:
                    logits = self.ar_predict_layers[random_codebook](xy_dec[:, :, 0, :]).permute(0, 2, 1) # b t d -> b t num_classes -> b num_classes t 
                    logits_list.append(logits)
                elif self.parrallel_mode==2:
                    for j in range(self.num_quantizers//2):
                        logits = self.ar_predict_layers[self.num_quantizers//2*random_codebook+j](xy_dec[:, :, j, :]).permute(0, 2, 1) # b t d -> b t num_classes -> b num_classes t 
                        logits_list.append(logits)             

                logits = torch.stack(logits_list, dim=-1) # b num_classes t -> b num_classes t num_quantizers
            
            else:
                logits = self.ar_predict_layer(xy_dec).permute(0, 3, 1, 2) # b t q d -> b num_class t q  

        else:
            logits = self.ar_predict_layer(xy_dec).permute(0, 2, 1)

        if maskd_indices_batch.numel() != 0 : # only compute masked loss
            maskd_indices_batch = maskd_indices_batch.unsqueeze(-1).expand_as(targets)
            maskd_targets = targets * maskd_indices_batch.to(targets.dtype)  + ((1- maskd_indices_batch).to(targets.dtype)  * NUM_AUDIO_TOKENS)
        
        if y_correct!=None:
            infill_mask_indices_batch = targets!=codes
            infill_mask_indices_batch_int = infill_mask_indices_batch.type(torch.int64)
            infill_maskd_targets = targets * infill_mask_indices_batch_int.to(targets.dtype)  + ((1- infill_mask_indices_batch_int).to(targets.dtype)  * NUM_AUDIO_TOKENS)

        if only_comp_mask_loss is True:
            total_loss = F.cross_entropy(logits, maskd_targets, reduction=reduction, ignore_index=NUM_AUDIO_TOKENS)
        else:
            total_loss = F.cross_entropy(logits, targets, reduction=reduction)

        if self.only_autoregressive is True:

            if self.parrallel_mode==0:
                # 注意soundstorm就是计算mask的loss
                for j in range(self.num_quantizers):
                    metrics[f"ArTop10Accuracy_{j}"] = self.ar_accuracy_metric(
                    logits.detach()[...,j], maskd_targets[...,j]
                    ).item() * y_lens.sum().type(torch.float32)

                    metrics[f"ArTop10Accuracy_top1_{j}"] = self.ar_accuracy_metric_top1(
                    logits.detach()[...,j], maskd_targets[...,j]
                    ).item() * y_lens.sum().type(torch.float32)   
                    
                    if y_correct!=None:
                        metrics[f"ArTop10Accuracy_infill_{j}"] = self.ar_accuracy_metric(
                        logits.detach()[...,j], infill_maskd_targets[...,j]
                        ).item() * y_lens.sum().type(torch.float32)

                        metrics[f"ArTop10Accuracy_infill_top1_{j}"] = self.ar_accuracy_metric_top1(
                        logits.detach()[...,j], infill_maskd_targets[...,j]
                        ).item() * y_lens.sum().type(torch.float32)
            elif self.parrallel_mode==1:
                metrics[f"ArTop10Accuracy_{random_codebook}"] = self.ar_accuracy_metric(
                logits.detach(), maskd_targets
                ).item() * y_lens.sum().type(torch.float32)

                metrics[f"ArTop10Accuracy_top1_{random_codebook}"] = self.ar_accuracy_metric_top1(
                logits.detach(), maskd_targets
                ).item() * y_lens.sum().type(torch.float32)  
            elif self.parrallel_mode==2:
                for j in range(self.num_quantizers//2):
                    metrics[f"ArTop10Accuracy_{random_codebook*self.num_quantizers//2+j}"] = self.ar_accuracy_metric(
                    logits.detach()[...,j], maskd_targets[...,j]
                    ).item() * y_lens.sum().type(torch.float32)

                    metrics[f"ArTop10Accuracy_top1_{random_codebook*self.num_quantizers//2+j}"] = self.ar_accuracy_metric_top1(
                    logits.detach()[...,j], maskd_targets[...,j]
                    ).item() * y_lens.sum().type(torch.float32)          


        if self.parrallel_mode==0:
            metrics["ArTop10Accuracy"] = self.ar_accuracy_metric(
            logits.detach(), maskd_targets
            ).item() * y_lens.sum().type(torch.float32)

            metrics["ArTop10Accuracy_top1"] = self.ar_accuracy_metric_top1(
            logits.detach(), maskd_targets
            ).item() * y_lens.sum().type(torch.float32)

            if y_correct!=None:
                metrics["ArTop10Accuracy_infill"] = self.ar_accuracy_metric(
                logits.detach(), infill_maskd_targets
                ).item() * y_lens.sum().type(torch.float32)

                metrics["ArTop10Accuracy_infill_top1"] = self.ar_accuracy_metric_top1(
                logits.detach(), infill_maskd_targets
                ).item() * y_lens.sum().type(torch.float32)   
        return ((x, codes), total_loss, metrics)

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "sinine":
            return lambda r: 1- np.sin(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError
  

    def mask_by_random_topk(self, mask_len, probs_update, temperature=4.5, along_time=False):
        # mask_len b 
        # probs_update b x T
        if along_time is True:
            batch_size, T = probs_update.size()  
            # print(f"T:{T}")
            # print(F'T - mask_len:{T - mask_len}')
            mask = torch.arange(T).unsqueeze(0).to(probs_update.device) >= (T -mask_len)  
            # 扩展mask以匹配probs_update的batch维度  
            masking = mask.expand(batch_size, T)  
        else:
            confidence = torch.log(probs_update)
            # confidence = torch.log(probs_update) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs_update.shape).to(probs_update.device)
            sorted_confidence, _ = torch.sort(confidence, -1)
            cut_off = torch.take_along_dim(sorted_confidence, mask_len.unsqueeze(dim=-1).to(torch.long), dim=-1)
            # Masks tokens with lower confidence.
            masking = (confidence < cut_off)
        return masking

    def inference(
        self,
        total_x: torch.Tensor,
        total_x_lens: torch.Tensor,
        prompt_y: torch.Tensor,
        top_k,
        top_p,
        temperature,
        initial_temp=4.5, 
        top_k_know_token=None,
        known_token_update=False,
        T=16,
    ) -> torch.Tensor:
        """
        Args:
            x:
            A 2-D tensor of shape (1, S).
            x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
            y:
            A 3-D tensor of shape (1, T, 8).
            top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
            temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
            Return the predicted audio code matrix.
        """
        ## 这里x,y 长度没有设定为 一致
        # print(f"T:{T}")
        PROBS_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(total_x.device)
        total_x_lens_mask = make_pad_mask(total_x_lens).to(total_x_lens.device)
        b_y, T_y = total_x.shape[0], total_x.shape[1]-prompt_y.shape[1]


        if self.semantic_num_quantizers==1:  # input is phoneme or hubert token
            total_x_emb = self.ar_text_embedding(total_x) 
        
        prompt_y_flat = rearrange(prompt_y, 'b ... -> b (...)') # b x T x 16-> bx (T x 16)
        offsets = (NUM_AUDIO_TOKENS+1) * torch.arange(self.num_quantizers, device = prompt_y_flat.device) # [0, 501, 1002, ...]
        offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(prompt_y_flat.shape[-1], self.num_quantizers))  # n==T
        offsets = offsets[:, :prompt_y_flat.shape[-1]]
        prompt_y_offsets = prompt_y_flat + offsets
        prompt_y_emb = self.ar_audio_embedding(prompt_y_offsets)
        prompt_y_emb = rearrange(prompt_y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers)

        y = torch.ones((b_y, T_y, self.num_quantizers), dtype=prompt_y.dtype, device=prompt_y.device)*NUM_AUDIO_TOKENS

        unknown_tokens_nums = (y==NUM_AUDIO_TOKENS)[:, :, 0].sum(dim=-1)

        # print(f"unknown_tokens_nums:{unknown_tokens_nums}")
        # compute attention mask

        bsz = b_y
        src_len = total_x_emb.shape[1]   ## 这里x,y 长度没有设定为 一致
        x_attn_mask = torch.zeros(bsz * self.num_heads, src_len, src_len, dtype=prompt_y.dtype).to(prompt_y.device)
        x_padding_mask = (
        total_x_lens_mask.view(bsz, 1, 1, src_len)
        .expand(-1, self.num_heads, -1, -1)
        .reshape(bsz * self.num_heads, 1, src_len)
        )
        # _xy_padding_mask: 400 1 392 xy_attn_mask: 400 392 392 -> 400 392 392
        final_x_attn_mask = x_attn_mask.logical_or(x_padding_mask)  #only_padding_mask

        new_attn_mask = torch.zeros_like(final_x_attn_mask, dtype=total_x.dtype)
        new_attn_mask = new_attn_mask.float()
        new_attn_mask.masked_fill_(final_x_attn_mask, float("-inf"))
        final_x_attn_mask = new_attn_mask

        for t in range(T):
            # print(f"T:{T}")
            # print(f"t:{t}")
            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = self.gamma_func(mode="cosine")(ratio)

            # choice 1
            topk_real = top_k

            # # choice 2   # cosine decay
            # mask_ratio_topk = self.gamma_func(mode="cosine")(ratio)
            # topk_real = int(np.floor(top_k*mask_ratio_topk)) # b x t
            # if topk_real==0:
            #     topk_real=1

            # # choice 3 # sinine decay
            # mask_ratio_topk = self.gamma_func(mode="sinine")(ratio)
            # topk_real = int(np.floor(top_k*mask_ratio_topk)) # b x t
            # if topk_real==0:
            #     topk_real=1

            # # choice 3 # line decay
            # mask_ratio_topk = self.gamma_func(mode="linear")(ratio)
            # topk_real = int(np.floor(top_k*mask_ratio_topk)) # b x t
            # if topk_real==0:
            #     topk_real=1

            # print(f"topk_real:{topk_real}")
            # print(mask_ratio)
            # torch.set_printoptions(threshold=float('inf'))  
            # f.write(str(y[:,:,0]))
            # f.write('\n')
            unknow_loc = (y==NUM_AUDIO_TOKENS)[:, :, :1].squeeze(-1)  # unkown 的位置需要更新
            y_flat = rearrange(y, 'b ... -> b (...)') # b x T x 16-> bx (T x 16)
            offsets = (NUM_AUDIO_TOKENS+1) * torch.arange(self.num_quantizers, device = y_flat.device) # [0, 501, 1002, ...]
            offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(y_flat.shape[-1], self.num_quantizers))  # n==T
            offsets = offsets[:, :y_flat.shape[-1]]
            y_offsets = y_flat + offsets
            y_emb = self.ar_audio_embedding(y_offsets)
            y_emb = rearrange(y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers)

            total_y_emb = torch.concat([prompt_y_emb, y_emb], dim=1)
            total_x_y_emb = total_x_emb + total_y_emb
            total_x_y_emb = self.ar_audio_position(total_x_y_emb)    

            xy_dec, _ = self.ar_decoder(
                (total_x_y_emb, None),
                mask=final_x_attn_mask,
            )  
            xy_dec = xy_dec[:, prompt_y.shape[1]:, :]
            logits_list = []
            if self.only_autoregressive is True:
                xy_dec = rearrange(xy_dec, 'b t (q d) -> b t q d', q = self.num_quantizers)
                if self.shared_linear is False:
                    for j in range(self.num_quantizers):
                        logits = self.ar_predict_layers[j](xy_dec[:, :, j, :]).permute(0, 2, 1) # b t d -> b t num_classes -> b num_classes t 
                        logits_list.append(logits)
                    logits = torch.stack(logits_list, dim=-1) # b num_classes t -> b num_classes t num_quantizers
                else:
                    logits = self.ar_predict_layer(xy_dec).permute(0, 3, 1, 2) # b t q d -> b num_class t q  

            # y_all_update = torch.argmax(logits, dim=1)
            samples = []
            bb, dd, tt, qq = logits.shape
            logits_reshape = rearrange(logits.clone(), 'b d t q -> (b t q) d')

            unknow_loc_quantizer = unknow_loc.unsqueeze(-1).expand(-1, -1, self.num_quantizers) # b t q

            print(f"temperature:{temperature}")
            samples = topk_sampling(
                logits_reshape.clone(), top_k=topk_real, top_p=top_p, temperature=temperature
            )
            samples = samples.squeeze(-1)
            y_all_update_unknow_token = rearrange(samples, '(b t q)-> b t q', b=bb, t=tt)

            print(f'top_k_know_token:{top_k_know_token}')
            if top_k_know_token is not None:
                samples_know_token = topk_sampling(
                    logits_reshape.clone(), top_k=top_k_know_token, top_p=top_p, temperature=temperature
                )
                samples_know_token = samples_know_token.squeeze(-1)            
                y_all_update_know_token = rearrange(samples_know_token, '(b t q)-> b t q', b=bb, t=tt)
                y_all_update = torch.where(unknow_loc_quantizer, y_all_update_unknow_token, y_all_update_know_token)
            else:
                y_all_update = y_all_update_unknow_token.clone()
            
            if known_token_update is False:
                print(f'known_token_update:{known_token_update}')
                # # 是否只update已有的token
                y_all_update = torch.where(unknow_loc_quantizer, y_all_update, y) # b t q 

            if t ==T-1:
                y = y_all_update
                break

            probs = F.softmax(logits, dim=1) # b num_class t q 
            
            probs = probs.permute(0, 2, 3, 1)
            probs_update = torch.take_along_dim(probs, y_all_update.unsqueeze(-1), dim=-1).squeeze(-1)  # b t q -> b t

            probs_update = torch.where(unknow_loc_quantizer, probs_update, PROBS_OF_KNOWN_TOKENS)
            probs_update = torch.prod(probs_update, dim=-1)  # b t q

            mask_len = torch.floor(unknown_tokens_nums*mask_ratio) # b x t
            # f.write(f"mask token len:{str(len(mask_len))}")
            # print(f'mask_ratio:{mask_ratio}')

            masking = self.mask_by_random_topk(mask_len, probs_update, temperature=initial_temp*(1. - mask_ratio))
            # print(f"masking:{masking}")

            masking = masking.unsqueeze(-1).expand(-1, -1, self.num_quantizers)
            y= torch.where(masking, NUM_AUDIO_TOKENS, y_all_update)
        return y
    def inference_along_time(
        self,
        total_x: torch.Tensor,
        total_x_lens: torch.Tensor,
        prompt_y: torch.Tensor,
        top_k,
        top_p,
        temperature,
        initial_temp=4.5, 
        top_k_know_token=None,
        known_token_update=False,
        T=16,
    ) -> torch.Tensor:
        """
        Args:
            x:
            A 2-D tensor of shape (1, S).
            x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
            y:
            A 3-D tensor of shape (1, T, 8).
            top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
            temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
            Return the predicted audio code matrix.
        """
        ## 这里x,y 长度没有设定为 一致
        # print(f"T:{T}")
        PROBS_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(total_x.device)
        total_x_lens_mask = make_pad_mask(total_x_lens).to(total_x_lens.device)
        b_y, T_y = total_x.shape[0], total_x.shape[1]-prompt_y.shape[1]


        if self.semantic_num_quantizers==1:  # input is phoneme or hubert token
            total_x_emb = self.ar_text_embedding(total_x) 
        
        prompt_y_flat = rearrange(prompt_y, 'b ... -> b (...)') # b x T x 16-> bx (T x 16)
        offsets = (NUM_AUDIO_TOKENS+1) * torch.arange(self.num_quantizers, device = prompt_y_flat.device) # [0, 501, 1002, ...]
        offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(prompt_y_flat.shape[-1], self.num_quantizers))  # n==T
        offsets = offsets[:, :prompt_y_flat.shape[-1]]
        prompt_y_offsets = prompt_y_flat + offsets
        prompt_y_emb = self.ar_audio_embedding(prompt_y_offsets)
        prompt_y_emb = rearrange(prompt_y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers)

        y = torch.ones((b_y, T_y, self.num_quantizers), dtype=prompt_y.dtype, device=prompt_y.device)*NUM_AUDIO_TOKENS

        unknown_tokens_nums = (y==NUM_AUDIO_TOKENS)[:, :, 0].sum(dim=-1)

        # print(f"unknown_tokens_nums:{unknown_tokens_nums}")
        # compute attention mask

        bsz = b_y
        src_len = total_x_emb.shape[1]   ## 这里x,y 长度没有设定为 一致
        x_attn_mask = torch.zeros(bsz * self.num_heads, src_len, src_len, dtype=prompt_y.dtype).to(prompt_y.device)
        x_padding_mask = (
        total_x_lens_mask.view(bsz, 1, 1, src_len)
        .expand(-1, self.num_heads, -1, -1)
        .reshape(bsz * self.num_heads, 1, src_len)
        )
        # _xy_padding_mask: 400 1 392 xy_attn_mask: 400 392 392 -> 400 392 392
        final_x_attn_mask = x_attn_mask.logical_or(x_padding_mask)  #only_padding_mask

        new_attn_mask = torch.zeros_like(final_x_attn_mask, dtype=total_x.dtype)
        new_attn_mask = new_attn_mask.float()
        new_attn_mask.masked_fill_(final_x_attn_mask, float("-inf"))
        final_x_attn_mask = new_attn_mask

        # with open(f"known_unknow_top_k_{top_k}.txt", "w") as f:
        for t in range(T):
            # print(f"T:{T}")
            # print(f"t:{t}")
            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = self.gamma_func(mode="cosine")(ratio)

            # choice 1
            topk_real = top_k

            unknow_loc = (y==NUM_AUDIO_TOKENS)[:, :, :1].squeeze(-1)  # unkown 的位置需要更新
            y_flat = rearrange(y, 'b ... -> b (...)') # b x T x 16-> bx (T x 16)
            offsets = (NUM_AUDIO_TOKENS+1) * torch.arange(self.num_quantizers, device = y_flat.device) # [0, 501, 1002, ...]
            offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(y_flat.shape[-1], self.num_quantizers))  # n==T
            offsets = offsets[:, :y_flat.shape[-1]]
            y_offsets = y_flat + offsets
            y_emb = self.ar_audio_embedding(y_offsets)
            y_emb = rearrange(y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers)

            total_y_emb = torch.concat([prompt_y_emb, y_emb], dim=1)
            total_x_y_emb = total_x_emb + total_y_emb
            total_x_y_emb = self.ar_audio_position(total_x_y_emb)    

            xy_dec, _ = self.ar_decoder(
                (total_x_y_emb, None),
                mask=final_x_attn_mask,
            )  
            xy_dec = xy_dec[:, prompt_y.shape[1]:, :]
            logits_list = []
            if self.only_autoregressive is True:
                xy_dec = rearrange(xy_dec, 'b t (q d) -> b t q d', q = self.num_quantizers)
                if self.shared_linear is False:
                    for j in range(self.num_quantizers):
                        logits = self.ar_predict_layers[j](xy_dec[:, :, j, :]).permute(0, 2, 1) # b t d -> b t num_classes -> b num_classes t 
                        logits_list.append(logits)
                    logits = torch.stack(logits_list, dim=-1) # b num_classes t -> b num_classes t num_quantizers
                else:
                    logits = self.ar_predict_layer(xy_dec).permute(0, 3, 1, 2) # b t q d -> b num_class t q  

            # y_all_update = torch.argmax(logits, dim=1)
            samples = []
            bb, dd, tt, qq = logits.shape
            logits_reshape = rearrange(logits.clone(), 'b d t q -> (b t q) d')

            unknow_loc_quantizer = unknow_loc.unsqueeze(-1).expand(-1, -1, self.num_quantizers) # b t q

            samples = topk_sampling(
                logits_reshape.clone(), top_k=topk_real, top_p=top_p, temperature=temperature
            )
            samples = samples.squeeze(-1)
            y_all_update_unknow_token = rearrange(samples, '(b t q)-> b t q', b=bb, t=tt)

            print(f'top_k_know_token:{top_k_know_token}')
            if top_k_know_token is not None:
                samples_know_token = topk_sampling(
                    logits_reshape.clone(), top_k=top_k_know_token, top_p=top_p, temperature=temperature
                )
                samples_know_token = samples_know_token.squeeze(-1)            
                y_all_update_know_token = rearrange(samples_know_token, '(b t q)-> b t q', b=bb, t=tt)
                y_all_update = torch.where(unknow_loc_quantizer, y_all_update_unknow_token, y_all_update_know_token)
            else:
                y_all_update = y_all_update_unknow_token.clone()
            
            if known_token_update is False:
                print(f'known_token_update:{known_token_update}')
                # # 是否只update已有的token
                y_all_update = torch.where(unknow_loc_quantizer, y_all_update, y) # b t q 

            if t ==T-1:
                y = y_all_update
                break

            probs = F.softmax(logits, dim=1) # b num_class t q 
            
            probs = probs.permute(0, 2, 3, 1)

            probs_update = torch.take_along_dim(probs, y_all_update.unsqueeze(-1), dim=-1).squeeze(-1)  # b t q -> b t

            probs_update = torch.where(unknow_loc_quantizer, probs_update, PROBS_OF_KNOWN_TOKENS)
            probs_update = torch.prod(probs_update, dim=-1)  # b t q

            # print(f'mask_ratio:{mask_ratio}')
            mask_len = torch.floor(unknown_tokens_nums*mask_ratio) # b x t
            # f.write(f"mask token len:{str(len(mask_len))}")

            masking = self.mask_by_random_topk(mask_len, probs_update, temperature=initial_temp*(1. - mask_ratio))

            masking = masking.unsqueeze(-1).expand(-1, -1, self.num_quantizers)
            y= torch.where(masking, NUM_AUDIO_TOKENS, y_all_update)

        return y

    def inference_group_ar(
            self,
            total_x: torch.Tensor,
            total_x_lens: torch.Tensor,
            prompt_y: torch.Tensor,
            top_k,
            top_p,
            temperature,
            initial_temp=4.5, 
            T=16,
        ) -> torch.Tensor:
            """
            Args:
                x:
                A 2-D tensor of shape (1, S).
                x_lens:
                A 1-D tensor of shape (1,). It contains the number of tokens in `x`
                before padding.
                y:
                A 3-D tensor of shape (1, T, 8).
                top_k: (`optional`) int
                The number of highest probability tokens to keep for top-k-filtering. Default to -100.
                temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
            Returns:
                Return the predicted audio code matrix.
            """
            ## 这里x,y 长度没有设定为 一致
            # print(f"T:{T}")
            # print(f"top_k:{top_k}")
            # print(f"top_p:{top_p}")
            # print(f"total_x:{total_x}")
            # print(f"prompt_y:{prompt_y}")
            # print(f"total_x.shape:{total_x.shape}")
            # print(f"prompt_y.shape:{prompt_y.shape}")
            # print(f"top_k:{top_k}")
            # print(f"top_p:{top_p}")
            # print(f"temperature:{temperature}")
            # print(f"T:{T}")
            
            total_x_lens_mask = make_pad_mask(total_x_lens).to(total_x_lens.device)
            b_y, T_y = total_x.shape[0], total_x.shape[1]-prompt_y.shape[1]

            if self.semantic_num_quantizers==1:  # input is phoneme or hubert token
                total_x_emb = self.ar_text_embedding(total_x) 
            
            prompt_y_flat = rearrange(prompt_y, 'b ... -> b (...)') # b x T x 16-> bx (T x 16)
            offsets = (NUM_AUDIO_TOKENS+1) * torch.arange(self.num_quantizers, device = prompt_y_flat.device) # [0, 501, 1002, ...]
            offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(prompt_y_flat.shape[-1], self.num_quantizers))  # n==T
            offsets = offsets[:, :prompt_y_flat.shape[-1]]
            prompt_y_offsets = prompt_y_flat + offsets
            prompt_y_emb = self.ar_audio_embedding(prompt_y_offsets)
            prompt_y_emb = rearrange(prompt_y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers)

            y = torch.ones((b_y, T_y, self.num_quantizers), dtype=prompt_y.dtype, device=prompt_y.device)*NUM_AUDIO_TOKENS

            # compute attention mask

            bsz = b_y
            src_len = total_x_emb.shape[1]   ## 这里x,y 长度没有设定为 一致
            x_attn_mask = torch.zeros(bsz * self.num_heads, src_len, src_len, dtype=prompt_y.dtype).to(prompt_y.device)
            x_padding_mask = (
            total_x_lens_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.num_heads, -1, -1)
            .reshape(bsz * self.num_heads, 1, src_len)
            )
            # _xy_padding_mask: 400 1 392 xy_attn_mask: 400 392 392 -> 400 392 392
            final_x_attn_mask = x_attn_mask.logical_or(x_padding_mask)  #only_padding_mask

            new_attn_mask = torch.zeros_like(final_x_attn_mask, dtype=total_x.dtype)
            new_attn_mask = new_attn_mask.float()
            new_attn_mask.masked_fill_(final_x_attn_mask, float("-inf"))
            final_x_attn_mask = new_attn_mask

            cut_distance = math.ceil(T_y/T)
            
            update_steps = list(range(0, T_y, cut_distance))    

            for t in range(T):

                yy = torch.zeros(T_y).to(prompt_y.device)
                yy[update_steps]=NUM_AUDIO_TOKENS
                yy = yy.unsqueeze(0).expand(bsz, -1)
                unknow_loc = (yy==NUM_AUDIO_TOKENS)
                y_flat = rearrange(y, 'b ... -> b (...)') # b x T x 16-> bx (T x 16)
                offsets = (NUM_AUDIO_TOKENS+1) * torch.arange(self.num_quantizers, device = y_flat.device) # [0, 501, 1002, ...]
                offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(y_flat.shape[-1], self.num_quantizers))  # n==T
                offsets = offsets[:, :y_flat.shape[-1]]
                y_offsets = y_flat + offsets
                y_emb = self.ar_audio_embedding(y_offsets)
                y_emb = rearrange(y_emb, 'b (t q) c -> b t (q c)', q = self.num_quantizers)

                total_y_emb = torch.concat([prompt_y_emb, y_emb], dim=1)
                total_x_y_emb = total_x_emb + total_y_emb
                total_x_y_emb = self.ar_audio_position(total_x_y_emb)

                xy_dec, _ = self.ar_decoder(
                    (total_x_y_emb, None),
                    mask=final_x_attn_mask,
                )  
                
                xy_dec = xy_dec[:, prompt_y.shape[1]:, :]
                logits_list = []
                if self.only_autoregressive is True:
                    xy_dec = rearrange(xy_dec, 'b t (q d) -> b t q d', q = self.num_quantizers)
                    if self.shared_linear is False:
                        for j in range(self.num_quantizers):
                            logits = self.ar_predict_layers[j](xy_dec[:, :, j, :]).permute(0, 2, 1) # b t d -> b t num_classes -> b num_classes t 
                            logits_list.append(logits)
                        logits = torch.stack(logits_list, dim=-1) # b num_classes t -> b num_classes t num_quantizers
                    else:
                        logits = self.ar_predict_layer(xy_dec).permute(0, 3, 1, 2) # b t q d -> b num_class t q  

                # y_all_update = torch.argmax(logits, dim=1)
                samples = []
                bb, dd, tt, qq = logits.shape
                logits_reshape = rearrange(logits.clone(), 'b d t q -> (b t q) d')

                samples = topk_sampling(
                    logits_reshape, top_k=top_k, top_p=top_p, temperature=temperature
                )
                samples = samples.squeeze(-1)
                y_all_update = rearrange(samples, '(b t q)-> b t q', b=bb, t=tt)

                unknow_loc_quantizer = unknow_loc.unsqueeze(-1).expand(-1, -1, self.num_quantizers)

                y = torch.where(unknow_loc_quantizer, y_all_update, y) # b t q 

                if t ==T-1:
                    y = y_all_update
                    break

                update_steps = [x + 1 for x in update_steps]
                
                if update_steps[-1] ==T_y:
                    update_steps = update_steps[:-1]

 
            return y


class VALLE_NAR(VALLF):

    
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        prefix_mode: int = 0,
        share_embedding: bool = True,
        nar_scale_factor: float = 1.0,
        only_autoregressive: bool = False,
        is_pretrain: bool = False,
        pret_mode: int = 0, 
        **kwargs,
    ):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super(VALLE_NAR, self).__init__(
            d_model,
            nhead,
            num_layers,
            norm_first=norm_first,
            add_prenet=add_prenet,
            decoder_cls=TransformerEncoder,
            decoder_layer_cls=TransformerEncoderLayer,
            prefix_mode=prefix_mode,
            share_embedding=share_embedding,
            nar_scale_factor=nar_scale_factor,
            only_autoregressive=only_autoregressive,
            is_pretrain=is_pretrain,
            pret_mode=pret_mode,
            **kwargs,
        )
        self.ar_text_prenet = None
        self.ar_audio_prenet = None
        self.ar_audio_position = None

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: Union[torch.Tensor, PromptedFeatures],
        y_lens: Union[torch.Tensor, PromptedFeatures],
        reduction: str = "sum",
        train_stage: int = 0,
        maskd_indices_batch: List =[],
        **kwargs,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
          x: if tts-> x is phoneme else: semantic tokens
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y: if pretrain and ac finetune -> y is semantic tokens else: acoustic tokens
            A 3-D tensor of shape (N, T, 8).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          train_stage:
            0: AR & NAR modules, 1: AR modules, 2: NAR modules
        Returns:
          Return the predicted audio code matrix, cross-entropy loss and Top-10 accuracy.
        """
        # print(f"x:{x}")


        # NOTE: x has been padded in TextTokenCollater
        # for attention
        x_mask = make_pad_mask(x_lens).to(x.device)    
        x_mask_int = x_mask.type(torch.int64)
        y_mask = make_pad_mask(y_lens).to(y.device)
        y_mask_int = y_mask.type(torch.int64)

        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))
        x_len = x_lens.max()
        bsz, src_len = x.shape[0], x.shape[1]

        text = x
        # torch.set_printoptions(threshold=float('inf'))  # threshold参数设置为大于你张量中元素总数的值 
        targets = codes

        metrics = {}
        total_loss = 0.0
        torch.set_printoptions(threshold=float('inf'))  

        if self.semantic_num_quantizers==1:  # input is phoneme or hubert token
            x = self.ar_text_embedding(text) 
        elif self.semantic_num_quantizers==2: # input is tfnet semantic tokens
            x = rearrange(x, 'b ... -> b (...)') # b x T x 2-> bx (T x 2)
            # 有待商榷
            offsets_x = (NUM_SEMANTIC_TOKENS_TFNET+1) * torch.arange(self.semantic_num_quantizers, device = x.device) # [0, 501, 1002, ...]
            offsets_x = repeat(offsets_x, 'q -> 1 (n q)', n = ceil_div(x.shape[-1], self.semantic_num_quantizers))  # n==T
            offsets_x = offsets_x[:, :x.shape[-1]]
            x = x + offsets_x
            x = self.ar_text_embedding(x)
            x = rearrange(x, 'b (t q) c -> b t (q c)', q = self.semantic_num_quantizers)

            x = self.ar_text_embedding(x)
        x_emb = self.ar_text_position(x)

        # attention mask
        x_attn_mask = torch.zeros(bsz * self.num_heads, x_lens.max(), x_lens.max(), dtype=x.dtype).to(x.device)
        x_padding_mask = (
        x_mask.view(bsz, 1, 1, src_len)
        .expand(-1, self.num_heads, -1, -1)
        .reshape(bsz * self.num_heads, 1, src_len)
        )
        # _xy_padding_mask: 400 1 392 xy_attn_mask: 400 392 392 -> 400 392 392
        final_x_attn_mask = x_attn_mask.logical_or(x_padding_mask)  #only_padding_mask
        new_attn_mask = torch.zeros_like(final_x_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(final_x_attn_mask, float("-inf"))
        final_x_attn_mask = new_attn_mask

        xy_dec, _ = self.ar_decoder(
            (x_emb, None),
            mask=final_x_attn_mask,

        )
        logits_list = []

        if self.only_autoregressive is True:
            xy_dec = rearrange(xy_dec, 'b t (q d) -> b t q d', q = self.num_quantizers)
            if self.shared_linear is False:
                for j in range(self.num_quantizers):
                    logits = self.ar_predict_layers[j](xy_dec[:, :, j, :]).permute(0, 2, 1) # b t d -> b t num_classes -> b num_classes t 
                    logits_list.append(logits)

                logits = torch.stack(logits_list, dim=-1) # b num_classes t -> b num_classes t num_quantizers
            
            else:
                logits = self.ar_predict_layer(xy_dec).permute(0, 3, 1, 2) # b t q d -> b num_class t q  

        else:
            logits = self.ar_predict_layer(xy_dec).permute(0, 2, 1)

        if maskd_indices_batch.numel() != 0: # only compute masked loss
            maskd_indices_batch = maskd_indices_batch.unsqueeze(-1).expand_as(targets)
            maskd_targets = targets * maskd_indices_batch.to(targets.dtype)  + ((1- maskd_indices_batch).to(targets.dtype)  * NUM_AUDIO_TOKENS)

            total_loss = F.cross_entropy(logits, maskd_targets, reduction=reduction, ignore_index=NUM_AUDIO_TOKENS)

        else:
            total_loss = F.cross_entropy(logits, targets, reduction=reduction)

        if self.only_autoregressive is True:
            for j in range(self.num_quantizers):
                metrics[f"ArTop10Accuracy_{j}"] = self.ar_accuracy_metric(
                logits.detach()[...,j], targets[...,j]
                ).item() * y_lens.sum().type(torch.float32)

                metrics[f"ArTop10Accuracy_top1_{j}"] = self.ar_accuracy_metric_top1(
                logits.detach()[...,j], targets[...,j]
                ).item() * y_lens.sum().type(torch.float32)

                if maskd_indices_batch.numel() != 0:
                    metrics[f"ArTop10Accuracyignore_{j}"] = self.ar_accuracy_metric_ignore(
                    logits.detach()[...,j], maskd_targets[...,j]
                    ).item() * y_lens.sum().type(torch.float32)

                    metrics[f"ArTop10Accuracyignore_top1_{j}"] = self.ar_accuracy_metric_top1_ignore(
                    logits.detach()[...,j], maskd_targets[...,j]
                    ).item() * y_lens.sum().type(torch.float32)        

        metrics["ArTop10Accuracy"] = self.ar_accuracy_metric(  # it maybe casuse error if batch_size if large.
        logits.detach(), targets
        ).item() * y_lens.sum().type(torch.float32)

        metrics["ArTop10Accuracy_top1"] = self.ar_accuracy_metric_top1(
        logits.detach(), targets
        ).item() * y_lens.sum().type(torch.float32)

        if maskd_indices_batch.numel() != 0:
            metrics["ArTop10Accuracyignore"] = self.ar_accuracy_metric_ignore(
            logits.detach(), maskd_targets
            ).item() * y_lens.sum().type(torch.float32)

            metrics["ArTop10Accuracyignore_top1"] = self.ar_accuracy_metric_top1_ignore(
            logits.detach(), maskd_targets
            ).item() * y_lens.sum().type(torch.float32)


        return ((x, codes), total_loss, metrics)

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError
  

    def mask_by_random_topk(self, mask_len, probs_update, temperature):
        # mask_len b 
        # probs_update b x T
        confidence = torch.log(probs_update)

        # confidence = torch.log(probs_update) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs_update.shape).to(probs_update.device)
        sorted_confidence, _ = torch.sort(confidence, -1)
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.unsqueeze(dim=-1).to(torch.long), dim=-1)
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking

    def inference(
        self,
        total_x: torch.Tensor,
        total_x_lens: torch.Tensor,
        top_k,
        top_p,
        temperature,
    ) -> torch.Tensor:
        """
        Args:
            x:
            A 2-D tensor of shape (1, S).
            x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
            y:
            A 3-D tensor of shape (1, T, 8).
            top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
            temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
            Return the predicted audio code matrix.
        """
        ## 这里x,y 长度没有设定为 一致

        total_x_lens_mask = make_pad_mask(total_x_lens).to(total_x_lens.device)

        if self.semantic_num_quantizers==1:  # input is phoneme or hubert token
            total_x_emb = self.ar_text_embedding(total_x) 
        bsz = total_x.shape[0]
        # compute attention mask
        src_len = total_x_emb.shape[1]   ## 这里x,y 长度没有设定为 一致
        x_attn_mask = torch.zeros(bsz * self.num_heads, src_len, src_len, dtype=total_x.dtype).to(total_x.device)
        x_padding_mask = (
        total_x_lens_mask.view(bsz, 1, 1, src_len)
        .expand(-1, self.num_heads, -1, -1)
        .reshape(bsz * self.num_heads, 1, src_len)
        )
        # _xy_padding_mask: 400 1 392 xy_attn_mask: 400 392 392 -> 400 392 392
        final_x_attn_mask = x_attn_mask.logical_or(x_padding_mask)  #only_padding_mask

        new_attn_mask = torch.zeros_like(final_x_attn_mask, dtype=total_x.dtype)
        new_attn_mask = new_attn_mask.float()
        new_attn_mask.masked_fill_(final_x_attn_mask, float("-inf"))
        final_x_attn_mask = new_attn_mask

        total_x_emb = self.ar_text_position(total_x_emb)

        xy_dec, _ = self.ar_decoder(
            (total_x_emb, None),
            mask=final_x_attn_mask,
        )  
        logits_list = []
        if self.only_autoregressive is True:
            xy_dec = rearrange(xy_dec, 'b t (q d) -> b t q d', q = self.num_quantizers)
            if self.shared_linear is False:
                for j in range(self.num_quantizers):
                    logits = self.ar_predict_layers[j](xy_dec[:, :, j, :]).permute(0, 2, 1) # b t d -> b t num_classes -> b num_classes t 
                    logits_list.append(logits)
                logits = torch.stack(logits_list, dim=-1) # b num_classes t -> b num_classes t num_quantizers
            else:
                logits = self.ar_predict_layer(xy_dec).permute(0, 3, 1, 2) # b t q d -> b num_class t q  

            # y_all_update = torch.argmax(logits, dim=1)
            samples = []
            bb, dd, tt, qq = logits.shape
            logits_reshape = rearrange(logits, 'b d t q -> (b t q) d')
            print(f"top_k:{top_k}")
            samples = topk_sampling(
                logits_reshape, top_k=top_k, top_p=top_p, temperature=temperature
            )
            samples = samples.squeeze(-1)
            y = rearrange(samples, '(b t q)-> b t q', b=bb, t=tt)

        
        return y
