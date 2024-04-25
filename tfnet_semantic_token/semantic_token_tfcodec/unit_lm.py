import torch
import torch.nn.functional as F

# import argparse
from beartype.typing import Optional
# from language_model.audiolm_module import *
from tqdm import tqdm
from einops import rearrange, repeat
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torch import nn, einsum
from .audiolm_module import *
import numpy as np
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            causal=False,
            dim_head=64,
            dim_context=None,
            heads=8,
            norm_context=False,
            num_null_kv=0,
            dropout=0.1,
            scale=8,
            cond_wind_size=0,
            cond_bugfix=True,
    ):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.causal = causal
        inner_dim = dim_head * heads
        self.cond_wind_size = cond_wind_size
        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.num_null_kv = num_null_kv
        if num_null_kv > 0:
            self.null_kv = nn.Parameter(torch.randn(2, num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )
        self.cond_bugfix = cond_bugfix

    def forward(
            self,
            x,
            context=None,
            mask=None,
            attn_bias=None,
            prefix_context=None,
            prefix_context_mask=None,
            prefix_attn_bias = None,
    ):
        b, n, _, device = *x.shape, x.device

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        # take care of prefix-based self attention conditioning
        # make sure to either concat the to the self attention mask or lengthen it accordingly

        if exists(prefix_context):
            kv_input = torch.cat((prefix_context, kv_input), dim=-2)
            prefix_seq_len = prefix_context.shape[-2]

            if not exists(mask):
                mask = torch.ones((b, n), device=device, dtype=torch.bool)

            if exists(prefix_context_mask):
                mask = torch.cat((prefix_context_mask, mask), dim=-1)
            else:
                mask = F.pad(mask, (prefix_seq_len, 0), value=True)

            if exists(attn_bias):
                if prefix_attn_bias is None:
                    attn_bias = F.pad(attn_bias, (prefix_seq_len, 0), value=0.)
                else:
                    attn_bias = torch.cat((prefix_attn_bias, attn_bias), dim=-1)

        # prenorm

        x = self.norm(x)

        # project for queries, keys, values

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        # null key / values

        if self.num_null_kv > 0:
            null_k, null_v = repeat(self.null_kv, 'kv n d -> kv b n d', b=b).unbind(dim=0)
            k = torch.cat((null_k, k), dim=-2)
            v = torch.cat((null_v, v), dim=-2)

        # split for multi-headed attention

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        # new technique, rmsnormed queries and keys, first used by 22B parameter model successfully https://arxiv.org/abs/2302.05442

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale

        if exists(attn_bias):
            attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.)
            sim = sim + attn_bias

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)


        if self.cond_bugfix:
            if self.cond_wind_size > 0 and (exists(context) or exists(prefix_context)):
                i, j = sim.shape[-2:]
                condition_len = context.size()[1] if exists(context) else prefix_context.size()[1]
                condition_mask1 = torch.ones((i, condition_len), dtype=torch.bool, device=x.device).tril()
                condition_mask2 = torch.ones((i, condition_len), dtype=torch.bool, device=x.device).tril(-self.cond_wind_size)
                condition_mask = condition_mask1 ^ condition_mask2
                condition_mask = ~condition_mask
                if self.training:
                    # assert (i == (condition_len + 1))  # without cond_end_token
                    assert (i == condition_len)  # with cond_end_token
                condition_mask = torch.cat([condition_mask,torch.zeros((i, i), dtype=torch.bool, device=x.device)],dim=1)
                sim = sim.masked_fill(condition_mask, -torch.finfo(sim.dtype).max)
        else:
            if self.cond_wind_size > 0 and (exists(context) or exists(prefix_context)):
                i, j = sim.shape[-2:]
                condition_len = context.size()[1] if exists(context) else prefix_context.size()[1]
                condition_mask1 = torch.ones((i, condition_len), dtype=torch.bool, device=x.device).tril(-1)
                condition_mask2 = torch.ones((i, condition_len), dtype=torch.bool, device=x.device).tril(-self.cond_wind_size-1)
                condition_mask = condition_mask1 ^ condition_mask2
                condition_mask = ~condition_mask
                if self.training:
                    # assert (i == (condition_len + 1))  # without cond_end_token
                    assert (i == condition_len)  # with cond_end_token
                condition_mask = torch.cat([condition_mask,torch.zeros((i, i), dtype=torch.bool, device=x.device)],dim=1)
                sim = sim.masked_fill(condition_mask, -torch.finfo(sim.dtype).max)
        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)




# transformer

class Transformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            heads,
            dim_context=None,
            cross_attend=False,
            attn_dropout=0.,
            ff_dropout=0.,
            grad_shrink_alpha=0.1,
            cond_as_self_attn_prefix=False,
            rel_pos_bias=True,
            # codec_config = None,
            rel_pos_bias_prefix = False,
            cond_wind_size=-1,
            cond_bugfix=True,
            **kwargs
    ):
        super().__init__()
        # self.codec_config = codec_config

        assert not (cross_attend and cond_as_self_attn_prefix)


        self.dim_context = default(dim_context, dim)

        self.cond_as_self_attn_prefix = cond_as_self_attn_prefix

        self.grad_shrink = partial(grad_shrink, alpha=grad_shrink_alpha)

        self.layers = nn.ModuleList([])

        self.rel_pos_bias = RelativePositionBias(dim=dim // 2, heads=heads) if rel_pos_bias else None


        self.rel_pos_bias_prefix = RelativePositionBias(dim=dim // 2,heads=heads) if rel_pos_bias_prefix and cond_as_self_attn_prefix and cond_wind_size<=0 else None

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dropout=attn_dropout, causal=True,cond_wind_size = cond_wind_size,cond_bugfix=cond_bugfix,**kwargs),
                Attention(dim=dim, heads=heads, dropout=attn_dropout, dim_context=dim_context, num_null_kv=1,
                          norm_context=True, cond_wind_size = cond_wind_size,cond_bugfix=cond_bugfix,**kwargs) if cross_attend else None,
                FeedForward(dim=dim, dropout=ff_dropout)
            ]))

        self.norm = LayerNorm(dim)
        self.depth = depth
    def forward(
            self,
            x,
            self_attn_mask=None,
            context=None,
            context_mask=None,
            context_attn_bias=None,
            attn_bias=None
    ):

        assert not (self.cond_as_self_attn_prefix and not exists(context))
        assert not (exists(context) and context.shape[
            -1] != self.dim_context), f'you had specified a conditioning dimension of {self.dim_context}, yet what was received by the transformer has dimension of {context.shape[-1]}'

        n, device = x.shape[1], x.device

        x = self.grad_shrink(
            x)  # from cogview paper, adopted by GLM 130B LLM, decreases likelihood of attention net instability

        if exists(attn_bias):
            rel_pos_bias = attn_bias
        else:
            rel_pos_bias = maybe(self.rel_pos_bias)(n)

        if exists(context_attn_bias):
            context_rel_pos_bias = context_attn_bias
        elif exists(context):
            context_rel_pos_bias = maybe(self.rel_pos_bias_prefix)(n, context.shape[1])
        else:
            context_rel_pos_bias = None

        self_attn_kwargs = dict()
        if self.cond_as_self_attn_prefix:
            self_attn_kwargs = dict(
                prefix_context=context,
                prefix_context_mask=context_mask,
                prefix_attn_bias=context_rel_pos_bias,
            )
        layer_results = []

        for depth, layer in enumerate(self.layers):
            attn, cross_attn, ff = layer
            x = attn(x, attn_bias=rel_pos_bias, mask=self_attn_mask, **self_attn_kwargs) + x

            if exists(cross_attn):
                assert exists(context)

                x = cross_attn(x, context=context, mask=context_mask) + x

            x = ff(x) + x
            if depth == (self.depth-1):
                layer_results.append(self.norm(x))
            else:
                layer_results.append(x)

        return self.norm(x), layer_results



# @beartype
class SemanticTransformerWithGroup(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            num_semantic_tokens,
            num_groups,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            t5_name=DEFAULT_T5_NAME,
            cond_dim=None,
            has_condition=False,
            audio_text_condition=False,
            cond_as_self_attn_prefix=False,
            cond_drop_prob=0.5,
            grad_shrink_alpha=0.1,
            codec_config=None,
            rel_pos_bias_prefix=False,
            cond_wind_size=-1,
            **kwargs
    ):
        super().__init__()
        self.num_semantic_tokens = num_semantic_tokens
        self.num_groups = num_groups
        self.cond_wind_size = cond_wind_size

        if audio_text_condition:
            has_condition = True
            cond_dim = default(cond_dim, dim)

        self.has_condition = has_condition
        self.embed_text = partial(t5_encode_text, name=t5_name)
        self.cond_drop_prob = cond_drop_prob

        self.start_token = nn.Parameter(torch.randn(dim))
        self.token_dim = dim
        self.codec_config = codec_config
        if self.codec_config['pred_mode'] == 'sequential_VQ':
            self.semantic_embedding = nn.Embedding(num_groups * (num_semantic_tokens + 1), dim)
            self.eos_id = num_semantic_tokens
            self.quantize_embedding = nn.Embedding(num_groups, dim)
            # self.to_logits = nn.Parameter(torch.randn(num_groups, (num_semantic_tokens + 1), dim))
            self.to_logits = nn.Linear(dim, num_semantic_tokens + 1)

        elif self.codec_config['pred_mode'] == 'parallel_VQ':
            self.semantic_embedding = nn.Embedding(num_groups * (num_semantic_tokens + 1), dim)
            # self.to_logits = nn.Parameter(torch.randn(num_groups, (num_semantic_tokens + 1), dim))
            self.to_logits = nn.Linear(dim, num_semantic_tokens + 1)
            self.eos_id = num_semantic_tokens

        elif self.codec_config['pred_mode'] == 'joint_VQ':
            if self.codec_config['share_eos']:
                self.semantic_embedding = nn.Embedding(num_groups * num_semantic_tokens + 1, dim // num_groups)
                self.eos_id = num_groups * num_semantic_tokens
            else:
                self.semantic_embedding = nn.Embedding(num_groups * (num_semantic_tokens + 1), dim // num_groups)
                self.eos_id = num_semantic_tokens

            self.to_logits = nn.Linear(dim, num_groups * (num_semantic_tokens + 1))

        elif self.codec_config['pred_mode'] == 'seperate_VQ':
            self.semantic_embedding = nn.Embedding(num_groups * (num_semantic_tokens + 1), dim)
            # self.to_logits = nn.Parameter(torch.randn(num_groups, (num_semantic_tokens + 1), dim))
            self.to_logits = nn.Linear(dim, num_semantic_tokens + 1)
            self.eos_id = num_semantic_tokens


        text_dim = default(cond_dim, get_encoded_dim(t5_name))
        self.proj_text_embed = nn.Linear(text_dim, dim, bias=False) if text_dim != dim else nn.Identity()
        self.codec_config = codec_config
        self.cond_bugfix = codec_config.get('cond_bugfix',True)
        if self.cond_wind_size > 0 and self.cond_bugfix:
            self.cond_end_token = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            cross_attend=has_condition and not cond_as_self_attn_prefix,
            cond_as_self_attn_prefix=cond_as_self_attn_prefix,
            rel_pos_bias_prefix=rel_pos_bias_prefix,
            grad_shrink_alpha=grad_shrink_alpha,
            cond_wind_size = cond_wind_size,
            cond_bugfix = self.cond_bugfix,
            **kwargs
        )

        # if self.codec_config['tgt_ASR']:
        #     self.lm_head = nn.Linear(dim, 33) ## vocabsize = 33

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1,
            **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)

        if cond_scale == 1 or not self.has_condition:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
            self,
            *,
            ids=None,
            return_loss=False,
            text: Optional[List[str]] = None,
            text_embeds=None,
            self_attn_mask=None,
            cond_drop_prob=None,
            unique_consecutive=None,
            text_mask=None,
            target_text=None,
            target_lang_emb=None,
    ):
        device = self.device
        result = {}
        b = ids.shape[0]

        has_text = exists(text) or exists(text_embeds)
        assert not (self.has_condition ^ has_text)

        text_mask = text_mask
        if not exists(text_embeds) and exists(text):
            with torch.no_grad():
                text_embeds = self.embed_text(text, output_device=device)
                text_mask = torch.any(text_embeds != 0, dim=-1)

        if exists(text_embeds):
            text_embeds = self.proj_text_embed(text_embeds)
            if self.cond_wind_size > 0 and self.cond_bugfix:
                cond_end_token = repeat(self.cond_end_token, 'd -> b n d', b = text_embeds.shape[0], n = text_embeds.shape[1])
                text_embeds = torch.where(~repeat(text_mask, 'b n -> b n d', d = text_embeds.shape[2]), cond_end_token, text_embeds)
                text_embeds = torch.cat((text_embeds, repeat(self.cond_end_token, 'd -> b 1 d', b = text_embeds.shape[0])), dim = 1)
                text_mask = F.pad(text_mask, (1, 0), value = True)

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        if exists(text_mask) and cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device=device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        if return_loss: ## always false
            labels, ids = ids.clone(), ids[:, :-self.num_groups]


        if self.codec_config['pred_mode'] == 'sequential_VQ':
            tokens = get_embeds(self.semantic_embedding, ids)  # [B, T*group, C]
            quantize_tokens = repeat(self.quantize_embedding.weight, 'g d -> (t g) d',
                                     t=ceil_div(ids.shape[-1], self.num_groups))
            quantize_tokens = quantize_tokens[:ids.shape[-1], ...]
            tokens = tokens + quantize_tokens
        elif self.codec_config['pred_mode'] == 'parallel_VQ':
            ids = rearrange(ids,'b (t g) -> (b g) t', g=self.num_groups)
            tokens = get_embeds(self.semantic_embedding, ids)  # [B*group, T, C]
        elif self.codec_config['pred_mode'] in ('joint_VQ','seperate_VQ'):
            tokens = get_embeds(self.semantic_embedding, ids)  # [B, T*group, C/group]
            tokens = tokens.reshape((b, -1, self.token_dim)) # [B, T, C]

        if self.codec_config['use_langIDs']:
            start_tokens = repeat(target_lang_emb, 'b d -> b 1 d', b=ids.shape[0])
        else:
            start_tokens = repeat(self.start_token, 'd -> b 1 d', b=ids.shape[0])

        if exists(self_attn_mask):
            self_attn_mask = F.pad(self_attn_mask, (1, 0), value=True)

        tokens = torch.cat((start_tokens, tokens), dim=1)

        tokens, layer_results = self.transformer(tokens, context=text_embeds, self_attn_mask=self_attn_mask, context_mask=text_mask)

        if self.codec_config['pred_mode'] == 'sequential_VQ':
            result['logits'] = self.to_logits(tokens)

        elif self.codec_config['pred_mode'] == 'parallel_VQ':
            tokens = rearrange(tokens, '(b g) t d -> b (t g) d', b=b, g=self.num_groups)
            result['logits'] = self.to_logits(tokens)

        elif self.codec_config['pred_mode'] in ('joint_VQ','seperate_VQ'):
            result['logits'] = self.to_logits(tokens)  # [B, T, C]

        # if self.codec_config['tgt_ASR2']:
        #     feats = layer_results[int(self.depth//2)-1]
        #     logits = self.lm_head(feats)

            # loss = None
            # labels = target_text
            # if labels is not None:
            #     # retrieve loss input_lengths from attention_mask
            #     attention_mask = (
            #         self_attn_mask if self_attn_mask is not None else torch.ones_like(logits, dtype=torch.long)
            #     )
            #     # input_lengths = self.wav2vec2._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long) ## ?
            #     input_lengths = (attention_mask > 0).sum(-1)
            #     # assuming that padded tokens are filled with -100
            #     # when not being attended to
            #     labels_mask = labels > 0
            #     target_lengths = labels_mask.sum(-1)
            #     flattened_targets = labels.masked_select(labels_mask)
            #
            #     # ctc_loss doesn't support fp16
            #     log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            #
            #     with torch.backends.cudnn.flags(enabled=False):
            #         loss = nn.functional.ctc_loss(
            #             log_probs,
            #             flattened_targets,
            #             input_lengths,
            #             target_lengths,
            #             blank=0,
            #             reduction='mean',
            #             zero_infinity=True,
            #         )
            # result['ctc_loss'] = loss
            # result['logits_ctc'] = logits
        result['layer_results'] = layer_results
        return result

    def generate_tokens(
            self,
            *,
            ids=None,
            return_loss=False,
            text: Optional[List[str]] = None,
            text_embeds=None,
            self_attn_mask=None,
            cond_drop_prob=None,
            unique_consecutive=None,
            text_mask=None,
            source_text=None,
            target_text=None,
            target_lang_emb=None,
    ):
        device = self.device
        result = {}
        b = ids.shape[0]

        has_text = exists(text) or exists(text_embeds)
        assert not (self.has_condition ^ has_text)

        text_mask = text_mask
        if not exists(text_embeds) and exists(text):
            with torch.no_grad():
                text_embeds = self.embed_text(text, output_device=device)
                text_mask = torch.any(text_embeds != 0, dim=-1)

        if exists(text_embeds):
            text_embeds = self.proj_text_embed(text_embeds)

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        if exists(text_mask) and cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device=device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        if return_loss: ## always false
            labels, ids = ids.clone(), ids[:, :-self.num_groups]


        if self.codec_config['pred_mode'] == 'sequential_VQ':
            tokens = get_embeds(self.semantic_embedding, ids)  # [B, T*group, C]
            quantize_tokens = repeat(self.quantize_embedding.weight, 'g d -> (t g) d',
                                     t=ceil_div(ids.shape[-1], self.num_groups))
            quantize_tokens = quantize_tokens[:ids.shape[-1], ...]
            tokens = tokens + quantize_tokens
        elif self.codec_config['pred_mode'] == 'parallel_VQ':
            ids = rearrange(ids,'b (t g) -> (b g) t', g=self.num_groups)
            tokens = get_embeds(self.semantic_embedding, ids)  # [B*group, T, C]
        elif self.codec_config['pred_mode'] in ('joint_VQ','seperate_VQ'):
            tokens = get_embeds(self.semantic_embedding, ids)  # [B, T*group, C/group]
            tokens = tokens.reshape((b, -1, self.token_dim)) # [B, T, C]

        if self.codec_config['use_langIDs']:
            start_tokens = repeat(target_lang_emb, 'b d -> b 1 d', b=ids.shape[0])
        else:
            start_tokens = repeat(self.start_token, 'd -> b 1 d', b=ids.shape[0])

        if exists(self_attn_mask):
            self_attn_mask = F.pad(self_attn_mask, (1, 0), value=True)

        tokens = torch.cat((start_tokens, tokens), dim=1)

        return tokens

# class LabelSmoothingLoss(nn.Module):
#     def __init__(self, classes, smoothing=0.0, dim=-1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.cls = classes
#         self.dim = dim
#
#     def forward(self, pred, target, ignore_index):
#         pred = pred.log_softmax(dim=self.dim)
#         with torch.no_grad():
#             # true_dist = pred.data.clone()
#             true_dist = torch.zeros_like(pred)
#             true_dist.fill_(self.smoothing / (self.cls - 1))
#             true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
# training wrappers
# @beartype
class SemanticTransformerWithGroupWrapper(nn.Module):
    def __init__(
            self,
            *,
            transformer: SemanticTransformerWithGroup,
            wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]] = None,
            audio_conditioner: Optional[AudioConditionerBase] = None,
            pad_id=-1,
            unique_consecutive=True,
            mask_prob=0.15,
            codec_config=None,
    ):
        super().__init__()
        self.wav2vec = wav2vec
        self.transformer = transformer
        self.audio_conditioner = audio_conditioner

        assert not (exists(
            audio_conditioner) and not transformer.has_condition), 'if conditioning on audio embeddings from mulan, transformer has_condition must be set to True'

        assert not exists(
            self.wav2vec) or self.wav2vec.codebook_size == transformer.num_semantic_tokens, f'num_semantic_tokens on SemanticTransformer must be set to {self.wav2vec.codebook_size}'

        self.unique_consecutive = unique_consecutive
        self.pad_id = pad_id
        self.eos_id = transformer.eos_id
        self.mask_prob = mask_prob
        self.codec_config = codec_config
        if 'use_label_smoothing' in self.codec_config.keys():
            self.use_label_smoothing = codec_config['use_label_smoothing']
        else:
            self.use_label_smoothing = False
        if self.use_label_smoothing:
            self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self.pad_id,label_smoothing=0.2)


    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.no_grad()
    # @beartype
    def generate(
            self,
            *,
            max_length,
            text: Optional[List[str]] = None,
            text_embeds=None,
            prime_wave=None,
            prime_ids=None,
            batch_size=1,
            cond_scale=1,
            filter_thres=0.9,
            temperature=1.,
            include_eos_in_output=True,  # if doing hierarchical sampling, eos must be kept for an easy time
            beam_size=10,
            do_beam_search=False,
            sample_criteria='gumbel',
            force_align = False,
            **kwargs
    ):
        device = self.device

        # derive wav2vec ids from the input wave

        if exists(prime_wave):
            assert not exists(prime_ids)
            assert exists(self.wav2vec)
            ids = self.wav2vec(prime_wave, flatten=False)
        elif exists(prime_ids):
            ids = prime_ids
        else:
            ids = torch.empty((batch_size, 0, self.transformer.num_groups), dtype=torch.long, device=device)

        # if self.unique_consecutive:
        #     ids = batch_unique_consecutive_wgroup(ids, pad_value=self.pad_id)

        # derive joint audio-text embeddings if needed

        if exists(self.audio_conditioner) and exists(prime_wave):
            assert not exists(text) and not exists(text_embeds)
            text_embeds = self.audio_conditioner(wavs=prime_wave, namespace='semantic')

        # derive text embeddings if needed
        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.no_grad():
                text_embeds = self.transformer.embed_text(text, output_device=device)

        if exists(text_embeds):
            b, t1, c1 = text_embeds.size()
            cross_atten_mask = torch.full((b, t1), 1,device=device, dtype=torch.bool)
        # start length and get running id output

        batch = ids.shape[0]
        start_length = ids.shape[1]

        last_logit_indices = (ids[:, :, 0] != self.pad_id).sum(dim=-1).long()

        ## init beam search
        beams = [[{'tokens': ids[i,:,j], 'score': 0.0} for i in range(batch)] for j in range(self.transformer.num_groups)]
        ids = torch.stack([torch.stack([beam['tokens'] for beam in beams[g]],dim=0) for g, beam_group in enumerate(beams)],dim=-1)
        sample_semantic_ids = ids.clone()

        sample_semantic_ids = rearrange(sample_semantic_ids, 'b ... -> b (...)')

        # sample from transformer
        shortest_length = 0
        layer_feats = []
        layer_feats2 = []
        # for ind in tqdm(range(start_length, max_length), desc='generating semantic',position=0):
        # max_length = text_embeds.shape[1] if force_align else max_length
        # max_length = text_embeds.shape[1]
        for ind in range(start_length, max_length):
            import time
            # time.sleep(0.1)
            # repeat(sample_semantic_ids, 'b t c -> (repeat b) t c', repeat=ids.shape[0])
            input_ids = convert_token_ids_for_embeddings2(sample_semantic_ids,
                                                         self.transformer.num_groups,
                                                         self.transformer.num_semantic_tokens, self.eos_id,
                                                         self.pad_id,self.codec_config['share_eos'])  # (B, T*G)

            out = self.transformer.forward_with_cond_scale(  # (B, T, G*(C+1))
                ids=input_ids,
                text_embeds=repeat(text_embeds, 'b t c -> (repeat b) t c', repeat=input_ids.shape[0]) if exists(text_embeds) else text_embeds,
                cond_scale=cond_scale,
                text_mask=cross_atten_mask,
                **kwargs
            )
            logits = out['logits']
            logits_ctc = out['logits_ctc'] if 'logits_ctc' in out.keys() else None
            layer_results = out['layer_results']
            # last_logit_indices_expanded = repeat(last_logit_indices, 'b -> (repeat b) 1 c', b=batch,repeat=logits.shape[0], c=logits.shape[-1])
            # last_logits = logits.gather(1, last_logit_indices_expanded)
            if self.codec_config['pred_mode'] == 'parallel_VQ':
                logits = rearrange(logits, 'b (t q) c -> b t q c', q=self.transformer.num_groups)
                last_logits = logits[:, -1, ...]
                last_logits = rearrange(last_logits, 'b q c -> (b q) c')
            else:
                last_logits  = logits[:, -1, : ]
            # last_logits = rearrange(last_logits, 'b 1 c -> b c')
            last_logits = last_logits.reshape((-1, self.transformer.num_semantic_tokens + 1))
            if len(layer_results)>8:
                layer_feat,layer_feat2 = layer_results[5],layer_results[8]
                layer_feats.append(layer_feat[:, -1, : ].unsqueeze(1))
                layer_feats2.append(layer_feat2[:, -1, : ].unsqueeze(1))

            if do_beam_search:
                last_logits = last_logits.reshape((-1, self.transformer.num_groups, self.transformer.num_semantic_tokens + 1))
                for g in range(self.transformer.num_groups):
                    candidates = []
                    for i, beam in enumerate(beams[g]):
                        logits = last_logits[i, g, :]
                        # filtered_logits = top_k(logits, thres=filter_thres)
                        # log_probs = nn.functional.log_softmax(filtered_logits / temperature, dim=-1)
                        scores, idxs = logits.topk(beam_size)
                        for j in range(beam_size):
                            idx = idxs[j].view(-1)
                            idx= torch.tensor([self.eos_id]).to(idx) if idx.item() == self.transformer.num_semantic_tokens else idx
                            score = scores[j].item()
                            new_beam = {'tokens': torch.cat([beam['tokens'], idx]), 'score': beam['score'] + score}
                            candidates.append(new_beam)
                    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:beam_size]
                    beams[g] = candidates.copy()

                sample_semantic_ids = torch.stack(
                    [torch.stack([beam['tokens'] for beam in beams[g]], dim=0) for g, beam_group in enumerate(beams)],
                    dim=-1)

            else:
                if ind < (shortest_length+start_length):
                    last_logits[:, -1] = float('-inf')  # prevent from eos if not last quantizer step, but move this to masking logic within the transformer at some point, for both training and eval
                filtered_logits = top_k(last_logits, thres=filter_thres)

                if sample_criteria == "gumbel":
                    sampled = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
                elif sample_criteria == 'random':
                    sampled = torch.randint(last_logits.size(1), (last_logits.size(0),)).cuda()
                else:
                    sampled = filtered_logits.argmax(dim=-1)

                sampled = rearrange(sampled, 'b -> b 1')
                sampled[sampled == self.transformer.num_semantic_tokens] = self.eos_id
                if self.codec_config['pred_mode'] in ('joint_VQ','parallel_VQ'):
                    sampled = sampled.reshape((-1,self.transformer.num_groups))
                sample_semantic_ids = torch.cat((sample_semantic_ids, sampled), dim=1)


            if all_rows_have_eos_id_wgroup(sample_semantic_ids, self.eos_id):
                break

            last_logit_indices += 1
        if sample_semantic_ids.dim() == 2:
            length = int(sample_semantic_ids.shape[-1] // self.transformer.num_groups )*  self.transformer.num_groups
            sample_semantic_ids = rearrange(sample_semantic_ids[:,:length], 'b (t q) -> b t q', q = self.transformer.num_groups)
        sample_semantic_ids = mask_out_after_eos_id_wgroup(sample_semantic_ids, self.eos_id,
                                                           num_groups=self.transformer.num_groups, keep_eos=False)
        if do_beam_search:
            sample_semantic_ids = sample_semantic_ids[0].unsqueeze(0)
        if len(layer_results)>8:
            layer_feats = torch.cat(layer_feats,dim=1)
            layer_feats2 = torch.cat(layer_feats2,dim=1)
        else:
            layer_feats,layer_feats2=None,None
        return sample_semantic_ids,[layer_feats,layer_feats2]

    def generate_batch( ##
            self,
            *,
            max_length,
            text: Optional[List[str]] = None,
            text_embeds=None,
            prime_wave=None,
            prime_ids=None,  # [B,T,N]
            batch_size=1,
            cond_scale=3,
            filter_thres=0.9,
            temperature=1.,
            sample_criteria: Optional[Union["max", "gumbel"]] = "gumbel",
            include_eos_in_output=False,  # if doing hierarchical sampling, eos must be kept for an easy time
            self_atten_mask=None,
            **kwargs
    ):
        b, t = prime_ids.shape[0], prime_ids.shape[1]
        # self_atten_mask = None
        # if self.mask_prob > 0.:
        #     self_atten_mask = generate_mask_with_prob([b, t], self.mask_prob, prime_ids.device)
        if exists(text_embeds):
            b, t1, c1 = text_embeds.size()
            cross_atten_mask = torch.full((b, t1), 1,device=prime_ids.device, dtype=torch.bool)
        result = self.forward(semantic_token_ids=prime_ids,  # [B,T,N]
                              raw_wave=None,
                              text=None,
                              text_embeds=text_embeds,
                              return_loss=True,
                              self_atten_mask=self_atten_mask,
                              cross_atten_mask=cross_atten_mask,
                              **kwargs
                              )

        logits = result['logits']  # (b, t, g*c)
        b, n, g, d = logits.shape[0], logits.shape[
            1], self.transformer.num_groups, self.transformer.num_semantic_tokens + 1

        logits = logits.reshape((b, -1, g, d))
        logits[:, :-1, :, -1] = float('-inf')  # prevent from eos if not last quantizer step
        logits = logits.reshape(-1, n * g, d).squeeze(0)
        filtered_logits = top_k(logits, thres=filter_thres)
        if sample_criteria == "gumbel":
            sampled = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
        else:
            sampled = filtered_logits.argmax(dim=-1)

        sampled[sampled == self.transformer.num_semantic_tokens] = self.eos_id  # set shared eos id
        sampled = sampled.reshape((b, n, g))
        sample_semantic_ids = mask_out_after_eos_id_wgroup(sampled, self.eos_id, num_groups=self.transformer.num_groups,
                                                           keep_eos=include_eos_in_output)
        layer_results = result['layer_results']
        return sample_semantic_ids, layer_results

    def forward(
            self,
            *,
            semantic_token_ids=None,  # [B,T,N]
            raw_wave=None,
            text=None,
            text_embeds=None,
            return_loss=False,
            self_atten_mask=None,
            cross_atten_mask=None,
            target_text=None,
            **kwargs
    ):
        assert exists(raw_wave) or exists(
            semantic_token_ids), 'either raw waveform (raw_wave) is given or semantic token ids are given (semantic_token_ids)'

        if exists(self.audio_conditioner):
            assert exists(raw_wave)
            assert not exists(text) and not exists(text_embeds)
            text_embeds = self.audio_conditioner(wavs=raw_wave, namespace='semantic')

        if not exists(semantic_token_ids):
            assert exists(self.wav2vec), 'VQWav2Vec must be be provided if given raw wave for training'
            semantic_token_ids = self.wav2vec(raw_wave, flatten=False)


        if return_loss:
            # semantic_token_ids = append_eos_id_wgroup(semantic_token_ids, self.eos_id)
            semantic_token_ids = append_eos_id_wgroup2(semantic_token_ids,self.eos_id,self.pad_id,self_atten_mask)


        if self.unique_consecutive:
            semantic_token_ids = batch_unique_consecutive_wgroup(semantic_token_ids, pad_value=self.pad_id)


        B, T, G  = semantic_token_ids.shape[0],semantic_token_ids.shape[1], semantic_token_ids.shape[2]
        semantic_token_ids = rearrange(semantic_token_ids, 'b ... -> b (...)')

        input_ids = semantic_token_ids
        if return_loss:
            if self.codec_config['pred_mode'] == 'sequential_VQ':
                input_ids = semantic_token_ids[:, :-1]
            else:
                input_ids = semantic_token_ids[:, :-self.transformer.num_groups]

        mask = torch.cumsum(torch.eq(input_ids.clone(), self.pad_id), dim=1) > 0
        self_atten_mask = ~mask
        b = self_atten_mask.shape[0]


        if self.codec_config['pred_mode'] == 'sequential_VQ':
            time_step = input_ids.shape[1]
        elif self.codec_config['pred_mode'] == 'parallel_VQ':
            time_step = input_ids.shape[1] // self.transformer.num_groups
            self_atten_mask = rearrange(self_atten_mask,'b (t g) -> (b g) t', b=b,g=self.transformer.num_groups)
        elif self.codec_config['pred_mode'] in ('joint_VQ','seperate_VQ'):
            time_step = input_ids.shape[1] // self.transformer.num_groups
            self_atten_mask = rearrange(self_atten_mask,'b (t g) -> b t g', b=b,g=self.transformer.num_groups)[:,:,0]
            # self_atten_mask = self_atten_mask.reshape((b, -1, self.transformer.num_groups))[:,:,0]
        self_atten_mask_orig = self_atten_mask
        # self_attn_mask = None
        if self.mask_prob > 0. and self.training:
            self_atten_mask2 = generate_mask_with_prob(
                [self_atten_mask.shape[0], time_step], self.mask_prob,
                self_atten_mask.device)

            self_atten_mask = self_atten_mask & self_atten_mask2


        input_ids2 = convert_token_ids_for_embeddings2(input_ids, self.transformer.num_groups,
                                                          self.transformer.num_semantic_tokens, self.eos_id, self.pad_id, share_eos=self.codec_config['share_eos'])

        result = self.transformer(
            ids=input_ids2,
            text=text,
            text_embeds=text_embeds,
            self_attn_mask= self_atten_mask,
            text_mask = cross_atten_mask,
            target_text = target_text,
            **kwargs
        )

        logits = result['logits'] ## b,t*g,c / b,t,c
        if not return_loss:
            return result

        b = semantic_token_ids.shape[0]
        s_ids = semantic_token_ids.clone()


        if self.codec_config['pred_mode'] in ('joint_VQ','seperate_VQ'):
            logits = logits.reshape((b, -1, self.transformer.num_groups, self.transformer.num_semantic_tokens + 1))
            logits = rearrange(logits, 'b t g d -> b (t g) d')
            # relabel eos_id for loss
            if self.codec_config['share_eos']:
                s_ids[semantic_token_ids == self.eos_id] = self.transformer.num_semantic_tokens

        if self.use_label_smoothing:
            loss = self.cross_entropy_loss(rearrange(logits, 'b n c -> b c n'),s_ids)  ## n=t*gs_ids,ignore_index=self.pad_id)
        else:
            loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'),s_ids,ignore_index=self.pad_id)

        pad_mask = (s_ids.view(-1) != self.pad_id)  # 创建一个掩码，将 pad_id 值对应的位置置为 False
        correct = torch.sum(
            (torch.argmax(rearrange(logits, 'b n c -> (b n) c'), -1) == s_ids.view(-1)).type(torch.float)*pad_mask
        )  # 乘以掩码后只保留非 pad_id 的位置
        # correct = accuracy * pad_mask
        total = torch.sum(pad_mask)  # 非 pad_id 的总数
        accuracy = correct / total  # 计算准确率


        # accuracy = torch.mean(
        #     (torch.argmax(rearrange(logits, 'b n g c -> (b n g) c'), -1) == s_ids.view(-1)).type(
        #         torch.float))
        result['loss'] = loss
        result['accuracy'] = accuracy

        G, C = self.transformer.num_groups, self.transformer.num_semantic_tokens + 1
        # logits[:, :-1, :, (C-1):C] = float('-inf') # prevent from eos if not last step
        logits = logits.reshape(b, -1, C)
        pred_ids = torch.argmax(logits[:, :, :-1], dim=-1).reshape(b, -1, G)
        result['pred_ids'] = pred_ids[:, :-1, :]  # remove last step
        result['self_atten_mask'] = F.pad(self_atten_mask_orig, (1, 0), value=True)


        return result
