from functools import partial, wraps
import sys
sys.path.append("/mnt/users/jiazhijun/valle_23_4_22")

import math
import torch
from torch import nn, einsum
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from beartype.typing import Optional, Union, List
from beartype import beartype
from einops import rearrange, repeat, reduce
from tqdm import tqdm

# from .t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME
from tfnet.tfnet_models_mp_lm.utils import default, exists, maybe, ceil_div, prob_mask_like, get_embeds, eval_decorator, batch_unique_consecutive, append_eos_id, generate_mask_with_prob
# from tfnet_models_mp_lm.utils import top_k, all_rows_have_eos_id, mask_out_after_eos_id, gumbel_sample, round_down_nearest_multiple
# from tfnet_models_mp_lm.utils import AudioConditionerBase
# from tfnet_models_mp_lm.AudioConditioner import TFNet_encoder, TFNet_encoder_channel_split_v2, TFNet_encoder_channel_split_v4
# from .transformer import TransformerS
# from .hubert_kmeans import HubertWithKmeans
from tfnet.tfnet_models_mp.tfnet_v2i_vqvae import TFNet as TFNetV2_interl_VQVAE
# from tfnet_models_mp.tfnet_v4i_vqvae import TFNet as TFNetV4_interl_VQVAE



# @beartype
# class CodecTransformer(nn.Module):
#     def __init__(
#         self,
#         *,
#         codebook_size,
#         num_quantizers,
#         dim,
#         depth,
#         heads = 8,
#         attn_dropout = 0.,
#         ff_dropout = 0.,S
#         t5_name = DEFAULT_T5_NAME,
#         has_condition = False,
#         cond_dim = None,
#         audio_text_condition = False,
#         cond_as_self_attn_prefix = False,
#         cond_drop_prob = 0.5,
#         grad_shrink_alpha = 0.1,
#         **kwargs
#     ):
#         super().__init__()
#         if audio_text_condition:
#             has_condition = True
#             cond_dim = default(cond_dim, dim)

#         self.has_condition = has_condition
#         self.embed_text = partial(t5_encode_text, name = t5_name)
#         self.cond_drop_prob = cond_drop_prob

#         self.start_token = nn.Parameter(torch.randn(dim))

#         self.eos_id = codebook_size
#         codebook_size_with_eos = codebook_size + 1
        
#         embedding_dim = dim // num_quantizers
#         assert embedding_dim * num_quantizers == dim, "The dimension of transformer must be divided by the number of quantizers!"
#         self.embedding = nn.Embedding(num_quantizers * codebook_size_with_eos, embedding_dim)
#         # self.quantize_embedding = nn.Embedding(num_quantizers, dim)

#         text_dim = default(cond_dim, get_encoded_dim(t5_name))
#         self.proj_text_embed = nn.Linear(text_dim, dim, bias = False) if text_dim != dim else nn.Identity()

#         self.transformer = Transformer(
#             dim = dim,
#             depth = depth,
#             heads = heads,
#             attn_dropout = attn_dropout,
#             ff_dropout = ff_dropout,
#             cross_attend = has_condition and not cond_as_self_attn_prefix,
#             cond_as_self_attn_prefix = cond_as_self_attn_prefix,
#             grad_shrink_alpha = grad_shrink_alpha,
#             **kwargs
#         )

#         self.codebook_size = codebook_size
#         self.num_quantizers = num_quantizers

#         self.logit_weights = nn.Parameter(torch.randn(num_quantizers, codebook_size_with_eos, embedding_dim))

#     @property
#     def device(self):
#         return next(self.parameters()).device

#     def forward_with_cond_scale(
#         self,
#         *args,
#         cond_scale = 3,
#         **kwargs
#     ):
#         logits = self.forward(*args, cond_drop_prob = 0., diagonal = 1, **kwargs)

#         if cond_scale == 1 or not self.has_condition:
#             return logits

#         null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
#         scaled_logits = null_logits + (logits - null_logits) * cond_scale
#         return scaled_logits

#     def forward(
#         self,
#         *,
#         token_ids,
#         self_attn_mask = None,
#         text: Optional[List[str]] = None,
#         text_embeds = None,
#         cond_drop_prob = None,
#         diagonal = None, # diagonal for causal cross attention
#     ):
#         b, device = token_ids.shape[0], token_ids.device

#         has_text = exists(text) or exists(text_embeds)
#         assert not (self.has_condition ^ has_text)

#         if not exists(text_embeds) and exists(text):
#             with torch.no_grad():
#                 text_embeds = self.embed_text(text, output_device = device)

#         text_mask = None
#         if exists(text_embeds):
#             text_mask = torch.any(text_embeds != 0, dim = -1)

#             text_embeds = self.proj_text_embed(text_embeds)

#         cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

#         if exists(text_mask) and cond_drop_prob > 0:
#             keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)
#             text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask
#         #1024
#         offsets = self.codebook_size * torch.arange(self.num_quantizers, device = device)
#         offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(token_ids.shape[-1], self.num_quantizers))
#         offsets = offsets[:, :token_ids.shape[-1]]
#         token_ids = token_ids + offsets

#         tokens = self.embedding(token_ids)#b (t q) c
#         tokens = rearrange(tokens, 'b (t q) c -> b t (q c)', q = self.num_quantizers)
        
#         # currently, we do not need quantizer embedding
#         # quantize_tokens = repeat(self.quantize_embedding.weight, 'q d -> (n q) d', n = ceil_div(token_ids.shape[-1], self.num_quantizers))
#         # quantize_tokens = quantize_tokens[:token_ids.shape[-1], ...]
#         # tokens = tokens + quantize_tokens
        
#         start_tokens = repeat(self.start_token, 'd -> b 1 d', b = b)

#         tokens = torch.cat((
#             start_tokens,
#             tokens
#         ), dim = 1)
#         # tokens = rearrange(tokens, 'b q t d -> (b q) t d')

#         if exists(self_attn_mask):
#             self_attn_mask = F.pad(self_attn_mask, (1, 0), value = True) 

#         pred_tokens = self.transformer(tokens, context = text_embeds, self_attn_mask = self_attn_mask, context_mask = text_mask, diagonal = diagonal)

#         # get coarse logits
#         pred_tokens = rearrange(pred_tokens, 'b t (q d) -> b t q d', q = self.num_quantizers)
#         logits = einsum('q c d, b t q d -> b t q c', self.logit_weights, pred_tokens)
#         logits = rearrange(logits, 'b t q c -> b (t q) c')

#         return logits

# @beartype
# class CodecTransformerWrapper(nn.Module):
#     def __init__(
#         self,
#         config,
#         transformer: Optional[CodecTransformer] = None,
#         tfnet: Optional[TFNetV2_interl_VQVAE] = None,
#         audio_conditioner: Optional[AudioConditionerBase] = None,
#         pad_id = -1,
#         unique_consecutive = False,
#         mask_prob = 0.15,
#         t5_name = DEFAULT_T5_NAME,
#     ):
#         super().__init__()
#         config_transformer = config["args"]["transformer"]

#         if tfnet is None:
#             tfnet_ckpt = config["tfnet_ckpt"]
#             config_tfnet = config["tfnet"]
            
#             if config_tfnet['model_type'] =='tfnetv2_interleave_vqvae':
#                 tfnet = TFNetV2_interl_VQVAE(config=config_tfnet, )
#             elif config_tfnet['model_type'] =='tfnetv4_interleave_vqvae':
#                 tfnet = TFNetV4_interl_VQVAE(config=config_tfnet, sampling_rate=config_tfnet["sampling_rate"], win_len=config_tfnet["dft_size"], hop_len=int(config_tfnet["dft_size"] * config_tfnet["hop_vqvae"]), n_fft=config_tfnet["dft_size"])

#             # load with tfnet_ckpt
#             tfnet.load(tfnet_ckpt)
#             tfnet.freeze_encoder()
#             tfnet.freeze_codebook()

#         if audio_conditioner is None:
#             if config_tfnet['model_type'] =='tfnetv2_interleave_vqvae':
#                 audio_conditioner = TFNet_encoder_channel_split_v2(config=config_tfnet, sampling_rate=config_tfnet["sampling_rate"], win_len=config_tfnet["dft_size"], hop_len=int(config_tfnet["dft_size"] * config_tfnet["hop_vqvae"])) 
#             elif config_tfnet['model_type'] =='tfnetv4_interleave_vqvae':
#                 audio_conditioner = TFNet_encoder_channel_split_v4(config=config_tfnet, sampling_rate=config_tfnet["sampling_rate"], win_len=config_tfnet["dft_size"], hop_len=int(config_tfnet["dft_size"] * config_tfnet["hop_vqvae"])) 

#         # decide how to merge context
#         cond_as_self_attn_prefix = False
#         self.cond_as_self_attn_prefix_causal = False
#         context_args = {}

#         context_merge_type = config_transformer["context_merge_type"]
#         if context_merge_type == "cond_as_self_attn_prefix":
#             cond_as_self_attn_prefix = True
#             context_args.update(
#                 {
#                     "use_causal_prefix": False,
#                     "num_null_context_embed": None,
#                 })
#         elif context_merge_type == "cond_as_self_attn_prefix_causal":
#             cond_as_self_attn_prefix = True
#             self.cond_as_self_attn_prefix_causal = True
#             context_args.update(
#                 {
#                     "use_causal_prefix": True,
#                     "num_null_context_embed": 1,
#                 })
#         elif context_merge_type == "cross_attend_causal":
#             context_args.update({"use_causal_cross_attend": True})
#         else:
#             context_args.update({"use_causal_cross_attend": False})
            
#         if transformer is None:
#             transformer = CodecTransformer(
#                         codebook_size = tfnet.codebook_size[0],
#                         num_quantizers = tfnet.codebook_num,
#                         dim = config_transformer["dim"],
#                         depth = config_transformer["depth"],
#                         has_condition = True,
#                         cond_as_self_attn_prefix = cond_as_self_attn_prefix,
#                         cond_dim = audio_conditioner.enc_feat_dim*tfnet.codebook_num,
#                         **context_args,
#             )

#         self.tfnet = tfnet
#         self.transformer = transformer
#         self.audio_conditioner = audio_conditioner
#         self.embed_text = partial(t5_encode_text, name = t5_name)

#         assert not (exists(audio_conditioner) and not transformer.has_condition), 'if conditioning on audio embeddings from mulan, transformer has_condition must be set to True'

#         self.unique_consecutive = unique_consecutive
#         self.pad_id = pad_id

#         self.num_quantizers = transformer.num_quantizers
#         self.eos_id = transformer.eos_id

#         self.mask_prob = mask_prob
#         self.softmax = nn.Softmax(dim=1)

#     @property
#     def device(self):
#         return next(self.parameters()).device

#     @eval_decorator
#     @torch.no_grad()
#     @beartype
#     def generate_frame(
#         self,
#         *,
#         text: Optional[List[str]] = None,
#         text_embeds = None,
#         enrollment_wave = None,
#         enrollment_embeds = None,
#         noisy_wave = None,
#         clean_wave = None,
#         clean_token_ids = None,
#         batch_size = 1,
#         cond_scale = 3.,
#         filter_thres = 0.9,
#         temperature = 1.,
#         reconstruct_wave = False,
#         **kwargs
#     ):
#         device = self.device
#         accuracy = None
#         target_wav = None

#         # derive tfnet ids from the input wave

#         if exists(clean_wave):
#             assert not exists(clean_token_ids)
#             assert exists(self.tfnet)
#             _, indices, _ = self.tfnet(clean_wave.unsqueeze(0), return_encoded = True)
#             ids = indices[..., :self.num_quantizers]
#             ids = rearrange(ids, 'b n q -> b (n q)', q = self.num_quantizers)
#         elif exists(clean_token_ids):
#             ids = clean_token_ids
#         else:
#             ids = torch.empty((batch_size, 0), dtype = torch.long, device = device)

#         assert exists(noisy_wave)
#         assert exists(self.audio_conditioner)
#         cond_embeds_orig = self.audio_conditioner(noisy_wave)
#         cond_embeds_orig = rearrange(cond_embeds_orig, 'b q t c -> b t (q c)')
#         pad_len = self.audio_conditioner.pad_len
#         max_time_steps = cond_embeds_orig.shape[-2] + 1
#         cond_embeds = cond_embeds_orig

#         # # derive text embeddings if needed
#         # if not exists(text_embeds) and exists(text):
#         #     with torch.no_grad():
#         #         text_embeds = self.transformer.embed_text(text, output_device = device)

#         if self.unique_consecutive:
#             ids = batch_unique_consecutive(ids, pad_value=self.pad_id)

#         # initialize
#         init_time_step = ids.shape[1]
#         sampled_token_ids = ids.clone()

#         for time_step in tqdm(range(init_time_step, max_time_steps), desc = 'generating clean codec feature'):
#             is_last_step = time_step == (max_time_steps - 1)
#             if self.cond_as_self_attn_prefix_causal:
#                 cond_embeds = cond_embeds_orig[:,:min(time_step + 1, max_time_steps - 1)]

#             logits = self.transformer.forward_with_cond_scale(
#                     token_ids = sampled_token_ids,
#                     text_embeds = cond_embeds,
#                     cond_scale = cond_scale,
#                     **kwargs
#             )

#             logits = rearrange(logits, 'b (t q) c -> b t q c', q = self.num_quantizers)

#             last_logits = logits[:, -1]
#             last_logits = rearrange(last_logits, 'b q c -> (b q) c')

#             if not is_last_step:
#                 last_logits[:, -1] = float('-inf') # prevent from eos if not last quantizer step, but move this to masking logic within the transformer at some point, for both training and eval

#             filtered_logits = top_k(last_logits, thres = filter_thres)
#             sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
#             sampled = rearrange(sampled, '(b q) -> b q', q = self.num_quantizers)
#             sampled_token_ids = torch.cat((sampled_token_ids, sampled), dim = -1)


#         sampled_token_ids = mask_out_after_eos_id(sampled_token_ids, self.eos_id, keep_eos = False)
#         sampled_token_ids = rearrange(sampled_token_ids[...,:-self.num_quantizers], 'b (t q) -> b t q', q = self.num_quantizers)

#         if not reconstruct_wave:
#             return sampled_token_ids

#         assert exists(self.tfnet)

#         wav, _ = self.tfnet.decode_from_codebook_indices(sampled_token_ids, pad_len=pad_len)
#         return wav, accuracy, target_wav

#     @eval_decorator
#     @torch.no_grad()
#     @beartype
#     def generate_batch(
#         self,
#         *,
#         clean_token_ids = None,
#         noisy_wave = None,
#         clean_wave= None,
#         text = None,
#         text_embeds = None,
#         enrollment_wave = None,
#         enrollment_embeds = None,
#         return_acc = False,
#         return_target = False,
#         sample_criteria: Optional[Union["max", "gumbel"]] = "max",
#         filter_thres = 0.9,
#         temperature = 1.,
#         split_win_len = -1,
#         split_hop_len = -1,
#         **kwargs
#     ):
#         assert exists(noisy_wave), 'noisy raw waveform (raw_wave) is given'
#         assert exists(clean_wave) or exists(clean_token_ids), 'either clean raw waveform (raw_wave) is given or token ids are given'

#         assert not all(map(exists, (clean_wave, clean_token_ids)))

#         pad_len = 0
#         use_win_infer = False
#         if split_win_len > 0:
#             use_win_infer = True

#         if exists(self.audio_conditioner):
#             assert exists(noisy_wave)
#             assert not exists(text) and not exists(text_embeds)
#             text_embeds = self.audio_conditioner(noisy_wave)
#             text_embeds = rearrange(text_embeds, 'b q t c -> b t (q c)')

#         if not exists(clean_token_ids):
#             assert exists(self.tfnet), 'TFNet must be provided if given raw wave for training'

#             with torch.no_grad():
#                 self.tfnet.eval()
#                 _, indices, _ = self.tfnet(clean_wave, return_encoded = True)
#                 clean_token_ids = indices[..., :self.num_quantizers]
#                 pad_len = self.tfnet.pad_len
        
#         # split into segments
#         clean_token_ids_orig = clean_token_ids.clone()
#         if use_win_infer:
#             b = clean_token_ids.shape[0]
#             token_len = clean_token_ids.shape[1]
#             seq_size = math.ceil((token_len - split_win_len) / split_hop_len) + 1
#             target_token_len = (seq_size - 1) * split_hop_len + split_win_len
#             split_pad_len = target_token_len - token_len
#             clean_token_ids = F.pad(clean_token_ids, (0, 0, 0, split_pad_len))
#             clean_token_ids = clean_token_ids.unfold(dimension=1, size=split_win_len, step=split_hop_len)
#             clean_token_ids = rearrange(clean_token_ids, 'b n q t -> (b n) t q')
#             if exists(text_embeds):
#                 text_embeds = F.pad(text_embeds, (0, 0, 0, split_pad_len))
#                 text_embeds = text_embeds.unfold(dimension=1, size=split_win_len, step=split_hop_len)
#                 text_embeds = rearrange(text_embeds, 'b n c t -> (b n) t c')

#         clean_token_ids = rearrange(clean_token_ids, 'b ... -> b (...)')
#         clean_token_ids = append_eos_id(clean_token_ids, self.transformer.eos_id)

#         if self.unique_consecutive:#maybe not reasonable for acoustic tokens
#             clean_token_ids = batch_unique_consecutive(clean_token_ids, pad_value = self.pad_id)

#         if return_acc:
#             labels = clean_token_ids.clone()
#             clean_token_ids = clean_token_ids[:, :-1]

#         # forgetful causal mask - structured dropout
#         self_attn_mask = None
#         if self.mask_prob > 0 and self.training:
#             clean_token_ids_tmp = rearrange(clean_token_ids, 'b (t q) -> b q t', q = self.num_quantizers)
#             b, _, t = clean_token_ids_tmp.shape
#             self_attn_mask = generate_mask_with_prob((b, t), self.mask_prob, device = clean_token_ids.device)

#         logits = self.transformer(
#             token_ids = clean_token_ids,
#             self_attn_mask = self_attn_mask,
#             text = text,
#             text_embeds = text_embeds,
#             cond_drop_prob = 0,
#             **kwargs
#         )
#         logits[:, :-self.num_quantizers, -1] = float('-inf') # prevent from eos if not last quantizer step, but move this to masking logic within the transformer at some point, for both training and eval
#         logits = rearrange(logits, 'b n c -> b c n')

#         if sample_criteria == "gumbel":
#             filtered_logits = top_k(logits, thres = filter_thres)
#             sampled_token_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
#             sampled_token_ids = rearrange(sampled_token_ids, 'b -> b 1')
#         else:
#             sampled_token_ids = torch.argmax(self.softmax(logits), 1)
#             sampled_token_ids = rearrange(sampled_token_ids[:,:-self.num_quantizers], 'b (t q) -> b t q', q=self.num_quantizers)

#         # merge segments into one clip
#         # if use_win_infer:
#         #     sampled_token_ids = rearrange(sampled_token_ids, '(b n) t q -> b n t q', b = b)
#         #     sampled_token_ids = torch.cat([sampled_token_ids[:, 0], sampled_token_ids[:, 1:, -1]], dim=1)
#         #     sampled_token_ids = sampled_token_ids[:,:token_len]

#         assert exists(self.tfnet)
#         wav, _ = self.tfnet.decode_from_codebook_indices(sampled_token_ids, pad_len=pad_len)
#         # accuracy
#         accuracy = None
#         if return_acc:
#             logits = logits[..., :-(self.num_quantizers - 1)]
#             accuracy = torch.mean((torch.argmax(self.softmax(logits), 1) == labels).type(torch.float))        
#         target_wav = None
#         if return_target:
#             target_wav, _ = self.tfnet.decode_from_codebook_indices(clean_token_ids_orig, pad_len=pad_len)

#         return wav, accuracy, target_wav

#     def forward(
#         self,
#         *,
#         clean_token_ids = None,
#         noisy_wave = None,
#         clean_wave= None,
#         text = None,
#         text_embeds = None,
#         enrollment_wave = None,
#         enrollment_embeds = None,
#         return_loss = False,
#         return_wave = False,
#         **kwargs
#     ):
#         assert exists(noisy_wave), 'noisy raw waveform (raw_wave) is given'
#         assert exists(clean_wave) or exists(clean_token_ids), 'either clean raw waveform (raw_wave) is given or token ids are given'
#         assert not all(map(exists, (clean_wave, clean_token_ids)))
        
#         if exists(self.audio_conditioner):
#             assert exists(noisy_wave)
#             assert not exists(text) and not exists(text_embeds)
#             text_embeds = self.audio_conditioner(noisy_wave)
#             text_embeds = rearrange(text_embeds, 'b q t c -> b t (q c)')

#         if not exists(clean_token_ids):
#             assert exists(self.tfnet), 'TFNet must be provided if given raw wave for training'

#             with torch.no_grad():
#                 self.tfnet.eval()
#                 _, indices, _ = self.tfnet(clean_wave, return_encoded = True)
#                 clean_token_ids = indices[..., :self.num_quantizers]
#                 pad_len = self.tfnet.pad_len

#         clean_token_ids = rearrange(clean_token_ids, 'b ... -> b (...)')

#         clean_token_ids = append_eos_id(clean_token_ids, self.transformer.eos_id)

#         if self.unique_consecutive:#maybe not reasonable for acoustic tokens
#             clean_token_ids = batch_unique_consecutive(clean_token_ids, pad_value = self.pad_id)

#         if return_loss:
#             labels = clean_token_ids.clone()
#             clean_token_ids = clean_token_ids[:, :-1]

#         # forgetful causal mask - structured dropout
#         self_attn_mask = None
#         if self.mask_prob > 0 and self.training:
#             clean_token_ids_tmp = rearrange(clean_token_ids, 'b (t q) -> b q t', q = self.num_quantizers)
#             b, _, t = clean_token_ids_tmp.shape
#             self_attn_mask = generate_mask_with_prob((b, t), self.mask_prob, device = clean_token_ids.device)

#         logits = self.transformer(
#             token_ids = clean_token_ids,
#             self_attn_mask = self_attn_mask,
#             text = text,
#             text_embeds = text_embeds,
#             cond_drop_prob = 0,
#             **kwargs
#         )

#         logits = logits[:,:-(self.num_quantizers - 1),:]
#         # whether to early return the logits

#         if not return_loss:
#             return logits

#         logits = rearrange(logits, 'b n c -> b c n')

#         loss = F.cross_entropy(
#             logits,
#             labels,
#             ignore_index = self.pad_id
#         )
#         # accuracy
#         accuracy = torch.mean((torch.argmax(self.softmax(logits), 1) == labels).type(torch.float))

#         wav = None
#         if return_wave:
#             if self.training:
#                 self.tfnet.train()
#             wav, _ = self.tfnet.decode_from_codebook_logits(logits, pad_len=pad_len)

#         return loss, accuracy, wav


if __name__=="__main__":
        import yaml
        with open("./tfnet/config_6k_tfnetv2_20msvq_hop5_combine4_rd_multi_lingual.yaml", "r") as config_file:  
            config = yaml.safe_load(config_file) 
        tfnet_ckpt = "/mnt/users/jiazhijun/models/tfcodec/tfnet_v2i_vqvae_combineVQ-iter-514000-val1-0.348327-entropy-118.968-mel-0.067548-recon-0.012822.ckpt"
        config_tfnet = config
        
        if config_tfnet['model_type'] =='tfnetv2_interleave_vqvae':
            tfnet = TFNetV2_interl_VQVAE(config=config_tfnet, )
        elif config_tfnet['model_type'] =='tfnetv4_interleave_vqvae':
            tfnet = TFNetV4_interl_VQVAE(config=config_tfnet, sampling_rate=config_tfnet["sampling_rate"], win_len=config_tfnet["dft_size"], hop_len=int(config_tfnet["dft_size"] * config_tfnet["hop_vqvae"]), n_fft=config_tfnet["dft_size"])

        # load with tfnet_ckpt
        tfnet.load(tfnet_ckpt)
        tfnet.freeze_encoder()
        tfnet.freeze_codebook()
        import librosa
        wav1, sr = librosa.load("/mnt/users/jiazhijun/data/test_vc/test_dns_v4/source/car1.wav", sr=16000)
        wav2, sr = librosa.load("/mnt/users/jiazhijun/data/test_vc/test_dns_v4/source/car3.wav", sr=16000)

        wav1 = torch.from_numpy(wav1)
        # wav1 = wav1.unsqueeze(0)
        wav2 = torch.from_numpy(wav2)
        # wav2 = wav2.unsqueeze(0)

        wav_total = []
        wav_total.append(wav1)
        wav_total.append(wav2)
        # wav1 = torch.cat((wav1, wav1), dim=0)
        from torch.nn.utils.rnn import pad_sequence
        wav_total = pad_sequence(wav_total, batch_first=True, padding_value=0)

        # print(wav_total.shape)
        # quit()
        with torch.no_grad():
            tfnet.eval()
            # with open(f'clean_samples_0.txt', 'w') as f:  
            #     for row in wav_total[0]:  
            #         f.write(' '.join([str(elem) for elem in row]) + '\n')  
                _, indices, _ = tfnet(wav_total)
            clean_token_ids = indices[..., :16]
        
        idx = 0 
        for clean_token_id in clean_token_ids:
            print(clean_token_id)
            # print(clean_token_ids.shape)
            # 假设 clean_token_id 是一个 PyTorch 张量  
            clean_token_id_np = clean_token_id.numpy()
            
            # 将 NumPy 数组写入文件  
            with open(f'clean_token_id_{idx}.txt', 'w') as f:  
                for row in clean_token_id_np:  
                    f.write(' '.join([str(elem) for elem in row]) + '\n')  
            idx+=1

        # clean_token_ids = rearrange(clean_token_ids, 'b ... -> b (...)')
        # clean_token_ids = append_eos_id(clean_token_ids, 1024)
        # token_ids = clean_token_ids
        # codebook_size = tfnet.codebook_size[0]
        # num_quantizers = tfnet.codebook_num
        # dim = 1024
        # embedding_dim = dim // num_quantizers

        # codebook_size_with_eos = codebook_size + 1
        # offsets = codebook_size * torch.arange(num_quantizers, device = token_ids.device)
        # dd = ceil_div(token_ids.shape[-1], num_quantizers)
        # offsets = repeat(offsets, 'q -> 1 (n q)', n = dd)
        # offsets = offsets[:, :token_ids.shape[-1]]
        # token_ids = token_ids + offsets
        # # quit()
        # embedding = nn.Embedding(num_quantizers * codebook_size_with_eos, embedding_dim)
        # tokens = embedding(token_ids)#b (t q) c
        # tokens = rearrange(tokens, 'b (t q) c -> b t (q c)', q = num_quantizers)

        # print(tokens.shape)
        # quit()
        idx=0
        for clean_token_id in clean_token_ids:
            clean_token_id =clean_token_id.unsqueeze(0)
            with open(f'clean_token_id_{idx}.txt', 'w') as f:  
                for row in clean_token_id_np:  
                    f.write(' '.join([str(elem) for elem in row]) + '\n')  

            # wav, _ = tfnet.decode_from_codebook_indices(clean_token_id, pad_len=0)
            # wav = wav.detach().numpy()[0]
            # import soundfile as sf
            # sf.write(f"/mnt/users/jiazhijun/valle_23_4_22/tfnet/test_codec_{idx}.wav", wav, 16000)
            idx+=1




 

    