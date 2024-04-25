
from pathlib import Path

import torch
from torch import nn
from einops import rearrange
from einops.parsing import ParsedExpression
from einops._backends import get_backend
from typing import List, Union, TypeVar, Tuple, Sequence
from einops import EinopsError
from einops import rearrange, repeat

import joblib
from functools import partial, wraps
import fairseq
import torch.nn.functional as F
from torchaudio.transforms import Resample as resample
from beartype.typing import Union, List
from beartype import beartype
from transformers import T5Tokenizer, T5EncoderModel, T5Config
from torch.nn.utils.rnn import pad_sequence

DEFAULT_T5_NAME = 'google/t5-v1_1-base'
MAX_LENGTH = 256
T5_CONFIGS = {}
Tensor = TypeVar('Tensor')
Shape = Union[Tuple[int, ...], List[int]]
def ceil_div(numer, denom):
    return (numer + denom - 1) // denom

def exists(val):
    return val is not None
def grad_shrink(t, alpha = 0.1):
    return t * alpha + t.detach() * (1 - alpha)
def default(val, d):
    return val if exists(val) else d
def get_embeds(
    embeddings: nn.Embedding,
    codes: torch.Tensor,
    pad_id = -1,
    return_mask = False,
    mask_pad_pos_to = 0
):
    pad_mask = codes == pad_id
    codes_without_pad = codes.masked_fill(pad_mask, 0) # just retrieve first code as dummy
    embeds = embeddings(codes_without_pad)

    if exists(mask_pad_pos_to):
        embeds = embeds.masked_fill(rearrange(pad_mask, '... -> ... 1'), mask_pad_pos_to)

    if return_mask:
        return embeds, ~pad_mask

    return embeds

def prod(x: Shape) -> int:
    result = 1
    for i in x:
        result *= i
    return result

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)





import logging
logging.root.setLevel(logging.ERROR)


def round_down_nearest_multiple(num, divisor):
    return num // divisor * divisor

def curtail_to_multiple(t, mult, from_left = False):
    data_len = t.shape[-1]
    rounded_seq_len = round_down_nearest_multiple(data_len, mult)
    seq_slice = slice(None, rounded_seq_len) if not from_left else slice(-rounded_seq_len, None)
    return t[..., seq_slice]

class HubertWithKmeans(nn.Module):
    """
    checkpoint and kmeans can be downloaded at https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
    or you can train your own
    """

    def __init__(
        self,
        checkpoint_path,
        kmeans_path,
        target_sample_hz = 16000,
        seq_len_multiple_of = None,
        output_layer = 9,
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of
        self.output_layer = output_layer

        model_path = Path(checkpoint_path)
        kmeans_path = Path(kmeans_path)

        assert model_path.exists(), f'path {checkpoint_path} does not exist'
        assert kmeans_path.exists(), f'path {kmeans_path} does not exist'

        checkpoint = torch.load(checkpoint_path)
        # print(checkpoint["cfg"]["feature_grad_mult"])
        # checkpoint["cfg"]["model"]["feature_grad_mult"] = 1
        load_model_input = {checkpoint_path: checkpoint}
        model, *_ = fairseq.checkpoint_utils.load_model_ensemble_and_task(load_model_input)

        self.model = model[0]
        self.model.feature_grad_mult = 0
        self.model.eval()

        kmeans = joblib.load(kmeans_path)
        self.kmeans = kmeans

    @property
    def groups(self):
        return 1

    @property
    def codebook_size(self):
        return self.kmeans.n_clusters

    # @torch.no_grad()
    def forward(
        self,
        wav_input,
        flatten = True,
        input_sample_hz = None,
        quantize = True,
        padding_mask = None,
    ):
        device = wav_input.device

        if exists(input_sample_hz):
            transform = resample(input_sample_hz, self.target_sample_hz)
            wav_input = transform(wav_input)

        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)

        embed = self.model(
            wav_input,
            features_only = True,
            mask = False,  # thanks to @maitycyrus for noticing that mask is defaulted to True in the fairseq code
            output_layer = self.output_layer,
            padding_mask = padding_mask,
        )

        # embed = torch.cat([embed['x']])
        if quantize:
            embed, packed_shape = pack([embed['x']], '* d')
            self.kmeans.verbose = False
            codebook_indices = self.kmeans.predict(embed.cpu().detach().numpy())

            codebook_indices = torch.from_numpy(codebook_indices).to(device).long()

            if flatten:
                return codebook_indices

            codebook_indices, = unpack(codebook_indices, packed_shape, '*')
            return codebook_indices
        else:
            return embed['x']

def analyze_pattern(pattern: str, opname: str) -> Tuple[int, int, int]:
    # Maybe some validation of identifiers?
    axes = pattern.split()
    axes_set = set(axes)
    if len(axes) != len(axes_set):
        raise EinopsError(f'Duplicates in axes names in {opname}(..., "{pattern}")')
    if '*' not in axes_set:
        raise EinopsError(f'No *-axis in {opname}(..., "{pattern}")')
    for axis in axes:
        if axis != '*':
            is_valid, reason = ParsedExpression.check_axis_name_return_reason(axis)
            if not is_valid:
                raise EinopsError(f'Invalid axis name {axis} in {opname}(..., "{pattern}")')
    n_axes_before = axes.index('*')
    n_axes_after = len(axes) - n_axes_before - 1
    min_axes = n_axes_before + n_axes_after
    return n_axes_before, n_axes_after, min_axes

def pack(tensors: Sequence[Tensor], pattern: str) -> Tuple[Tensor, List[Shape]]:
    n_axes_before, n_axes_after, min_axes = analyze_pattern(pattern, 'pack')

    # packing zero tensors is illegal
    backend = get_backend(tensors[0])

    reshaped_tensors: List[Tensor] = []
    packed_shapes: List[Shape] = []
    for i, tensor in enumerate(tensors):
        shape = backend.shape(tensor)
        if len(shape) < min_axes:
            raise EinopsError(f'packed tensor #{i} (enumeration starts with 0) has shape {shape}, '
                              f'while pattern {pattern} assumes at least {min_axes} axes')
        axis_after_packed_axes = len(shape) - n_axes_after
        packed_shapes.append(shape[n_axes_before:axis_after_packed_axes])
        reshaped_tensors.append(
            backend.reshape(tensor, (*shape[:n_axes_before], -1, *shape[axis_after_packed_axes:]))
        )

    return torch.cat(reshaped_tensors, dim=n_axes_before), packed_shapes

def unpack(tensor: Tensor, packed_shapes: List[Shape], pattern: str) -> List[Tensor]:

    n_axes_before, n_axes_after, min_axes = analyze_pattern(pattern, opname='unpack')

    backend = get_backend(tensor)
    input_shape = backend.shape(tensor)
    if len(input_shape) != n_axes_before + 1 + n_axes_after:
        raise EinopsError(f'unpack(..., {pattern}) received input of wrong dim with shape {input_shape}')

    unpacked_axis: int = n_axes_before

    lengths_of_composed_axes: List[int] = [
        -1 if -1 in p_shape else prod(p_shape)
        for p_shape in packed_shapes
    ]

    n_unknown_composed_axes = sum(x == -1 for x in lengths_of_composed_axes)
    if n_unknown_composed_axes > 1:
        raise EinopsError(
            f"unpack(..., {pattern}) received more than one -1 in {packed_shapes} and can't infer dimensions"
        )

    # following manipulations allow to skip some shape verifications
    # and leave it to backends

    # [[], [2, 3], [4], [-1, 5], [6]] < examples of packed_axis
    # split positions when computed should be
    # [0,   1,      7,   11,      N-6 , N ], where N = length of axis
    split_positions = [0] * len(packed_shapes) + [input_shape[unpacked_axis]]
    if n_unknown_composed_axes == 0:
        for i, x in enumerate(lengths_of_composed_axes[:-1]):
            split_positions[i + 1] = split_positions[i] + x
    else:
        unknown_composed_axis: int = lengths_of_composed_axes.index(-1)
        for i in range(unknown_composed_axis):
            split_positions[i + 1] = split_positions[i] + lengths_of_composed_axes[i]
        for j in range(unknown_composed_axis + 1, len(lengths_of_composed_axes))[::-1]:
            split_positions[j] = split_positions[j + 1] - lengths_of_composed_axes[j]

    shape_start = input_shape[:unpacked_axis]
    shape_end = input_shape[unpacked_axis + 1:]
    slice_filler = (slice(None, None),) * unpacked_axis
    try:
        return [
            backend.reshape(
                # shortest way slice arbitrary axis
                tensor[(*slice_filler, slice(split_positions[i], split_positions[i + 1]))],
                (*shape_start, *element_shape, *shape_end)
            )
            for i, element_shape in enumerate(packed_shapes)
        ]
    except BaseException:
        # this hits if there is an error during reshapes, which means passed shapes were incorrect
        raise RuntimeError(f'Error during unpack(..., "{pattern}"): could not split axis of size {split_positions[-1]}'
                           f' into requested {packed_shapes}')





class FairseqVQWav2Vec(nn.Module):
    """
    checkpoint path can be found at https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md#vq-wav2vec
    specifically download the kmeans model for now
    $ wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt
    """

    def __init__(
        self,
        checkpoint_path,
        target_sample_hz = 24000,
        seq_len_multiple_of = None
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of

        path = Path(checkpoint_path)
        assert path.exists(), f'path {checkpoint_path} does not exist'

        checkpoint = torch.load(checkpoint_path)
        load_model_input = {checkpoint_path: checkpoint}
        model, *_ = fairseq.checkpoint_utils.load_model_ensemble_and_task(load_model_input)

        self.model = model[0]
        self.model.eval()

        assert hasattr(self.model, 'vector_quantizer') and hasattr(self.model.vector_quantizer, 'embedding'), 'the vq wav2vec model does not seem to be valid'

    @property
    def groups(self):
        return self.model.vector_quantizer.groups

    @property
    def codebook_size(self):
        return self.model.vector_quantizer.embedding.shape[0]

    @torch.no_grad()
    def forward(
        self,
        wav_input,
        flatten = True,
        input_sample_hz = None
    ):
        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)

        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)

        embed = self.model.feature_extractor(wav_input)
        _, codebook_indices = self.model.vector_quantizer.forward_idx(embed)

        if not flatten:
            return codebook_indices

        return rearrange(codebook_indices, 'b ... -> b (...)')

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

class AudioConditionerBase(nn.Module):
    pass



def get_tokenizer(name):
    tokenizer = T5Tokenizer.from_pretrained(name)
    return tokenizer

def get_model(name):
    model = T5EncoderModel.from_pretrained(name)
    return model


def get_model_and_tokenizer(name):
    global T5_CONFIGS

    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()

    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = get_model(name)

    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']

@beartype
def t5_encode_text(
    texts: Union[str, List[str]],
    name = DEFAULT_T5_NAME,
    output_device = None
):
    if isinstance(texts, str):
        texts = [texts]

    t5, tokenizer = get_model_and_tokenizer(name)

    if torch.cuda.is_available():
        t5 = t5.cuda()

    device = next(t5.parameters()).device

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = 'pt',
        padding = 'longest',
        max_length = MAX_LENGTH,
        truncation = True
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    t5.eval()

    with torch.no_grad():
        output = t5(input_ids = input_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask[..., None].bool()

    if not exists(output_device):
        encoded_text = encoded_text.masked_fill(~attn_mask, 0.)
        return encoded_text

    encoded_text.to(output_device)
    attn_mask.to(output_device)

    encoded_text = encoded_text.masked_fill(~attn_mask, 0.)
    return encoded_text


def get_encoded_dim(name):
    if name not in T5_CONFIGS:
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config = config)

    elif "config" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["config"]

    elif "model" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["model"].config

    else:
        raise ValueError(f'unknown t5 name {name}')

    return config.d_model



class RelativePositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        layers = 3
    ):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(1, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, n, m=None):
        device = self.device
        pos = torch.arange(n, device=device)
        if m is not None:
            pos2 = torch.arange(m, device=device)
            rel_pos = (rearrange(pos, 'i -> i 1') - rearrange(pos2, 'j -> 1 j'))
            rel_pos += (m - 1)
            x = torch.arange(-m + 1, n, device=device).float()
        else:
            rel_pos = (rearrange(pos, 'i -> i 1') - rearrange(pos, 'j -> 1 j'))
            rel_pos += (n - 1)
            x = torch.arange(-n + 1, n, device=device).float()
        x = rearrange(x, '... -> ... 1')

        for layer in self.net:
            x = layer(x)

        x = x[rel_pos]
        return rearrange(x, 'i j h -> h i j')


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class CausalDSConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ds_conv = nn.Conv1d(dim, dim, 3, bias = False, groups = dim)

    def forward(self, x):
        x = rearrange(x, 'b n c -> b c n')
        x = F.pad(x, (2, 0))
        x = self.ds_conv(x)
        return rearrange(x, 'b c n -> b n c')

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x
def FeedForward(dim, mult = 4, dropout = 0.1):
    inner_dim = int(dim * 2 * mult / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        CausalDSConv(inner_dim * 2),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )


def generate_mask_with_prob(shape, mask_prob, device):
    seq = shape[-1]
    rand = torch.randn(shape, device = device)
    rand[:, 0] = -torch.finfo(rand.dtype).max
    num_mask = min(int(seq * mask_prob), seq - 1)
    indices = rand.topk(num_mask, dim = -1).indices
    mask = ~torch.zeros(shape, device = device).scatter(1, indices, 1.).bool()
    return mask


def maybe(fn):
    if not exists(fn):
        return always(None)

    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner


def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def l2norm(t):
    return F.normalize(t, dim = -1)


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob


def mask_out_after_eos_id_wgroup(t, eos_id, num_groups, mask_value=-1, keep_eos=True):  # [B, T, G]
    eos_mask = (t == eos_id).float()
    eos_mask = eos_mask.sum(dim=-1)  # [B, T]

    if keep_eos:
        eos_mask = F.pad(eos_mask, (1, -1))

    after_eos_mask = eos_mask.cumsum(dim=-1) > 0
    after_eos_mask = repeat(after_eos_mask, 'b t -> b t g', g=num_groups)
    return t.masked_fill(after_eos_mask, mask_value)


def all_rows_have_eos_id_wgroup(t, eos_id):  # [B, T, G]
    eos_mask = (t == eos_id)
    if eos_mask.dim() == 3:
        eos_mask = rearrange(eos_mask, 'b t g -> b (t g)')
    return torch.any(eos_mask, dim=-1).all()


def append_eos_id_wgroup(ids, eos_id):  # [B, T, G]
    b, g, device = ids.shape[0], ids.shape[-1], ids.device
    eos_ids = torch.ones(1, device=device).long() * eos_id
    eos_ids = repeat(eos_ids, '1 -> b 1 g', b=b, g=g)
    ids = torch.cat((ids, eos_ids), dim=1)
    return ids


def batch_unique_consecutive_wgroup(t, pad_value=0.):  # [B, T, G]
    unique_arr = [torch.unique_consecutive(el, dim=0) for el in t.unbind(dim=0)]
    return pad_sequence(unique_arr, batch_first=True, padding_value=pad_value)


# relabel token_ids according to group for getting embeddings g*c+1
def convert_token_ids_for_embeddings(x, g, c, eos_id, pad_id):
    ids = x.clone()
    b, t1 = x.shape[0], x.shape[1]
    ids = ids.reshape((b, -1, g))
    for b1 in range(b):
        for t1 in range(t1 // g):
            for g1 in range(g):
                if ids[b1, t1, g1] < eos_id and ids[b1, t1, g1] > pad_id:
                    ids[b1, t1, g1] = ids[b1, t1, g1] + g1 * c
    return ids.reshape((b, -1))

# same as convert_token_ids_for_embeddings except for eos_id
def convert_token_ids_for_embeddings2(x, g, c, eos_id, pad_id, share_eos=False):
    offsets = c * torch.arange(g, device=x.device)
    offsets = repeat(offsets, 'q -> 1 (n q)', n=ceil_div(x.shape[-1], g))
    offsets = offsets[:, :x.shape[-1]]
    x_offset = x + offsets
    x_offset[x==pad_id]=pad_id
    if share_eos:
        x_offset[x == eos_id] = eos_id
    return x_offset


def append_eos_id_wgroup2(ids,eos_id,pad_id,atten_mask):
    b, g, device = ids.shape[0], ids.shape[-1], ids.device
    eos_ids = torch.ones(1, device=device).long() * eos_id
    eos_input_ids = torch.cat((ids.clone(), repeat(eos_ids, '1 -> b 1 g', b=b, g=g)), dim=1)
    last_token_positions = atten_mask.sum(dim=1) - 1
    eos_positions = last_token_positions + 1
    eos_ids = repeat(eos_ids, '1 -> g', g=g)
    pad_ids = torch.ones(1, device=device).long() * pad_id
    pad_ids = repeat(pad_ids, '1 -> g', g=g)
    for i in range(ids.size(0)):
        eos_input_ids[i, eos_positions[i],:] = eos_ids
        eos_input_ids[i, eos_positions[i] + 1:,:] = pad_ids
    return eos_input_ids


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)