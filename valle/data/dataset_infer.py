# Copyright      2023                           (authors: Feiteng Li)
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
modified from lhoste.dataset.speech_synthesis.py
"""

from typing import Callable, Dict, List, Sequence, Union
import numpy as np
import torch
from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_audio
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import ifnone
from torch.nn.utils.rnn import pad_sequence
from valle.data.collation import TextTokenCollater
import random
import math

class SpeechSynthesisDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech synthesis(e.g. TTS) task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
            'audio_lens': (B, ) int tensor
            'text': str
            'audio_features': (B x NumFrames x NumFeatures) float tensor
            'audio_features_lens': (B, ) int tensor
            'text_tokens': (B x NumTextTokens) long tensor
            'text_tokens_lens': (B, ) int tensor
        }
    """

    def __init__(
        self,
        text_token_collater: TextTokenCollater,
        semantic_token_collater: TextTokenCollater,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        feature_input_strategy: BatchIO = PrecomputedFeatures(),
        feature_transforms: Union[Sequence[Callable], Callable] = None,
        semantic_depup=False,
        args = None,
        is_train = 1,
    ) -> None:
        super().__init__()
        self.is_train =is_train
        self.args = args
        self.text_token_collater = text_token_collater
        self.semantic_token_collater = semantic_token_collater
        self.cut_transforms = ifnone(cut_transforms, [])
        self.feature_input_strategy = feature_input_strategy
        if feature_transforms is None:
            feature_transforms = []
        elif not isinstance(feature_transforms, Sequence):
            feature_transforms = [feature_transforms]

        assert all(
            isinstance(transform, Callable) for transform in feature_transforms
        ), "Feature transforms must be Callable"
        self.feature_transforms = feature_transforms
        self.semantic_depup = semantic_depup
        self.num_quantizers = self.args.num_quantizers

        if self.args.semantic_type==0:
            assert self.args.pret_token==500
        elif self.args.semantic_type==1:
            assert self.args.pret_token==256

    
    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        
        try:
            text_tokens, text_tokens_lens = self.text_token_collater(
                [cut.supervisions[0].custom["tokens"]["text"] for cut in cuts]
            )
        except:
            text_tokens, text_tokens_lens = torch.tensor([]), torch.tensor([])

        semantic_tokens = []
        storage_path = []
        maskd_indices_batch = []

    # input semantic token. If TTS, semantic_token need to None
        for cut in cuts:
            storage_path.append(cut.recording.sources[0].source)
            if self.args.semantic_type==0:  # hubert tokens as semantic 
                semantic_token = cut.supervisions[0].custom["tokens"]["semantic_tokens"].copy()
                
            elif self.args.semantic_type==1: # tfnet tokens as semantic 
                semantic_token = cut.supervisions[0].custom["tokens"]["semantic_tokens_tfnet"].copy()  # [[[a, b], [c, d]]]

            if self.semantic_depup is True: # semantic token 是否去重
                semantic_token=depup(semantic_token)


            semantic_tokens.append(semantic_token)

            semantic_tokens,  semantic_tokens_lens = self.semantic_token_collater( # add bos, eos, pad
                semantic_tokens
            )

        # vc tts
        return {
            "utt_id": [cut.id for cut in cuts],
            "text": [cut.supervisions[0].text for cut in cuts],
            "text_tokens": text_tokens,
            "text_tokens_lens": text_tokens_lens,
            "semantic_tokens": semantic_tokens,
            "semantic_tokens_lens": semantic_tokens_lens,
        }


def validate_for_tts(cuts: CutSet) -> None:
    validate(cuts)
    for cut in cuts:
        assert (
            len(cut.supervisions) == 1
        ), "Only the Cuts with single supervision are supported."
from collections import Counter  

def longest_common_subsequence(seq1, seq2):  
    m, n = len(seq1), len(seq2)  
    dp = [[0] * (n + 1) for _ in range(m + 1)]  
  
    for i in range(1, m + 1):  
        for j in range(1, n + 1):  
            if seq1[i - 1] == seq2[j - 1]:  
                dp[i][j] = dp[i - 1][j - 1] + 1  
            else:  
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  
  
    lcs = []  
    i, j = m, n  
    while i > 0 and j > 0:  
        if seq1[i - 1] == seq2[j - 1]:  
            lcs.append(seq1[i - 1])  
            i -= 1  
            j -= 1  
        elif dp[i - 1][j] > dp[i][j - 1]:  
            i -= 1  
        else:  
            j -= 1  
  
    return lcs[::-1]  

def depup(semantic_token):
    unique_tokens = []  
    for token in semantic_token:  
        if unique_tokens==[] or token != unique_tokens[-1]:  
            unique_tokens.append(token)
    return unique_tokens

def mark_and_count_lcs_tokens(a, lcs_ab):  
    lcs_index = 0  
    lcs_token_counts = [0 for xx in range(len(lcs_ab))]

    ll = len(a)
    i=0
    while True:
        if i==ll:
            break
        if lcs_index < len(lcs_ab) and a[i] == lcs_ab[lcs_index]:  
            lcs_token_counts[lcs_index] += 1  
            i+=1
            while i<ll and a[i]==lcs_ab[lcs_index] :
                lcs_token_counts[lcs_index] += 1 
                i+=1
            lcs_index += 1

            i-=1   
        i+=1
    return lcs_token_counts 


def update_lcs_token(a, lcs_ab, lcs_token_counts_a, lcs_token_counts_b, update_nums, update_type=0):

    # update_type 0 del 1 add
    lcs_index = 0  
    lcs_token_counts = [0 for xx in range(len(lcs_ab))]
    updated_tokens = []
    ll = len(a)
    i=0
    while True:
        if i==ll:
            break
        if lcs_index < len(lcs_ab) and a[i] == lcs_ab[lcs_index]:  
            if lcs_token_counts_a[lcs_index] > lcs_token_counts_b[lcs_index] and update_type==0: 
                can_update_nums = lcs_token_counts_a[lcs_index]- lcs_token_counts_b[lcs_index]
                if can_update_nums <= update_nums:
                    nums = lcs_token_counts_b[lcs_index]
                    update_nums-=can_update_nums
                else:
                    nums = lcs_token_counts_a[lcs_index]
            elif update_type==0:
                nums = lcs_token_counts_a[lcs_index]

            if lcs_token_counts_a[lcs_index] < lcs_token_counts_b[lcs_index] and update_type==1:
                can_update_nums = lcs_token_counts_b[lcs_index]- lcs_token_counts_a[lcs_index]
                if can_update_nums <= update_nums:
                    nums = lcs_token_counts_b[lcs_index]
                    update_nums-=can_update_nums
                else:
                    nums = lcs_token_counts_a[lcs_index]
            elif update_type==1:
                nums = lcs_token_counts_a[lcs_index]

            updated_tokens+=[lcs_ab[lcs_index]]*nums
            i+=1
            while i<ll and a[i]==lcs_ab[lcs_index] :
                i+=1
            lcs_index += 1
            i-=1   
        else:
            updated_tokens+=[a[i]]
        i+=1
    return updated_tokens, update_nums


def update_non_lcs_token(a, lcs_ab, non_lcs_token_counts_a, update_nums, update_type=0):

    lcs_index = 0  
    lcs_token_counts = [0 for xx in range(len(lcs_ab))]
    if update_type==0:
        non_lcs_token_counts_a = del_count_elements(non_lcs_token_counts_a[:], update_nums)  # 使用切片创建一个副本，以免修改原始序列
    elif update_type==1:
        non_lcs_token_counts_a = add_count_elements(non_lcs_token_counts_a[:], update_nums)  # 使用切片创建一个副本，以免修改原始序列
   
    updated_tokens = []
    ll = len(a)
    i=0
    while True:
        if i==ll:
            break
        if lcs_index < len(lcs_ab) and a[i] == lcs_ab[lcs_index]:  
            nums = non_lcs_token_counts_a[lcs_index]

            updated_tokens+=[lcs_ab[lcs_index]]*nums
            i+=1
            while i<ll and a[i]==lcs_ab[lcs_index] :
                i+=1
            lcs_index += 1
            i-=1   
        else:
            updated_tokens+=[a[i]]
        i+=1
    return updated_tokens


# bug exists
def get_non_lcs_tokens(a, lcs_ab):  
    lcs_index = 0  
    non_lcs_tokens = []
    updated_tokens = []
    ll = len(a)
    i=0
    while True:
        if i==ll:
            break
        if lcs_index < len(lcs_ab) and a[i] == lcs_ab[lcs_index]:  
            i+=1
            while i<ll and a[i]==lcs_ab[lcs_index]:
                i+=1
            lcs_index += 1

            i-=1   
        else:
            if len(non_lcs_tokens) == 0:
                non_lcs_tokens+=[a[i]]
            elif a[i]!=non_lcs_tokens[-1] or a[i-1]!=non_lcs_tokens[-1]:
                non_lcs_tokens+=[a[i]]
        i+=1
    
    lcs_index = 0  

    depup_non_lcs_tokens = non_lcs_tokens

    non_lcs_token_counts = mark_and_count_lcs_tokens(a, depup_non_lcs_tokens)

    return depup_non_lcs_tokens, non_lcs_token_counts

from heapq import heapify, heappop, heappush  
def del_count_elements(sequence, total_to_subtract):
    # 创建一个索引堆，以便知道哪个元素被减去
    index_heap = [(-val, i) for i, val in enumerate(sequence)]
    heapify(index_heap)  # 建立最大堆

    # 从最大的数字开始逐一减去1
    for _ in range(total_to_subtract):
        if not index_heap:
            break  # 如果堆为空，则停止
        # 弹出最大的数字
        max_val, max_index = heappop(index_heap)
        if sequence[max_index] > 0:  # 如果该数字已经是0，则不再减去
            sequence[max_index] -= 1  # 减去1
        if sequence[max_index] > 0:  # 如果减去1后大于0，则放回堆中
            heappush(index_heap, (-sequence[max_index], max_index))

    return sequence

def add_count_elements(sequence, total_to_add):  
    # 创建一个索引堆，以便知道哪个元素被增加  
    index_heap = [(val, i) for i, val in enumerate(sequence)]  
    heapify(index_heap)  # 建立最小堆  
  
    # 从最小的数字开始逐一增加1  
    for _ in range(total_to_add):  
        if not index_heap:  
            break  # 如果堆为空，则停止  
        # 弹出最小的数字  
        min_val, min_index = heappop(index_heap)  
        sequence[min_index] += 1  # 增加1  
        # 增加1后，放回堆中  
        heappush(index_heap, (sequence[min_index], min_index))  
  
    return sequence

def get_a_larger_b(a, b):

    lcs_ab = longest_common_subsequence(a, b)  
    lcs_ab_depup = depup(lcs_ab)
    
    # 标记序列a中属于最大公共子序列的token，并统计数量  
    lcs_token_counts_a = mark_and_count_lcs_tokens(a, lcs_ab_depup)  

    lcs_token_count_b = mark_and_count_lcs_tokens(b, lcs_ab_depup)  

    updated_tokens, update_nums = update_lcs_token(a, lcs_ab_depup, lcs_token_counts_a, lcs_token_count_b, len(a)-len(b))

    non_lcs_tokens_a, non_lcs_token_counts_a = get_non_lcs_tokens(a, lcs_ab_depup)

    final_tokens = update_non_lcs_token(updated_tokens, non_lcs_tokens_a, non_lcs_token_counts_a, update_nums)
    return final_tokens

def get_a_smaller_b(a, b):

    lcs_ab = longest_common_subsequence(a, b)  
    lcs_ab_depup = depup(lcs_ab)
    
    # 标记序列a中属于最大公共子序列的token，并统计数量  
    lcs_token_counts_a = mark_and_count_lcs_tokens(a, lcs_ab_depup)  

    lcs_token_count_b = mark_and_count_lcs_tokens(b, lcs_ab_depup)  

    updated_tokens, update_nums = update_lcs_token(a, lcs_ab_depup, lcs_token_counts_a, lcs_token_count_b, len(b)-len(a), update_type=1)

    non_lcs_tokens_a, non_lcs_token_counts_a = get_non_lcs_tokens(a, lcs_ab_depup)

    # print(f"second_update_nums:{update_nums}")
    final_tokens = update_non_lcs_token(updated_tokens, non_lcs_tokens_a, non_lcs_token_counts_a, update_nums, update_type=1)
    return final_tokens


def aligh_input2output(semantic_tokens, semantic_features):
    update_semantic_tokens = []
    for source_list, tgt_tensor in zip(semantic_tokens, semantic_features):
        tgt_list = list(tgt_tensor)
        tgt_list = [int(t.item()) for t in tgt_list]  # .numpy()方法将Tensor转换为NumPy数组，然后自动转换为Python标量或列表  

        if len(source_list) > len(tgt_list):
            semantic_tokens = get_a_larger_b(source_list, tgt_list)
        elif len(source_list) < len(tgt_list):
            semantic_tokens = get_a_smaller_b(source_list, tgt_list)
        else:
            semantic_tokens = source_list
        # bug need to repair
        if len(semantic_tokens)!=len(tgt_list):
            if len(semantic_tokens) < len(tgt_list):
                semantic_tokens += [semantic_tokens[-1]]*(len(tgt_list)-len(semantic_tokens))
            elif len(semantic_tokens) > len(tgt_list):
                semantic_tokens = semantic_tokens[:len(tgt_list)]

        update_semantic_tokens.append(semantic_tokens)
    return update_semantic_tokens