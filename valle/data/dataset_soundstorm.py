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


    def mask_transform(self, semantic_token, prob=0.5, mask_token=500):

        # 设置替换为501的概率（0到1之间）  
        replacement_probability = 0.5  
        # 遍历列表并根据概率替换元素  
        for i, token in enumerate(semantic_token):  
            if random.random() < replacement_probability:  
                semantic_token[i] = mask_token  
        return semantic_token

    def apply_masking_strategy(self, item, mask_token=500, random_token=500, force_mask=False):      
            if force_mask:  
                return mask_token, True  
            else:  
                prob = random.random()      
                if prob < 0.8:   
                    if self.args.semantic_type==0:
                        return mask_token, True
                    elif self.args.semantic_type==1:
                        return [mask_token, mask_token], True
                else:      
                    prob2 = random.random()      
                    if prob2 < 0.5:      
                        return item, False  
                    else:
                        if self.args.semantic_type==0:
                            return random.randint(0, random_token), True
                        elif self.args.semantic_type==1:
                            return [random.randint(0, random_token), random.randint(0, random_token)], True
    
    def mask_transform_v2(self, input_list, masking_ratio=0.15, mask_token=500, random_token=500): 
        

        num_masked = int(len(input_list) * masking_ratio)    
        masked_indices = random.sample(range(len(input_list)), num_masked)
        masked_list = []
        masked_final_indices = []  
        for i, item in enumerate(input_list):  
            if i not in masked_indices:
                masked_final_indices.append(0)
                masked_list.append(item)  
            else:  
                
                masked_item, is_masked = self.apply_masking_strategy(item, mask_token, random_token=random_token)  
                masked_list.append(masked_item)

                if is_masked:
                    masked_final_indices.append(1)
                else:
                    masked_final_indices.append(0)
        
        if self.args.prepend_bos is True:
            masked_final_indices = masked_final_indices+[0]
        else:
            masked_final_indices = masked_final_indices[1:]+[0]                                 

        return masked_list, masked_final_indices  

    def del_transform(self, numbers, prob=0.5, mask_token=500):  
        result = []  
        for number in numbers:  
            if random.random() >= prob:
                result.append(number)
        return result

    def del_transform_v2(self, numbers, prob=0.5):  
        num_to_delete = int(len(numbers) * prob)  
        indices_to_delete = random.sample(range(len(numbers)), num_to_delete)  
        
        result = [number for i, number in enumerate(numbers) if i not in indices_to_delete]
    
        return result 
    
    def poisson_sampling(self, lam=3, size=1):  
        return np.random.poisson(lam, size)

    import random  

    def text_infilling(self, tokens, prob=0.5, lam=5, mask_token=500):  
        n = len(tokens)  
        i = 0  
        result = []  
        
        while i < n:  
            # 以一定概率执行Text Infilling操作  
            if random.random() < prob:  
                span_length = self.poisson_sampling(lam)[0]  
                if span_length == 0:  
                    result.append(mask_token)  
                    i += 1  
                else:  
                    if i + span_length <= n:  
                        result.append(mask_token)  
                        i += span_length  
                    else:  
                        result.append(tokens[i])  
                        i += 1  
            else:  
                result.append(tokens[i])  
                i += 1  
        return result
    
    import random  
    import numpy as np  
    
    
    def replace_tokens_with_probability(self, tokens, p, lam=3, random_token=499):  
        num_masked = int(len(tokens) * p)  
        masked_indices = sorted(random.sample(range(len(tokens)), num_masked))  
    
        for i in range(len(masked_indices)):  
            idx = masked_indices[i]  
            token1_random = random.randint(0, random_token)  
            num_continuous_replacements = self.poisson_sampling(lam=lam)[0]  
    
            for j in range(num_continuous_replacements):  
                next_index = masked_indices[i + 1] if i + 1 < len(masked_indices) else len(tokens)  
                if idx + j < next_index:  
                    tokens[idx + j] = token1_random  
                else:  
                    break  
    
        return tokens  

    def mask_transform_v2_span(self, input_list, masking_ratio=0.3, lam=3, mask_token=500, random_token=500):

        num_masked = int(len(input_list) * masking_ratio)      
        masked_indices = sorted(random.sample(range(len(input_list)), num_masked))
        masked_list = []      
        masked_final_indices = []    
        i = 0  
        idx = 0  
        while i < len(input_list):  
            if i not in masked_indices or idx >= len(masked_indices):

                masked_final_indices.append(0)
                masked_list.append(input_list[i])  
                i += 1  
            else:    
                num_replacements = self.poisson_sampling(lam,1)[0]
                force_mask = True if random.random() < 0.8 else False  
                  
                if force_mask:  
                    for _ in range(num_replacements):  
                        if i < len(input_list) and idx < len(masked_indices) :
                            if idx ==len(masked_indices)-1 or i< masked_indices[idx+1]:
                                masked_item, _ = self.apply_masking_strategy(input_list[i], mask_token, random_token=random_token, force_mask=force_mask) 
                                
                                masked_list.append(masked_item)  
                                masked_final_indices.append(1)  
                                i += 1  
                        else:  
                            break  
                else:  
                    masked_final_indices.append(0)  
                    masked_list.append(input_list[i])  
                    i += 1  
                  
                idx += 1  
        
        if self.args.prepend_bos is True:
            masked_final_indices = masked_final_indices+[0]
        else:
            masked_final_indices = masked_final_indices[1:]+[0]
        return masked_list, masked_final_indices  
    
    def pretrain_transform(self, tokens, mode=0, prob=0.5, lam=5, mask_token=500, random_token=500):
        random_token = mask_token  # attention
        mask_indices = []
        if mode==0:
            tokens = self.mask_transform(tokens, prob=prob, mask_token=mask_token)
        elif mode==1:
            tokens = self.del_transform(tokens, prob=prob, mask_token=mask_token)
        elif mode==2:
            tokens = self.text_infilling(tokens, prob=prob, lam=lam, mask_token=mask_token)
        elif mode==3 or mode == 5:
            tokens, mask_indices = self.mask_transform_v2(tokens, masking_ratio=prob, mask_token=mask_token, random_token=random_token)
        elif mode==4:
            tokens = self.del_transform_v2(tokens, prob=prob)
        elif mode==6:
            tokens = self.replace_tokens_with_probability(tokens, p=prob, lam=lam)
        elif mode==7:
            tokens, mask_indices = self.mask_transform_v2_span(tokens, masking_ratio=prob, lam=lam,mask_token=mask_token, random_token=random_token)

        return tokens, mask_indices

    def pad_acoustic_features(self, acousitc_tokens):
        EPSILON = 1e-10
        LOG_EPSILON = math.log(EPSILON)
        original_lengths = torch.tensor([tensor.shape[0] for tensor in acousitc_tokens], dtype=torch.int32)
        padded_output = pad_sequence(acousitc_tokens, batch_first=True, padding_value=LOG_EPSILON)  

        return padded_output, original_lengths
    
    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        validate_for_tts(cuts)

        for transform in self.cut_transforms:
            cuts = transform(cuts)
        
        if False:  # not used
            audio, audio_lens = collate_audio(cuts)
        else:  # for sharing tokenized features in different machines
            audio, audio_lens = None, None

        audio_features, audio_features_lens = self.feature_input_strategy(cuts, self.args.manifest_dir)
        audio_features = audio_features[:, :, :self.num_quantizers]
        for transform in self.feature_transforms:
            audio_features = transform(audio_features)
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
        if self.args.input_semantic is True:
            for cut in cuts:
                storage_path.append(cut.recording.sources[0].source)
                if self.args.semantic_type==0:  # hubert tokens as semantic 
                    semantic_token = cut.supervisions[0].custom["tokens"]["semantic_tokens"].copy()
                    
                elif self.args.semantic_type==1: # tfnet tokens as semantic 
                    semantic_token = cut.supervisions[0].custom["tokens"]["semantic_tokens_tfnet"].copy()  # [[[a, b], [c, d]]]

                if self.semantic_depup is True: # semantic token 是否去重
                    semantic_token=depup(semantic_token)

                # maskd_indices for semantic mask loss
                if self.args.is_pretrain:  # if pretrain mode， input semantic tokens need to be masked. if tfnet_semantic token :pret_token=256 hubert_semantic token :pret_token=500 
                    semantic_token, maskd_indices = self.pretrain_transform(semantic_token, mode=self.args.pret_mode, prob=self.args.pret_prob, lam=self.args.pret_lam, mask_token=self.args.pret_token) 

                elif self.args.ac_native_mask and self.is_train==1 and "arctic" not in cut.id : # ac_native_mask 代表 finetune 阶段加上native的数据， masked native semantic tokens -> native semantic tokens
                    semantic_token, maskd_indices = self.pretrain_transform(semantic_token, mode=self.args.pret_mode, prob=self.args.pret_prob, lam=self.args.pret_lam, mask_token=self.args.pret_token)
                    maskd_indices = []  

                semantic_tokens.append(semantic_token)
                if self.args.pret_mode==5 and self.args.is_pretrain is True: # mode5 need to compute masked loss, the other need to be total loss
                    maskd_indices_batch.append(torch.tensor(maskd_indices, dtype=torch.float32))
            

            if maskd_indices_batch != []:
                maskd_indices_batch = pad_sequence(maskd_indices_batch, batch_first=True, padding_value=0)
            else:
                maskd_indices_batch = torch.tensor([])

            bef_semantic_tokens=semantic_tokens # semantic_tokens before semantic_token_collater

            
            semantic_tokens,  semantic_tokens_lens = self.semantic_token_collater( # add bos, eos, pad
                semantic_tokens
            )
        else:
            semantic_tokens, semantic_tokens_lens, bef_semantic_tokens, maskd_indices_batch = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # target semantic token
        if self.args.semantic_remove:   # ac finetune (accent semantic -> native semantic token )and semantic pretrain (corrupted -> original)
            semantic_features = []
            
            if self.args.ac_tune_mode==0:
                tgt_semantic_tokens_tab ="native_semantic_tokens"
            elif self.args.ac_tune_mode==1:
                tgt_semantic_tokens_tab ="indian_semantic_tokens"
            for cut in cuts:
                speaker_names = cut.supervisions[0].custom[tgt_semantic_tokens_tab].keys()

                if self.args.random_tgt_spk:  # ac finetune. each target speaker is random
                    random_tgt_spkers = self.args.random_tgt_spkers
                    spk_name = random.sample(speaker_names, random_tgt_spkers)                  
                else:
                    tgt_spk_names_dic = self.args.tgt_spk_names.split(',')
                    set_A = set(speaker_names)
                    set_B = set(tgt_spk_names_dic)  
                    C = set_A & set_B
                    spk_names = list(C)
                    try:
                        spk_name = random.sample(spk_names, 1) 
                    except Exception as e:
                        print(f"random speaker: {cut.id}")
                        spk_name = random.sample(speaker_names, 1) 

                if self.args.is_pretrain: # pretrain
                    if self.args.semantic_type==0:  # hubert semantic token 
                        native_semantic_token = cut.supervisions[0].custom["tokens"]["semantic_tokens"]  # pretrain : input and output is myself
                    elif self.args.semantic_type==1: #tfnet semantic token
                        native_semantic_token = cut.supervisions[0].custom["tokens"]["semantic_tokens_tfnet"]
                else:  # ac finetune 
                    
                    if tgt_semantic_tokens_tab not in cut.supervisions[0].custom.keys(): # say input is native speech. only l1-l2 accent data has native_semantic_tokens attribute
                        native_semantic_token = cut.supervisions[0].custom["tokens"]["semantic_tokens"].copy()  # correct phrase. corrupted->original
                    else: # ac input: accent semantic token -> native semantic token
                        if self.args.semantic_type==0:
                            if self.args.random_tgt_spkers==1:
                                native_semantic_token = cut.supervisions[0].custom[tgt_semantic_tokens_tab][spk_name[0]] # input is non-native -> output is native

                            elif self.args.random_tgt_spkers==2:
                                native_semantic_token1 = cut.supervisions[0].custom[tgt_semantic_tokens_tab][spk_name[0]]
                                native_semantic_token2 = cut.supervisions[0].custom[tgt_semantic_tokens_tab][spk_name[1]]
                                native_semantic_token = longest_common_subsequence(native_semantic_token1, native_semantic_token2)

                        elif self.args.semantic_type==1:
                            native_semantic_token = cut.supervisions[0].custom["native_semantic_tfnet_tokens"][spk_name]

                if self.semantic_depup is True:  # pretrain and ac finetune phrase. target whether need to be depup
                    unique_tokens = depup(native_semantic_token)
                    semantic_features.append(torch.tensor(unique_tokens, dtype=torch.float32))
                else:
                    semantic_features.append(torch.tensor(native_semantic_token, dtype=torch.float32))

            semantic_features, semantic_features_lens = self.pad_acoustic_features(semantic_features)
            # audio_features_another, audio_features_another_lens = self.pad_acoustic_features([torch.tensor(cut.supervisions[0].custom["native_semantic_tokens"], dtype=torch.float32) for cut in cuts])
            
            if self.args.semantic_type==0:  # hubert
                semantic_features = semantic_features.unsqueeze(2)
            # ac finetune and pretrain        
            audio_features = semantic_features
            audio_features_lens = semantic_features_lens

        # vc tts
        return {
            "utt_id": [cut.id for cut in cuts],
            "text": [cut.supervisions[0].text for cut in cuts],
            "audio": audio,
            "audio_lens": audio_lens,
            "audio_features": audio_features,
            "audio_features_lens": audio_features_lens,
            "text_tokens": text_tokens,
            "text_tokens_lens": text_tokens_lens,
            "semantic_tokens": semantic_tokens,
            "semantic_tokens_lens": semantic_tokens_lens,
            "bef_semantic_tokens": bef_semantic_tokens,
            "storage_path": storage_path,
            'maskd_indices_batch': maskd_indices_batch
        }


def validate_for_tts(cuts: CutSet) -> None:
    validate(cuts)
    for cut in cuts:
        assert (
            len(cut.supervisions) == 1
        ), "Only the Cuts with single supervision are supported."
def depup(semantic_token):
    unique_tokens = []  
    for token in semantic_token:  
        if unique_tokens==[] or token != unique_tokens[-1]:  
            unique_tokens.append(token)
    return unique_tokens

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