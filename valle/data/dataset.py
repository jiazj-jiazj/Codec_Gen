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
NUM_AUDIO_TOKENS=1024
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

    def mask_in_group(self, mask, loc, audio_feature_len, audio_features, replace_prob=0.15, replace_all_prob=0.05, each_code_book_rep=5):
        
        bef_audio_feature = audio_features[loc, :, :]
        # print(f"bef_audio_feature:{bef_audio_feature}")
        # print(bef_audio_feature.shape)
        batch_size, total_len, code_book = audio_features.shape  
        # 找到没有被mask的区域  
        unmasked_indices = torch.nonzero(mask == 0, as_tuple=False).squeeze(1)
        # print(f"unmasked_indices:{unmasked_indices}")
        # quit()
        # 计算要替换的token数量                                                                                                                         
        # print(f'unmasked_indices:{unmasked_indices}')
        ## no work
        num_to_replace = int(len(unmasked_indices) * replace_prob)  
        # print(f"num_to_replace:{num_to_replace}")
        # 随机选择一些token进行替换  
        indices_to_replace = unmasked_indices[torch.randperm(len(unmasked_indices))[:num_to_replace]]  
        # 使用[0,1024)之间的随机数替换选定的token  
        # print(f'indices_to_replace:{indices_to_replace}')
        for idx in indices_to_replace:  
            num_to_replace_cb = torch.randint(1, each_code_book_rep, (1,)).item()  
            cb_indices_to_replace = torch.randperm(code_book)[:num_to_replace_cb]  
            # cb_indices_to_replace = min(cb_indices_to_replace, )
            # print(f"cb_indices_to_replace:{cb_indices_to_replace}")

            bef_audio_feature[idx.item(), cb_indices_to_replace] = torch.randint(0, 1024, (num_to_replace_cb,)).float()
            # bef_audio_feature[idx.item(), cb_indices_to_replace] = torch.full((num_to_replace_cb,), NUM_AUDIO_TOKENS).float()  
        
        if self.args.group_in_mask_replace_all_varible is True:
            replace_all_prob = random.uniform(0, replace_all_prob)  
        num_time_dim_to_replace = int(len(mask) * replace_all_prob)
        indices_to_replace_total = unmasked_indices[torch.randperm(len(unmasked_indices))[:num_time_dim_to_replace]]  
        # print(f'time_dim_to_replace: {time_dim_to_replace}') 

        indice_to_replace_cb = torch.randint(0, batch_size, (len(indices_to_replace_total),))
        indice_to_replace_cb_one_batch = torch.randint(0, len(mask), (len(indices_to_replace_total),))
        prob = random.random()
        if prob < 0.5:
            bef_audio_feature[indices_to_replace_total, :] = audio_features[indice_to_replace_cb, 0, :].squeeze(1)  # 假设aa_bb是预先定义好的 
        else:
            bef_audio_feature[indices_to_replace_total, :] = bef_audio_feature[indice_to_replace_cb_one_batch, :].squeeze(1)  # 假设aa_bb是预先定义好的 

        return bef_audio_feature


    # 0 baseline 1 group_ar 2: 0and1 3:0:random_group_ar
    def semantic_alighwith_acoustic(self, semantic_tokens, audio_features, audio_features_lens, nar_mask_type=0, pad_token=500, nar_mask_ratio=0.5, replace_prob=0.15, replace_all_prob=0.05):  # nar_mask_type =0 soundstorm baseline 1: group
        aft_semantic_tokens = []
    
        for semantic, audio_features_len in zip(semantic_tokens, audio_features_lens):
            if len(semantic) < audio_features_len:
                semantic+=[semantic[-1]* (audio_features_len.item()-len(semantic))]
            aft_semantic_tokens.append(torch.tensor(semantic, dtype=torch.int64))
        
        padded_output = pad_sequence(aft_semantic_tokens, batch_first=True, padding_value=pad_token)  

        masks = []
        # print(f"audio_features_lens:{audio_features_lens}")
        for loc, audio_features_len in enumerate(audio_features_lens):
            
            if nar_mask_type in [0, 1, 2, 3]:
                prob = random.random()  
                if nar_mask_type==0: # 只走baseline分支
                    prob =0.2
                elif nar_mask_type==1: # 只走下分支
                    prob =0.7
                if prob<nar_mask_ratio:
                    u = random.random()*math.pi*0.5
                    p = math.cos(u)
                    #print(p)
                    mask = np.random.binomial(1, p, size=audio_features_len)
                    #print(mask)
                    t = random.randint(0,audio_features_len.item()-1) # -1保证不是所有的utterance都是prompt
                    mask[:t] = 0
                    if not np.any(mask[:audio_features_len]):  
                        # 如果都是0，随机选择一个位置赋值为1  
                        random_index = random.randint(0, audio_features_len.item()-1)  
                        mask[random_index] = 1
                    mask = torch.tensor(mask, dtype=torch.int32)
                    # masks.append(torch.tensor(mask, dtype=torch.int32))
                else:

                    if nar_mask_type==2:
                        prompt_t = random.randint(0,audio_features_len.item()-1) # -1保证不是所有的utterance都是prompt ,0说明可能没有prompt
                        len_y = audio_features_len.item()-prompt_t
                        u_2 = random.random()*math.pi*0.5
                        # # total_step 1-cosi
                        # total_steps_choice_1 = np.floor((1- math.cos(u_2))*len_y)  # total_steps
                        # total_steps = int(max(1.0, total_steps_choice_1))
                    
                        # # total_step line 
                        total_steps = int(random.sample(range(1, len_y+1), 1)[0])

                        group = int(np.ceil(len_y/total_steps))
                        mask_group_len = int(random.sample(range(1, total_steps+1), 1)[0])
                        mask_single = torch.cat((torch.zeros(total_steps - mask_group_len), torch.ones(mask_group_len)))  
                        mask_single_seq = mask_single.repeat(group)[:len_y]
                        prompt_zeros = torch.zeros(prompt_t)  
                        mask = torch.cat((prompt_zeros, mask_single_seq)) 
                        mask = mask.to(dtype=torch.int32)
                        # masks.append(mask)
                    elif nar_mask_type==3:
                        # prompt_2_4 = random.randint(100,200) 
                        prompt_t = random.randint(0,audio_features_len.item()-1) # -1保证不是所有的utterance都是prompt ,0说明可能没有prompt
                        # prompt_t = min(prompt_t, prompt_2_4)
                        len_y = audio_features_len.item()-prompt_t
                        u_2 = random.random()*math.pi*0.5
                        # # total_step 1-cosi
                        # total_steps_choice_1 = np.floor((1- math.cos(u_2))*len_y)  # total_steps
                        # total_steps = int(max(1.0, total_steps_choice_1))
                    
                        # # total_step line 
                        total_steps = int(random.sample(range(1, len_y+1), 1)[0])

                        group = int(np.ceil(len_y/total_steps))
                        # single_group_prob = random.random()

                        # if single_group_prob<0.5:
                        #     group=1
                        mask_single_seq = torch.zeros(len_y, dtype=torch.int32)

                        if group == 1:  
                            indices = torch.tensor([0, len_y])  
                        else:  
                            if len_y-2 < group-1:
                                group =len_y-1
                            indices = torch.randperm(len_y-2)[:group-1] + 1  # +1 是为了避免生成0  
                            indices = torch.sort(indices).values  
                            indices = torch.cat((torch.tensor([0]), indices, torch.tensor([len_y]))) 
                        # print(f"indices:{indices}")
                        for i in range(group):    
                            # 对每个块，确定mask的起始位置  
                            start = torch.randint(indices[i], indices[i+1], (1,)).item()  
                            # 将起始位置到块末尾的部分设置为1    
                            mask_single_seq[start:indices[i+1]] = 1  
                        prompt_zeros = torch.zeros(prompt_t) 
                        mask = torch.cat((prompt_zeros, mask_single_seq)) 
                        mask = mask.to(dtype=torch.int32)

                        # masks.append(mask)    
            elif nar_mask_type==4:
                    prompt_2_4 = random.randint(100,200) 
                    prompt_t = random.randint(0,audio_features_len.item()-1) # -1保证不是所有的utterance都是prompt ,0说明可能没有prompt
                    prompt_t = min(prompt_t, prompt_2_4)

                    len_y = audio_features_len.item()-prompt_t
                    u_2 = random.random()*math.pi*0.5
                    # # total_step 1-cosi
                    # total_steps_choice_1 = np.floor((1- math.cos(u_2))*len_y)  # total_steps
                    # total_steps = int(max(1.0, total_steps_choice_1))
                
                    # # total_step line 
                    total_steps = int(random.sample(range(1, len_y+1), 1)[0])
                    group = 1
                    mask_single_seq = torch.zeros(len_y, dtype=torch.int32)
                    indices = torch.tensor([0, len_y])  

                    # print(f"indices:{indices}")
                    for i in range(group):    
                        # 对每个块，确定mask的起始位置  
                        start = torch.randint(indices[i], indices[i+1], (1,)).item()  
                        # 将起始位置到块末尾的部分设置为1    
                        mask_single_seq[start:indices[i+1]] = 1  
                    prompt_zeros = torch.zeros(prompt_t) 
                    mask = torch.cat((prompt_zeros, mask_single_seq)) 
                    mask = mask.to(dtype=torch.int32)
                    # masks.append(mask)    
            elif nar_mask_type==5:
                    prompt_2_4 = random.randint(100,200) 
                    prompt_t = random.randint(0,audio_features_len.item()-1) # -1保证不是所有的utterance都是prompt ,0说明可能没有prompt
                    prompt_t = min(prompt_t, prompt_2_4)

                    len_y = audio_features_len.item()-prompt_t
                    mask_single_seq = torch.ones(len_y, dtype=torch.int32)

                    prompt_zeros = torch.zeros(prompt_t) 
                    mask = torch.cat((prompt_zeros, mask_single_seq)) 
                    mask = mask.to(dtype=torch.int32)

            # print(f"len(mask):{len(mask)}")
            # print(f"audio_features_len:{audio_features_len}")
            if self.args.group_in_mask is True:
                audio_feature_sub = self.mask_in_group(mask.clone(), loc, audio_features_len, audio_features.clone(), replace_prob=replace_prob, replace_all_prob=replace_all_prob)
                audio_features[loc, :, :] = audio_feature_sub

            masks.append(mask)

        masks = pad_sequence(masks, batch_first=True, padding_value=0)

        return padded_output, audio_features, audio_features_lens, masks

    def semantic_aligh_valle_nar(self, semantic_tokens, pad_token=500):  # nar_mask_type =0 soundstorm baseline 1: group
        aft_semantic_tokens = []
        lengths = [len(token) for token in semantic_tokens]
        lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
        for semantic in semantic_tokens:
            semantic_aft = torch.tensor(semantic, dtype=torch.int64)
            aft_semantic_tokens.append(semantic_aft)

        padded_output = pad_sequence(aft_semantic_tokens, batch_first=True, padding_value=pad_token) 

        return padded_output, lengths_tensor

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
        try:
            audio_features, audio_features_lens = self.feature_input_strategy(cuts, self.args.manifest_dir)
            audio_features = audio_features[:, :, :self.num_quantizers]
            for transform in self.feature_transforms:
                audio_features = transform(audio_features)
        except:
            pass
        try:
            text_tokens, text_tokens_lens = self.text_token_collater(
                [cut.supervisions[0].custom["tokens"]["text"] for cut in cuts]
            )
        except:
            text_tokens, text_tokens_lens = torch.tensor([]), torch.tensor([])

        semantic_tokens = []
        storage_path = []
        maskd_indices_batch = []
        only_comp_mask_loss=self.args.only_comp_mask_loss
        audio_features_correct = []
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

            if self.args.model_name.lower()=="soundstorm":
                audio_features_correct = audio_features.clone()

                semantic_tokens,  audio_features, semantic_tokens_lens, maskd_indices_batch = self.semantic_alighwith_acoustic( # add bos, eos, pad
                    semantic_tokens, audio_features.clone(), audio_features_lens, nar_mask_type=self.args.nar_mask_type, pad_token=500, nar_mask_ratio=self.args.nar_mask_ratio, replace_prob=self.args.group_in_mask_replace_prob, replace_all_prob=self.args.group_in_mask_replace_all_prob
                )
                
            if self.args.model_name.lower()=="valle_nar" and self.args.is_pretrain:  # 注意如果不是pretrain， 要先和tgt对齐 再pad
                semantic_tokens,  semantic_tokens_lens= self.semantic_aligh_valle_nar(semantic_tokens, pad_token=500)
                maskd_indices_batch = torch.tensor([])
            if self.args.model_name.lower()=="valle_nar" and not self.args.is_pretrain:  # 避免执行semantic_token_collater
                pass
            elif self.args.model_name.lower()!="soundstorm": # 只有VALLE VALLFE 需要add bos eos
                semantic_tokens,  semantic_tokens_lens = self.semantic_token_collater( # add bos, eos, pad  # 注意执行这一行，semantic_remove 要设置为false
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
                if not self.args.is_pretrain: # pretrain
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
            
            if self.args.model_name.lower()=="valle_nar" and not self.args.is_pretrain: 
                # because model is nar, len(input) need to be equal to len(output)
                semantic_tokens = aligh_input2output(semantic_tokens, semantic_features)
                semantic_tokens,  semantic_tokens_lens= self.semantic_aligh_valle_nar(semantic_tokens, pad_token=500)
                maskd_indices_batch = torch.tensor([])
            else:
                # print(f"before semantic_tokens shape:{semantic_tokens.shape}")  
                semantic_tokens = aligh_input2output(semantic_tokens, semantic_features) #先对齐
                semantic_tokens,  semantic_tokens_lens = self.semantic_token_collater( # add bos, eos, pad
                    semantic_tokens
                )

            semantic_features, semantic_features_lens = self.pad_acoustic_features(semantic_features)

            # audio_features_another, audio_features_another_lens = self.pad_acoustic_features([torch.tensor(cut.supervisions[0].custom["native_semantic_tokens"], dtype=torch.float32) for cut in cuts])
            
            if self.args.semantic_type==0:  # hubert
                semantic_features = semantic_features.unsqueeze(2)
            # ac finetune and pretrain        
            audio_features = semantic_features
            audio_features_lens = semantic_features_lens

        # print(f"semantic_tokens.shape:{semantic_tokens.shape}")
        # print(f"audio_features:{audio_features.shape}")
        # vc tts
        if audio_features_correct == []:
            audio_features_correct = None

        return {
            "utt_id": [cut.id for cut in cuts],
            "text": [cut.supervisions[0].text for cut in cuts],
            "audio": audio,
            "audio_lens": audio_lens,
            "audio_features": audio_features,
            "audio_features_correct": audio_features_correct, 
            "audio_features_lens": audio_features_lens,
            "text_tokens": text_tokens,
            "text_tokens_lens": text_tokens_lens,
            "semantic_tokens": semantic_tokens,
            "semantic_tokens_lens": semantic_tokens_lens,
            "bef_semantic_tokens": bef_semantic_tokens,
            "storage_path": storage_path,
            'maskd_indices_batch': maskd_indices_batch,
            "only_comp_mask_loss":only_comp_mask_loss,
        }


class SpeechSynthesisDataset_infer(torch.utils.data.Dataset):
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
        
        text_tokens, text_tokens_lens = self.text_token_collater(
            [cut.supervisions[0].custom["tokens"]["text"] for cut in cuts]
        )

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