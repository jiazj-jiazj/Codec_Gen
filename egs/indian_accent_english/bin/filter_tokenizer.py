import os  
import json
import sys
# 获取当前工作目录  
current_working_directory = os.getcwd()  
sys.path.insert(0, current_working_directory) 
from lhotse import CutSet

import re
from valle.data import (
    AudioTokenConfig,
    AudioTokenExtractor,
    AudioTokenConfig_16k,
    AudioTokenExtractor_16k,
    AudioTokenExtractor_16k_tfcodec,
    TextTokenizer,
    tokenize_text,
    ApplyKmeans,
    HubertFeatureReader
)
from valle.data.collation import get_text_token_collater


cuts = CutSet.from_file("/home/v-zhijunjia/zhijundata_small_v2/data/IndicTTS/lhotse_data/filter_Indic_TTS_cuts_all.jsonl.gz")

cuts =cuts.to_eager()

print(len(cuts))
quit()
text_token_collater=get_text_token_collater("/home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols")

ignore_lists = []
for cut in cuts:
    try:
        text_tokens, text_tokens_lens = text_token_collater(
            [cut.supervisions[0].custom["tokens"]["text"]]
        )
    except Exception as e:
        # print(e)
        # print(cut.id)
        # print(cut.supervisions[0].custom["tokens"]["text"])
        ignore_lists.append(cut.id)

cuts_update = cuts.filter(lambda c: c.id not in ignore_lists)

cuts_update.to_file("/scratch/indian_accent_datasets/indictts-english/IndicTTS_lhotse/cutset_data/filter_Indic_TTS_cuts_all.jsonl.gz")

print(len(cuts))
print(len(cuts_update))
# Seventh october,  KSO