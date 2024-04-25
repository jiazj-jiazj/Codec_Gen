import os  
import json
import sys
# 获取当前工作目录  
current_working_directory = os.getcwd()  
sys.path.insert(0, current_working_directory) 
from lhotse import CutSet
import os  
import shutil  
import string
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


cuts = CutSet.from_file("/home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token/filter_add_sem_filter_Indic_TTScuts_all.jsonl.gz")

cuts =cuts.to_eager()

cuts_update_fem = cuts.filter(lambda c: abs(len(c.supervisions[0].custom["native_semantic_tokens"]["speaker0"]) -len(c.supervisions[0].custom["tokens"]["semantic_tokens"]))>=100 and str(c.id).split('-')[0] in ["indictts_phase2_assamese_fem_speaker1_english"])
cuts_update_mal = cuts.filter(lambda c: abs(len(c.supervisions[0].custom["native_semantic_tokens"]["speaker0"]) -len(c.supervisions[0].custom["tokens"]["semantic_tokens"]))>=100 and str(c.id).split('-')[0] in ["indictts_phase2_assamese_male_speaker1_english"])

# cuts_update.describe()

root = "/scratch/indian_accent_datasets/indictts-english/IndicTTS"
tgt_dir = "/home/v-zhijunjia/zhijundata_small_v2/data_local/accent_wer/val_ac_models/IndicTTS_l1_l2_arctic_test_100cases"
tgt_dir_gt_txt = "/home/v-zhijunjia/zhijundata_small_v2/data_local/accent_wer/val_ac_models/IndicTTS_l1_l2_arctic_test_100cases_txt"

i=0
for cut in cuts_update_fem:
    dir = cut.id.split('-', maxsplit=1)[0]
    file_name = cut.id.split('-', maxsplit=1)[1]
    file_name = file_name.rsplit('-', maxsplit=1)[0]
    dir = dir.replace("indictts_phase2_assamese_fem_speaker1_english", "IndicTTS_Phase2_Assamese_fem_Speaker1_english").replace("indictts_phase2_assamese_male_speaker1_english", "IndicTTS_Phase2_Assamese_male_Speaker1_english")
    wav_path = os.path.join(root, dir, "english", "wav", file_name+".wav")
    text = cut.supervisions[0].text
    # 复制 wav 文件  
    tgt_wav_path = os.path.join(tgt_dir, file_name + ".wav")  
    shutil.copy(wav_path, tgt_wav_path)  
      
    # 去除文本中的标点符号  
    text_no_punctuation = text.translate(str.maketrans('', '', string.punctuation))  
    # 保存文本文件  
    tgt_txt_path = os.path.join(tgt_dir_gt_txt, file_name + ".txt")  
    with open(tgt_txt_path, 'w') as text_file:  
        text_file.write(text_no_punctuation)
    i+=1
    if i>=49:
        break
i=0
for cut in cuts_update_mal:
    dir = cut.id.split('-', maxsplit=1)[0]
    file_name = cut.id.split('-', maxsplit=1)[1]
    file_name = file_name.rsplit('-', maxsplit=1)[0]
    dir = dir.replace("indictts_phase2_assamese_fem_speaker1_english", "IndicTTS_Phase2_Assamese_fem_Speaker1_english").replace("indictts_phase2_assamese_male_speaker1_english", "IndicTTS_Phase2_Assamese_male_Speaker1_english")
    wav_path = os.path.join(root, dir, "english", "wav", file_name+".wav")
    text = cut.supervisions[0].text
    # 复制 wav 文件  
    tgt_wav_path = os.path.join(tgt_dir, file_name + ".wav")  
    shutil.copy(wav_path, tgt_wav_path)  
      
    # 去除文本中的标点符号  
    text_no_punctuation = text.translate(str.maketrans('', '', string.punctuation))  
    # 保存文本文件  
    tgt_txt_path = os.path.join(tgt_dir_gt_txt, file_name + ".txt")  
    with open(tgt_txt_path, 'w') as text_file:
        text_file.write(text_no_punctuation)
    i+=1
    if i>=49:
        break
# print(len(cuts_update))

# cuts_update.to_file("/home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token/filter_bad_case_filter_add_sem_filter_Indic_TTScuts_all.jsonl.gz")


# # Seventh october,  KSO