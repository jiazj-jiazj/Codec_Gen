import os  
import sys
# 获取当前工作目录  
current_working_directory = os.getcwd()  
from tqdm import tqdm
# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory) 

from lhotse import CutSet

input_file = "/home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/lhotse_data_vc_target_semantic_native_all_accents/tokenized/cuts_test.jsonl.gz"
out_put_file_name = input_file.split("/")[-1]

indian_cutset = CutSet.from_file(input_file)
indian_cutset = indian_cutset.to_eager()

import json

output_dir = "/home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/lhotse_data_vc_target_semantic_native_all_accents_with_indian_semantic/tokenized/"
os.makedirs(output_dir, exist_ok=True)
# 设置文件路径
file_path = '/scratch/data/l1_l2_arctic/combine_L1_L2/tokens_dic/hubert_indian_l1_l2_arctic_semantic_dic.json'

# 打开并读取JSON文件
with open(file_path, 'r') as json_file:
    data = json.load(json_file)
    for i, cut_set in tqdm(enumerate(indian_cutset), total=len(indian_cutset)):
        wav_path = cut_set.recording.sources[0].source
        file_name = wav_path.split('/')[-1]
        print(cut_set.supervisions[0].custom["accent"])
        continue
        try:
            cut_set.supervisions[0].custom["indian_semantic_tokens"] =data[file_name]
        except Exception as e:
            print(f"file_name not exist")

indian_cutset.to_file(os.path.join(output_dir, out_put_file_name))