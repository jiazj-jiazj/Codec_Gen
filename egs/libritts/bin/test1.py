import torch
import sys
import os
current_working_directory = os.getcwd()  
  
# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory) 


from lhotse import CutSet

cutsets = CutSet.from_file("/home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/lhotse_data_v2/lhotse_data_all_accents_native_l1_l2_arctic_v2/l1_l2_arctic_cuts_all.jsonl.gz")
for cut in cutsets:
    print(cut)
    break

