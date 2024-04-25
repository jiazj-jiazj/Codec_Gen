import os  
import sys
# 获取当前工作目录  
current_working_directory = os.getcwd()  
  
# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory)

from lhotse import CutSet

cutset = CutSet.from_file("/home/v-zhijunjia/zhijundata_small_v2/data/mls/mls_train_lhotse_dataset/small_small_mls-english_cuts_train_0_1_2.jsonl.gz")

cutset_small = cutset.subset(first=30)
cutset_small.to_file("/home/v-zhijunjia/zhijundata_small_v2/data/mls/mls_train_lhotse_dataset/small_small_small_mls-english_cuts_train_0_1_2.jsonl.gz")