import os  
import sys
# 获取当前工作目录  
current_working_directory = os.getcwd()  
# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory) 

from lhotse import CutSet

cutset = CutSet.from_file("data/mls/mls_train_lhotse_dataset/mls-english_cuts_train_0_1_2.jsonl.gz")
cutset = cutset.subset(first=3000)
cutset.to_file("data/mls/mls_train_lhotse_dataset/mls-english_cuts_train_0_1_2_small_3000.jsonl.gz")
