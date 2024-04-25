import os  
import sys
# 获取当前工作目录  
current_working_directory = os.getcwd()  
  
# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory) 

from lhotse import CutSet
cut_set = CutSet.from_file("/scratch/data/Libritts/tokenized_tfnet_semantic_token/cuts_dev.jsonl.gz")
cut_set_1000 = cut_set.subset(first=1000)
cut_set_1000.to_file("/scratch/data/Libritts/tokenized_tfnet_semantic_token/cuts_dev_1000.jsonl.gz")
