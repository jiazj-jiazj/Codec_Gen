import sys
import os
import argparse
current_working_directory = os.getcwd()  
# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory) 
import re

from lhotse import CutSet

cuts = CutSet.from_file("/home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/lhotse_data_v2/lhotse_data_all_accents_native_l1_l2_arctic_v2/l1_l2_arctic_cuts_all.jsonl.gz")
cuts.describe()


def check_cut_id_in_dev(cut):
    # if cut_super_id not in 
    cut_super_id = cut.supervisions[0].id
    if 'arctic_b' not in cut_super_id:
        return False
    match = re.search(r'arctic_b(\d+)', cut_super_id)
    num = int(match.group(1))
    if num>=390 and num<=489:
        return True
    return False

def check_cut_id_in_test(cut):
    cut_super_id = cut.supervisions[0].id
    if 'arctic_b' not in cut_super_id:
        return False
    match = re.search(r'arctic_b(\d+)', cut_super_id)
    num = int(match.group(1))
    if num>=490 and num<=539:
        return True
    return False

cuts_indian = cuts.filter(lambda c: (c.supervisions[0].custom['accent'] in ["Hindi"]))

cuts_dev_indian = cuts_indian.filter(lambda c: check_cut_id_in_dev(c))
cuts_dev_indian.describe()

for cut in cuts_dev_indian:
    print(cut.id)

cuts_test_indian = cuts_indian.filter(lambda c: check_cut_id_in_test(c))
cuts_test_indian.describe()

for cut in cuts_test_indian:
    print(cut.id)
cuts_train_indian = cuts_indian.filter(lambda c: check_cut_id_in_test(c)==False and check_cut_id_in_dev(c)==False)
cuts_train_indian.describe()
for cut in cuts_train_indian:
    print(cut.id)
dir_ ="/home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/lhotse_data_v2/indian2native/"
os.makedirs(dir_, exist_ok=True)

cuts_train_indian.to_file(dir_+ "cuts_train.jsonl.gz")
cuts_dev_indian.to_file(dir_+ "cuts_dev.jsonl.gz")
cuts_test_indian.to_file(dir_+ "cuts_test.jsonl.gz")