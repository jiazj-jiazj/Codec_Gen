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

# native->indian val test

val_test_speakers = ["cmu_us_bdl_arctic", "cmu_us_clb_arctic", "cmu_us_slt_arctic", "cmu_us_rms_arctic"]
def check_cut_id_in_dev(cut):
    # if cut_super_id not in 
    if cut.supervisions[0].speaker not in val_test_speakers:
        return False
    cut_super_id = cut.supervisions[0].id
    if 'arctic_b' not in cut_super_id:
        return False
    match = re.search(r'arctic_b(\d+)', cut_super_id)
    num = int(match.group(1))
    if num>=390 and num<=489:
        return True
    return False

def check_cut_id_in_test(cut):
    if cut.supervisions[0].speaker not in val_test_speakers:
        return False
    cut_super_id = cut.supervisions[0].id
    if 'arctic_b' not in cut_super_id:
        return False
    match = re.search(r'arctic_b(\d+)', cut_super_id)
    num = int(match.group(1))
    if num>=490 and num<=539:
        return True
    return False

def check_cut_id_in_train_part(cut):
    if cut.supervisions[0].speaker in val_test_speakers:
        return True
    cut_super_id = cut.supervisions[0].id
    if 'arctic_b' not in cut_super_id:
        return True
    match = re.search(r'arctic_b(\d+)', cut_super_id)
    num = int(match.group(1))
    if num>=390 and num<=539:
        return False
    return True

cuts_all_native = cuts.filter(lambda c: (c.supervisions[0].custom["indian_semantic_tokens"]!=None))
cuts_all_native = cuts_all_native.filter(lambda c: (c.supervisions[0].custom['accent'] in ["may_native", "Scottish", "Canadian"]))

cuts_dev_native = cuts_all_native.filter(lambda c: check_cut_id_in_dev(c))
cuts_dev_native.describe()
for cut in cuts_dev_native:
    print(cut.id)

cuts_test_native = cuts_all_native.filter(lambda c: check_cut_id_in_test(c))
cuts_test_native.describe()
for cut in cuts_test_native:
    print(cut.id)
cuts_train_native = cuts_all_native.filter(lambda c: check_cut_id_in_test(c)==False and check_cut_id_in_dev(c)==False and check_cut_id_in_train_part(c)==True)
cuts_train_native.describe()
for cut in cuts_train_native:
    print(cut.id)
dir_ ="/home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/lhotse_data_v2/native2indian/"
os.makedirs(dir_, exist_ok=True)

cuts_train_native.to_file(dir_+ "cuts_train.jsonl.gz")
cuts_dev_native.to_file(dir_+ "cuts_dev.jsonl.gz")
cuts_test_native.to_file(dir_+ "cuts_test.jsonl.gz")

