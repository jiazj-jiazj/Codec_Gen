import os
import sys

print(sys.path)
current_working_directory = os.getcwd()  
  
# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory) 
from lhotse.recipes.mls import prepare_mls
from lhotse import CutSet

# parser = argparse.ArgumentParser(description="Process command line parameters.")
# parser.add_argument("--input_corpus_path", type=str, default="/raid/dataset/mls_files")
# parser.add_argument("--output_path", type=str, default="/raid/dataset/lhotse_dataset_test")
# parser.add_argument("--is_opus_file", default="False")  
 
# config = {}
# args = parser.parse_args()
# prepare_mls(args.input_corpus_path, args.output_path, args.is_opus_file)


cut_set = CutSet.from_file("/home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/lhotse_data_vc_target_semantic_native_all_accents/initial_tokenized/cuts_train.jsonl.gz")

cut_set.describe()
quit()
for cut in cut_set:
    print(cut.supervisions[0].custom["accent"])

