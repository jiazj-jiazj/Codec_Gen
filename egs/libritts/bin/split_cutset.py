import sys
import os
import argparse
current_working_directory = os.getcwd()  
# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory) 

from lhotse import CutSet

def get_args():
    parser = argparse.ArgumentParser()

    # 替换成 bash tokenized_16k_mls_tfcodec_train_total.sh 处理好的文件
    parser.add_argument(
        "--src-path",  
        type=str,
        default="/scratch/indian_accent_datasets/indictts-english/IndicTTS_lhotse/cutset_data/filter_Indic_TTS_cuts_all.jsonl.gz",  
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/scratch/indian_accent_datasets/indictts-english/IndicTTS_lhotse/splited_lhotse_data",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="filter_Indic_TTS",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="jsonl.gz",
    )
    return parser.parse_args()
# 替换成 bash tokenized_16k_mls_tfcodec_train_total.sh 处理好的文件

args = get_args()
src_path = args.src_path
prefix = args.prefix
partition= args.partition
suffix = args.suffix
output_dir = args.output_dir

cut_set = CutSet.from_file(src_path)

cut_set_shuffled = cut_set.shuffle()
split_into_20 = cut_set_shuffled.split(num_splits=10)
for i, cut_set in enumerate(split_into_20):
    cuts_filename = f"{prefix}cuts_{partition}_{i}.{suffix}"
    cut_set.to_file(os.path.join(output_dir, cuts_filename))