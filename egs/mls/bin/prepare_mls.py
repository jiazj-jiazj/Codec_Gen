import os
import sys

current_working_directory = os.getcwd()  
  
# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory) 
from lhotse.recipes.mls import prepare_mls

parser = argparse.ArgumentParser(description="Process command line parameters.")
parser.add_argument("--input_corpus_path", type=str, default="/raid/dataset/mls_files")
parser.add_argument("--output_path", type=str, default="/raid/dataset/lhotse_dataset_test")
parser.add_argument("--is_opus_file", default="False")  
 
config = {}
args = parser.parse_args()
prepare_mls(args.input_corpus_path, args.output_path, args.is_opus_file)
