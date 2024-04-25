import sys
import os

path_now = os.getcwd()
sys.path.append(path_now)
from lhotse import CutSet

from lhotse.manipulation import combine as combine_manifests
from lhotse.serialization import load_manifest_lazy_or_eager

file1_ = "/home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/lhotse_data_v2/IndianTTS_l1l2indian2native/cuts_train.jsonl.gz"
file2_ = "/home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token/filter_bad_case_filter_add_sem_filter_Indic_TTScuts_all.jsonl.gz"
manifests = [file1_, file2_]
output_manifest = "/home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/lhotse_data_v2/IndianTTS_l1l2indian2native/filter_bad_case_cuts_train_l1_l2_indianTTS.jsonl.gz"

data_set = combine_manifests(*[load_manifest_lazy_or_eager(m) for m in manifests])
data_set.to_file(output_manifest)

# cut_set = CutSet.from_file("/home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token/filter_bad_case_filter_add_sem_filter_Indic_TTScuts_all.jsonl.gz")
# cut_set.describe()
# print(cut_set[0])