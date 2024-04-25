import sys
sys.path.append("/home/v-zhijunjia/valle-4-23")
from lhotse import CutSet, manipulation
import lhotse

cuts_ac = CutSet.from_file('/scratch/data/l1_l2_arctic/lhotse_data_vc_target_semantic_native/tokenized/cuts_dev.jsonl.gz')
for cut in cuts_ac:
    print(cut.id)

quit()

cuts_libri = CutSet.from_file('/scratch/data/Libritts/tokenized/cuts_dev.jsonl.gz')
for cut in cuts_libri:
    print(cut)
    break
# cuts_native = CutSet.from_file('/scratch/data/LibriTTS/vc_tokenized_16k_tfcodec_16codes/cuts_train.jsonl.gz')
# # cuts_native = cuts_native.to_eager()
# cuts_native = cuts_native.shuffle()
# cuts_ac = cuts_ac.shuffle()

# clean100_filter_cuts_native = cuts_native.filter(lambda r: 'clean-100' in r.recording.sources[0].source)

# first_100_filter_cuts_native = clean100_filter_cuts_native.subset(first=5000)


# combine_native_non_nat = manipulation.combine([first_100_filter_cuts_native, cuts_ac])

# combine_native_non_nat = combine_native_non_nat.shuffle()
# combine_native_non_nat = combine_native_non_nat.to_eager()
# print(len(combine_native_non_nat))

# combine_native_non_nat.to_file("/scratch/data/libritts_clean_100_5000_l1l2/tokenized/cuts_train.jsonl.gz")


# #149736
# #33236 clean-100
# #4900

# cuts_ac = CutSet.from_file('/scratch/data/l1_l2_arctic/lhotse_data_vc_target_semantic_native/tokenized/cuts_dev.jsonl.gz')
# cuts_native = CutSet.from_file('/scratch/data/LibriTTS/vc_tokenized_16k_tfcodec_16codes/cuts_dev.jsonl.gz')

# print(len(cuts_ac))
# quit()


# cuts_native = cuts_native.to_eager()

# cuts_ac = cuts_ac.shuffle()
# cuts_native = cuts_native.shuffle()

# clean100_filter_cuts_native = cuts_native.filter(lambda r: 'clean' in r.recording.sources[0].source)
# first_500_filter_cuts_native = clean100_filter_cuts_native.subset(first=500)
# print(len(first_500_filter_cuts_native))

# combine_native_non_nat = manipulation.combine([first_500_filter_cuts_native, cuts_ac])
# combine_native_non_nat = combine_native_non_nat.shuffle()

# combine_native_non_nat.to_file("/scratch/data/libritts_clean_100_5000_l1l2/tokenized/cuts_dev.jsonl.gz")

# print(len(combine_native_non_nat))
# quit()
# print(len(cuts_ac))
# print(len(cuts_native))
# quit()
# cuts_native = cuts_native.shuffle()

#10349 all

#500