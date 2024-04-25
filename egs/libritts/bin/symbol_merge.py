import sys
import os
path_now = os.getcwd()
sys.path.append(path_now)
from valle.utils import SymbolTable
# sys.path.insert(0, "/dev_huaying/zhijun/fairseq")

file1 = "/raid/dataset/lhotse_dataset/mls_train_lhotse_dataset_phone_processed/unique_text_tokens_0.k2symbols"
file2 = "/raid/dataset/lhotse_dataset/mls_train_lhotse_dataset_phone_processed/unique_text_tokens_1.k2symbols"
file3 = "/raid/dataset/lhotse_dataset/mls_train_lhotse_dataset_phone_processed/unique_text_tokens_2.k2symbols"
file4 = "/raid/dataset/lhotse_dataset/mls_train_lhotse_dataset/unique_text_tokens_dev__test.k2symbols"


unique_phonemes1 = SymbolTable().from_file(file1)
unique_phonemes2 = SymbolTable().from_file(file2)
unique_phonemes3 = SymbolTable().from_file(file3)
unique_phonemes4 = SymbolTable().from_file(file4)

sep_phonemes = [unique_phonemes1, unique_phonemes2, unique_phonemes3, unique_phonemes4]

total_symbol = SymbolTable()
unique_symbols = set()
for unique_phoneme in sep_phonemes:
    symbols = unique_phoneme.symbols

    unique_symbols.update(symbols)
    
for s in sorted(list(unique_symbols)):
    total_symbol.add(s)

total_symbol.to_file("/raid/dataset/lhotse_dataset/mls_train_lhotse_dataset/unique_text_tokens_train_0_1_2_dev_test.k2symbols")
# unique_phonemes1.load("")
# for s in sorted(list(unique_symbols)):
#     unique_phonemes.add(s)
# logging.info(f"{len(unique_symbols)} unique phonemes: {unique_symbols}")

# unique_phonemes_file = f"{args.output_dir}/unique_text_tokens_{'_'.join(dataset_parts)}.k2symbols"
# unique_phonemes.to_file(unique_phonemes_file)


# if __name__ == "__main__":
#     formatter = (
#         "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
#     )

#     # hubert chinese
#     logging.basicConfig(format=formatter, level=logging.INFO)
#     main()
