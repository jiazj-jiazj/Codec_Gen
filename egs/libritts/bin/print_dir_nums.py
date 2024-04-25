import os  
  
directory = "/home/v-zhijunjia/data/benchmark_librispeech_10speakers/converted_pretrain_mode5_015_epoch29_noshared_linear"  
  
file_count = sum(os.path.isfile(os.path.join(directory, f)) for f in os.listdir(directory))  
  
print(f"There are {file_count} files in the directory.")  
