import os  
  
directory = "/home/v-zhijunjia/data/benchmark_librispeech_10speakers/converted_pretrain_mode5_015_epoch29_noshared_linear"  
  
for filename in os.listdir(directory):  
    if filename.endswith(".txt"):  
        file_path = os.path.join(directory, filename)  
        try:  
            os.remove(file_path)  
            print(f"Deleted file: {file_path}")  
        except OSError as e:  
            print(f"Error deleting file: {file_path}. Reason: {e}")  
