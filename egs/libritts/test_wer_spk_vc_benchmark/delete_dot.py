import os  
import glob  
  
folder_path = '/dev_huaying/zhijun/data/benchmark_librispeech_10speakers/asr_encodec_vc'  
file_pattern = os.path.join(folder_path, '*.*')  
  
files = glob.glob(file_pattern)  
  
for file in files:  
    file_name, file_ext = os.path.splitext(os.path.basename(file))  
    cleaned_file_name = file_name.replace('.', '') + file_ext  
    cleaned_file_path = os.path.join(folder_path, cleaned_file_name)  
      
    os.rename(file, cleaned_file_path)  
