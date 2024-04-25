import os  
import shutil  
  
folder_path = "/home/v-zhijunjia/data/benchmark_librispeech_10speakers/converted_pretrain_mode5_015_epoch29_noshared_linear"  
  
for file_name in os.listdir(folder_path):  
    if file_name.endswith('.wav'):  # Assuming the audio files are in WAV format  
        file_path = os.path.join(folder_path, file_name)  
        name_without_ext, ext = os.path.splitext(file_name) 
        print(name_without_ext)
        new_name = name_without_ext.replace(".", "") + ext  
        new_path = os.path.join(folder_path, new_name)  
          
        # Rename the file  
        shutil.move(file_path, new_path)  
