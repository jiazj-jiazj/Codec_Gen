import os  
import re  
  
def rename_wav_files(src_folder):  
    for root, dirs, files in os.walk(src_folder):  
        for file in files:  
            if file.endswith(".wav"):  
                # Replace 'prompt' with 'speaker' at the beginning of the filename  
                new_file_name = re.sub(r'^prompt', 'speaker', file)  
  
                # Remove 'ar_' and the following characters, keeping the .wav extension  
                new_file_name = re.sub(r'ar_.+(?=\.wav)', '', new_file_name)  
  
                # Rename the file  
                src_file_path = os.path.join(root, file)  
                dest_file_path = os.path.join(root, new_file_name)  
                os.rename(src_file_path, dest_file_path)  
  
src_folder = "/home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/converted_vc_onlyar_v0_parallel"  
rename_wav_files(src_folder)  
