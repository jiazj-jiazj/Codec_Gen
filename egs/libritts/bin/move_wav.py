import os  
import shutil  
  
def move_wav_files_recursively(src_folder, dest_folder):  
    for root, dirs, files in os.walk(src_folder):  
        for file in files:  
            if file.endswith(".wav"):  
                src_file_path = os.path.join(root, file)  
                dest_file_path = os.path.join(dest_folder, file)  
                shutil.copy(src_file_path, dest_file_path)  
  
src_folder = "/home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/converted_vc_onlyar_v0_parallel"  
dest_folder = "/home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/converted_vc_onlyar_v0_parallel_1dir"  
os.makedirs(dest_folder, exist_ok=True)
move_wav_files_recursively(src_folder, dest_folder)  
