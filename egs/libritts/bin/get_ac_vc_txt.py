import os  
import shutil  
  
wav_folders = [  
    "/home/v-zhijunjia/data/data_update/benchmark_l1l2_vc_india/prompt",  
    "/home/v-zhijunjia/data/data_update/benchmark_l1l2_vc_india/source",  
]  
transcripts_folder = "/scratch/data/test_vc_arctic/cmu_us_bdl_arctic/transcript"  
output_txt_folder = "/home/v-zhijunjia/data/data_update/benchmark_l1l2_vc_india/txt"  
os.makedirs(output_txt_folder, exist_ok=True)  
  
# Collect all .wav files from the given folders  
wav_files = []  
for folder in wav_folders:  
    for root, dirs, files in os.walk(folder):  
        for file in files:  
            if file.endswith(".wav"):  
                wav_files.append(os.path.basename(file))  
  
# Find and copy corresponding text files to the output folder  
for wav_file in wav_files:  
    text_name = wav_file.split("_")[-1].split(".")[0]  # Get the text name (starts with 'arctic')  
    for root, dirs, files in os.walk(transcripts_folder):  
        for file in files:  
            if text_name in file:  
                src_txt_path = os.path.join(root, file)  
                dest_txt_path = os.path.join(output_txt_folder, file)  
                shutil.copy(src_txt_path, dest_txt_path)  
                print(f"Copied {src_txt_path} to {dest_txt_path}")  
                break  
