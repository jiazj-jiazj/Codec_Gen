import os  
import random  
import shutil  
  
src_folder = "/scratch/data/BWAVN"  
dst_folder = "/home/v-zhijunjia/data/data_update/benchmark_librispeech_noise_10speakers/source"  
prompt_folder = "/home/v-zhijunjia/data/data_update/benchmark_librispeech_noise_10speakers/prompt"  
file_prefix = "gt_"  
  
# 列出所有 wav 文件  
all_files = [f for f in os.listdir(src_folder) if f.endswith(".wav")]  
  
# 提取所有的 speaker  
speakers = list(set([f.split("-")[0] for f in all_files]))  
  
# 从具有至少 4 个音频文件的 speaker 中随机抽取 10 个  
valid_speakers = [s for s in speakers if len([f for f in all_files if f.startswith(s)]) >= 4]  
selected_speakers = random.sample(valid_speakers, 10)  
  
# 对于每个选定的 speaker，随机选择 4 个 wav 文件并将其移动到目标文件夹  
for speaker in selected_speakers:  
    # 筛选出当前 speaker 的所有 wav 文件  
    speaker_files = [f for f in all_files if f.startswith(speaker)]  
  
    # 随机选择 4 个 wav 文件  
    selected_files = random.sample(speaker_files, 4)  
  
    # 将选定的文件移动到目标文件夹，并在文件名前加上 "gt_"  
    for file in selected_files:  
        src_path = os.path.join(src_folder, file)  
        dst_path = os.path.join(dst_folder, file_prefix + file)  
        shutil.copy(src_path, dst_path)  
  
    # 从每个 speaker 的 wav 文件中随机选择一个文件并将其移动到 prompt 文件夹  
    prompt_file = random.choice(speaker_files)  
    prompt_src_path = os.path.join(src_folder, prompt_file)  
    prompt_dst_path = os.path.join(prompt_folder, prompt_file)  
    shutil.copy(prompt_src_path, prompt_dst_path)  
