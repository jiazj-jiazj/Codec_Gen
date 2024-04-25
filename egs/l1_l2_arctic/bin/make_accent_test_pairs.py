
import os
import shutil

import soundfile as sf  

import os  
  
# # 指定文件夹路径  
# folder_path = '/home/v-zhijunjia/data/accent_icml/100pairs_accent_classifier/source_native_wavs'  
  
# # 列举文件夹中的所有文件  
# files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]  
  
# # 计算文件数量  
# file_count = len(files)  
  
# print(f"该文件夹中有 {file_count} 个文件。")  


root_dir = "/scratch/data/Libritts/raw_files/LibriTTS/test-clean"
tgt_dir = "/home/v-zhijunjia/data/accent_icml/100pairs_accent_classifier/prompt_native_wavs_7s_larger"

num =0
for dir_path, dir_paths, file_names in os.walk(root_dir):
    for file_name in file_names:
        if file_name.endswith(".wav"):
            file_path = os.path.join(dir_path, file_name)
            with sf.SoundFile(file_path) as f:  
                # 获取音频时长  
                duration = len(f) / f.samplerate  
            tgt_dir_file = os.path.join(tgt_dir, file_name)
            # 判断音频长度是否在7秒到10秒之间  
            if 7 <= duration <= 10:  
                shutil.copy(file_path, tgt_dir_file)
                num+=1
                if num==100:
                    print("over")
                    quit()

## make source native 
# root_dir = "/scratch/data/l1_l2_arctic/combine_L1_L2/train_native/total/cmu_us_clb_arctic/wav"
# tgt_dir = "/home/v-zhijunjia/data/accent_icml/100pairs_accent_classifier/source_native_wavs"

# num =0
# for dir_path, dir_paths, file_names in os.walk(root_dir):
#     for file_name in file_names:
#         if file_name.endswith(".wav"):
#             file_path = os.path.join(dir_path, file_name)
#             tgt_dir_file = os.path.join(tgt_dir, "clb_"+file_name)
#             shutil.copy(file_path, tgt_dir_file)
#             num+=1
#             if num==50:
#                 quit()

