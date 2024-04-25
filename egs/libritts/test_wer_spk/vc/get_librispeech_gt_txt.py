import os  
import shutil  
  
asr_txt_folder = "/dev_huaying/zhijun/data/test_vc/benchmark_librispeech_10speakers/asr_txt"  
gt_txt_folder = "/dev_huaying/zhijun/data/test_vc/benchmark_librispeech_10speakers/gt_txt"  
original_txt_folder = "/dev_huaying/zhijun/data/LibriSpeech/filtered_4s_10s_test_clean_add_txt"  
os.makedirs(gt_txt_folder, exist_ok=True)
  
# 遍历asr_txt_folder中的所有txt文件  
for file in os.listdir(asr_txt_folder):  
    if file.endswith(".txt") and file.startswith("asr_"):  
        corresponding_file = file.replace("asr_", "", 1)  
        
        for root, dirs ,origin_files in os.walk(original_txt_folder):
            for origin_file in origin_files:
                if origin_file == corresponding_file:

                    src_file = os.path.join(root, corresponding_file)  
                    dest_file = os.path.join(gt_txt_folder, corresponding_file)  
          
                    # 将文件从original_txt_folder移动到gt_txt_folder  
                    shutil.copy(src_file, dest_file)  
  
print("文件已移动并修改文件名。")  
