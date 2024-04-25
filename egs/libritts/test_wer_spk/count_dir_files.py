import os  
  
def count_files_in_folder(folder_path):  
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]  
    return len(files)  
  
folder_path = "/mnt/zhijun/Accents/combine_L1_L2/train_native_split/train/cmu_us_bdl_arctic/wav"  # 替换为您要计算文件数量的文件夹路径  
file_count = count_files_in_folder(folder_path)  
print("文件数量:", file_count)  
