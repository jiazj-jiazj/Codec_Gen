import os  
import shutil  
  
# 指定文件夹路径  
folder_path = "/path/to/benchmark_vc_ac_v2/converted_nopretrain_l1l2_ac_vc_2023_09_19_16_22_21"  
  
# 创建新文件夹  
v1_folder = os.path.join(folder_path, "converted_nopretrain_l1l2_ac_vc_2023_09_19_16_22_21_v1")  
v2_folder = os.path.join(folder_path, "converted_nopretrain_l1l2_ac_vc_2023_09_19_16_22_21_v2")  
  
os.makedirs(v1_folder, exist_ok=True)  
os.makedirs(v2_folder, exist_ok=True)  
  
# 遍历文件夹中的文件  
for file in os.listdir(folder_path):  
    file_path = os.path.join(folder_path, file)  
      
    # 如果是wav文件  
    if file.endswith(".wav"):  
        # 根据文件名最后一位将文件复制到相应的文件夹  
        if file[-5] == "0":  
            shutil.copy(file_path, os.path.join(v1_folder, file))  
        elif file[-5] == "1":  
            shutil.copy(file_path, os.path.join(v2_folder, file))  
