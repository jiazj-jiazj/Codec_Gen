import os  
import shutil  
import re
def get_last_three_chars(filename): 
    pattern = r"(arctic_b\d{4})"  
    filename = re.search(pattern, filename).group(1) 
    return filename  
def get_last_three_chars_v2(filename): 
    pattern = r"(arctic_b\d{4})"  
    filename = re.search(pattern, filename).group(1) 
    return filename

# 两个源文件夹的路径  
folder1 = "/home/v-zhijunjia/data/accent_iclr/iclr_ac100cases_chongchongchong/wav"  
folder2 = "/scratch/data/l1_l2_arctic/combine_L1_L2/train_native_split/val/cmu_us_rms_arctic/wav"  
  
# 目标文件夹的路径  
destination_folder = "/home/v-zhijunjia/data/accent_iclr/iclr_ac100cases_chongchongchong/native_rms_cases"  
  
# 检查目标文件夹是否存在，如果不存在则创建  
if not os.path.exists(destination_folder):  
    os.makedirs(destination_folder)  
  
# 获取文件夹1和文件夹2中的文件名  
folder1_files = os.listdir(folder1)  
folder2_files = os.listdir(folder2)  

# 建立文件名后三位到文件名的映射  
folder1_mapping = {get_last_three_chars_v2(file): file for file in folder1_files}
print(folder1_mapping)
# 遍历文件夹2中的文件  
for file in folder2_files:  
    last_three_chars = get_last_three_chars(file)  
    # 检查文件名后三位是否在文件夹1中存在  
    if last_three_chars in folder1_mapping:  
        src_file_path = os.path.join(folder2, file)  
        dst_file_path = os.path.join(destination_folder, file)

        shutil.copy(src_file_path, dst_file_path)  
  
print(f"已复制符合条件的文件到 {destination_folder}")  
