import os  
import glob  
  
# 指定目录路径  
directory = "/home/v-zhijunjia/CodecGen/egs/libritts/bin"  
  
# 使用 glob 模块查找以 "azure" 开头的文件  
files_to_delete = glob.glob(os.path.join(directory, "azure*"))  
  
# 遍历找到的文件并删除它们  
for file_path in files_to_delete:  
    if os.path.isfile(file_path):  
        os.remove(file_path)  
        print(f"Deleted: {file_path}")  
