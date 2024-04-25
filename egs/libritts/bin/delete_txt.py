import os  
import glob  
  
def delete_files_with_extension(folder_path, extension):  
    files_to_delete = glob.glob(f"{folder_path}/*{extension}")  
    for file_path in files_to_delete:  
        os.remove(file_path)  
        print(f"删除文件: {file_path}")  
  
folder_path = "/dev_huaying/zhijun/data/test_vc/benchmark_librispeech/convertedvc_yourtts_v0"  # 替换为您要删除文件的文件夹路径  
delete_files_with_extension(folder_path, ".txt")  
