import os  
  
def get_files_with_size(root_folder):  
    files_with_size = []  
  
    # 遍历文件夹及其子文件夹  
    for folder, _, filenames in os.walk(root_folder):  
        for filename in filenames:  
            # 获取文件的相对路径  
            relative_path = os.path.join(folder, filename)  
              
            # 获取文件大小  
            file_size = os.path.getsize(relative_path)  
  
            # 添加文件路径和大小到列表中  
            files_with_size.append((relative_path, file_size))  
  
    return files_with_size  
  
def print_files_sorted_by_size(files_with_size):  
    # 按文件大小排序  
    sorted_files = sorted(files_with_size, key=lambda x: x[1])  
  
    # 打印文件路径和大小  
    for file_path, file_size in sorted_files:  
        print(f"File: {file_path}, Size: {file_size} bytes")  
  
root_folder = "/home/v-zhijunjia/valle-4-23"  
files_with_size = get_files_with_size(root_folder)  
print_files_sorted_by_size(files_with_size)  
