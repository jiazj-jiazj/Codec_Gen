import os  
  
def get_files_sorted_by_size(folder_path):  
    files_with_sizes = []  
  
    for root, _, files in os.walk(folder_path):  
        for file in files:  
            file_path = os.path.join(root, file)  
            file_size = os.path.getsize(file_path)  
            files_with_sizes.append((file_path, file_size))  
  
    files_with_sizes.sort(key=lambda x: x[1])  
  
    for file_path, size in files_with_sizes:  
        print(f"{file_path}: {size} bytes")  
  
folder_path = "/dev_huaying/zhijun/valle_23_4_22"  
get_files_sorted_by_size(folder_path)  
