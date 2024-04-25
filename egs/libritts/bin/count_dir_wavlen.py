import os  
import json  
from tqdm import tqdm  
  
def count_subdirectories(directory):  
    subdir_count = 0  
    for _, dirnames, _ in os.walk(directory):
        print(dirnames)
        subdir_count += len(dirnames)  
    return subdir_count  
  
def process_directory_multiprocessing(directory):  
    total_mins = 0  
    i=0
    # 处理子文件夹并更新进度条  
    for root, _, files in os.walk(directory):  
        for file in files:  
            try:
                if file.endswith('.json'):  
                    i+=1
                    if i%20==0:
                        print(root)

                    file_path = os.path.join(root, file)  
                    with open(file_path, 'r') as f:  
                        data = json.load(f)
                        time_secs = data['book_meta']['totaltimesecs']
                        total_mins += (time_secs / 60)
            except Exception as e:
                pass
            
    return total_mins  
  
if __name__ == "__main__":  
    directory = "/home/v-zhijunjia/zhijundata_small/data_local/dataset/librilight/mid_data/medium"  
    total_minutes = process_directory_multiprocessing(directory)  
    print(f"Total minutes: {total_minutes}")  
