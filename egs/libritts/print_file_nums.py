
import os  
  
def count_files_in_subfolders(folder_path):  
    for root, dirs, files in os.walk(folder_path):  
        if root == folder_path:  
            for subdir in dirs:  
                total_files = 0
                subdir_path = os.path.join(root, subdir)  
                # print(subdir_path)
                # quit()
                for root_x, dirs, files in os.walk(subdir_path):  
                    print(root_x)
                    total_files += len(files)  
  
                print(f"Subfolder '{subdir}' contains {total_files} files.")  
  
folder_path = "/mnt/zhijun/Accents/combine_l1_l2_all_accents/train_accents_all_split/train/cmu_us_ksp_arctic"  # 请替换为你的文件夹路径  
count_files_in_subfolders(folder_path)
