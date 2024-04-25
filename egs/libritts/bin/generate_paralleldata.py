import os  
import shutil  
import re  
  
def copy_files_to_folders(src_folder, dest_folder):  
    for file in os.listdir(src_folder):
        if file.endswith(".wav"):  
            # Extract the string between 'gt_' and '_ar'  
            match = re.search(r'gt_(.+?)_ar', file) 

            if match:  

                group_string = match.group(1)  
                # Create a new folder with the format "speaker1_{}" and increment if it exists  
                folder_name = f"{group_string}"  
                i = 1  
                if os.path.exists(os.path.join(dest_folder, folder_name)):  
                    i += 1  
                    folder_name = f"{group_string}"  
                # Create the f  older if it doesn't exist  
                folder_path = os.path.join(dest_folder, folder_name)  
                os.makedirs(folder_path, exist_ok=True)  
  
                # Copy the file to the new folder  
                src_file_path = os.path.join(src_folder, file)  
                dest_file_path = os.path.join(folder_path, file)  
                shutil.copy(src_file_path, dest_file_path)  
            else:
                print(f"{file} not match")
  
src_folder = "/home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/converted_vc_onlyar_v0"  
dest_folder = "/home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/converted_vc_onlyar_v0_parallel"  
copy_files_to_folders(src_folder, dest_folder)  
