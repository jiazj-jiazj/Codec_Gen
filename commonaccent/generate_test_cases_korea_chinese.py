import os  
import shutil  
import random  
  
def copy_random_files(source_dir, target_dir, num_files, prefix):  
    # Get a list of all files in the source directory  
    files = os.listdir(source_dir)  
      
    # Randomly select a subset of files  
    selected_files = random.sample(files, num_files)  
      
    # Make sure the target directory exists  
    os.makedirs(target_dir, exist_ok=True)  
      
    # Copy each selected file to the target directory  
    for file in selected_files:  
        # Add the prefix to the filename  
        new_filename = prefix + "_" + file  
        # Get the full paths for the source and target files  
        source_file = os.path.join(source_dir, file)  
        target_file = os.path.join(target_dir, new_filename)  
        # Copy the file  
        shutil.copyfile(source_file, target_file)  
  
# Use the function  
source_dir = '/home/v-zhijunjia/zhijundata_small_v3/data/l1_l2_arctic/data/Accents/L2_arctic_v2/TXHC/wav'  
target_dir = '/home/v-zhijunjia/data/icml_more_accent/korean/source_accent_cases'  
num_files = 25  
prefix = 'TXHC'  
copy_random_files(source_dir, target_dir, num_files, prefix)
