import os  
  
base_folder = "/scratch/data/l2_test"  
  
for root, dirs, files in os.walk(base_folder):  
    for file in files:  
        if file.endswith(".wav"):  
            speaker = os.path.basename(os.path.dirname(root))  
            new_file = f"gt_{speaker}_{file}"  
            old_file_path = os.path.join(root, file)  
            new_file_path = os.path.join(root, new_file)  
            os.rename(old_file_path, new_file_path)  
            print(f"Renamed {old_file_path} to {new_file_path}")  
