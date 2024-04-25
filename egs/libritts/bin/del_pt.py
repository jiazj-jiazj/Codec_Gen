import os  
import re  
  
folder_path = "/home/v-zhijunjia/data/valle-tensorboard-models/pretrain_finetune/mode_2_mask_5_5000libri_8000arc_mask_libri_input_tune_mode_3_p_15"  
files_to_keep = ["epoch-1.pt", "epoch-2.pt", "epoch-11.pt"]  # Replace with the files you want to keep  
  
# Iterate through the files in the folder  
for file in os.listdir(folder_path):
    # Check if the file name matches the pattern "epoch-x.pt"  
    if re.match(r"epoch-\d+\.pt", file) and file not in files_to_keep:  
        file_path = os.path.join(folder_path, file)  
        os.remove(file_path)  
        print(f"Deleted: {file_path}")  
