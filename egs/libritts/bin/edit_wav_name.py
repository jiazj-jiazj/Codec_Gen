import os  
  
folder_path = "/home/v-zhijunjia/data/accent_iclr/ASI/mode_5_mask015_source1spker_300cases_tgt1spker_lr_0005_source-topk-2-epoch14_2023-09-25_21:04:48_txt"  
  
for file in os.listdir(folder_path):  
    file_path = os.path.join(folder_path, file)  
    if file.endswith(".txt"):  
        new_file_name = file[:12] + '_' + file[-7] + ".txt"

        new_file_path = os.path.join(folder_path, new_file_name)  
        if file != new_file_name:  
            os.rename(file_path, new_file_path)  
            print(f"文件名从 {file_path} 更改为 {new_file_path}")  
        else:  
            print(f"保留文件：{file_path}")  
