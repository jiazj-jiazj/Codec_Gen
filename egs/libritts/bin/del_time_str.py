import os  
  
folder_path = "/home/v-zhijunjia/demo/iclr_final/converted_test_can_del/hubert_depup_v2_top_102023-11-20_22:42:55"  
prefix = "tfcodec_vc_"  
  
for filename in os.listdir(folder_path):  
    # 检查文件是否为音频文件，这里我们假设音频文件的扩展名为.mp3  
    if filename.endswith(".wav"):  
        # 获取文件的旧路径和新路径  
        old_filepath = os.path.join(folder_path, filename)  
        new_filepath = os.path.join(folder_path, prefix + filename)  
  
        # 重命名文件  
        os.rename(old_filepath, new_filepath)  
