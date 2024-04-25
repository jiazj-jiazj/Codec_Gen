import os  
import shutil  
  
# 在 train, val, test 文件夹中创建 speaker 子文件夹  
base_folders = ["/mnt/zhijun/Accents/combine_l1_l2_all_accents/train_accents_all_split/train", "/mnt/zhijun/Accents/combine_l1_l2_all_accents/train_accents_all_split/val", "/mnt/zhijun/Accents/combine_l1_l2_all_accents/train_accents_all_split/test"]  
for base_folder in base_folders:  
    os.makedirs(base_folder, exist_ok=True)  
  
# 列出 total 文件夹中的所有 speaker 文件夹  
total_folder = "/mnt/zhijun/Accents/combine_l1_l2_all_accents/train_accents_all"  
speaker_folders = [folder for folder in os.listdir(total_folder) if os.path.isdir(os.path.join(total_folder, folder))]  
print(speaker_folders)
# 处理每个 speaker 文件夹  
for speaker_folder in speaker_folders:  
    wav_files = sorted([f for f in os.listdir(os.path.join(total_folder, speaker_folder, "wav")) if f.endswith(".wav")])  
    # txt_files = sorted([f for f in os.listdir(os.path.join(total_folder, speaker_folder, "transcript")) if f.endswith(".txt")])  
    
    train_files = []
    val_files = []
    test_files = []
    for file_name in wav_files:  
        # 检查文件名是否以'arctic_b'开头  
        if file_name.startswith("arctic_b"):  
            # 获取数字部分并转换为整数  
            file_number = int(file_name[8:12]) 
            print(file_number) 
    
            # 判断文件属于哪个集合  
            if 440 <= file_number <= 489:  
                val_files.append(file_name)
            elif 490 <= file_number <= 539:  
                test_files.append(file_name)  
            else:  
                train_files.append(file_name)
        else:
            train_files.append(file_name)

  
    for base_folder, wav_list in zip(base_folders, [train_files, val_files, test_files]):  
        os.makedirs(os.path.join(base_folder, speaker_folder, "wav"), exist_ok=True)  
        # os.makedirs(os.path.join(base_folder, speaker_folder, "transcript"), exist_ok=True)  
  
        for wav_file in wav_list:  
            src_wav = os.path.join(total_folder, speaker_folder, "wav", wav_file)  
            dst_wav = os.path.join(base_folder, speaker_folder, "wav", wav_file)  
            shutil.copy(src_wav, dst_wav)  
  
            # src_txt = os.path.join(total_folder, speaker_folder, "transcript", txt_file)  
            # dst_txt = os.path.join(base_folder, speaker_folder, "transcript", txt_file)  
            # shutil.copy(src_txt, dst_txt)  
