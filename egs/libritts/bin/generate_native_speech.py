import os
import random
import shutil

# 设置源目录和目标目录
source_dir = "/home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/data/Accents/combine_L1_L2/train_native_split/test"
dest_dir_wavs = "/home/v-zhijunjia/zhijundata_small_v2/data_local/accent_wer/val_ac_models_native2indian/native_accent_test_arctics_50cases"
dest_dir_texts = "/home/v-zhijunjia/zhijundata_small_v2/data_local/accent_wer/val_ac_models_native2indian/native_accent_test_arctics_50cases_gt_txt"

# 确保目标目录存在
os.makedirs(dest_dir_wavs, exist_ok=True)
os.makedirs(dest_dir_texts, exist_ok=True)

# 获取所有speaker目录
speaker_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

# 对每个speaker目录进行操作
for speaker in speaker_dirs:
    speaker_wav_dir = os.path.join(source_dir, speaker, "wav")
    speaker_text_dir = os.path.join(source_dir, speaker, "transcript")
    
    # 确保wav目录存在
    if os.path.exists(speaker_wav_dir) and os.path.exists(speaker_text_dir):
        # 获取所有wav文件
        wav_files = [f for f in os.listdir(speaker_wav_dir) if f.endswith('.wav')]
        # 随机选择10个wav文件
        selected_wavs = random.sample(wav_files, min(10, len(wav_files)))
        
        # 复制选中的wav文件和对应的文本文件到目标目录
        for wav_file in selected_wavs:
            # 复制wav文件
            source_wav_path = os.path.join(speaker_wav_dir, wav_file)
            dest_wav_name = "{}_{}".format(speaker, wav_file)
            dest_wav_path = os.path.join(dest_dir_wavs, dest_wav_name)
            shutil.copy2(source_wav_path, dest_wav_path)
            
            # 复制对应的文本文件
            text_file = wav_file.replace('.wav', '.txt')
            source_text_path = os.path.join(speaker_text_dir, text_file)
            if os.path.exists(source_text_path):  # 确保文本文件存在
                dest_text_path = os.path.join(dest_dir_texts, text_file)
                shutil.copy2(source_text_path, dest_text_path)

print("Wav and text files have been copied successfully.")
