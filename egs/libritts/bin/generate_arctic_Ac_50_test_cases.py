import os
import shutil
import random

# 定义源目录和目标目录
source_dir = "/scratch/data/l1_l2_arctic/combine_L1_L2/train_india_split/test"
target_wav_dir = "/home/v-zhijunjia/zhijundata_small_v2/data_local/accent_wer/indian_accent_test_arctics_50cases"
target_txt_dir = "/home/v-zhijunjia/zhijundata_small_v2/data_local/accent_wer/indian_accent_test_arctics_50cases_gt_txt"

# 确保目标目录存在
os.makedirs(target_wav_dir, exist_ok=True)
os.makedirs(target_txt_dir, exist_ok=True)

# 遍历每个speaker目录
for speaker in os.listdir(source_dir):
    speaker_dir = os.path.join(source_dir, speaker)
    wav_dir = os.path.join(speaker_dir, 'wav')
    transcript_dir = os.path.join(speaker_dir, 'transcript')

    if os.path.isdir(wav_dir) and os.path.isdir(transcript_dir):
        # 获取所有wav文件的列表
        wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
        # 随机选择10个wav文件
        selected_wavs = random.sample(wav_files, min(10, len(wav_files)))

        for wav_file in selected_wavs:
            # 拷贝wav文件
            source_wav_path = os.path.join(wav_dir, wav_file)
            target_wav_path = os.path.join(target_wav_dir, f"{speaker}_{wav_file}")
            shutil.copy(source_wav_path, target_wav_path)
            
            # 拷贝对应的txt文件
            txt_file = wav_file.replace('.wav', '.txt')
            source_txt_path = os.path.join(transcript_dir, txt_file)
            target_txt_path = os.path.join(target_txt_dir, f"{speaker}_{txt_file}")
            shutil.copy(source_txt_path, target_txt_path)

print("完成复制选定的wav和txt文件。")
