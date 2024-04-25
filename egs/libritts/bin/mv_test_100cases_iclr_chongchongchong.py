import os  
import shutil  
  
src_dir = "/scratch/data/l1_l2_arctic/combine_L1_L2/train_india_split/val"  
dst_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_ac100cases_chongchongchong"  
  
# 初始化文本范围  
start_file_index = 1  
num_files_to_copy = 20  
  
# 为目标目录创建wav和transcript文件夹  
os.makedirs(os.path.join(dst_dir, "wav"), exist_ok=True)  
os.makedirs(os.path.join(dst_dir, "transcript"), exist_ok=True)  
  
# 遍历源目录中的每个speaker文件夹  
for speaker in os.listdir(src_dir):  
    speaker_path = os.path.join(src_dir, speaker)  
    if os.path.isdir(speaker_path):  
        wav_src_path = os.path.join(speaker_path, "wav")  
        transcript_src_path = os.path.join(speaker_path, "transcript")  
  
        # 获取wav和transcript文件列表，并按文件名排序  
        sorted_wav_files = sorted(os.listdir(wav_src_path))  
        sorted_transcript_files = sorted(os.listdir(transcript_src_path))  
  
        # 从每个speaker文件夹中拷贝指定范围的wav文件和相应的transcript文件  
        for wav_file, transcript_file in zip(sorted_wav_files[start_file_index - 1:start_file_index + num_files_to_copy - 1], sorted_transcript_files[start_file_index - 1:start_file_index + num_files_to_copy - 1]):  
            # 为wav文件添加speaker名称前缀  
            new_wav_file = f"{speaker}_{wav_file}"  
  
            # 复制wav文件和transcript文件到目标目录  
            shutil.copy(os.path.join(wav_src_path, wav_file), os.path.join(dst_dir, "wav", new_wav_file))  
            shutil.copy(os.path.join(transcript_src_path, transcript_file), os.path.join(dst_dir, "transcript", transcript_file))  
  
        # 更新文本范围  
        start_file_index += num_files_to_copy  
