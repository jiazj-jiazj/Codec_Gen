import os  
import shutil  
  
# 源文件夹的路径  
source_wav_folder = "/scratch/data/l1_l2_arctic/combine_L1_L2/train_india/total/ASI/wav"  
source_transcript_folder = "/scratch/data/l1_l2_arctic/combine_L1_L2/train_india/total/ASI/transcript"  
  
# 目标文件夹的路径  
destination_folder = "/home/v-zhijunjia/data/accent_iclr/ASI"  
destination_wav_folder = os.path.join(destination_folder, "wav")  
destination_transcript_folder = os.path.join(destination_folder, "transcript")  
  
# 检查目标文件夹是否存在，如果不存在则创建  
if not os.path.exists(destination_wav_folder):  
    os.makedirs(destination_wav_folder)  
  
if not os.path.exists(destination_transcript_folder):  
    os.makedirs(destination_transcript_folder)  
  
# 获取源文件夹中的wav和transcript文件名  
wav_files = sorted(os.listdir(source_wav_folder))  
transcript_files = sorted(os.listdir(source_transcript_folder))  
  
# 获取wav文件名后30个文件  
wav_files_to_move = wav_files[-30:]  
  
# 遍历要移动的wav文件  
for wav_file in wav_files_to_move:  
    # 移动wav文件  
    src_wav_path = os.path.join(source_wav_folder, wav_file)  
    dst_wav_path = os.path.join(destination_wav_folder, wav_file)  
    shutil.copy(src_wav_path, dst_wav_path)  
  
    # 移动对应的transcript文件  
    transcript_file = wav_file.replace(".wav", ".txt")  
    src_transcript_path = os.path.join(source_transcript_folder, transcript_file)  
    dst_transcript_path = os.path.join(destination_transcript_folder, transcript_file)  
    shutil.copy(src_transcript_path, dst_transcript_path)  
  
print(f"已移动 {len(wav_files_to_move)} 对 wav 和 transcript 文件到 {destination_folder}")  
