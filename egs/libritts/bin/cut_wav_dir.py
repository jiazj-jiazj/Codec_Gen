import os  
import librosa  
import numpy as np  
import soundfile as sf  
  
# 设置源文件夹和目标文件夹路径  
source_folder = "/home/v-zhijunjia/data/data_update/benchmark_vc_tts_9s/prompt_9s_39spkers"  
target_folder = "/home/v-zhijunjia/data/data_update/benchmark_vc_tts_9s/prompt_9s_39spkers_6s"  
  
# 目标秒数  
target_duration = 6.0  # 目标秒数  
  
# 如果目标文件夹不存在，则创建它  
if not os.path.exists(target_folder):  
    os.makedirs(target_folder)  
  
# 获取所有的.wav文件  
wav_files = [f for f in os.listdir(source_folder) if f.endswith('.wav')]  
  
for wav_file in wav_files:  
    # 构建完整的文件路径  
    file_path = os.path.join(source_folder, wav_file)  
      
    # 使用librosa加载音频文件  
    audio, sr = librosa.load(file_path, sr=None)  
      
    # 计算目标裁剪长度对应的样本数  
    target_length = int(sr * target_duration)  
      
    # 如果音频文件短于目标长度，则跳过  
    if len(audio) < target_length:  
        continue  
      
    # 随机选择开始裁剪的位置  
    start = np.random.randint(0, len(audio) - target_length)  
      
    # 裁剪音频  
    cropped_audio = audio[start:start + target_length]  
      
    # 构建目标文件路径  
    target_file_path = os.path.join(target_folder, wav_file)  
      
    # 保存裁剪后的音频文件  
    sf.write(target_file_path, cropped_audio, sr)  
  
print("Finished trimming audio files.")  
