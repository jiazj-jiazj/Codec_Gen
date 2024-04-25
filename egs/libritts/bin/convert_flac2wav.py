import librosa
import soundfile as sf
import os

# 设置你的源文件夹和目标文件夹
source_folder = '/home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer_min'  # 替换成你的.flac文件所在的文件夹路径
target_folder = '/home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer_min_wav'  # 替换成你想要保存.wav文件的目标文件夹路径

# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的所有文件
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith('.flac'):
            # 构建完整的文件路径
            flac_file_path = os.path.join(root, file)
            # 创建与.flac文件同名的.wav文件路径
            wav_file_path = os.path.join(target_folder, os.path.splitext(file)[0] + '.wav')
            
            # 使用librosa进行转换
            audio, sr = librosa.load(flac_file_path, sr=None)
            sf.write(wav_file_path, audio, sr)

print("转换完成！")
