import librosa  
import soundfile as sf  
  
def trim_audio(input_file, output_file, duration_s):  
    # 加载音频文件  
    y, sr = librosa.load(input_file, sr=None)  
  
    # 计算裁剪的样本数  
    end_sample = int(duration_s * sr)  
  
    # 裁剪音频文件  
    y_trimmed = y[:end_sample]  
  
    # 保存裁剪后的音频文件  
    sf.write(output_file, y_trimmed, sr)  
  
# 使用示例  
input_file = '/home/v-zhijunjia/demo/update/vctk/prompt1_p248_233_sys2_p248_233_model3_ar_epoch-10pt_nar_epoch-40pt_2024_02_20_08_32_54_14_0.wav'  # 替换为你的音频文件路径  
output_file = '/home/v-zhijunjia/demo/update/vctk/trimmed_prompt1_p248_233_sys2_p248_233_model3_ar_epoch-10pt_nar_epoch-40pt_2024_02_20_08_32_54_14_0.wav'  # 替换为你想保存的新文件路径  
duration_s = 1.4  # 保留的秒数  
  
trim_audio(input_file, output_file, duration_s)  
