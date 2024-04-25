import librosa  
import soundfile as sf  
  
def trim_audio(input_file, output_file, start_time, end_time):  
    audio, sample_rate = librosa.load(input_file)  
    start_sample = int(start_time * sample_rate)  
    end_sample = int(end_time * sample_rate)  
    trimmed_audio = audio[start_sample:end_sample]  
    sf.write(output_file, trimmed_audio, sample_rate)  
  
if __name__ == "__main__":  
    input_file = "/home/v-zhijunjia/data/test_accent_baseline_paper/final_cases/p248_003_20230916_091910_p248_003_ar_best-valid-losspt_nar_epoch-40pt_2023_09_16_09_17_37_3_0.wav"  # 替换为您要裁剪的音频文件名  
    output_file = "/home/v-zhijunjia/data/test_accent_baseline_paper/final_cases/clip_2_47_p248_003_20230916_091910_p248_003_ar_best-valid-losspt_nar_epoch-40pt_2023_09_16_09_17_37_3_0.wav"  # 替换为您要保存的新文件名（WAV 格式）  
    start_time = 0  # 裁剪开始时间（秒）  
    end_time = 2.47  # 裁剪结束时间（秒）  
  
    trim_audio(input_file, output_file, start_time, end_time)  
