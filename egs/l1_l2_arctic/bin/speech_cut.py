import librosa  
import soundfile as sf  
import numpy as np
# 音频文件路径  
file_path = '/home/v-zhijunjia/data/accent_iclr/converted_can_del/vctk_20_cases_IndicTTS_indian_native2all_native_txt_infilling_all_cases_tgt_4_speakers_lr_0_001_topk_2_epoch__top_k_stage2_30_2024-01-31_23:51:43/prompt1_p248_004_20240131_160103_sys2_p248_004_model3_ar_epoch-10pt_nar_epoch-40pt_2024_01_31_15_58_34_0_0.wav'  
  
# 加载音频文件  
y, sr = librosa.load(file_path, sr=None)  
  

# 计算要丢弃
# 的样本数量，sr是采样率  
samples_to_cut = int(sr * 2.4)  
samples_to_end = int(sr * 3.1) 
# 裁剪音频数组，丢弃前0.75秒  
y_trimmed = np.concatenate((y[:samples_to_cut], y[samples_to_end:]))  
  
# （可选）保存裁剪后的音频到新文件  
trimmed_file_path = '/home/v-zhijunjia/data/accent_icml/cases_analysis/proposed_p248_004_trimmed.wav'  
sf.write(trimmed_file_path, y_trimmed, sr)  
  
print(f"Trimmed audio saved to {trimmed_file_path}")  
