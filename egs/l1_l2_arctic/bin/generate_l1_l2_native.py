import os  
import shutil
# 定义文件夹路径  
directory = "/home/v-zhijunjia/data/accent_iclr/iclr_final/arctic/asi_source"  
source_dir = "/scratch/data/l1_l2_arctic/combine_L1_L2/train_native/total"
tgt_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/arctic/gt_native"
# 遍历目录并获取所有.wav文件的路径  
wav_files = []  
for root, dirs, files in os.walk(directory):  
    for file in files:  
        if file.endswith(".wav"):  
            wav_files.append(file)  
  
# 确保wav_files只包含100个文件  
wav_files = wav_files[:100]  
  
# 定义说话者  
speakers = ["cmu_us_bdl_arctic", "cmu_us_clb_arctic", "cmu_us_rms_arctic", "cmu_us_slt_arctic"]  
  
# 初始化词典，每个说话者对应一个空列表  
speaker_wav_dict = {speaker: [] for speaker in speakers}  
  
# 分配文件给每个说话者  
for i, speaker in enumerate(speakers):  
    # 为每个说话者分配25个wav文件  
    speaker_wav_dict[speaker] = wav_files[i*25:(i+1)*25]  
  
# 打印结果  
for speaker in speaker_wav_dict:  
    speaker_wavs = speaker_wav_dict[speaker]
    for wav in speaker_wavs:
        wav = wav.split("_",maxsplit=1)[1]
        source_file = os.path.join(source_dir, speaker, "wav", wav)
        tgt_file_name = os.path.join(tgt_dir, f"{speaker}_"+wav)
        print(source_file)
        print(tgt_file_name)
        shutil.copy(source_file, tgt_file_name)



