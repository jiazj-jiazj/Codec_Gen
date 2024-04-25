import os  
import random  
import shutil  
import re

import torch
import torchaudio
from transformers import Wav2Vec2Processor, HubertForCTC
from datasets import load_dataset
import torchaudio.transforms as T  
import fire

def resample_audio(audio_path, target_sample_rate):  
    waveform, sr = torchaudio.load(audio_path)  
    if sr != target_sample_rate:  
        resampler = T.Resample(sr, target_sample_rate)  
        waveform = resampler(waveform)  
    return waveform, target_sample_rate

def transcript(cropped_audio, sr, target_sample_rate=16000):
    cropped_audio_copy = cropped_audio.clone()  # 创建一个新的张量副本  

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")    
    if sr != target_sample_rate:  
        resampler = T.Resample(sr, target_sample_rate)  
        waveform = resampler(waveform)  
    input_values =  cropped_audio_copy
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription


def find_files(path, folder_B, ext1=".flac", ext2=".txt", need2continue='.trans.txt'):  
    for root, dirs, files in os.walk(path):  
        wav_files = []  # 用于存储当前文件夹下的所有.wav文件  
        for file in files:  
            if file.endswith(ext1):  # 如果文件是.wav文件，将其添加到wav_files列表中  
                wav_files.append(file)  
  
        if len(wav_files) <= 1:  # 如果文件夹中只有一个或没有.wav文件，则跳过当前文件夹  
            continue  
  
        for file in files:  
            if file.endswith(ext2):  # 如果文件是.normalized.txt文件 
                if file.endswith(need2continue):
                    continue
                txt_file = os.path.join(root, file)  
                corresponding_wav_file = file.replace(ext2, ext1)  # 找到与.normalized.txt文件名相同的.wav文件  

                wav_files.remove(corresponding_wav_file)  # 从wav_files列表中移除该.wav文件  
  
                random_wav_file = random.choice(wav_files)  # 从剩余的.wav文件中随机选择一个  
                random_wav_path = os.path.join(root, random_wav_file)  # 构建随机.wav文件的完整路径  
                wav_files.append(corresponding_wav_file)  # 将之前移除的.wav文件添加回wav_files列表  
  
                # 重命名文件并移动到文件夹B  
                subfolder = os.path.relpath(root, path)  # 获取子文件夹的相对路径  
                os.makedirs(os.path.join(folder_B, subfolder), exist_ok=True)  # 在文件夹B中创建相应的子文件夹  
  
                # 找到与random_wav_file名称相同的.normalized.txt文件  
                same_name_txt_file = random_wav_file.replace(ext1, ext2)  

                file_base = re.sub(r'(\..*)', '', file)  
                prompt_wav_file = file_base + ext1  
                prompt_txt_file = file_base + ext2
                # 构建新文件名  
                prompt_wav_file = "prompt_" + prompt_wav_file  
                prompt_txt_file = "prompt_" + prompt_txt_file  
                gen_txt_file = "gen_" + file  
                gt_wav_file = "gt_" + corresponding_wav_file  

                import torchaudio  
    
                def random_crop_audio(audio_path, crop_duration):  
                    waveform, sr = torchaudio.load(audio_path)  
                    total_duration = waveform.shape[1] / sr  
                    if total_duration <= crop_duration:  
                        return None  
                    max_start = waveform.shape[1] - int(crop_duration * sr)  
                    start = random.randint(0, max_start)  
                    end = start + int(crop_duration * sr)  
                    return waveform[:, start:end], sr
                

                # 在复制并重命名文件之前添加以下代码  
                cropped_audio, sr = random_crop_audio(os.path.join(root, random_wav_file), 3)  

                # must >= 3s between 4s and 10s
                if cropped_audio is None:  
                    continue  

                txt = transcript(cropped_audio, sr, target_sample_rate=16000)


                os.makedirs(os.path.join(folder_B, subfolder), exist_ok=True)
                torchaudio.save(os.path.join(folder_B, subfolder, prompt_wav_file), cropped_audio, sr)  

                with open(os.path.join(folder_B, subfolder, prompt_txt_file), "w") as f:  
                    f.write(txt) 
                
                # 移动并重命名文件  
                # shutil.copy(os.path.join(root, random_wav_file), os.path.join(folder_B, subfolder, prompt_wav_file))  
                # shutil.copy(os.path.join(root, same_name_txt_file), os.path.join(folder_B, subfolder, prompt_txt_file))  
                shutil.copy(os.path.join(root, file), os.path.join(folder_B, subfolder, gen_txt_file))  
                shutil.copy(os.path.join(root, corresponding_wav_file), os.path.join(folder_B, subfolder, gt_wav_file))   


if __name__ == "__main__":
    fire.Fire(find_files)
