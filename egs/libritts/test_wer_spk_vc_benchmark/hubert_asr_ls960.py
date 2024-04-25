
import torch
import torchaudio
from transformers import Wav2Vec2Processor, HubertForCTC
from datasets import load_dataset
import torchaudio.transforms as T  
import os
torch.backends.cudnn.enabled = False  
import fire

def resample_audio(audio_path, target_sample_rate):  
    waveform, sr = torchaudio.load(audio_path)  
    if sr != target_sample_rate:  
        resampler = T.Resample(sr, target_sample_rate)  
        waveform = resampler(waveform)  
    return waveform, target_sample_rate

def asr_dir(input_dir, is_gt, output_dir, asr_type="hubert_asr_ls960"):
    if asr_type =="hubert_asr_ls960":
        os.makedirs(output_dir, exist_ok=True)
        processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        # 检查GPU是否可用，并将模型移动到GPU  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        model = model.to(device) 
        for root, dirs, files in os.walk(input_dir):  
            
            # 处理文件  
            for file_name in files:  
                file_path = os.path.join(root, file_name)  
                # if is_gt is True:
                #     if file_name.startswith("gt") is not True:
                #         continue

                # 检查文件后缀是否为 .wav  
                if file_path.endswith(".wav") or file_path.endswith(".flac"):  
                    print(f"Processing: {file_path}")  
                    waveform, sr = resample_audio(file_path, 16000)  
                    # 将音频数据移动到GPU  
                    input_values = waveform.to(device)  
                    # 使用模型进行预测  
                    with torch.no_grad():  
                        logits = model(input_values).logits 
                        predicted_ids = torch.argmax(logits, dim=-1)
                        transcription = processor.decode(predicted_ids[0]) 
                        
                        print(transcription)
                        # 替换后缀为 .txt 并创建输出文件路径  
                        # txt_file_name = file_name.replace(".wav", ".txt").replace(".flac", ".txt") 
                        if is_gt is True:
                            txt_file_name = file_name[:-4]+".txt" #p225_001.wav
                        else:
                            # txt_file_name = file_name[:17]+".txt" #p225_001.wav
                            txt_file_name = file_name.replace('.wav', '.txt')
                        # print(txt_file_name)
                        output_file_path = os.path.join(output_dir, txt_file_name)
                        # print(output_file_path)

                        # 将transcript写入到新的txt文件中  
                        with open(output_file_path, "w") as output_file:  
                            output_file.write(transcription) 
                        # print(output_file_path)
    elif asr_type =="espnet_common_voice":

        for root, dirs, files in os.walk(input_dir):  
            
            # 处理文件  
            for file_name in files:  
                file_path = os.path.join(root, file_name)  
                # if is_gt is True:
                #     if file_name.startswith("gt") is not True:
                #         continue

                # 检查文件后缀是否为 .wav  
                if file_path.endswith(".wav") or file_path.endswith(".flac"):  
                    print(f"Processing: {file_path}")  
                    waveform, sr = resample_audio(file_path, 16000)  
                    # 将音频数据移动到GPU  
                    input_values = waveform.to(device)  
                    # 使用模型进行预测  
                    with torch.no_grad():  
                        logits = model(input_values).logits 
                        predicted_ids = torch.argmax(logits, dim=-1)
                        transcription = processor.decode(predicted_ids[0]) 
                        
                        print(transcription)
                        # 替换后缀为 .txt 并创建输出文件路径  
                        # txt_file_name = file_name.replace(".wav", ".txt").replace(".flac", ".txt") 
                        if is_gt is True:
                            txt_file_name = file_name[:-4]+".txt" #p225_001.wav
                        else:
                            # txt_file_name = file_name[:17]+".txt" #p225_001.wav
                            txt_file_name = file_name.replace('.wav', '.txt')
                        # print(txt_file_name)
                        output_file_path = os.path.join(output_dir, txt_file_name)
                        # print(output_file_path)

                        # 将transcript写入到新的txt文件中  
                        with open(output_file_path, "w") as output_file:  
                            output_file.write(transcription) 


if __name__ == "__main__":  
    fire.Fire(asr_dir)   