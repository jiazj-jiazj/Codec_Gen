
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

def asr_dir(file_path):
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    # 检查GPU是否可用，并将模型移动到GPU  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    model = model.to(device) 
  
    waveform, sr = resample_audio(file_path, 16000)  
    # 将音频数据移动到GPU  
    input_values = waveform.to(device)  
    # 使用模型进行预测  
    with torch.no_grad():  
        logits = model(input_values).logits 
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0]) 
        
        print(transcription) 
        # print(output_file_path)


if __name__ == "__main__":  
    fire.Fire(asr_dir)   