import torch
from transformers import Wav2Vec2Processor, HubertForCTC
from datasets import load_dataset

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# 使用批量大小为 4 的样本  
batch_size = 4  
audio_samples = [ds[i]["audio"]["array"] for i in range(batch_size)] 

input_values = processor(audio_samples, return_tensors="pt", padding=True).input_values

logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)  
  
# 将预测的 ID 解码为文本  
transcriptions = [processor.decode(pred_ids) for pred_ids in predicted_ids]  
print(transcriptions)
# ->"A MAN SAID TO THE UNIVERSE SIR I EXIST"
