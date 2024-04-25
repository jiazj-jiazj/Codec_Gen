# Let's see how to retrieve time steps for a model
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
from datasets import load_dataset
import datasets
import torch
import librosa

# import model, feature extractor, tokenizer
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

def calculate_mean_variance(numbers):  
    # 计算均值  
    mean_value = sum(numbers) / len(numbers)  
    # 计算方差  
    variance_value = sum((x - mean_value) ** 2 for x in numbers) / (len(numbers) - 1)  
      
    return mean_value, variance_value 
total_i = {}
list_i = {}
for ii in range(10):
    file_path = f'/home/v-zhijunjia/data/accent_icml/cases_analysis/source_p248_004_trimmed.wav'  
    audio_data, sampling_rate = librosa.load(file_path, sr=16_000)  # 确保采样率为16kHz  

    input_values = feature_extractor(audio_data, return_tensors="pt", sampling_rate=sampling_rate).input_values  

    logits = model(input_values).logits[0]  
    pred_ids = torch.argmax(logits, axis=-1)

    # retrieve word stamps (analogous commands for `output_char_offsets`)
    outputs = tokenizer.decode(pred_ids, output_word_offsets=True)
    # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
    time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate

    word_offsets = [
        {
            "word": d["word"],
            "start_time": round(d["start_offset"] * time_offset, 2),
            "end_time": round(d["end_offset"] * time_offset, 2),
            "total_time": round(d["end_offset"] * time_offset, 2) - round(d["start_offset"] * time_offset, 2),
        }
        for d in outputs.word_offsets
    ]
    # compare word offsets with audio `en_train_0/common_voice_en_19121553.mp3` online on the dataset viewer:
    # https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/en

    last_end_time = 0
    ans_list = []
    if len(word_offsets)==15:

        num_0 = 0 
        for word in word_offsets:
            end_time = word["end_time"]
            total_time = end_time - last_end_time
            last_end_time = end_time
            ans_list.append(round(total_time, 2))
            if num_0 not in total_i.keys():
                list_i[num_0]= []
                total_i[num_0]=0.0

            total_i[num_0]+=round(total_time, 2)
            list_i[num_0].append(total_time)
            num_0+=1

final_list_mean = []
final_list_vari = []
for key in list_i.keys():
    list = list_i[key]
    mean, vari = calculate_mean_variance(list)
    final_list_mean.append(round(mean, 2))
    final_list_vari.append(vari)

print(f"final_list_mean:{final_list_mean}")
print(f"final_list_vari:{final_list_vari}")