import os
import gradio as gr
import torch
import pydub
import torchaudio
from torchaudio.sox_effects import apply_effects_tensor
import numpy as np
import re
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
torch.backends.cudnn.enabled = False  
import torchaudio.transforms as T  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def load_audio(file_name):
    audio = pydub.AudioSegment.from_file(file_name)
    arr = np.array(audio.get_array_of_samples(), dtype=np.float32)
    arr = arr / (1 << (8 * audio.sample_width - 1))
    return arr.astype(np.float32), audio.frame_rate

model_name = "microsoft/wavlm-base-plus-sv"
cosine_sim = torch.nn.CosineSimilarity(dim=-1)

EFFECTS_other = [
    ["remix", "-"],
    ["channels", "1"],
    ["rate", "16000"],  
    ["gain", "-1.0"],
    ["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
    ["trim", "0", "10"],
]
EFFECTS_16k = [
    ["remix", "-"],
    ["channels", "1"],
    ["gain", "-1.0"],
    ["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
    ["trim", "0", "10"],
]

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioXVector.from_pretrained(model_name).to(device)
THRESHOLD = 0.85

def similarity_fn(path1, path2):
    if not (path1 and path2):
        return '<b style="color:red">ERROR: Please record audio for *both* speakers!</b>'
    
    wav1, sr1 = load_audio(path1)
    if sr1==16000:
        wav1, _ = apply_effects_tensor(torch.tensor(wav1).unsqueeze(0), sr1, EFFECTS_16k)
    else:
        wav1, _ = apply_effects_tensor(torch.tensor(wav1).unsqueeze(0), sr1, EFFECTS_other)

    wav2, sr2 = load_audio(path2)

    if sr2==16000:
        wav2, _ = apply_effects_tensor(torch.tensor(wav2).unsqueeze(0), sr2, EFFECTS_16k)
    else:
        wav2, _ = apply_effects_tensor(torch.tensor(wav2).unsqueeze(0), sr2, EFFECTS_other)

    input1 = feature_extractor(wav1.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values.to(device)
    input2 = feature_extractor(wav2.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values.to(device)

    with torch.no_grad():
        emb1 = model(input1).embeddings
        emb2 = model(input2).embeddings
    emb1 = torch.nn.functional.normalize(emb1, dim=-1).cpu()
    emb2 = torch.nn.functional.normalize(emb2, dim=-1).cpu()
    similarity = cosine_sim(emb1, emb2).numpy()[0]

    return similarity

def get_file_path(folder, speaker, chapter, file_name):  
    return os.path.join(folder, speaker, chapter, file_name)  
  
def find_matching_gen_file_names(speaker, chapter, gt_file_name, prefix="begin1", is_gt=False):  
    original_file_name = gt_file_name[len("gt_"):].split(".")[0]  
    if is_gt is True:
        pattern = f"{prefix}_{original_file_name}.*\\.flac$"
    else:
        pattern = f"{prefix}_{original_file_name}_.*\\.wav$"  
      
    gen_speaker_chapter_folder = os.path.join(gen_folder, speaker, chapter)  
    matching_files = [f for f in os.listdir(gen_speaker_chapter_folder) if re.match(pattern, f)]  
      
    return matching_files  

if __name__ == "__main__":

    gt_folder = "/dev_huaying/zhijun/data/LibriSpeech/filtered_4s_10s-test-spk-wer"
    gen_folder = "/dev_huaying/zhijun/data/LibriSpeech/filtered_4s_10s-test-spk-wer"
    out_folder = "/dev_huaying/zhijun/valle_23_4_22/egs/libritts/log"
    os.makedirs(out_folder, exist_ok=True)
    is_gt = True
    prefix = "prompt"

    total_simm = 0
    num = 0
    for root, dirs, files in os.walk(gt_folder):
        for file_name in files:
            if file_name.startswith("gt_") and (file_name.endswith(".flac") or file_name.endswith(".wav")):
                print(file_name)
                relative_path = os.path.relpath(root, gt_folder)
                speaker, chapter = relative_path.split(os.sep)
                gt_file_path = get_file_path(gt_folder, speaker, chapter, file_name)
                print(gt_file_path)  
                gen_file_names = find_matching_gen_file_names(speaker, chapter, file_name, prefix, is_gt)                                                 
                print(gen_file_names)

                for gen_file_name in gen_file_names:
                    gen_file_path = get_file_path(gen_folder, speaker, chapter, gen_file_name)

                    simm = similarity_fn(gt_file_path, gen_file_path)
                    total_simm+=simm
                    num+=1
                    print(simm)
                    
                #     with open(os.path.join(out_folder, prefix+"spk.txt"), "w") as f:
                #         f.write(str(total_simm / num)) 
                # quit()

    with open(os.path.join(out_folder, prefix+"spk.txt"), "w") as f:
        f.write(str(total_simm / num))

