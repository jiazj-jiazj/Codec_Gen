import soundfile as sf
import torch
import fire
import torch.nn.functional as F
from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL
import os
import re

MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]
def get_file_path(folder, speaker, chapter, file_name):  
    return os.path.join(folder, speaker, chapter, file_name)  
  
def find_matching_gen_file_names(gen_folder, speaker, chapter, gt_file_name, prefix="begin1", is_gt=False):  
    original_file_name = gt_file_name[len("gt_"):].split(".")[0]  
    if is_gt is True:
        pattern = f"{prefix}_{original_file_name}.*\\.flac$"
    else:
        pattern = f"{prefix}_{original_file_name}_.*\\.wav$"  
      
    gen_speaker_chapter_folder = os.path.join(gen_folder, speaker, chapter)  
    matching_files = [f for f in os.listdir(gen_speaker_chapter_folder) if re.match(pattern, f)]  
      
    return matching_files  

def init_model(model_name, checkpoint=None):
    if model_name == 'unispeech_sat':
        config_path = 'config/unispeech_sat.th'
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='unispeech_sat', config_path=config_path)
    elif model_name == 'wavlm_base_plus':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=config_path)
    elif model_name == 'wavlm_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
    elif model_name == 'hubert_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=config_path)
    elif model_name == 'wav2vec2_xlsr':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=config_path)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model

def similarity_fn(model, wav1, wav2, use_gpu=True):
    wav1, sr1 = sf.read(wav1)
    wav2, sr2 = sf.read(wav2)

    wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
    wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    resample2 = Resample(orig_freq=sr2, new_freq=16000)
    wav1 = resample1(wav1)
    wav2 = resample2(wav2)

    if use_gpu:
        model = model.cuda()
        wav1 = wav1.cuda()
        wav2 = wav2.cuda()

    model.eval()
    with torch.no_grad():
        emb1 = model(wav1)
        emb2 = model(wav2)

    sim = F.cosine_similarity(emb1, emb2)
    return sim[0].item()

def verification(model_name,  gt_folder, gen_folder, out_folder, is_gt, prefix, use_gpu=True, checkpoint=None):

    assert model_name in MODEL_LIST, 'The model_name should be in {}'.format(MODEL_LIST)
    model = init_model(model_name, checkpoint)
    os.makedirs(out_folder, exist_ok=True)    
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
                gen_file_names = find_matching_gen_file_names(gen_folder, speaker, chapter, file_name, prefix, is_gt)                                                 
                print(gen_file_names)

                for gen_file_name in gen_file_names:
                    gen_file_path = get_file_path(gen_folder, speaker, chapter, gen_file_name)

                    simm = similarity_fn(model, gt_file_path, gen_file_path, use_gpu)
                    total_simm+=simm
                    num+=1


    with open(os.path.join(out_folder, prefix+"_wav_tdnn_spk.txt"), "w") as f:
        f.write(str(total_simm / num))
        f.write(str(num))




if __name__ == "__main__":
    fire.Fire(verification)

