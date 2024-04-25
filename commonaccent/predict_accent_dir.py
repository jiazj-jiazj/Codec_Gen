import torchaudio
from speechbrain.pretrained.interfaces import foreign_class

classifier = foreign_class(source="Jzuluaga/accent-id-commonaccent_xlsr-en-english", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

# US Accent Example

import os  
  
def classify_folder(classifier, folder_path):  
    # Get a list of all files in the folder  
    files = os.listdir(folder_path)  
      
    # Filter the list to include only .wav files  
    wav_files = [f for f in files if f.endswith('.wav')]  
    
    totol_lens = len(wav_files)
    # Initialize a dictionary to store the results  
    results = {}  
    native_number = 0
    # Classify each file  
    for wav_file in wav_files:  
        full_path = os.path.join(folder_path, wav_file)  
        out_prob, score, index, text_lab = classifier.classify_file(full_path)  
        results[wav_file] = (out_prob, score, index, text_lab)  

        print(text_lab)
        if text_lab[0] in ['us', 'england', 'canada', 'scotland']:
            native_number+=1
    print(f'native_numbe:{native_number}')
    print(f'totol_lens:{totol_lens}')
    return native_number/totol_lens
  
# Use the function  
folder_path = '/home/v-zhijunjia/data/icml_more_accent/Korean_/converted_can_del/dns_vctk_20_cases_IndicTTS_indian_native2all_native_txt_infilling_all_cases_tgt_4_speakers_lr_0_001_topk_2_epoch__top_k_stage2_10_2024-03-28_17:09:23'  
results = classify_folder(classifier, folder_path)  

print(results)
# # Print the results  
# for file, result in results.items():  
#     print(f'File: {file}, Result: {result}')  
