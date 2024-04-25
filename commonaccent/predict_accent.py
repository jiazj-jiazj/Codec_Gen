import torchaudio
from speechbrain.pretrained.interfaces import foreign_class

classifier = foreign_class(source="Jzuluaga/accent-id-commonaccent_xlsr-en-english", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

# US Accent Example
out_prob, score, index, text_lab = classifier.classify_file('/home/v-zhijunjia/data/icml_more_accent/converted_can_del/dns_vctk_20_cases_IndicTTS_indian_native2all_native_txt_infilling_all_cases_tgt_4_speakers_lr_0_001_topk_2_epoch__top_k_stage2_10_2024-03-28_15:52:15/prompt1_arctic_a0003_20240328_075421_sys2_arctic_a0003_model3_ar_epoch-10pt_nar_epoch-40pt_2024_03_28_07_52_17_1_0.wav')
print(text_lab)

# Philippines Example
out_prob, score, index, text_lab = classifier.classify_file('/home/v-zhijunjia/data/icml_more_accent/converted_can_del/dns_vctk_20_cases_IndicTTS_indian_native2all_native_txt_infilling_all_cases_tgt_4_speakers_lr_0_001_topk_2_epoch__top_k_stage2_10_2024-03-28_15:52:15/prompt1_arctic_a0005_sys2_arctic_a0005_model3_ar_epoch-10pt_nar_epoch-40pt_2024_03_28_07_52_17_1_0.wav')
print(text_lab)
