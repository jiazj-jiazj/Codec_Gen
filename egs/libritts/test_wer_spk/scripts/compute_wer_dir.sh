#!/bin/bash  

# gt
gen_folder="/home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/gen_wavs_tfcodec_ours"  
is_gt=False  
prefix="valle_ours"  

# # valle_ours
# gen_folder="/home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/gen_wavs_styletts_v2"  
# is_gt=False  
# prefix="valle_ours"  

  
CUDA_VISIBLE_DEVICES=0 python egs/libritts/test_wer_spk/compute_wer_dir.py \
    --gt_folder "/home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/filtered_4s_10s-test-spk-wer" \
    --gen_folder "${gen_folder}" \
    --is_gt "${is_gt}" \
    --prefix "${prefix}"
