#!/bin/bash  

# gt
# gen_folder="/dev_huaying/zhijun/data/test_vc/benchmark_librispeech/asr_yourtts_v0"
# gen_folder="/mnt/users/jiazhijun/data/test_vc/bench_mark_vctk/source_wav_txt"  
# gen_folder="/dev_huaying/zhijun/data/benchmark_librispeech_10speakers/asr_yourtts_v0"
# gen_folder="/home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/asr_pretrain_mode_5_mask_0_15"

# gen_folder="/home/v-zhijunjia/data/accent_iclr/ASI/tune_30mins/ASI_10ases_tgt1spker_lr_0005_source-topk-2-epoch-6_pretrain_2023-09-26_16:05:31"



# is_gt=False  
# prefix="prompt"  
# dataset="p248"
# gt_folder="/scratch/data/l1_l2_arctic/combine_L1_L2/train_native/total/cmu_us_bdl_arctic/transcript"

# # valle_ours
# gen_folder="/home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/gen_wavs_tfcodec_ours"  
# is_gt=False  
# prefix="valle_ours"  

# tfnet_sem_ours

# # vctk ac_benchmark
# gen_folder="/home/v-zhijunjia/data/accent_iclr/IndicTTS_l1_l2_txt"  
# is_gt=False  
# # prefix="valle_ours"  
# dataset="tts_benchmark_one_dir"
# gt_folder="/home/v-zhijunjia/data/accent_iclr/p248_txt"
# CUDA_VISIBLE_DEVICES=0 python -u egs/libritts/test_wer_spk_vc_benchmark/compute_wer_dir.py \
#     --gt_folder "${gt_folder}" \
#     --gen_folder "${gen_folder}" \
#     --is_gt "${is_gt}" \
#     --prefix "${prefix}" \
#     --dataset "${dataset}"
#     # > /dev_huaying/zhijun/valle_23_4_22/egs/libritts/test_wer_spk/log/${prefix}_wer.log 2>&1 & 

# ac_benchmark
gen_folder="/home/v-zhijunjia/zhijundata_small_v2/data_local/accent_wer/val_ac_models/IndicTTS_indian_native2all_native_txt_infilling_all_cases_wav2vec2_txt"  
is_gt=False  
# prefix="valle_ours"  
dataset="tts_benchmark_one_dir"
gt_folder="/home/v-zhijunjia/zhijundata_small_v2/data_local/accent_wer/val_ac_models/indian_accent_test_arctics_50cases_gt_txt"
CUDA_VISIBLE_DEVICES=0 python -u egs/libritts/test_wer_spk_vc_benchmark/compute_wer_dir.py \
    --gt_folder "${gt_folder}" \
    --gen_folder "${gen_folder}" \
    --is_gt "${is_gt}" \
    --prefix "${prefix}" \
    --dataset "${dataset}"
    # > /dev_huaying/zhijun/valle_23_4_22/egs/libritts/test_wer_spk/log/${prefix}_wer.log 2>&1 & 

# # tts_benchmark
# gen_folder="/home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/converted_can_del/no-prompt-tts-2stages_topk_stage1_3_2024-01-23_00:07:07_txt"  
# is_gt=False  
# # prefix="valle_ours"  
# dataset="tts_benchmark_one_dir"
# gt_folder="/home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/gt_gen_txt"
# CUDA_VISIBLE_DEVICES=0 python -u egs/libritts/test_wer_spk_vc_benchmark/compute_wer_dir.py \
#     --gt_folder "${gt_folder}" \
#     --gen_folder "${gen_folder}" \
#     --is_gt "${is_gt}" \
#     --prefix "${prefix}" \
#     --dataset "${dataset}"
#     # > /dev_huaying/zhijun/valle_23_4_22/egs/libritts/test_wer_spk/log/${prefix}_wer.log 2>&1 & 

# # vc_benchmark
# gen_folder="/home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/converted_test_can_del/soundstorm_test_steps_16_topk_70_txt"  
# is_gt=False  
# # prefix="valle_ours"  
# dataset="vc_benchmark"
# gt_folder="/home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/gt_txt"
# CUDA_VISIBLE_DEVICES=0 python -u egs/libritts/test_wer_spk_vc_benchmark/compute_wer_dir.py \
#     --gt_folder "${gt_folder}" \
#     --gen_folder "${gen_folder}" \
#     --is_gt "${is_gt}" \
#     --prefix "${prefix}" \
#     --dataset "${dataset}"
#     # > /dev_huaying/zhijun/valle_23_4_22/egs/libritts/test_wer_spk/log/${prefix}_wer.log 2>&1 &  
# result = model.transcribe('/home/v-zhijunjia/data/accent_iclr/ac_baseline_20cases/p248_003.wav')
# model = wenet.load_model('english')
