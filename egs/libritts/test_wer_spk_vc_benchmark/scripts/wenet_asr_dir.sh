is_gt=False
input_dir="/home/v-zhijunjia/zhijundata_small_v2/data_local/accent_wer/val_ac_models/IndicTTS_indian_native2all_native_txt_infilling_all_cases"
output_dir="/home/v-zhijunjia/zhijundata_small_v2/data_local/accent_wer/val_ac_models/IndicTTS_indian_native2all_native_txt_infilling_all_cases_wenet_txt"
# input_dir="/mnt/users/jiazhijun/data/test_vc/bench_mark_vctk/source_wav_txt"
CUDA_VISIBLE_DEVICES=0 python egs/libritts/test_wer_spk_vc_benchmark/wenet_asr_ls960.py \
    --is_gt ${is_gt} \
    --input_dir ${input_dir} \
    --output_dir ${output_dir}