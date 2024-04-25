is_gt=False
input_dir="/home/v-zhijunjia/zhijundata_small_v2/data_local/accent_wer/val_ac_models/native_ac_50cases"
output_dir="/home/v-zhijunjia/zhijundata_small_v2/data_local/accent_wer/val_ac_models/native_ac_50cases_txt_v2"
# input_dir="/mnt/users/jiazhijun/data/test_vc/bench_mark_vctk/source_wav_txt"
CUDA_VISIBLE_DEVICES=0 python egs/libritts/test_wer_spk_vc_benchmark/hubert_asr_ls960.py \
    --is_gt ${is_gt} \
    --input_dir ${input_dir} \
    --output_dir ${output_dir}