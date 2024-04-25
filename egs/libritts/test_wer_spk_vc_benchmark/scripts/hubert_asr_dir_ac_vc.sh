# is_gt=False
# input_dir="/home/v-zhijunjia/data/accent_iclr/ASI/tune_30mins/ASI_10ases_tgt1spker_lr_0005_source-topk-1-epoch-5_pretrain_2023-09-26_15:26:10"

# output_dir=${input_dir}_txt
# # input_dir="/mnt/users/jiazhijun/data/test_vc/bench_mark_vctk/source_wav_txt"
# CUDA_VISIBLE_DEVICES=0 python egs/libritts/test_wer_spk_vc_benchmark/hubert_asr_ls960.py \
#     --is_gt ${is_gt} \
#     --input_dir ${input_dir} \
#     --output_dir ${output_dir}

# for checkpoint in $(seq 20 20 220); do  
input_dir=/home/v-zhijunjia/data/data_update/benchmark_vc_tts_9s/vc_source
output_base_dir=${input_dir}_txt  

CUDA_VISIBLE_DEVICES=0 python egs/libritts/test_wer_spk_vc_benchmark/hubert_asr_ls960.py \
    --is_gt ${is_gt} \
    --input_dir ${input_dir} \
    --output_dir ${output_base_dir}
# done  
