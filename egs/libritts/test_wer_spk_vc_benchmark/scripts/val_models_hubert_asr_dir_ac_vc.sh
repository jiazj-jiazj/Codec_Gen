# is_gt=False
# input_dir="/home/v-zhijunjia/data/accent_iclr/ASI/tune_30mins/ASI_10ases_tgt1spker_lr_0005_source-topk-1-epoch-5_pretrain_2023-09-26_15:26:10"

# output_dir=${input_dir}_txt
# # input_dir="/mnt/users/jiazhijun/data/test_vc/bench_mark_vctk/source_wav_txt"
# CUDA_VISIBLE_DEVICES=0 python egs/libritts/test_wer_spk_vc_benchmark/hubert_asr_ls960.py \
#     --is_gt ${is_gt} \
#     --input_dir ${input_dir} \
#     --output_dir ${output_dir}

is_gt=False

  
for epoch in {13..14}; do  
    input_dir=/home/v-zhijunjia/data/accent_iclr/val_ac_models/converted_can_del/ac_benchmark_VALLFE_cases_pretrain_initial_epoch_${epoch}_2023-11-29_22:33:48  
    output_base_dir=${input_dir}_wav_txt 
    # current_input_dir="${input_dir}-epoch-${epoch}_pretrain_2023-09-26_15:26:10"  
    # output_dir="${current_input_dir}_txt"  
    CUDA_VISIBLE_DEVICES=0 python egs/libritts/test_wer_spk_vc_benchmark/hubert_asr_ls960.py \
        --is_gt ${is_gt} \
        --input_dir ${input_dir} \
        --output_dir ${output_base_dir}
done  
