# num_runs=12 
# top_k=-1
# add_prenet="False"
# for ((i=1; i<=num_runs; i++))  
# do  
#   echo "Running iteration $i"  
#   CUDA_VISIBLE_DEVICES=0 python egs/libritts/bin/combine_ar_nar.py --output-dir /dev_huaying/zhijun/data/test_valle_styleTTS_yourtts_naturalspeech2/nar_v2_prefix-mode_1_top_${top_k}_${add_prenet}_prenet_best_valid_best_valid --model-name valle --norm-first true --add-prenet ${add_prenet} --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
#   --text-prompts "looked out and tens the fives。" --text-tokens /dev_huaying/zhijun/data/valle-tensorboard-models/en_unique_text_tokens.k2symbols \
#   --audio-prompts /dev_huaying/zhijun/data/test_valle_naturalspeech2_yourtts_styleTTS/test1/reference_LibriSpeech_1st_txt_looktown.wav \
#   --text "And lay me down in thy cold bed and leave my shining lot." \
#   --prefix-mode 1 \
#   --checkpoint1 /dev_huaying/zhijun/data/valle-tensorboard-models/ar/no_prenet_Name_VALLE_max-duration_80_dtype_float32_base-lr_0.01_world-size_8_train-stage_1_echo_50_start_echo_1_accumulate_grad_steps_4_2023_06_14_16_56_24/epoch-50.pt \
#   --checkpoint2 /dev_huaying/zhijun/data/valle-tensorboard-models/nar/noprenet_Name_VALLE_max-duration_70_dtype_float32_base-lr_0.01_world-size_8_train-stage_2_echo_200_start_echo_1_accumulate_grad_steps_4_2023_06_14_16_32_03/best-valid-loss.pt \
#   --top-k ${top_k} --temperature 1.0
# done  

# test vc base
# for steps in 1 2 4 8 32 64; do  
# top_k_know_token_stage2=10

# vc group_in_mask 
# for steps in 1 2 4 8 32 64; do
top_k_know_token=10
for steps in 16 32; do  
    for top_k_know_token in 30 50 70; do
        basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
        num_runs=1
        add_prenet="False"  
        top_k=70
        CUDA_VISIBLE_DEVICES=0 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py --model-name soundstorm --norm-first true --add-prenet "False" \
        --decoder-dim 1024 --nhead 16 \
        --num-decoder-layers 12 \
        --share-embedding true \
        --nums ${num_runs} \
        --semantic-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
        --semantic-sys-dir /home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/source \
        --audio-prompts-dir /home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/prompt \
        --input-semantic True \
        --only-autoregressive True \
        --prefix-mode 1 \
        --checkpoint1 /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_50_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True_2024_03_11_02_25_30_2024_03_11_02_25_30/epoch-50.pt \
        --shared-linear False \
        --temperature 1.0 \
        --num-quantizers 16 \
        --input-codec 2 \
        --target-mode 0 \
        --accent-remove False \
        --prompt-pre-cut False \
        --semantic-depup False \
        --semantic-type 0 \
        --mode 0 \
        --mode-stage2 0 \
        --top-k ${top_k} \
        --top-k-know-token ${top_k_know_token} \
        --soundstorm-steps ${steps} \
        --outputdir-name converted_test_can_del/soundstorm_baseinfer_steps_${steps}_topk_${top_k}_top_k_know_token_${top_k_know_token} \
        > /home/v-zhijunjia/CodecGen/egs/libritts/log/soundstorm_gr_ar_test_steps_${steps}_topk_${top_k}_${basestr}.txt 2>&1 &
        # --outputdir-name converted_test_can_del/soundstorm_test_steps_${steps}_topk_${top_k} > /home/v-zhijunjia/CodecGen/egs/libritts/log/soundstorm_test_steps_${steps}_topk_${top_k}_${basestr}.txt 2>&1 &
    done
done


# # vc benchmark
# # for steps in 1 2 4 8 32 64; do  
# for steps in 16 32; do  
#     basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
#     num_runs=1
#     add_prenet="False"  
#     top_k=70
#     CUDA_VISIBLE_DEVICES=0 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py --model-name soundstorm --norm-first true --add-prenet "False" \
#     --decoder-dim 1024 --nhead 16 \
#     --num-decoder-layers 12 \
#     --share-embedding true \
#     --nums ${num_runs} \
#     --semantic-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
#     --semantic-sys-dir /home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/source \
#     --audio-prompts-dir /home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/prompt \
#     --input-semantic True \
#     --only-autoregressive True \
#     --prefix-mode 1 \
#     --checkpoint1 /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_100_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_2024_01_15_15_45_24/epoch-40.pt \
#     --shared-linear False \
#     --temperature 1.0 \
#     --num-quantizers 16 \
#     --input-codec 2 \
#     --target-mode 0 \
#     --accent-remove False \
#     --prompt-pre-cut False \
#     --semantic-depup False \
#     --semantic-type 0 \
#     --mode 0 \
#     --mode-stage2 0 \
#     --top-k ${top_k} \
#     --soundstorm-steps ${steps} \
#     --outputdir-name converted_test_can_del/soundstorm_baseinfer_steps_${steps}_topk_${top_k}_ \
#     > /home/v-zhijunjia/CodecGen/egs/libritts/log/soundstorm_gr_ar_test_steps_${steps}_topk_${top_k}_${basestr}.txt 2>&1 &
#     # --outputdir-name converted_test_can_del/soundstorm_test_steps_${steps}_topk_${top_k} > /home/v-zhijunjia/CodecGen/egs/libritts/log/soundstorm_test_steps_${steps}_topk_${top_k}_${basestr}.txt 2>&1 &
# done
# converted_test_can_del/3times_3s_search_concact_prompt_files_3_3_6s_p_0_6_test_LibrSpeech_6s_prompt_0_6_prompt_3_3s_vc_few_shot_${top_k}_topp_${top_p}_${basestr}  
# /mnt/users/jiazhijun/dat/valle-tensorboard-models/accents/ar_accents_restore/restore_from_vc_libritts.pt

# # soundstorm train random infer_group_ar
# for steps in 32 64; do  
#     basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
#     num_runs=1
#     add_prenet="False"  
#     top_k=70
#     CUDA_VISIBLE_DEVICES=0 python egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py --model-name soundstorm --norm-first true --add-prenet "False" \
#     --decoder-dim 1024 --nhead 16 \
#     --num-decoder-layers 12 \
#     --share-embedding true \
#     --nums ${num_runs} \
#     --semantic-tokens /scratch/data/Libritts/tokenized_tfnet_semantic_token/unique_semantic_tokens.k2symbols \
#     --semantic-sys-dir /home/v-zhijunjia/data/daata_update/benchmark_librispeech_10speakers/source \
#     --audio-prompts-dir /home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/prompt \
#     --input-semantic True \
#     --only-autoregressive True \
#     --prefix-mode 1 \
#     --checkpoint1 /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_100_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_2024_01_15_15_45_24/epoch-40.pt \
#     --shared-linear False \
#     --temperature 1.0 \
#     --num-quantizers 16 \
#     --input-codec 2 \
#     --target-mode 0 \
#     --accent-remove False \
#     --prompt-pre-3s False \
#     --semantic-depup False \
#     --semantic-type 0 \
#     --mode 0 \
#     --mode-stage2 0 \
#     --top-k ${top_k} \
#     --soundstorm-steps ${steps} \
#     --outputdir-name converted_test_can_del/soundstorm_test_steps_${steps}_topk_${top_k}_group_ar 
#     # > /home/v-zhijunjia/CodecGen/egs/libritts/log/soundstorm_gr_ar_test_steps_${steps}_topk_${top_k}_${basestr}.txt 2>&1 &
#     # --outputdir-name converted_test_can_del/soundstorm_test_steps_${steps}_topk_${top_k} > /home/v-zhijunjia/CodecGen/egs/libritts/log/soundstorm_test_steps_${steps}_topk_${top_k}_${basestr}.txt 2>&1 &
# done


# CUDA_VISIBLE_DEVICES=0 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py --model-name valle --norm-first true --add-prenet ${add_prenet} --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
# --share-embedding true \
# --nums ${num_runs} \
# --semantic-tokens /mnt/zhijun/LibriTTS/data/vc_tokenized_16k_tfcodec/unique_semantic_tokens.k2symbols \
# --semantic-sys-dir /dev_huaying/zhijun/data/test_vc/test_demo/source \
# --audio-prompts-dir /dev_huaying/zhijun/data/test_vc/test_demo/prompt \
# --input-semantic True \
# --only-autoregressive True \
# --prefix-mode 1 \
# --checkpoint1 /dev_huaying/zhijun/data/valle-tensorboard-models/vc/only_ar/epoch-40.pt \
# --top-k ${top_k} --temperature 1.0 \
# --num-quantizers 16 \
# --input-codec 1 \
# --outputdir-name "converted_vc_onlyar_v0" \
# > egs/libritts/log/vc_dir_libritts_tfcodecs_onlyar_${basestr}.txt 2>&1 &
# # /mnt/users/jiazhijun/data/valle-tensorboard-models/accents/ar_accents_restore/restore_from_vc_libritts.pt

