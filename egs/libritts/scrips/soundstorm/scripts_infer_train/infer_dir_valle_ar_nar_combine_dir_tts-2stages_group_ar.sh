basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  

# num_runs=3
# num_runs_stage2=1
# top_k=2
# top_k_stage2=70
# add_prenet="False"
# for top_k in $(seq 2 1 2); do  
#     CUDA_VISIBLE_DEVICES=0 python egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#     --model-name vallfe \
#     --model-name-stage2 soundstorm \
#     --soundstorm-steps 16 \
#     --norm-first true \
#     --add-prenet ${add_prenet} \
#     --decoder-dim 1024 --nhead 16 \
#     --decoder-dim-stage2 1024 --nhead-stage2 16 \
#     --encoder-num-layers 6 \
#     --decoder-num-layers 6 \
#     --num-decoder-layers 12 \
#     --num-decoder-layers-stage2 12 \
#     --share-embedding true \
#     --nums ${num_runs} \
#     --nums-stage2 ${num_runs_stage2} \
#     --semantic-tokens /scratch/data/Libritts/tokenized/unique_semantic_tokens.k2symbols \
#     --text-tokens /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#     --test-benchmark False \
#     --semantic-sys-dir /home/v-zhijunjia/data/data_update/expore_gap_bettwen_tts_vc/tts/source \
#     --audio-prompts-dir /home/v-zhijunjia/data/data_update/expore_gap_bettwen_tts_vc/tts/prompt \
#     --task-id 1 \
#     --input-semantic True \
#     --only-autoregressive True \
#     --prefix-mode 1 \
#     --checkpoint1 /home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
#     --checkpoint2 /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_100_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_nar_mask_type_1_2024_01_18_13_58_20/epoch-40.pt \
#     --top-k ${top_k} \
#     --prepend-bos True \
#     --prepend-bos-stage2 False \
#     --semantic-depup False \
#     --top-k-stage2 ${top_k_stage2} \
#     --shared-linear-stage2 False \
#     --temperature 1.0 \
#     --prompt-pre-cut False \
#     --num-quantizers 1 \
#     --num-quantizers-stage2 16 \
#     --input-codec 1 \
#     --target-mode 2 \
#     --accent-remove True \
#     --mode 0 \
#     --mode-stage2 0 \
#     --is-pretrain True \
#     --pret-mode 0 \
#     --outputdir-name converted_can_del/tts-2stages-no-prompt-tts-2stages_topk_stage1_${top_k}_${basestr}
# done    

# tts  nar_mask_r_0.5_gp_i_mask_True
num_runs=1
num_runs_stage2=1
top_k=2
top_k_stage2=20
steps=16
add_prenet="False"
for steps in 12 14 18; do
    for top_k_stage2 in 15 25; do  
        CUDA_VISIBLE_DEVICES=0 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
        --model-name vallfe \
        --model-name-stage2 soundstorm \
        --soundstorm-steps ${steps} \
        --norm-first true \
        --add-prenet ${add_prenet} \
        --decoder-dim 1024 --nhead 16 \
        --decoder-dim-stage2 1024 --nhead-stage2 16 \
        --encoder-num-layers 6 \
        --decoder-num-layers 6 \
        --num-decoder-layers 12 \
        --num-decoder-layers-stage2 12 \
        --share-embedding true \
        --nums ${num_runs} \
        --nums-stage2 ${num_runs_stage2} \
        --nums-stage2 ${num_runs_stage2} \
        --semantic-tokens /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
        --text-tokens /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
        --test-benchmark True \
        --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer_min \
        --task-id 1 \
        --input-semantic True \
        --only-autoregressive True \
        --prefix-mode 1 \
        --checkpoint1 /home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
        --checkpoint2 /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_50_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True_2024_03_11_02_25_30_2024_03_11_02_25_30/epoch-50.pt \
        --top-k ${top_k} \
        --prepend-bos True \
        --prepend-bos-stage2 False \
        --semantic-depup False \
        --top-k-stage2 ${top_k_stage2} \
        --shared-linear-stage2 False \
        --temperature 1.0 \
        --prompt-pre-cut False \
        --num-quantizers 1 \
        --num-quantizers-stage2 16 \
        --input-codec 1 \
        --target-mode 2 \
        --accent-remove True \
        --mode 0 \
        --mode-stage2 0 \
        --soundstorm-type 1 \
        --is-pretrain True \
        --pret-mode 0 \
        --outputdir-name converted_can_del_tts/group_ar/nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True/no-prompt-tts-2stages_topk_stage1_${top_k}_topk_stage2_${top_k_stage2}_steps_${steps}_groupar_${basestr} \
        > /home/v-zhijunjia/CodecGen/egs/libritts/log/tts-2stages-soundstorm-group-ar_topk_stage1_${top_k}_${basestr}_step_${steps}.txt 2>&1 &
    done   
done

# # tts  nar_mask_t_5_nar_mask_r_0_5
# num_runs=1
# num_runs_stage2=1
# top_k=2
# top_k_stage2=20
# steps=16
# add_prenet="False"
# for steps in 1 2; do
#     for top_k_stage2 in 20; do  
#         CUDA_VISIBLE_DEVICES=0 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#         --model-name vallfe \
#         --model-name-stage2 soundstorm \
#         --soundstorm-steps ${steps} \
#         --norm-first true \
#         --add-prenet ${add_prenet} \
#         --decoder-dim 1024 --nhead 16 \
#         --decoder-dim-stage2 1024 --nhead-stage2 16 \
#         --encoder-num-layers 6 \
#         --decoder-num-layers 6 \
#         --num-decoder-layers 12 \
#         --num-decoder-layers-stage2 12 \
#         --share-embedding true \
#         --nums ${num_runs} \
#         --nums-stage2 ${num_runs_stage2} \
#         --nums-stage2 ${num_runs_stage2} \
#         --semantic-tokens /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
#         --text-tokens /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#         --test-benchmark True \
#         --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer_min \
#         --task-id 1 \
#         --input-semantic True \
#         --only-autoregressive True \
#         --prefix-mode 1 \
#         --checkpoint1 /home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
#         --checkpoint2 /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_40_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_nar_mask_t_5_nar_mask_r_0.5_tostep_line_2024_03_06_15_51_11/epoch-40.pt \
#         --top-k ${top_k} \
#         --prepend-bos True \
#         --prepend-bos-stage2 False \
#         --semantic-depup False \
#         --top-k-stage2 ${top_k_stage2} \
#         --shared-linear-stage2 False \
#         --temperature 1.0 \
#         --prompt-pre-cut False \
#         --num-quantizers 1 \
#         --num-quantizers-stage2 16 \
#         --input-codec 1 \
#         --target-mode 2 \
#         --accent-remove True \
#         --mode 0 \
#         --mode-stage2 0 \
#         --soundstorm-type 1 \
#         --is-pretrain True \
#         --pret-mode 0 \
#         --outputdir-name converted_can_del_tts/group_ar/nar_mask_t_5_nar_mask_r_0_5/no-prompt-tts-2stages_topk_stage1_${top_k}_topk_stage2_${top_k_stage2}_steps_${steps}_groupar_${basestr} \
#         > /home/v-zhijunjia/CodecGen/egs/libritts/log/tts-2stages-soundstorm-group-ar_topk_stage1_${top_k}_${basestr}_step_${steps}.txt 2>&1 &
#     done   
# done

# # tts  nar_mask_t_4_nar_mask_r_0_5
# num_runs=1
# num_runs_stage2=1
# top_k=2
# top_k_stage2=20
# steps=16
# add_prenet="False"
# for steps in 16 32 64 128; do
#     for top_k_stage2 in 20; do  
#         CUDA_VISIBLE_DEVICES=0 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#         --model-name vallfe \
#         --model-name-stage2 soundstorm \
#         --soundstorm-steps ${steps} \
#         --norm-first true \
#         --add-prenet ${add_prenet} \
#         --decoder-dim 1024 --nhead 16 \
#         --decoder-dim-stage2 1024 --nhead-stage2 16 \
#         --encoder-num-layers 6 \
#         --decoder-num-layers 6 \
#         --num-decoder-layers 12 \
#         --num-decoder-layers-stage2 12 \
#         --share-embedding true \
#         --nums ${num_runs} \
#         --nums-stage2 ${num_runs_stage2} \
#         --nums-stage2 ${num_runs_stage2} \
#         --semantic-tokens /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
#         --text-tokens /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#         --test-benchmark True \
#         --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer_min \
#         --task-id 1 \
#         --input-semantic True \
#         --only-autoregressive True \
#         --prefix-mode 1 \
#         --checkpoint1 /home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
#         --checkpoint2 /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_40_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_nar_mask_t_4_nar_mask_r_0.5_tostep_line_2024_03_06_15_38_09/epoch-40.pt \
#         --top-k ${top_k} \
#         --prepend-bos True \
#         --prepend-bos-stage2 False \
#         --semantic-depup False \
#         --top-k-stage2 ${top_k_stage2} \
#         --shared-linear-stage2 False \
#         --temperature 1.0 \
#         --prompt-pre-cut False \
#         --num-quantizers 1 \
#         --num-quantizers-stage2 16 \
#         --input-codec 1 \
#         --target-mode 2 \
#         --accent-remove True \
#         --mode 0 \
#         --mode-stage2 0 \
#         --soundstorm-type 1 \
#         --is-pretrain True \
#         --pret-mode 0 \
#         --outputdir-name converted_can_del_tts/group_ar/nar_mask_t_4_nar_mask_r_0_5/no-prompt-tts-2stages_topk_stage1_${top_k}_topk_stage2_${top_k_stage2}_steps_${steps}_groupar_${basestr} \
#         > /home/v-zhijunjia/CodecGen/egs/libritts/log/tts-2stages-soundstorm-group-ar_topk_stage1_${top_k}_${basestr}_step_${steps}.txt 2>&1 &
#     done   
# done

# # tts  nar_mask_t_3_nar_mask_r_0.5_tostep_line_prompt_2_4s
# num_runs=1
# num_runs_stage2=1
# top_k=2
# top_k_stage2=20
# steps=16
# add_prenet="False"
# for steps in 16 32; do
#     for top_k_stage2 in 15 20; do  
#         CUDA_VISIBLE_DEVICES=0 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#         --model-name vallfe \
#         --model-name-stage2 soundstorm \
#         --soundstorm-steps ${steps} \
#         --norm-first true \
#         --add-prenet ${add_prenet} \
#         --decoder-dim 1024 --nhead 16 \
#         --decoder-dim-stage2 1024 --nhead-stage2 16 \
#         --encoder-num-layers 6 \
#         --decoder-num-layers 6 \
#         --num-decoder-layers 12 \
#         --num-decoder-layers-stage2 12 \
#         --share-embedding true \
#         --nums ${num_runs} \
#         --nums-stage2 ${num_runs_stage2} \
#         --nums-stage2 ${num_runs_stage2} \
#         --semantic-tokens /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
#         --text-tokens /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#         --test-benchmark True \
#         --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer_min \
#         --task-id 1 \
#         --input-semantic True \
#         --only-autoregressive True \
#         --prefix-mode 1 \
#         --checkpoint1 /home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
#         --checkpoint2 /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_40_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_nar_mask_t_3_nar_mask_r_0.5_tostep_line_prompt_2_4s_2024_03_06_07_29_19/epoch-40.pt \
#         --top-k ${top_k} \
#         --prepend-bos True \
#         --prepend-bos-stage2 False \
#         --semantic-depup False \
#         --top-k-stage2 ${top_k_stage2} \
#         --shared-linear-stage2 False \
#         --temperature 1.0 \
#         --prompt-pre-cut False \
#         --num-quantizers 1 \
#         --num-quantizers-stage2 16 \
#         --input-codec 1 \
#         --target-mode 2 \
#         --accent-remove True \
#         --mode 0 \
#         --mode-stage2 0 \
#         --soundstorm-type 1 \
#         --is-pretrain True \
#         --pret-mode 0 \
#         --outputdir-name converted_can_del_tts/group_ar/nar_mask_t_3_nar_mask_r_0_5_prompt_2_4s/no-prompt-tts-2stages_topk_stage1_${top_k}_topk_stage2_${top_k_stage2}_steps_${steps}_groupar_${basestr} \
#         > /home/v-zhijunjia/CodecGen/egs/libritts/log/tts-2stages-soundstorm-group-ar_topk_stage1_${top_k}_${basestr}_step_${steps}.txt 2>&1 &
#     done   
# done


# # tts  nar_mask_t_3_nar_mask_r_0.5_tostep_line_
# num_runs=1
# num_runs_stage2=1
# top_k=2
# top_k_stage2=20
# steps=16
# add_prenet="False"
# for steps in 14 18 22 26 32; do
#     for top_k_stage2 in 15 20 25 30; do  
#         CUDA_VISIBLE_DEVICES=0 python egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#         --model-name vallfe \
#         --model-name-stage2 soundstorm \
#         --soundstorm-steps ${steps} \
#         --norm-first true \
#         --add-prenet ${add_prenet} \
#         --decoder-dim 1024 --nhead 16 \
#         --decoder-dim-stage2 1024 --nhead-stage2 16 \
#         --encoder-num-layers 6 \
#         --decoder-num-layers 6 \
#         --num-decoder-layers 12 \
#         --num-decoder-layers-stage2 12 \
#         --share-embedding true \
#         --nums ${num_runs} \
#         --nums-stage2 ${num_runs_stage2} \
#         --nums-stage2 ${num_runs_stage2} \
#         --semantic-tokens /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
#         --text-tokens /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#         --test-benchmark True \
#         --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer_min \
#         --task-id 1 \
#         --input-semantic True \
#         --only-autoregressive True \
#         --prefix-mode 1 \
#         --checkpoint1 /home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
#         --checkpoint2 /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_40_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_nar_mask_t_3_nar_mask_r_0.5_tostep_line_2024_03_06_15_09_07/epoch-40.pt \
#         --top-k ${top_k} \
#         --prepend-bos True \
#         --prepend-bos-stage2 False \
#         --semantic-depup False \
#         --top-k-stage2 ${top_k_stage2} \
#         --shared-linear-stage2 False \
#         --temperature 1.0 \
#         --prompt-pre-cut False \
#         --num-quantizers 1 \
#         --num-quantizers-stage2 16 \
#         --input-codec 1 \
#         --target-mode 2 \
#         --accent-remove True \
#         --mode 0 \
#         --mode-stage2 0 \
#         --soundstorm-type 1 \
#         --is-pretrain True \
#         --pret-mode 0 \
#         --outputdir-name converted_can_del_tts/group_ar/nar_mask_t_3_nar_mask_r_0_5_xijie/no-prompt-tts-2stages_topk_stage1_${top_k}_topk_stage2_${top_k_stage2}_steps_${steps}_groupar_${basestr} 
#         # > /home/v-zhijunjia/CodecGen/egs/libritts/log/tts-2stages-soundstorm-group-ar_topk_stage1_${top_k}_${basestr}_step_${steps}.txt 2>&1 &
#     done   
# done
# # tts benchmark
# num_runs=1
# num_runs_stage2=1
# top_k=2
# top_k_stage2=20
# steps=16
# add_prenet="False"
# for steps in $(seq 16 16 16); do  
#     CUDA_VISIBLE_DEVICES=0 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#     --model-name vallfe \
#     --model-name-stage2 soundstorm \
#     --soundstorm-steps ${steps} \
#     --norm-first true \
#     --add-prenet ${add_prenet} \
#     --decoder-dim 1024 --nhead 16 \
#     --decoder-dim-stage2 1024 --nhead-stage2 16 \
#     --encoder-num-layers 6 \
#     --decoder-num-layers 6 \
#     --num-decoder-layers 12 \
#     --num-decoder-layers-stage2 12 \
#     --share-embedding true \
#     --nums ${num_runs} \
#     --nums-stage2 ${num_runs_stage2} \
#     --nums-stage2 ${num_runs_stage2} \
#     --semantic-tokens /scratch/data/Libritts/tokenized/unique_semantic_tokens.k2symbols \
#     --text-tokens /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#     --test-benchmark True \
#     --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/filtered_4s_10s-test-spk-wer_min \
#     --task-id 1 \
#     --input-semantic True \
#     --only-autoregressive True \
#     --prefix-mode 1 \
#     --checkpoint1 /home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
#     --checkpoint2 /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_100_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_nar_mask_type_1_2024_01_18_13_58_20/epoch-40.pt \
#     --top-k ${top_k} \
#     --prepend-bos True \
#     --prepend-bos-stage2 False \
#     --semantic-depup False \
#     --top-k-stage2 ${top_k_stage2} \
#     --shared-linear-stage2 False \
#     --temperature 1.0 \
#     --prompt-pre-cut False \
#     --num-quantizers 1 \
#     --num-quantizers-stage2 16 \
#     --input-codec 1 \
#     --target-mode 2 \
#     --accent-remove True \
#     --mode 0 \
#     --mode-stage2 0 \
#     --soundstorm-type 1 \
#     --is-pretrain True \
#     --pret-mode 0 \
#     --outputdir-name converted_can_del/no-prompt-tts-2stages_topk_stage1_${top_k}_topk_stage2_${top_k_stage2}_steps_${steps}_groupar_${basestr} > /home/v-zhijunjia/CodecGen/egs/libritts/log/tts-2stages-soundstorm-group-ar_topk_stage1_${top_k}_${basestr}_step_${steps}.txt 2>&1 &
# done    
# /home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/filtered_4s_10s-test-spk-wer

# --semantic-sys-dir /home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/source \
# --audio-prompts-dir /home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/prompt \


# ## ar+ar
# basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  

# num_runs=3
# num_runs_stage2=1
# top_k=2
# top_k_stage2=10
# add_prenet="False"
# for checkpoint in $(seq 60 60 60); do  
  
#     CUDA_VISIBLE_DEVICES=0 python egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#     --model-name vallfe \
#     --model-name-stage2 valle \
#     --norm-first true --add-prenet ${add_prenet} \
#     --decoder-dim 1024 --nhead 16 \
#     --decoder-dim-stage2 1024 --nhead-stage2 16 \
#     --encoder-num-layers 6 \
#     --decoder-num-layers 6 \
#     --num-decoder-layers 12 \
#     --num-decoder-layers-stage2 12 \
#     --share-embedding true \
#     --nums ${num_runs} \
#     --nums-stage2 ${num_runs_stage2} \
#     --nums-stage2 ${num_runs_stage2} \
#     --semantic-tokens /scratch/data/Libritts/tokenized/unique_semantic_tokens.k2symbols \
#     --text-tokens /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#     --test-benchmark True \
#     --dir-need2test /home/v-zhijunjia/zhijundata_small_v2/data_local/data/tts_test_txt2sem_prompt \
#     --task-id 1 \
#     --input-semantic True \
#     --only-autoregressive True \
#     --prefix-mode 1 \
#     --checkpoint1 /home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
#     --checkpoint2 /home/v-zhijunjia/data/valle-tensorboard-models/vc/only_ar/epoch-40.pt \
#     --top-k ${top_k} \
#     --prepend-bos True \
#     --prepend-bos-stage2 False \
#     --semantic-depup False \
#     --top-k-stage2 ${top_k_stage2} \
#     --shared-linear-stage2 False \
#     --temperature 1.0 \
#     --num-quantizers 1 \
#     --num-quantizers-stage2 16 \
#     --input-codec 1 \
#     --target-mode 2 \
#     --accent-remove True \
#     --mode 0 \
#     --mode-stage2 0 \
#     --is-pretrain True \
#     --pret-mode 0 \
#     --outputdir-name converted_can_del/no-prompt-tts-2stages_${epoch}_${basestr}
# done    