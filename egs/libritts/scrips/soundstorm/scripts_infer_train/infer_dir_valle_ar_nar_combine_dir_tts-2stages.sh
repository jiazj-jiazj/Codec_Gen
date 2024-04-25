basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  


# top_k_know_token_stage2 fixed
# tts nar_mask_r_0.5_gp_i_mask_True_g_in_m_rep_p_0.0001_g_in_m_rep_al_p_0.05_onlymask_False
num_runs=1
num_runs_stage2=1
top_k=2
top_k_know_token_stage2=70
known_token_update="True"
top_k_stage2=20
steps=16
add_prenet="False"
for steps in 16 32; do
    for epoch_ckpt in 50; do 
        for top_k_stage2 in 140; do
            for top_p_stage2 in 1; do
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
                    --semantic-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
                    --text-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
                    --test-benchmark True \
                    --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer \
                    --task-id 1 \
                    --input-semantic True \
                    --only-autoregressive True \
                    --prefix-mode 1 \
                    --checkpoint1 /home/v-zhijunjia/zhijundata_small_v3/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
                    --checkpoint2 /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_m-dur_80_b-lr_0.05_eo_100_s_i_seman_True_nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True_g_in_m_rep_p_0_g_in_m_rep_al_p_1_g_in_m_r_a_v_True_2024_04_17_03_00_31/epoch-${epoch_ckpt}.pt \
                    --top-k ${top_k} \
                    --top-p 1 \
                    --prepend-bos True \
                    --known-token-update ${known_token_update} \
                    --prepend-bos-stage2 False \
                    --semantic-depup False \
                    --top-k-stage2 ${top_k_stage2} \
                    --top-p-stage2 ${top_p_stage2} \
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
                    --soundstorm-type 0 \
                    --is-pretrain True \
                    --pret-mode 0 \
                    --sem-read True \
                    --txt-sem-path /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer_sem \
                    --outputdir-name converted_can_del_tts_first_top2_fixed_seed_v2/ours/v17_update_gp_i_mask_True_g_in_m_rep_p_0_g_in_m_rep_al_p_1/epoch-${epoch_ckpt}-base-line-no-prompt-tts-2stages_topk_stage1_${top_k}_topk_stage2_${top_k_stage2}_top_k_know_token_stage2_${top_k_know_token_stage2}_steps_${steps}_known_token_update_${known_token_update}_${basestr} \
                    > /home/v-zhijunjia/CodecGen/egs/libritts/log/rep_p_0.0001_g_in_m_rep_al_p_0.15_tts-2stages-soundstorm-baseline_topk_stage1_${top_k}_${basestr}_step_${steps}.txt 2>&1 &
            done
        done
    done
done   
# # # test
# # top_k_know_token_stage2 fixed
# # tts nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True
# num_runs=1
# num_runs_stage2=1
# top_k=2
# top_k_know_token_stage2=70
# known_token_update="True"
# top_k_stage2=20
# steps=16
# add_prenet="False"
# for steps in 16; do
#     for top_k_stage2 in 10 30 50; do
#         for top_p_stage2 in 1; do
#                 CUDA_VISIBLE_DEVICES=0 python egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#                 --model-name vallfe \
#                 --model-name-stage2 soundstorm \
#                 --soundstorm-steps ${steps} \
#                 --norm-first true \
#                 --add-prenet ${add_prenet} \
#                 --decoder-dim 1024 --nhead 16 \
#                 --decoder-dim-stage2 1024 --nhead-stage2 16 \
#                 --encoder-num-layers 6 \
#                 --decoder-num-layers 6 \
#                 --num-decoder-layers 12 \
#                 --num-decoder-layers-stage2 12 \
#                 --share-embedding true \
#                 --nums ${num_runs} \
#                 --nums-stage2 ${num_runs_stage2} \
#                 --semantic-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
#                 --text-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#                 --test-benchmark True \
#                 --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer \
#                 --task-id 1 \
#                 --input-semantic True \
#                 --only-autoregressive True \
#                 --prefix-mode 1 \
#                 --checkpoint1 /home/v-zhijunjia/zhijundata_small_v3/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
#                 --checkpoint2 /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_50_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True_2024_03_11_02_25_30_2024_03_11_02_25_30/epoch-50.pt \
#                 --top-k ${top_k} \
#                 --top-p 1 \
#                 --prepend-bos True \
#                 --known-token-update ${known_token_update} \
#                 --prepend-bos-stage2 False \
#                 --semantic-depup False \
#                 --top-k-stage2 ${top_k_stage2} \
#                 --top-p-stage2 ${top_p_stage2} \
#                 --shared-linear-stage2 False \
#                 --temperature 1.0 \
#                 --prompt-pre-cut False \
#                 --num-quantizers 1 \
#                 --num-quantizers-stage2 16 \
#                 --input-codec 1 \
#                 --target-mode 2 \
#                 --accent-remove True \
#                 --mode 0 \
#                 --mode-stage2 0 \
#                 --soundstorm-type 0 \
#                 --is-pretrain True \
#                 --pret-mode 0 \
#                 --sem-read True \
#                 --txt-sem-path /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer_sem \
#                 --outputdir-name converted_can_del_tts_first_top2_fixed_seed/baseline/update_all_nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True_knowtoken_topk_update/base-line-no-prompt-tts-2stages_topk_stage1_${top_k}_topk_stage2_${top_k_stage2}_top_k_know_token_stage2_${top_k_know_token_stage2}_steps_${steps}_known_token_update_${known_token_update}_${basestr}
#         done
#     done
# done   

# # top_k=2 fixed-sem

# # tts benchmark
# num_runs=1
# num_runs_stage2=1
# top_k=2
# top_k_stage2=70
# steps=16
# top_k_know_token_stage2=70
# add_prenet="False"
# for steps in 16 32; do
#     for top_k_stage2 in 140; do  
#         for top_p_stage2 in 1; do
#             for top_k_know_token_stage2 in 140; do
#                 CUDA_VISIBLE_DEVICES=0 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#                 --model-name vallfe \
#                 --model-name-stage2 soundstorm \
#                 --soundstorm-steps ${steps} \
#                 --norm-first true \
#                 --add-prenet ${add_prenet} \
#                 --decoder-dim 1024 --nhead 16 \
#                 --decoder-dim-stage2 1024 --nhead-stage2 16 \
#                 --encoder-num-layers 6 \
#                 --decoder-num-layers 6 \
#                 --num-decoder-layers 12 \
#                 --num-decoder-layers-stage2 12 \
#                 --share-embedding true \
#                 --nums ${num_runs} \
#                 --nums-stage2 ${num_runs_stage2} \
#                 --nums-stage2 ${num_runs_stage2} \
#                 --semantic-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
#                 --text-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#                 --test-benchmark True \
#                 --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer \
#                 --task-id 1 \
#                 --input-semantic True \
#                 --only-autoregressive True \
#                 --prefix-mode 1 \
#                 --checkpoint1 /home/v-zhijunjia/zhijundata_small_v3/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
#                 --checkpoint2 /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_m-dur_80_b-lr_0.05_eo_50_s_i_seman_True_nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_False_g_in_m_rep_p_0.15_g_in_m_rep_al_p_0.05_2024_04_03_06_39_44/epoch-50.pt \
#                 --top-k ${top_k} \
#                 --top-p 1 \
#                 --prepend-bos True \
#                 --prepend-bos-stage2 False \
#                 --semantic-depup False \
#                 --top-k-stage2 ${top_k_stage2} \
#                 --top-p-stage2 ${top_p_stage2} \
#                 --top-k-know-token-stage2 ${top_k_know_token_stage2} \
#                 --shared-linear-stage2 False \
#                 --temperature 1.0 \
#                 --prompt-pre-cut False \
#                 --num-quantizers 1 \
#                 --num-quantizers-stage2 16 \
#                 --input-codec 1 \
#                 --target-mode 2 \
#                 --accent-remove True \
#                 --mode 0 \
#                 --mode-stage2 0 \
#                 --soundstorm-type 0 \
#                 --sem-read True \
#                 --txt-sem-path /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer_sem \
#                 --is-pretrain True \
#                 --pret-mode 0 \
#                 --outputdir-name converted_can_del_tts_first_top2_fixed_seed_v2/baseline/base-line-no-prompt-tts-2stages_topk_stage1_${top_k}_topk_stage2_${top_k_stage2}_steps_${steps}_${basestr} \
#                 > /home/v-zhijunjia/CodecGen/egs/libritts/log/tts-2stages-soundstorm-baseline_topk_stage1_${top_k}_${basestr}_step_${steps}.txt 2>&1 &
#             done
#         done
#     done
# done    


# # top_k_know_token_stage2 fixed
# # tts nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True
# num_runs=1
# num_runs_stage2=1
# top_k=2
# top_k_know_token_stage2=70
# known_token_update="True"
# top_k_stage2=20
# steps=16
# temperature_stage2=1
# add_prenet="False"
# for steps in 16; do
#     for top_k_stage2 in 140; do
#         for top_p_stage2 in 1; do
#             for temperature_stage2 in 1; do
#                 CUDA_VISIBLE_DEVICES=0 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#                 --model-name vallfe \
#                 --model-name-stage2 soundstorm \
#                 --soundstorm-steps ${steps} \
#                 --norm-first true \
#                 --add-prenet ${add_prenet} \
#                 --decoder-dim 1024 --nhead 16 \
#                 --decoder-dim-stage2 1024 --nhead-stage2 16 \
#                 --encoder-num-layers 6 \
#                 --decoder-num-layers 6 \
#                 --num-decoder-layers 12 \
#                 --num-decoder-layers-stage2 12 \
#                 --share-embedding true \
#                 --nums ${num_runs} \
#                 --nums-stage2 ${num_runs_stage2} \
#                 --semantic-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
#                 --text-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#                 --test-benchmark True \
#                 --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer \
#                 --task-id 1 \
#                 --input-semantic True \
#                 --only-autoregressive True \
#                 --prefix-mode 1 \
#                 --checkpoint1 /home/v-zhijunjia/zhijundata_small_v3/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
#                 --checkpoint2 /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_m-dur_80_b-lr_0.05_eo_50_s_i_seman_True_nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True_g_in_m_rep_p_0.3_g_in_m_rep_al_p_0.15_2024_04_03_08_24_42/epoch-48.pt \
#                 --top-k ${top_k} \
#                 --top-p 1 \
#                 --prepend-bos True \
#                 --known-token-update ${known_token_update} \
#                 --prepend-bos-stage2 False \
#                 --semantic-depup False \
#                 --top-k-stage2 ${top_k_stage2} \
#                 --top-p-stage2 ${top_p_stage2} \
#                 --shared-linear-stage2 False \
#                 --temperature 1.0 \
#                 --temperature-stage2 ${temperature_stage2} \
#                 --prompt-pre-cut False \
#                 --num-quantizers 1 \
#                 --num-quantizers-stage2 16 \
#                 --input-codec 1 \
#                 --target-mode 2 \
#                 --accent-remove True \
#                 --mode 0 \
#                 --mode-stage2 0 \
#                 --soundstorm-type 0 \
#                 --is-pretrain True \
#                 --pret-mode 0 \
#                 --sem-read True \
#                 --txt-sem-path /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer_sem \
#                 --outputdir-name converted_can_del_tts_first_top2_fixed_seed_v2/ours/v16_update_gp_i_mask_True_g_in_m_rep_p_0.3_g_in_m_rep_al_p_0.15/tem-stage2-${temperature_stage2}-topp-stage2-${top_p_stage2}-base-line-no-prompt-tts-2stages_topk_stage1_${top_k}_topk_stage2_${top_k_stage2}_top_k_know_token_stage2_${top_k_know_token_stage2}_steps_${steps}_known_token_update_${known_token_update}_${basestr} \
#                 > /home/v-zhijunjia/CodecGen/egs/libritts/log/tts-2stages-soundstorm-baseline_topk_stage1_${top_k}_${basestr}_step_${steps}.txt 2>&1 &
#             done
#         done
#     done
# done   

# # top_k_know_token_stage2 change
# # tts nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True
# num_runs=1
# num_runs_stage2=1
# top_k=2
# top_k_know_token_stage2=70
# known_token_update="True"
# top_k_stage2=20
# steps=16
# add_prenet="False"
# for steps in 16; do
#     for top_k_stage2 in 140; do
#         for top_p_stage2 in 1; do
#             for top_k_know_token_stage2 in 10 50 70 130 140 150; do
#                 CUDA_VISIBLE_DEVICES=0 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#                 --model-name vallfe \
#                 --model-name-stage2 soundstorm \
#                 --soundstorm-steps ${steps} \
#                 --norm-first true \
#                 --add-prenet ${add_prenet} \
#                 --decoder-dim 1024 --nhead 16 \
#                 --decoder-dim-stage2 1024 --nhead-stage2 16 \
#                 --encoder-num-layers 6 \
#                 --decoder-num-layers 6 \
#                 --num-decoder-layers 12 \
#                 --num-decoder-layers-stage2 12 \
#                 --share-embedding true \
#                 --nums ${num_runs} \
#                 --nums-stage2 ${num_runs_stage2} \
#                 --semantic-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
#                 --text-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#                 --test-benchmark True \
#                 --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer \
#                 --task-id 1 \
#                 --input-semantic True \
#                 --only-autoregressive True \
#                 --prefix-mode 1 \
#                 --checkpoint1 /home/v-zhijunjia/zhijundata_small_v3/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
#                 --checkpoint2 /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_m-dur_80_b-lr_0.05_eo_50_s_i_seman_True_nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True_g_in_m_rep_p_0.3_g_in_m_rep_al_p_0.15_2024_04_03_08_24_42/epoch-48.pt \
#                 --top-k ${top_k} \
#                 --top-p 1 \
#                 --prepend-bos True \
#                 --known-token-update ${known_token_update} \
#                 --prepend-bos-stage2 False \
#                 --semantic-depup False \
#                 --top-k-stage2 ${top_k_stage2} \
#                 --top-p-stage2 ${top_p_stage2} \
#                 --top-k-know-token-stage2 ${top_k_know_token_stage2} \
#                 --shared-linear-stage2 False \
#                 --temperature 1.0 \
#                 --prompt-pre-cut False \
#                 --num-quantizers 1 \
#                 --num-quantizers-stage2 16 \
#                 --input-codec 1 \
#                 --target-mode 2 \
#                 --accent-remove True \
#                 --mode 0 \
#                 --mode-stage2 0 \
#                 --soundstorm-type 0 \
#                 --is-pretrain True \
#                 --pret-mode 0 \
#                 --sem-read True \
#                 --txt-sem-path /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer_sem \
#                 --outputdir-name converted_can_del_tts_first_top2_fixed_seed_v2/ours/v9_update_gp_i_mask_True_g_in_m_rep_p_0.3_g_in_m_rep_al_p_0.15/base-line-no-prompt-tts-2stages_topk_stage1_${top_k}_topk_stage2_${top_k_stage2}_top_k_know_token_stage2_${top_k_know_token_stage2}_steps_${steps}_known_token_update_${known_token_update}_${basestr} \
#                 > /home/v-zhijunjia/CodecGen/egs/libritts/log/tts-2stages-soundstorm-baseline_topk_stage1_${top_k}_${basestr}_step_${steps}.txt 2>&1 &
#             done
#         done
#     done
# done   
# # tts nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True
# num_runs=1
# num_runs_stage2=1
# top_k=2
# top_k_know_token_stage2=10
# top_k_stage2=20
# steps=16
# add_prenet="False"
# for steps in 16; do
#     for top_k_stage2 in 70; do
#         for top_p_stage2 in 1; do
#             for top_k_know_token_stage2 in 70; do
#                 CUDA_VISIBLE_DEVICES=0 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#                 --model-name vallfe \
#                 --model-name-stage2 soundstorm \
#                 --soundstorm-steps ${steps} \
#                 --norm-first true \
#                 --add-prenet ${add_prenet} \
#                 --decoder-dim 1024 --nhead 16 \
#                 --decoder-dim-stage2 1024 --nhead-stage2 16 \
#                 --encoder-num-layers 6 \
#                 --decoder-num-layers 6 \
#                 --num-decoder-layers 12 \
#                 --num-decoder-layers-stage2 12 \
#                 --share-embedding true \
#                 --nums ${num_runs} \
#                 --nums-stage2 ${num_runs_stage2} \
#                 --semantic-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
#                 --text-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#                 --test-benchmark True \
#                 --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer \
#                 --task-id 1 \
#                 --input-semantic True \
#                 --only-autoregressive True \
#                 --prefix-mode 1 \
#                 --checkpoint1 /home/v-zhijunjia/zhijundata_small_v3/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
#                 --checkpoint2 /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_50_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True_2024_03_11_02_25_30_2024_03_11_02_25_30/epoch-50.pt \
#                 --top-k ${top_k} \
#                 --top-p 1 \
#                 --prepend-bos True \
#                 --prepend-bos-stage2 False \
#                 --semantic-depup False \
#                 --top-k-stage2 ${top_k_stage2} \
#                 --top-p-stage2 ${top_p_stage2} \
#                 --top-k-know-token-stage2 ${top_k_know_token_stage2} \
#                 --shared-linear-stage2 False \
#                 --temperature 1.0 \
#                 --prompt-pre-cut False \
#                 --num-quantizers 1 \
#                 --num-quantizers-stage2 16 \
#                 --input-codec 1 \
#                 --target-mode 2 \
#                 --accent-remove True \
#                 --mode 0 \
#                 --mode-stage2 0 \
#                 --soundstorm-type 0 \
#                 --is-pretrain True \
#                 --pret-mode 0 \
#                 --sem-read True \
#                 --txt-sem-path /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer_sem \
#                 --outputdir-name converted_can_del_tts_first_top2_fixed_seed/baseline/update_all_nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True_knowtoken_topk_update/base-line-no-prompt-tts-2stages_topk_stage1_${top_k}_topk_stage2_${top_k_stage2}_top_k_know_token_stage2_${top_k_know_token_stage2}_steps_${steps}_${basestr} \
#                 > /home/v-zhijunjia/CodecGen/egs/libritts/log/tts-2stages-soundstorm-baseline_topk_stage1_${top_k}_${basestr}_step_${steps}.txt 2>&1 &
#             done
#         done
#     done
# done   

# > /home/v-zhijunjia/CodecGen/egs/libritts/log/tts-2stages-soundstorm-baseline_topk_stage1_${top_k}_${basestr}_step_${steps}.txt 2>&1 &

# # top_k=2


# # tts nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True
# num_runs=1
# num_runs_stage2=1
# top_k=2
# top_k_know_token_stage2=10
# top_k_stage2=20
# steps=16
# add_prenet="False"
# for steps in 16; do
#     for top_k_stage2 in 20; do
#         for top_p_stage2 in 1; do
#             for top_k_know_token_stage2 in 10 20 30 50 70; do
#                 CUDA_VISIBLE_DEVICES=0 python egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#                 --model-name vallfe \
#                 --model-name-stage2 soundstorm \
#                 --soundstorm-steps ${steps} \
#                 --norm-first true \
#                 --add-prenet ${add_prenet} \
#                 --decoder-dim 1024 --nhead 16 \
#                 --decoder-dim-stage2 1024 --nhead-stage2 16 \
#                 --encoder-num-layers 6 \
#                 --decoder-num-layers 6 \
#                 --num-decoder-layers 12 \
#                 --num-decoder-layers-stage2 12 \
#                 --share-embedding true \
#                 --nums ${num_runs} \
#                 --nums-stage2 ${num_runs_stage2} \
#                 --nums-stage2 ${num_runs_stage2} \
#                 --semantic-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
#                 --text-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#                 --test-benchmark True \
#                 --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer_min \
#                 --task-id 1 \
#                 --input-semantic True \
#                 --only-autoregressive True \
#                 --prefix-mode 1 \
#                 --checkpoint1 /home/v-zhijunjia/zhijundata_small_v3/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
#                 --checkpoint2 /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_50_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True_2024_03_11_02_25_30_2024_03_11_02_25_30/epoch-50.pt \
#                 --top-k ${top_k} \
#                 --top-p 1 \
#                 --prepend-bos True \
#                 --prepend-bos-stage2 False \
#                 --semantic-depup False \
#                 --top-k-stage2 ${top_k_stage2} \
#                 --top-p-stage2 ${top_p_stage2} \
#                 --top-k-know-token-stage2 ${top_k_know_token_stage2} \
#                 --shared-linear-stage2 False \
#                 --temperature 1.0 \
#                 --prompt-pre-cut False \
#                 --num-quantizers 1 \
#                 --num-quantizers-stage2 16 \
#                 --input-codec 1 \
#                 --target-mode 2 \
#                 --accent-remove True \
#                 --mode 0 \
#                 --mode-stage2 0 \
#                 --soundstorm-type 0 \
#                 --is-pretrain True \
#                 --pret-mode 0 \
#                 --outputdir-name converted_can_del_tts_first_top1/baseline/update_all_nar_mask_t_0_nar_mask_r_0.5_gp_i_mask_True_knowtoken_topk_update/base-line-no-prompt-tts-2stages_topk_stage1_${top_k}_topk_stage2_${top_k_stage2}_top_k_know_token_stage2_${top_k_know_token_stage2}_steps_${steps}_${basestr} \
#                 > /home/v-zhijunjia/CodecGen/egs/libritts/log/tts-2stages-soundstorm-baseline_topk_stage1_${top_k}_${basestr}_step_${steps}.txt 2>&1 &
#             done
#         done
#     done
# done    
# # tts nar_mask_t_3_nar_mask_r_0.5
# num_runs=1
# num_runs_stage2=1
# top_k=2
# top_k_stage2=70
# steps=16
# add_prenet="False"
# for steps in 16 32; do
#     for top_k_stage2 in 70; do  
#         for top_p_stage2 in 1; do
#             CUDA_VISIBLE_DEVICES=0 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#             --model-name vallfe \
#             --model-name-stage2 soundstorm \
#             --soundstorm-steps ${steps} \
#             --norm-first true \
#             --add-prenet ${add_prenet} \
#             --decoder-dim 1024 --nhead 16 \
#             --decoder-dim-stage2 1024 --nhead-stage2 16 \
#             --encoder-num-layers 6 \
#             --decoder-num-layers 6 \
#             --num-decoder-layers 12 \
#             --num-decoder-layers-stage2 12 \
#             --share-embedding true \
#             --nums ${num_runs} \
#             --nums-stage2 ${num_runs_stage2} \
#             --nums-stage2 ${num_runs_stage2} \
#             --semantic-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
#             --text-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#             --test-benchmark True \
#             --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/nar_test/filtered_4s_10s-test-spk-wer_min \
#             --task-id 1 \
#             --input-semantic True \
#             --only-autoregressive True \
#             --prefix-mode 1 \
#             --checkpoint1 /home/v-zhijunjia/zhijundata_small_v3/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
#             --checkpoint2 /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_40_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_nar_mask_t_3_nar_mask_r_0.5_tostep_line_2024_03_06_15_09_07/epoch-40.pt \
#             --top-k ${top_k} \
#             --top-p 1 \
#             --prepend-bos True \
#             --prepend-bos-stage2 False \
#             --semantic-depup False \
#             --top-k-stage2 ${top_k_stage2} \
#             --top-p-stage2 ${top_p_stage2} \
#             --shared-linear-stage2 False \
#             --temperature 1.0 \
#             --prompt-pre-cut False \
#             --num-quantizers 1 \
#             --num-quantizers-stage2 16 \
#             --input-codec 1 \
#             --target-mode 2 \
#             --accent-remove True \
#             --mode 0 \
#             --mode-stage2 0 \
#             --soundstorm-type 0 \
#             --is-pretrain True \
#             --pret-mode 0 \
#             --outputdir-name converted_can_del_tts/baseline/nar_mask_t_3_nar_mask_r_0.5/base-line-no-prompt-tts-2stages_topk_stage1_${top_k}_topk_stage2_${top_k_stage2}_steps_${steps}_${basestr} \
#             > /home/v-zhijunjia/CodecGen/egs/libritts/log/tts-2stages-soundstorm-baseline_topk_stage1_${top_k}_${basestr}_step_${steps}.txt 2>&1 &
#         done
#     done
# done    

# # tts benchmark
# num_runs=1
# num_runs_stage2=1
# top_k=2
# top_k_stage2=70
# steps=16
# add_prenet="False"
# for steps in 16 28 32; do
#     for top_k_stage2 in 70; do  
#         for top_p_stage2 in 1; do
#             CUDA_VISIBLE_DEVICES=0 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#             --model-name vallfe \
#             --model-name-stage2 soundstorm \
#             --soundstorm-steps ${steps} \
#             --norm-first true \
#             --add-prenet ${add_prenet} \
#             --decoder-dim 1024 --nhead 16 \
#             --decoder-dim-stage2 1024 --nhead-stage2 16 \
#             --encoder-num-layers 6 \
#             --decoder-num-layers 6 \
#             --num-decoder-layers 12 \
#             --num-decoder-layers-stage2 12 \
#             --share-embedding true \
#             --nums ${num_runs} \
#             --nums-stage2 ${num_runs_stage2} \
#             --nums-stage2 ${num_runs_stage2} \
#             --semantic-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols \
#             --text-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#             --test-benchmark True \
#             --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/filtered_4s_10s-test-spk-wer \
#             --task-id 1 \
#             --input-semantic True \
#             --only-autoregressive True \
#             --prefix-mode 1 \
#             --checkpoint1 /home/v-zhijunjia/zhijundata_small_v3/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
#             --checkpoint2 /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_100_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_2024_01_15_15_45_24/epoch-40.pt \
#             --top-k ${top_k} \
#             --top-p 1 \
#             --prepend-bos True \
#             --prepend-bos-stage2 False \
#             --semantic-depup False \
#             --top-k-stage2 ${top_k_stage2} \
#             --top-p-stage2 ${top_p_stage2} \
#             --shared-linear-stage2 False \
#             --temperature 1.0 \
#             --prompt-pre-cut False \
#             --num-quantizers 1 \
#             --num-quantizers-stage2 16 \
#             --input-codec 1 \
#             --target-mode 2 \
#             --accent-remove True \
#             --mode 0 \
#             --mode-stage2 0 \
#             --soundstorm-type 0 \
#             --is-pretrain True \
#             --pret-mode 0 \
#             --outputdir-name converted_can_del_tts_all_librispeech/baseline/baseline/base-line-no-prompt-tts-2stages_topk_stage1_${top_k}_topk_stage2_${top_k_stage2}_steps_${steps}_${basestr} \
#             > /home/v-zhijunjia/CodecGen/egs/libritts/log/tts-2stages-soundstorm-baseline_topk_stage1_${top_k}_${basestr}_step_${steps}.txt 2>&1 &
#         done
#     done
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
#     --text-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
#     --test-benchmark True \
#     --dir-need2test /home/v-zhijunjia/zhijundata_small_v3/data_local/data/tts_test_txt2sem_prompt \
#     --task-id 1 \
#     --input-semantic True \
#     --only-autoregressive True \
#     --prefix-mode 1 \
#     --checkpoint1 /home/v-zhijunjia/zhijundata_small_v3/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
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