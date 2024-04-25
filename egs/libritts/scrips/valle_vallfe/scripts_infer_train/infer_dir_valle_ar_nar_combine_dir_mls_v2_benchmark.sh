num_runs=3
add_prenet="False"  
basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
  
for top_k in $(seq 100 20 300); do
  basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
  CUDA_VISIBLE_DEVICES=0 python egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
    --model-name valle --norm-first true --add-prenet ${add_prenet} --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
    --semantic-tokens /home/v-zhijunjia/zhijundata_small_v2/data/mls/mls_train_lhotse_dataset/unique_semantic_tokens.k2symbols \
    --share-embedding true \
    --text-tokens /home/v-zhijunjia/zhijundata_small_v2/data/mls/mls_train_lhotse_dataset_huxue/unique_text_tokens_train_all_dev_test.k2symbols \
    --prefix-mode 1 \
    --only-autoregressive True \
    --input-codec 1 \
    --num-quantizers 16 \
    --mode-stage2 0 \
    --nums ${num_runs} \
    --task-id 1 \
    --checkpoint1 /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/output_vc/tts_VALLE_ybos_False_m-dur_80_b-lr_0.05_eo_70_s_eo_1_p_md_1_i_seman_False_o_ar_True_n_qua_16_s_seps_400000_s_epoc_4_a_g_s_2_s_depup_False_s_rmv_False_dim_1024_n_16_n_d_lrs_12_2023_12_18_10_25_25/epoch-12.pt \
    --top-k ${top_k} --temperature 1.0 \
    --test-benchmark True \
    --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/filtered_4s_10s-test-spk-wer \
    --outputdir-name gen_wavs_mls_ours/gen_wavs_mls44k_ours_${basestr}_topk_${top_k} \
    > egs/libritts/log/tts_mls_6k_dir_benchmark_total_onlyar_test_best_${basestr}.txt 2>&1 &
done  
    # --outputdir-name converted_tts/tts_benchmark_results_${basestr}_topk_${top_k} \

  # > egs/libritts/log/tts_dir_total_onlyar${basestr}.txt 2>&1 &

    # --only-autoregressive True \
  # --input-codec 0 \


# # mls_4k

# for top_k in $(seq 60 20 80); do
#   basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
#   CUDA_VISIBLE_DEVICES=0 python egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
#     --model-name valle --norm-first true --add-prenet ${add_prenet} --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
#     --semantic-tokens /home/v-zhijunjia/zhijundata_small_v2/data/mls/mls_train_lhotse_dataset/unique_semantic_tokens.k2symbols \
#     --share-embedding true \
#     --text-tokens /home/v-zhijunjia/zhijundata_small_v2/data/mls/mls_train_lhotse_dataset/unique_text_tokens_train_0_1_2_dev_test.k2symbols \
#     --prefix-mode 1 \
#     --only-autoregressive True \
#     --input-codec 1 \
#     --num-quantizers 16 \
#     --mode-stage2 0 \
#     --nums ${num_runs} \
#     --task-id 1 \
#     --checkpoint1 /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/output_vc/tts_VALLE_ybos_False_m-dur_80_b-lr_0.05_eo_70_s_eo_1_p_md_1_i_seman_False_o_ar_True_n_qua_16_s_seps_50000_s_epoc_4_a_g_s_2_s_depup_False_s_rmv_False_dim_1024_n_16_n_d_lrs_12_2023_12_08_13_47_09/epoch-51.pt \
#     --top-k ${top_k} --temperature 1.0 \
#     --test-benchmark True \
#     --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/filtered_4s_10s-test-spk-wer \
#     --outputdir-name gen_wavs_mls6k_ours_${basestr}_topk_${top_k} \
#     > egs/libritts/log/tts_mls_6k_dir_benchmark_total_onlyar_test_best_${basestr}.txt 2>&1 &
# done  