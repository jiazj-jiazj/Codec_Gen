basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  

num_runs=7
num_runs_stage2=1
top_k=2
top_k_stage2=10
add_prenet="False"
for checkpoint in $(seq 60 60 60); do  
  
    CUDA_VISIBLE_DEVICES=0 python egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
    --model-name vallfe \
    --model-name-stage2 valle \
    --norm-first true --add-prenet ${add_prenet} \
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
    --semantic-tokens /scratch/data/Libritts/tokenized/unique_semantic_tokens.k2symbols \
    --text-tokens /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols \
    --test-benchmark True \
    --dir-need2test /home/v-zhijunjia/zhijundata_small_v2/data_local/data/tts_test_txt2sem_prompt \
    --task-id 1 \
    --input-semantic True \
    --only-autoregressive True \
    --txt2semantic-need-prompt True \
    --prefix-mode 1 \
    --checkpoint1 /home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/output_dir/txt_semantic/tune_txt2semantic_azureml/epoch-70.pt \
    --checkpoint2 /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/output_vc/Soundstorm_Soundstorm_m-dur_80_dtp_float32_b-lr_0.05_eo_100_s_i_seman_True_o_ar_True_n_qua_16_s_seps_5000_s_epoc_4_a_g_s_2_dim_1024_n_16_nar_mask_type_1_2024_01_18_13_58_20/epoch-77.pt \
    --top-k ${top_k} \
    --prompt-pre-3s False \
    --prepend-bos True \
    --prepend-bos-stage2 False \
    --semantic-depup False \
    --top-k-stage2 ${top_k_stage2} \
    --shared-linear-stage2 False \
    --temperature 1.0 \
    --num-quantizers 1 \
    --num-quantizers-stage2 16 \
    --input-codec 1 \
    --target-mode 2 \
    --accent-remove True \
    --mode 0 \
    --mode-stage2 0 \
    --is-pretrain True \
    --pret-mode 0 \
    --outputdir-name converted_can_del/aligh_sem-tts-2stages_${epoch}_${basestr}
done    

# /home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/filtered_4s_10s-test-spk-wer
# --semantic-sys-dir /home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/source \
# --audio-prompts-dir /home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/prompt \