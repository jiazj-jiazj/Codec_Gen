num_runs=5
add_prenet="False"  
basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
  
for top_k in {80..80..80}; do
  basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
  CUDA_VISIBLE_DEVICES=0 python egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py \
    --model-name valle --norm-first true --add-prenet ${add_prenet} --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
    --semantic-tokens /scratch/data/Libritts/tokenized/unique_semantic_tokens.k2symbols \
    --share-embedding true \
    --text-tokens /home/v-zhijunjia/data/valle-tensorboard-models/tts/en_unique_text_tokens_v2.k2symbols \
    --prefix-mode 1 \
    --only-autoregressive True \
    --input-codec 1 \
    --num-quantizers 16 \
    --mode-stage2 0 \
    --nums ${num_runs} \
    --task-id 1 \
    --checkpoint1 /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/output_vc/tts_Name_VALLE_max-duration_base-lr_0.05_train-stage_1_echo_60_input_semantic_False_only_ar_True_num_quantizers_16_sheduler_steps_5000_sheduler_epochs_4_accumulate_grad_steps_4_semantic_depup_False_2023_08_16_03_47_10/epoch-100.pt \
    --top-k ${top_k} --temperature 1.0 \
    --audio-prompts-dir /home/v-zhijunjia/zhijundata_small_v2/data_local/data/check_less_bright_voice/source_native \
    --semantic-sys-dir /home/v-zhijunjia/zhijundata_small_v2/data_local/data/check_less_bright_voice/texts_native \
    --outputdir-name converted_tts/tts_results \
    # > egs/libritts/log/tts_dir_total_onlyar_test_best_${basestr}.txt 2>&1 &
done  

  # > egs/libritts/log/tts_dir_total_onlyar${basestr}.txt 2>&1 &

    # --only-autoregressive True \
  # --input-codec 0 \