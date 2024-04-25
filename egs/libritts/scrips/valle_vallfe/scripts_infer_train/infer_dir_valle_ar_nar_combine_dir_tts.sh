num_runs=3  
add_prenet="False"  
basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  

infer_batch_size=10
for top_k in {90..10..110}; do
  CUDA_VISIBLE_DEVICES=0 python -u egs/libritts/bin/combine_ar_nar_dir_onlyar.py \
    --output-dir //home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/tts_reference_comparedwith_sota_top__${top_k}_epoch_44_v2_librispeech_test \
    --model-name valle --norm-first true --add-prenet ${add_prenet} --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
    --share-embedding true \
    --text-tokens /home/v-zhijunjia/data/valle-tensorboard-models/tts/en_unique_text_tokens_v2.k2symbols \
    --prefix-mode 1 \
    --only-autoregressive True \
    --input-codec 1 \
    --num-quantizers 16 \
    --mode-stage2 0 \
    --infer-batch-size ${infer_batch_size} \
    --checkpoint1 /home/v-zhijunjia/zhijundata_small/data_local/valle-tensorboard-models/tts/onlyar_tfcodec/epoch-100.pt \
    --checkpoint-NAR /home/v-zhijunjia/data/valle-tensorboard-models/tts/encodec/nar/best-valid-loss-95.pt \
    --top-k ${top_k} --temperature 1.0 \
    --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/filtered_4s_10s-test-spk-wer \
    --repeat-nums ${num_runs} 
    # > egs/libritts/log/tts_dir_total_onlyar${basestr}.txt 2>&1 &
done  

  # > egs/libritts/log/tts_dir_total_onlyar${basestr}.txt 2>&1 &

    # --only-autoregressive True \
  # --input-codec 0 \