num_runs=1 
top_k=10
add_prenet="False"

CUDA_VISIBLE_DEVICES=0 python egs/libritts/bin/combine_ar_nar_dir.py \
  --output-dir //home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/gen_encodec_test_topk10 \
  --model-name valle --norm-first true --add-prenet ${add_prenet} --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
  --share-embedding true \
  --text-tokens /home/v-zhijunjia/data/valle-tensorboard-models/tts/en_unique_text_tokens.k2symbols \
  --prefix-mode 1 \
  --checkpoint1 /home/v-zhijunjia/data/valle-tensorboard-models/tts/encodec/ar/best-train-loss-80.pt \
  --checkpoint2 /home/v-zhijunjia/data/valle-tensorboard-models/tts/encodec/nar/best-valid-loss-95.pt \
  --top-k ${top_k} --temperature 1.0 \
  --dir-need2test /home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/filtered_4s_10s-test-spk-wer \
  --repeat-nums ${num_runs}