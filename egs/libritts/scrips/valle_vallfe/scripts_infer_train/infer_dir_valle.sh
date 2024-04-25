
python bin/infer.py --output-dir /dev_huaying/zhijun/data/test_valle_styleTTS_yourtts_naturalspeech2/valle_error --model-name valle --norm-first true --add-prenet false --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
--text-prompts "looke out and tens the fives。" --text-tokens /mnt/zhijun/data/libritts/data/tokenized/unique_text_tokens.k2symbols --audio-prompts /dev_huaying/zhijun/data/test_valle_naturalspeech2_yourtts_styleTTS/test1/reference_LibriSpeech_1st_txt_looktown.wav \
--text "do you love me" --checkpoint /dev_huaying/zhijun/data/Name_VALLE_max-duration_90_dtype_float32_base-lr_0.01_world-size_8_train-stage_1_2023_05_18_11_05_05_copy/best-train-loss.pt



