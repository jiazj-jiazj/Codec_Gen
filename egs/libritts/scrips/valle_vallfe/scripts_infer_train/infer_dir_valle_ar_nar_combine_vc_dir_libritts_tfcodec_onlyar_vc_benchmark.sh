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
basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  

num_runs=1
top_k=10
add_prenet="False"  
CUDA_VISIBLE_DEVICES=0 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py --model-name valle --norm-first true --add-prenet ${add_prenet} \
--decoder-dim 1024 --nhead 16 \
--num-decoder-layers 12 \
--share-embedding true \
--nums ${num_runs} \
--semantic-tokens /scratch/data/Libritts/tokenized_tfnet_semantic_token/unique_semantic_tfnet_tokens.k2symbols \
--semantic-sys-dir /home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/source \
--audio-prompts-dir /home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/prompt \
--input-semantic True \
--only-autoregressive True \
--prefix-mode 1 \
--checkpoint1 /home/v-zhijunjia/zhijundata_small/data/LibriTTS/lhotse_vc/output_vc/sem_tp_1_Name_VALLE_max-duration_50_dtype_float32_base-lr_0.05_train-stage_1_echo_70_start_echo_1_prefix_mode_1_input_semantic_True_only_ar_True_num_quantizers_16_sheduler_steps_5000_sheduler_epochs_4_accumulate_grad_steps_4_2023_10_31_07_45_20/epoch-60.pt \
--top-k ${top_k} \
--shared-linear False \
--temperature 1.0 \
--num-quantizers 16 \
--input-codec 2 \
--target-mode 0 \
--accent-remove False \
--semantic-type 1 \
--mode 0 \
--mode-stage2 0 \
--outputdir-name benchmark_tfnet_vc/v2_top_${top_k}_${basestr} >/home/v-zhijunjia/data/log/infer_dir_valle_ar_nar_combine_vc_dir_libritts_librispeech_benchmark_${basestr}.txt 2>&1 &
# /mnt/users/jiazhijun/data/valle-tensorboard-models/accents/ar_accents_restore/restore_from_vc_libritts.pt

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