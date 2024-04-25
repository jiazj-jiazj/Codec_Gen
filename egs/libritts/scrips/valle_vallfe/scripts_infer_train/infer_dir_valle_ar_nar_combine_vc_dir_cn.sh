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
file_name="test_dns_v5"
CUDA_VISIBLE_DEVICES=3 nohup python -u egs/libritts/bin/combine_ar_nar_vc_dir.py --model-name valle --norm-first true --add-prenet ${add_prenet} --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
--share-embedding true \
--nums ${num_runs} \
--semantic-tokens ../data/valle-tensorboard-models/unique_semantic_tokens.k2symbols \
--semantic-sys-dir ../data/test_vc/${file_name}/source \
--audio-prompts-dir ../data/test_vc/${file_name}/prompt \
--input-semantic True \
--prefix-mode 1 \
--checkpoint1 ../data/valle-tensorboard-models/cn_vc/ar/Name_VALLE_max-duration_50_dtype_float32_base-lr_0.05_world-size_8_train-stage_1_echo_100_start_echo_1_accumulate_grad_steps_4_prefix_mode_1_input_semantic_True_2023_07_12_12_21_16/epoch-37.pt \
--checkpoint2 ../data/valle-tensorboard-models/cn_vc/nar/Name_VALLE_max-duration_50_dtype_float32_base-lr_0.05_world-size_8_train-stage_2_echo_150_start_echo_1_accumulate_grad_steps_4_prefix_mode_1_input_semantic_True_valid_interval_10000_2023_07_12_15_23_53/epoch-28.pt \
--input-language 1 \
--semantic-layer 9 \
--outputdir-name converted_vc_${top_k} \
--top-k ${top_k} --temperature 1.0 \
> egs/libritts/log/vc_dir_topk_${top_k}_${file_name}_${basestr}.txt 2>&1 &