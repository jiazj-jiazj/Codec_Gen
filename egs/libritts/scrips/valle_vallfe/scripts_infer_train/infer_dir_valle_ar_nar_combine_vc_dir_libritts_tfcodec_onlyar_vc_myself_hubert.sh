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
num_runs_stage2=1
top_k=10
top_k_stage2=1
top_p=1
top_p_stage2=1
add_prenet="False" 

for top_k in $(seq 10 1 10); do  
    CUDA_VISIBLE_DEVICES=0 python egs/libritts/bin/combine_ar_nar_vc_dir_onlyar.py --model-name valle --norm-first true --add-prenet ${add_prenet} \
    --decoder-dim 1024 --nhead 16 \
    --num-decoder-layers 12 \
    --share-embedding true \
    --nums ${num_runs} \
    --nums-stage2 ${num_runs_stage2} \
    --semantic-tokens /scratch/data/Libritts/tokenized_tfnet_semantic_token/unique_semantic_tokens.k2symbols \
    --semantic-sys-dir /home/v-zhijunjia/demo/iclr_final/arctic_v3/source \
    --input-semantic True \
    --only-autoregressive True \
    --prefix-mode 1 \
    --checkpoint1 /home/v-zhijunjia/data/valle-tensorboard-models/vc/only_ar/epoch-40.pt \
    --top-k ${top_k} \
    --top-k-stage2 ${top_k_stage2} \
    --top-p ${top_p} \
    --top-p-stage2 ${top_p_stage2} \
    --shared-linear False \
    --temperature 1.0 \
    --num-quantizers 16 \
    --input-codec 2 \
    --target-mode 0 \
    --accent-remove True \
    --semantic-depup False \
    --semantic-type 0 \
    --mode 0 \
    --mode-stage2 0 \
    --outputdir-name converted_test_can_del/vctk_vc_top_${top_k}_topp_${top_p}_${basestr}
done
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