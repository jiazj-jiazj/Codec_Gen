basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
dataset_parts="-p dev"  # debug
text_extractor="pypinyin_initials_finals"



out_dir_name="tokenized_16k_layer9"
audio_output_dir="/mnt/users/jiazhijun/data/AISHELL2_lhotse_data/${out_dir_name}"
nohup env CUDA_VISIBLE_DEVICES=2 python3 -u egs/libritts/bin/tokenizer.py --dataset-parts "${dataset_parts}" \
    --text-extractor ${text_extractor} \
    --audio-extractor Encodec \
    --prefix "aishell2" \
    --batch-duration 400 \
    --src-dir /mnt/users/jiazhijun/data/AISHELL2_lhotse_data/manifests \
    --acoustic-sample 16000 \
    --input-language 1 \
    --semantic-layer 9 \
    --output-dir ${audio_output_dir} > egs/libritts/log/aishell2_${out_dir_name}_${basestr}.txt 2>&1 &

# python3 -u /mnt/users/jiazhijun/valle_23_4_22/egs/libritts/bin/tokenizer.py --dataset-parts "${dataset_parts}" \
#     --text-extractor ${text_extractor} \
#     --audio-extractor Encodec \
#     --prefix "aishell2" \
#     --batch-duration 400 \
#     --src-dir /mnt/users/jiazhijun/data/AISHELL2_lhotse_data/manifests \
#     --acoustic-sample 16000 \
#     --input-language 1 \
#     --semantic-layer 9 \
#     --output-dir ${audio_output_dir}

