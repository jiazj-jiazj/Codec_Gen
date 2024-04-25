basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
dataset_parts="-p test"  # debug
# dataset_parts="-p test -p train"  # debug

text_extractor="pypinyin_initials_finals"



out_dir_name="tokenized_16k_layer9"
audio_output_dir="/mnt/users/jiazhijun/data/AISHELL3_lhotse_data/lhotse_data_new/lhotse_data/data/${out_dir_name}"
nohup python3 -u egs/libritts/bin/tokenizer.py --dataset-parts "${dataset_parts}" \
    --text-extractor ${text_extractor} \
    --audio-extractor Encodec \
    --prefix "aishell3" \
    --batch-duration 400 \
    --src-dir /mnt/users/jiazhijun/data/AISHELL3_lhotse_data/lhotse_data_new/lhotse_data/data/manifests \
    --acoustic-sample 16000 \
    --input-language 1 \
    --semantic-layer 9 \
    --output-dir ${audio_output_dir} > egs/libritts/log/aishell3_${out_dir_name}_${basestr}.txt 2>&1 &
# python3 egs/libritts/bin/tokenizer.py --dataset-parts "-p train" \
#     --text-extractor pypinyin_initials_finals \
#     --audio-extractor Encodec \
#     --prefix "aishell3" \
#     --batch-duration 400 \
#     --src-dir /mnt/users/jiazhijun/data/AISHELL3_lhotse_data/lhotse_data_new/lhotse_data/data/manifests \
#     --acoustic-sample 16000 \
#     --input-language 1 \
#     --semantic-layer 9 \
#     --output-dir /mnt/users/jiazhijun/data/AISHELL3_lhotse_data/lhotse_data_new/lhotse_data/data/tokenized_16k_layer9
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

