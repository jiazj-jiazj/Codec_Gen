basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
dataset_parts="-p all"  # debug

out_dir_name="cutset_data"
audio_output_dir=/scratch/indian_accent_datasets/indictts-english/IndicTTS_lhotse/${out_dir_name}_11
CUDA_VISIBLE_DEVICES=0 nohup python -u egs/indian_accent_english/bin/tokenizer.py \
    --dataset-parts "${dataset_parts}" \
    --audio-extractor Tfcodec \
    --prefix "Indic_TTS" \
    --batch-duration 200 \
    --src-dir /scratch/indian_accent_datasets/indictts-english/IndicTTS_lhotse \
    --acoustic-sample 16000 \
    --input-language 0 \
    --semantic-layer 9 \
    --add-semantic True \
    --output-dir ${audio_output_dir} > /home/v-zhijunjia/CodecGen/egs/indian_accent_english/log/tokenize_${basestr}.txt 2>&1 &
    # > egs/libritts/log/tokenized_tfcodec_l1_l2_arctic_all_accents_${basestr}.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python3 -u egs/libritts/bin/tokenizer.py --dataset-parts "${dataset_parts}" \
#     --audio-extractor Tfcodec \
#     --prefix "l1_l2_arctic" \
#     --batch-duration 400 \
#     --src-dir /mnt/users/jiazhijun/data/Accents/combine_L1_L2/manifests \
#     --acoustic-sample 16000 \
#     --input-language 0 \
#     --semantic-layer 9 \
#     --output-dir ${audio_output_dir} > egs/libritts/log/l1_l2_arctic_tfcodec_${out_dir_name}_${basestr}.txt 2>&1 &
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
