basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
audio_feats_dir="/mnt/users/jiazhijun/data/LibriTTS/data/vc_tokenized_16k"

CUDA_VISIBLE_DEVICES=3 nohup python3 -u /mnt/users/jiazhijun/valle_23_4_22/egs/libritts/bin/tokenizer.py --dataset-parts "-p train-other-500" \
    --audio-extractor Tfcodec \
    --batch-duration 400 \
    --prefix libritts \
    --src-dir /mnt/users/jiazhijun/data/LibriTTS/data/manifests \
    --acoustic-sample 16000 \
    --input-language 0 \
    --semantic-layer 9 \
    --wav-path /mnt/users/jiazhijun/data/LibriTTS/data/LibriTTS/ \
    --output-dir /mnt/users/jiazhijun/data/LibriTTS/data/vc_tokenized_16k_tfcodec > /mnt/users/jiazhijun/valle_23_4_22_23_7_30/egs/libritts/log/vc_tokenized_16k_tfcodec_train_other_500_${basestr}.txt 2>&1 &
touch ${audio_feats_dir}/.libritts.tokenize.done
