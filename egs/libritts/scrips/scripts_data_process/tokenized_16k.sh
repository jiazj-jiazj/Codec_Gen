basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
audio_feats_dir="/mnt/zhijun/LibriTTS/data/vc_tokenized_16k"
nohup python3 -u /dev_huaying/zhijun/valle_23_4_22/egs/libritts/bin/tokenizer_16k.py --dataset-parts all \
    --audio-extractor Encodec \
    --batch-duration 400 \
    --src-dir /mnt/zhijun/LibriTTS/data/manifests \
    --output-dir /mnt/zhijun/LibriTTS/data/vc_tokenized_16k > /dev_huaying/zhijun/valle_23_4_22/egs/libritts/log/vc_tokenized_16k_${basestr}.txt 2>&1 &
touch ${audio_feats_dir}/.libritts.tokenize.done
