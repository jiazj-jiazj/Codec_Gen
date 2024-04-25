basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
audio_feats_dir="/mnt/users/jiazhijun/data/LibriTTS/data/vc_tokenized_16k"

CUDA_VISIBLE_DEVICES=2 python3 egs/libritts/bin/tokenizer_train.py --dataset-parts "-p train" \
    --audio-extractor Tfcodec \
    --batch-duration 400 \
    --prefix mls-english \
    --train-part-path /raid/dataset/lhotse_dataset/mls/cut_lhotse/mls-english_cuts_train_16.jsonl.gz \
    --acoustic-sample 16000 \
    --input-language 0 \
    --semantic-layer 9 \
    --wav-path /raid/dataset/mls_files/mls_english \
    --output-dir /raid/dataset/lhotse_dataset/mls_train_lhotse_dataset \
    --tfnet-ckpt ../data/valle-tensorboard-models/other_models/tfcodec/890hrs16khz_tfnet_v2i_vqvae_20msVQ_hop5_combine4_rd1_6kbps/tfnet_v2i_vqvae_combineVQ-iter-514000-val1-0.348327-entropy-118.968-mel-0.067548-recon-0.012822.ckpt \
    # >../data/log/vc_tokenized_16k_tfcodec_split_mls_train2_20_${part_th}_${basestr}.txt 2>&1 &
# touch ${audio_feats_dir}/.libritts.tokenize.done
