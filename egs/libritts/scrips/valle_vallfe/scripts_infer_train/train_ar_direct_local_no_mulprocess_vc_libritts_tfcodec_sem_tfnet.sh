basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S") 
train_stage=1
exp_dir=exp

# --manifest-dir /scratch/data/Libritts/tokenized_tfnet_semantic_token \
# --text-tokens /scratch/data/Libritts/tokenized_tfnet_semantic_token/unique_text_tokens.k2symbols \
# --semantic-tokens /scratch/data/Libritts/tokenized_tfnet_semantic_token/unique_semantic_tfnet_tokens.k2symbols \

CUDA_VISIBLE_DEVICES=0 python -u egs/libritts/bin/train_direct.py --max-duration 50 \
    --filter-min-duration 0.5 --filter-max-duration 14 --train-stage ${train_stage} \
    --num-buckets 6 \
    --dtype float32 \
    --save-every-n 1000 \
    --valid-interval 2000 \
    --model-name valle \
    --share-embedding true \
    --norm-first true \
    --add-prenet false \
    --decoder-dim 1024 \
    --nhead 16 \
    --num-decoder-layers 12 \
    --prefix-mode 1 \
    --base-lr 0.01 \
    --warmup-steps 200 \
    --average-period 0 \
    --num-quantizers 16 \
    --num-epochs 20 \
    --start-epoch 1 \
    --start-batch 0 \
    --accumulate-grad-steps 4 \
    --num-quantizers 16 \
    --world-size 1 \
    --manifest-dir /scratch/data/Libritts/tokenized \
    --text-tokens /scratch/data/Libritts/tokenized/unique_text_tokens.k2symbols \
    --semantic-tokens /scratch/data/Libritts/tokenized/unique_semantic_tokens.k2symbols \
    --newfile-suffix test2 \
    --input-semantic True \
    --shared-linear False \
    --only-autoregressive True \
    --semantic-type 1 \
    --is-local True \
    --exp-dir ${exp_dir}
    # > egs/libritts/log/train_libritts_tfcodec_sem_tfnet_${basestr}.txt 2>&1 &
