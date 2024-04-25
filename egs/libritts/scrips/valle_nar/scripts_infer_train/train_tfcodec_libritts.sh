basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S") 
train_stage=1
exp_dir=exp

# libritts
CUDA_VISIBLE_DEVICES=0 python -u egs/libritts/bin_commit/multiprocess_caller.py --nproc-per-node 1 --nnodes 1 --max-duration 50 \
    --filter-min-duration 0.5 --filter-max-duration 14 --train-stage ${train_stage} \
    --num-buckets 6 \
    --dtype float32 \
    --save-every-n 20000000000 \
    --valid-interval 500000000 \
    --model-name valle_nar \
    --share-embedding true \
    --norm-first true \
    --add-prenet false \
    --decoder-dim 1024 \
    --nhead 16 \
    --num-decoder-layers 12 \
    --prefix-mode 1 \
    --base-lr 0.05 \
    --warmup-steps 200 \
    --average-period 0 \
    --num-quantizers 1 \
    --num-epochs 120 \
    --start-epoch 1 \
    --start-batch 0 \
    --accumulate-grad-steps 1 \
    --num-quantizers 1 \
    --world-size 1 \
    --manifest-dir /scratch/data/Libritts/tokenized_tfnet_semantic_token \
    --text-tokens /scratch/data/Libritts/tokenized_tfnet_semantic_token/unique_text_tokens.k2symbols \
    --semantic-tokens /scratch/data/Libritts/tokenized_tfnet_semantic_token/unique_semantic_tokens.k2symbols \
    --newfile-suffix test2 \
    --input-semantic True \
    --is-pretrain True \
    --semantic-remove True \
    --shared-linear False \
    --only-autoregressive True \
    --is-local True \
    --exp-dir ${exp_dir} \
    --pret-mode 5 \
    --pret-prob 0.15 \
