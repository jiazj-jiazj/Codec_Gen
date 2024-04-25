basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S") 
train_stage=1
exp_dir=exp

python egs/libritts/bin/multiprocess_caller.py --max-duration 200 \
    --nproc-per-node 1 \
    --filter-min-duration 0.5 \
    --filter-max-duration 14 --train-stage ${train_stage} \
    --num-buckets 6 \
    --dtype float32 \
    --save-every-n 200000000 \
    --log-interval 20 \
    --valid-interval 40 \
    --model-name vallf \
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
    --num-epochs 20 \
    --start-epoch 1 \
    --start-batch 0 \
    --sheduler-epochs 5 \
    --sheduler-steps 200 \
    --accumulate-grad-steps 4 \
    --num-quantizers 1 \
    --world-size 1 \
    --manifest-dir /scratch/data/Libritts/tokenized_tfnet_semantic_token \
    --text-tokens /scratch/data/Libritts/tokenized_tfnet_semantic_token/unique_text_tokens.k2symbols \
    --semantic-tokens /scratch/data/Libritts/tokenized_tfnet_semantic_token/unique_semantic_tokens.k2symbols \
    --train-dir-name cuts_train_1000.jsonl.gz\
    --val-dir-name cuts_dev_1000.jsonl.gz \
    --newfile-suffix test2 \
    --input-semantic True \
    --only-autoregressive True \
    --semantic-remove True \
    --is-local True \
    --exp-dir ${exp_dir} \
    --is-pretrain True \
    --pret-mode 5 \
    --pret-prob 0.15 \
    --pret-lam 3 \
    --pret-token 256

# /home/v-zhijunjia/valle-4-23/valle/data/dataset.py
