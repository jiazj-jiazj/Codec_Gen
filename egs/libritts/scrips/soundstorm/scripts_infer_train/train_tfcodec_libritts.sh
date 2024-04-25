basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S") 
train_stage=1
exp_dir=exp

# libritts
CUDA_VISIBLE_DEVICES=0 python -u egs/libritts/bin_commit/multiprocess_caller.py --nproc-per-node 1 --nnodes 1 --max-duration 50 \
    --filter-min-duration 0.5 --filter-max-duration 14 --train-stage ${train_stage} \
    --num-buckets 6 \
    --dtype float32 \
    --save-every-n 20000000000 \
    --valid-interval 100 \
    --model-name Soundstorm \
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
    --num-quantizers 16 \
    --num-epochs 120 \
    --start-epoch 1 \
    --start-batch 0 \
    --accumulate-grad-steps 1 \
    --num-quantizers 16 \
    --world-size 1 \
    --manifest-dir /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token \
    --text-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token/unique_text_tokens.k2symbols \
    --semantic-tokens /home/v-zhijunjia/zhijundata_small_v3/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token/unique_semantic_tokens.k2symbols \
    --newfile-suffix test2 \
    --input-semantic True \
    --shared-linear False \
    --only-autoregressive True \
    --is-local True \
    --exp-dir ${exp_dir} \
    --nar-mask-ratio 0.5 \
    --nar-mask-type 2 \
    --parrallel-mode 1 \
    --group-in-mask True \
    --group-in-mask-replace-prob 0 \
    --group-in-mask-replace-all-prob 0.15 \
    --train-dir-name "cuts_train_1000.jsonl.gz" \
    --val-dir-name "cuts_dev_1000.jsonl.gz"
    # --group-in-mask-replace-all-varible True \
