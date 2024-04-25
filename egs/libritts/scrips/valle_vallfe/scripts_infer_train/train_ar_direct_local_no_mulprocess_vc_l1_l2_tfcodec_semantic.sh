basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S") 
train_stage=1
exp_dir=exp

CUDA_VISIBLE_DEVICES=0 python egs/libritts/bin/train_direct.py --max-duration 5 \
    --filter-min-duration 0.5 --filter-max-duration 14 --train-stage ${train_stage} \
    --num-buckets 6 \
    --dtype float32 \
    --save-every-n 200 \
    --valid-interval 400 \
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
    --num-epochs 20 \
    --start-epoch 1 \
    --start-batch 0 \
    --accumulate-grad-steps 1 \
    --num-quantizers 1 \
    --world-size 1 \
    --manifest-dir /mnt/zhijun/Accents/combine_l1_l2_all_accents/tokenized_tfcodec/tokenized_24k_layer9 \
    --text-tokens /mnt/zhijun/Accents/combine_l1_l2_all_accents/tokenized_tfcodec/tokenized_24k_layer9/unique_text_tokens.k2symbols \
    --semantic-tokens /mnt/zhijun/Accents/combine_l1_l2_all_accents/tokenized_tfcodec/tokenized_24k_layer9/unique_semantic_tokens.k2symbols \
    --newfile-suffix test2 \
    --input-semantic True \
    --semantic-depup False \
    --tgt-spk-name "cmu_us_bdl_arctic" \
    --semantic-remove True \
    --only-autoregressive True \
    --shared-linear False \
    --is-local True \
    --exp-dir ${exp_dir} 

# nohup env CUDA_VISIBLE_DEVICES=1 python -u egs/libritts/bin/train_direct.py --max-duration 5 \
#     --filter-min-duration 0.5 --filter-max-duration 14 --train-stage ${train_stage} \
#     --num-buckets 6 \
#     --dtype float32 \
#     --save-every-n 200 \
#     --valid-interval 400 \
#     --model-name valle \
#     --share-embedding true \
#     --norm-first true \
#     --add-prenet false \
#     --decoder-dim 1024 \
#     --nhead 16 \
#     --num-decoder-layers 12 \
#     --prefix-mode 1 \
#     --base-lr 0.01 \
#     --warmup-steps 200 \
#     --average-period 0 \
#     --num-quantizers 8 \
#     --num-epochs 20 \
#     --start-epoch 1 \
#     --start-batch 0 \
#     --accumulate-grad-steps 4 \
#     --num-quantizers 16 \
#     --world-size 1 \
#     --manifest-dir /mnt/zhijun/Accents/combine_L1_L2/tokenized_tfcodec/tokenized_24k_layer9 \
#     --text-tokens /mnt/zhijun/Accents/combine_L1_L2/tokenized_tfcodec/tokenized_24k_layer9/unique_text_tokens.k2symbols \
#     --semantic-tokens /mnt/zhijun/Accents/combine_L1_L2/tokenized_tfcodec/tokenized_24k_layer9/unique_semantic_tokens.k2symbols \
#     --newfile-suffix test2 \
#     --input-semantic True \
#     --only-autoregressive True \
#     --is-local True \
#     --exp-dir ${exp_dir} \
#     > egs/libritts/log/train_libritts_tfcodec_${basestr}.txt 2>&1 &
