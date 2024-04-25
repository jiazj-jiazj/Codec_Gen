basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S") 
train_stage=1
exp_dir=exp

python egs/libritts/bin/multiprocess_caller.py --max-duration 200 \
    --nproc-per-node 1 \
    --filter-min-duration 0.5 --filter-max-duration 14 --train-stage ${train_stage} \
    --num-buckets 6 \
    --dtype float32 \
    --save-every-n 20 \
    --log-interval 10 \
    --valid-interval 30 \
    --model-name valle \
    --share-embedding true \
    --norm-first true \
    --add-prenet false \
    --decoder-dim 1024 \
    --nhead 16 \
    --num-decoder-layers 12 \
    --prefix-mode 1 \
    --base-lr 0.0002 \
    --warmup-steps 20 \
    --average-period 0 \
    --num-epochs 100 \
    --start-epoch 1 \
    --start-batch 0 \
    --sheduler-epochs 5 \
    --sheduler-steps 200 \
    --accumulate-grad-steps 1 \
    --num-quantizers 1 \
    --world-size 1 \
    --manifest-dir /scratch/data/l1_l2_arctic/lhotse_data_vc_target_semantic_native/tokenized \
    --text-tokens /scratch/data/l1_l2_arctic/lhotse_data_vc_target_semantic_native/tokenized/unique_text_tokens.k2symbols \
    --semantic-tokens /scratch/data/l1_l2_arctic/lhotse_data_vc_target_semantic_native/tokenized/unique_semantic_tokens.k2symbols \
    --newfile-suffix test2 \
    --input-semantic True \
    --only-autoregressive True \
    --semantic-remove True \
    --is-local True \
    --exp-dir "/home/v-zhijunjia/data/valle-tensorboard-models/pretrain_finetune/mode_5_mask015_source4spker_allcases_tgt1spker_lr_0002" \
    --restore True \
    --restore-file-name "pret-mode5-epoch-53.pt" \
    --random-tgt-spk False \
    --sec-dataset True \
    --manifest-dir-sec /scratch/data/Libritts/tokenized \
    --ac-native-mask False \
    --pret-mode 3 \
    --pret-prob 0.05 \
    --pret-lam 3 \
    --pret-token 500