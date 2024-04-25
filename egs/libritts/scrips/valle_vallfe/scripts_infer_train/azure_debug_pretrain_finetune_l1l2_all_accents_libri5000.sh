basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S") 
train_stage=1
exp_dir=exp

python egs/libritts/bin/multiprocess_caller.py --max-duration 200 \
    --nproc-per-node 1 \
    --filter-min-duration 0.5 --filter-max-duration 14 --train-stage ${train_stage} \
    --num-buckets 6 \
    --dtype float32 \
    --save-every-n 300 \
    --log-interval 20 \
    --valid-interval 40 \
    --model-name valle \
    --share-embedding true \
    --norm-first true \
    --add-prenet false \
    --decoder-dim 1024 \
    --nhead 16 \
    --num-decoder-layers 12 \
    --prefix-mode 1 \
    --base-lr 0.001 \
    --warmup-steps 20 \
    --average-period 0 \
    --num-epochs 50 \
    --start-epoch 1 \
    --start-batch 0 \
    --sheduler-epochs 5 \
    --sheduler-steps 200 \
    --accumulate-grad-steps 1 \
    --num-quantizers 1 \
    --world-size 1 \
    --manifest-dir /scratch/data/l1_l2_arctic/lhotse_data_vc_target_semantic_native_all_accents/initial_tokenized \
    --text-tokens /scratch/data/l1_l2_arctic/lhotse_data_vc_target_semantic_native_all_accents/initial_tokenized/unique_text_tokens.k2symbols \
    --semantic-tokens /scratch/data/l1_l2_arctic/lhotse_data_vc_target_semantic_native_all_accents/initial_tokenized/unique_semantic_tokens.k2symbols \
    --newfile-suffix test2 \
    --input-semantic True \
    --only-autoregressive True \
    --semantic-remove True \
    --is-local True \
    --exp-dir "/home/v-zhijunjia/data/valle-tensorboard-models/pretrain_finetune/mode_2_infilling_0_5_5_all_accent_l1_l2_libri5000" \
    --restore True \
    --restore-file-name "pret-infilling-epoch-28.pt" \
    --random-tgt-spk True \
    --sec-dataset True \
    --manifest-dir-sec /scratch/data/Libritts/tokenized
    # > egs/libritts/log/finetune_l1l2_all_accents_${basestr}.txt 2>&1 &



