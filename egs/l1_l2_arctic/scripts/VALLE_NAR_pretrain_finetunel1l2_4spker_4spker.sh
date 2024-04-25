basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S") 
train_stage=1
exp_dir=exp

python egs/libritts/bin_commit/multiprocess_caller.py --max-duration 150 \
    --nproc-per-node 1 \
    --filter-min-duration 0.5 --filter-max-duration 14 --train-stage ${train_stage} \
    --num-buckets 6 \
    --dtype float32 \
    --save-every-n 99900000 \
    --log-interval 10 \
    --valid-interval 80 \
    --model-name valle_nar \
    --share-embedding true \
    --norm-first true \
    --add-prenet false \
    --decoder-dim 1024 \
    --nhead 16 \
    --num-decoder-layers 12 \
    --prefix-mode 1 \
    --base-lr 0.002 \
    --warmup-steps 20 \
    --average-period 0 \
    --num-epochs 20 \
    --start-epoch 1 \
    --start-batch 0 \
    --sheduler-epochs 5 \
    --sheduler-steps 450 \
    --accumulate-grad-steps 1 \
    --num-quantizers 1 \
    --world-size 1 \
    --manifest-dir /home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/lhotse_data_v2/indian2native \
    --text-tokens /home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/lhotse_data_v2/indian2native/unique_text_tokens_all.k2symbols \
    --semantic-tokens /home/v-zhijunjia/zhijundata_small_v2/data/l1_l2_arctic/lhotse_data_v2/indian2native/unique_semantic_tokens.k2symbols \
    --newfile-suffix test2 \
    --input-semantic True \
    --only-autoregressive True \
    --semantic-remove True \
    --is-local True \
    --exp-dir "/home/v-zhijunjia/zhijundata_small_v2/data_local/valle-tensorboard-models/pretrain_finetune/VALLE_NAR_hubert_sem/mask_0_5/indian2native_lr_0_002" \
    --restore True \
    --restore-file-name "pret-p-0-5-epoch-45.pt" \
    --random-tgt-spk False \
    --tgt-spk-names cmu_us_bdl_arctic,cmu_us_clb_arctic,cmu_us_rms_arctic,cmu_us_slt_arctic \
    --sec-dataset False \
    --manifest-dir-sec /scratch/data/Libritts/tokenized \
    --ac-native-mask False \