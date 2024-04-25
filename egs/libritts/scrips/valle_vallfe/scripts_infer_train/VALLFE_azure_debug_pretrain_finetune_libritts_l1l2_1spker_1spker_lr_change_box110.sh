basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S") 
train_stage=1
exp_dir=exp

python egs/libritts/bin/multiprocess_caller.py --max-duration 200 \
    --nproc-per-node 1 \
    --filter-min-duration 0.5 --filter-max-duration 14 --train-stage ${train_stage} \
    --num-buckets 6 \
    --dtype float32 \
    --save-every-n 40000 \
    --log-interval 20 \
    --valid-interval 40 \
    --model-name VALLFE \
    --share-embedding true \
    --norm-first true \
    --add-prenet false \
    --decoder-dim 1024 \
    --nhead 16 \
    --num-decoder-layers 12 \
    --encoder-num-layers 6 \
    --decoder-num-layers 6 \
    --prepend-bos True \
    --prefix-mode 1 \
    --base-lr 0.001 \
    --warmup-steps 20 \
    --average-period 0 \
    --num-epochs 300 \
    --start-epoch 1 \
    --start-batch 0 \
    --sheduler-epochs 5 \
    --sheduler-steps 200 \
    --parts-req-gra 2 \
    --accumulate-grad-steps 1 \
    --num-quantizers 1 \
    --world-size 1 \
    --manifest-dir /home/v-zhijunjia/zhijundata_small/data/l1_l2_arctic/lhotse_data_vc_target_semantic_native/tokenized \
    --text-tokens /home/v-zhijunjia/zhijundata_small/data/l1_l2_arctic/lhotse_data_vc_target_semantic_native/tokenized/unique_text_tokens.k2symbols \
    --semantic-tokens /home/v-zhijunjia/zhijundata_small/data/l1_l2_arctic/lhotse_data_vc_target_semantic_native/tokenized/unique_semantic_tokens.k2symbols \
    --newfile-suffix test2 \
    --input-semantic True \
    --only-autoregressive True \
    --semantic-remove True \
    --is-local True \
    --exp-dir "/home/v-zhijunjia/zhijundata_small/data_local/valle-tensorboard-models/pretrain_finetune/VALLFE_hubert_sem/tune_encoder_first3layers_0_001" \
    --restore True \
    --restore-file-name "pret-epoch-70.pt" \
    --random-tgt-spk False \
    --sec-dataset False \
    --manifest-dir-sec /scratch/data/Libritts/tokenized \
    --ac-native-mask False \
    --pret-mode 3 \
    --pret-prob 0.05 \
    --pret-lam 3 \
    --pret-token 500


    # --prepend-bos True \
