exp_dir=exp/valle

## Train AR model
python3 bin/valid_direct.py --max-duration 20 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
      --num-buckets 6 --dtype "float16" --save-every-n 10000 --valid-interval 20000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.01 --warmup-steps 200 --average-period 0 \
      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
      --world-size 2 \
      --manifest-dir /mnt/zhijun/data/libritts/data/tokenized/ \
      --text-tokens /mnt/zhijun/data/libritts/data/tokenized/unique_text_tokens.k2symbols \
      --newfile_suffix test2 \
      --exp-dir ${exp_dir} \
      --model-path /dev_huaying/zhijun/data/Name_VALLE_max-duration_90_dtype_float32_base-lr_0.01_world-size_8_train-stage_1_2023_05_18_11_05_05_copy/best-train-loss.pt