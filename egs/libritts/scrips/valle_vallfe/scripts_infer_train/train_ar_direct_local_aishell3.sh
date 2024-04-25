# exp_dir=exp/valle

# ## Train AR model
# python3 bin/trainer_direct.py --max-duration 20 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
#       --num-buckets 6 --dtype "float16" --save-every-n 100 --valid-interval 200 \
#       --model-name valle --share-embedding true --norm-first true --add-prenet false \
#       --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
#       --base-lr 0.01 --warmup-steps 200 --average-period 0 \
#       --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
#       --world-size 2 \
#       --manifest-dir /home/aiscuser/data/data/tokenized/ \
#       --text-tokens /home/aiscuser/data/data/tokenized/unique_text_tokens.k2symbols \
#       --newfile_suffix test2 \
#       --exp-dir ${exp_dir}
# exp_dir=exp/valle

# ## Train AR model
# python3 -m torch.distributed.run bin/trainer_direct.py --nproc_per_node 2 --nnodes 1 \
#       --max-duration 20 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
#       --num-buckets 6 --dtype "float16" --save-every-n 100 --valid-interval 200 \
#       --model-name valle --share-embedding true --norm-first true --add-prenet false \
#       --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
#       --base-lr 0.01 --warmup-steps 200 --average-period 0 \
#       --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
#       --manifest-dir /home/aiscuser/data/data/tokenized/ \
#       --text-tokens /home/aiscuser/data/data/tokenized/unique_text_tokens.k2symbols \
#       --newfile_suffix test2 \
#       --exp-dir ${exp_dir}

exp_dir=exp/valle_aishell3

## Train AR model
python3 bin/multiprocess_caller.py --nproc-per-node 2 --nnodes 1 \
      --max-duration 10 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
      --num-buckets 6 --dtype "float16" --save-every-n 100 --valid-interval 200 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.01 --warmup-steps 200 --average-period 0 \
      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
      --manifest-dir /dev_huaying/zhijun/data/AISHELL-3/lhotse_data/data/tokenized/ \
      --text-tokens /dev_huaying/zhijun/data/AISHELL-3/lhotse_data/data/tokenized/unique_text_tokens.k2symbols \
      --newfile-suffix test2 \
      --world-size 2 \
      --is-local True \
      --exp-dir ${exp_dir}