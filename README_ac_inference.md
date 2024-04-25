# Introduction 
[VALL-E](https://arxiv.org/pdf/2301.02111.pdf)  
![Alt text](pictures/valle.png)
# Install
* VALL-E
    ```shell
        # environment.yml
        conda env create -f valle_23_4_22/conda_envs_files/environment.yml
        pip install + # the commited packages
    ```
    ```shell
            # PyTorch
        pip install torch==1.13.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
        pip install torchmetrics==0.11.1
        # fbank
        pip install librosa==0.8.1

        # phonemizer pypinyin
        apt-get install espeak-ng
        ## OSX: brew install espeak
        pip install phonemizer==3.2.1 pypinyin==0.48.0

        pip uninstall lhotse
        pip uninstall lhotse
        pip install git+https://github.com/lhotse-speech/lhotse

        # k2
        # find the right version in https://huggingface.co/csukuangfj/k2
        pip install https://huggingface.co/csukuangfj/k2/resolve/main/cuda/k2-1.23.4.dev20230224+cuda11.6.torch1.13.1-cp310-cp310-linux_x86_64.whl

        # icefall
        git clone https://github.com/k2-fsa/icefall
        cd icefall
        pip install -r requirements.txt
        export PYTHONPATH=`pwd`/../icefall:$PYTHONPATH
        echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.zshrc
        echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.bashrc
        cd -
        source ~/.zshrc

        # valle
        git clone https://msramc-audio@dev.azure.com/msramc-audio/CodecLM/_git/CodecLM
        cd valle
        pip install -e .
    ```

# VALL-E train&inference
First
```shell
    cd ~/CodecLM # it will upload the whole project to azureml 
```
Azureml Train 

* AR train:  
```shell
    python /dev_huaying/zhijun/CodecLM/egs/libritts/bin/azure_32g_ar.py
```
* NAR train:
```shell
    python /dev_huaying/zhijun/CodecLM/egs/libritts/bin/azure_32g_nar.py
```
Local Train
* Data prepare
```shell
    cd egs/libritts
    bash prepare.sh --stage -1 --stop-stage 3
```
* train & Inference
```shell
    exp_dir=exp/valle
    python3 egs/libritts/bin/train_direct.py --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
        --num-buckets 6 --dtype "float32" --save-every-n 2000 --valid-interval 4000 \
        --model-name valle --share-embedding true --norm-first true --add-prenet false \
        --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
        --base-lr 0.05 --warmup-steps 200 --average-period 0 \
        --num-epochs 100 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
        --exp-dir ${exp_dir}    --world-size 8


    ## Train NAR model
    python3 egs/libritts/bin/train_direct.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
        --num-buckets 6 --dtype "float32" --save-every-n 2000 --valid-interval 4000 \
        --model-name valle --share-embedding true --norm-first true --add-prenet false \
        --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
        --base-lr 0.05 --warmup-steps 200 --average-period 0 \
        --num-epochs 200 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
        --exp-dir ${exp_dir}    --world-size 8

    # inference 
    python3 egs/libritts/bin/combine_ar_nar_dir.py \
    --output-dir /dev_huaying/zhijun/data/LibriSpeech/gen_wavs_valle_ours \
    --model-name valle --norm-first true --add-prenet false --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
    --share-embedding true \
    --text-prompts "HE IS MY ESQUIRE EXCELLENCY RETURNED ROBIN WITH DIG." --text-tokens /dev_huaying/zhijun/data/valle-tensorboard-models/en_unique_text_tokens.k2symbols \
    --audio-prompts /dev_huaying/zhijun/data/LibriSpeech/filtered_4s_10s-test-spk-wer/61/70968/prompt_61-70968-0000.flac \
    --text "HE BEGAN A CONFUSED COMPLAINT AGAINST THE WIZARD WHO HAD VANISHED BEHIND THE CURTAIN ON THE LEFT.	" \
    --prefix-mode 1 \
    --checkpoint1 /dev_huaying/zhijun/data/valle-tensorboard-models/ar/Name_VALLE_max-duration_80_dtype_float32_base-lr_0.05_world-size_8_train-stage_1_echo_50_start_echo_1_accumulate_grad_steps_4_2023_06_19_15_03_10/best-train-loss-80.pt #AR model\
    --checkpoint2 /dev_huaying/zhijun/data/valle-tensorboard-models/nar/NAR_epoch200_from_epoch74_max_duration_70_base_lr_0.01_accumulate_grad_steps_4_prefix_mode_1/best-valid-loss-95.pt #NAR model\
    --top-k -1 --temperature 1.0 \
    --dir-need2test /dev_huaying/zhijun/data/LibriSpeech/filtered_4s_10s-test-spk-wer \
    --repeat-nums ${num_runs}
```
Train/Val Curve
* AR
![AR_train](pictures/AR_train.png)
![AR_valid](pictures/AR_valid.png)
* NAR
##curve after epoch74
![NAR_train](pictures/NAR_train.png)
![NAR_valid](pictures/NAR_valid.png)

# tokenize时修改manifest root前缀
valle_23_4_22/lhotse/audio.py