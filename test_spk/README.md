## Test speaker similarity

#### Environment Setup

1. `pip install --require-hashes -r requirements.txt`
2. Install fairseq code
   - For HuBERT_Large and Wav2Vec2.0 (XLSR), we should install the official [fairseq](https://github.com/pytorch/fairseq).
   - For UniSpeech-SAT large, we should install the [Unispeech-SAT](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech-SAT) fairseq code.
   - For WavLM, we should install the latest s3prl: `pip install s3prl@git+https://github.com/s3prl/s3prl.git@7ab62aaf2606d83da6c71ee74e7d16e0979edbc3#egg=s3prl`

### Local Test
```bash
basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
# #gt
# gen_folder="/dev_huaying/zhijun/data/LibriSpeech/filtered_4s_10s-test-spk-wer"
# prefix="prompt"
# is_gt=True

# # valle-ours
# gen_folder="/dev_huaying/zhijun/data/LibriSpeech/gen_wavs_valle_ours"
# prefix="valle_ours"
# is_gt=False

# # valle-3-6
# gen_folder="/dev_huaying/zhijun/data/LibriSpeech/gen_wavs_valle_3_6"
# prefix="valle_3_6"
# is_gt=False

# yourtts
gen_folder="/dev_huaying/zhijun/data/LibriSpeech/gen_wavs_yourtts"
prefix="yourtts"
is_gt=False

CUDA_VISIBLE_DEVICES=1 nohup python -u /dev_huaying/zhijun/UniSpeech/downstreams/speaker_verification/verification_dir.py --model_name wavlm_large \
    --gt_folder "/dev_huaying/zhijun/data/LibriSpeech/filtered_4s_10s-test-spk-wer" \
    --gen_folder ${gen_folder} \
    --out_folder "/dev_huaying/zhijun/valle_23_4_22/egs/libritts/log"\
    --prefix ${prefix} \
    --use_gpu True \
    --is_gt ${is_gt} \
    --checkpoint "/dev_huaying/zhijun/data/wavlm_large_finetune.pth" \
    > /dev_huaying/zhijun/UniSpeech/downstreams/speaker_verification/log/${prefix}_spk_${basestr}.log 2>&1 &
```

