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
