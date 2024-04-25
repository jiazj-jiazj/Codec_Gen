
model_name,  gt_folder, gen_folder, out_folder, out_folder, prefix, use_gpu=True, checkpoint=None
python /dev_huaying/zhijun/UniSpeech/downstreams/speaker_verification/verification.py --model_name wavlm_large \
    --gt_folder vox1_data/David_Faustino/hn8GyCJIfLM_0000012.wav \
    --gen_folder vox1_data/Josh_Gad/HXUqYaOwrxA_0000015.wav \
    --out_folder  \
    --prefix \
    --use_gpu True \
    --checkpoint "/dev_huaying/zhijun/data/wavlm_large_finetune.pth"