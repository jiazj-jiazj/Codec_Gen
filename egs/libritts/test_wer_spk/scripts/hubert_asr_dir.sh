is_gt=False
input_dir="/home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/tts_reference_comparedwith_sota_top__80_epoch_44_v2_librispeech"

python egs/libritts/test_wer_spk/hubert_asr_ls960.py \
    --is_gt ${is_gt} \
    --input_dir ${input_dir}