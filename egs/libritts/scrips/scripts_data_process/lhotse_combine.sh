audio_feats_dir=/mnt/users/jiazhijun/data/LibriTTS/data/vc_tokenized_16k_tfcodec

# lhotse combine \
#     ${audio_feats_dir}/libritts_cuts_dev-clean.jsonl.gz \
#     ${audio_feats_dir}/libritts_cuts_dev-other.jsonl.gz \
#     ${audio_feats_dir}/cuts_dev.jsonl.gz


lhotse combine \
    ${audio_feats_dir}/libritts_cuts_test-clean.jsonl.gz \
    ${audio_feats_dir}/libritts_cuts_test-other.jsonl.gz \
    ${audio_feats_dir}/cuts_test.jsonl.gz


# lhotse combine \
#     ${audio_feats_dir}/libritts_cuts_train-clean-100.jsonl.gz \
#     ${audio_feats_dir}/libritts_cuts_train-clean-360.jsonl.gz \
#     ${audio_feats_dir}/libritts_cuts_train-other-500.jsonl.gz \
#     ${audio_feats_dir}/cuts_train.jsonl.gz