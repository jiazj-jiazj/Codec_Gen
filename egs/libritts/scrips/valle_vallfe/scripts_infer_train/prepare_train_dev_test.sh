audio_feats_dir=/mnt/zhijun/LibriTTS/data/vc_tokenized


# train
lhotse combine \
${audio_feats_dir}/libritts_cuts_train-clean-100.jsonl.gz \
${audio_feats_dir}/libritts_cuts_train-clean-360.jsonl.gz \
${audio_feats_dir}/libritts_cuts_train-other-500.jsonl.gz \
${audio_feats_dir}/cuts_train.jsonl.gz

# dev
lhotse copy \
${audio_feats_dir}/libritts_cuts_dev-clean.jsonl.gz \
${audio_feats_dir}/cuts_dev.jsonl.gz

lhotse copy \
${audio_feats_dir}/libritts_cuts_test-clean.jsonl.gz \
${audio_feats_dir}/cuts_test.jsonl.gz

touch ${audio_feats_dir}/.libritts.train.done


python3 egs/libritts/bin/display_manifest_statistics.py --manifest-dir ${audio_feats_dir}