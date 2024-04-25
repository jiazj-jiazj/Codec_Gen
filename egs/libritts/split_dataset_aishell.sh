dataset_parts="-p train -p dev -p test"  # debug
text_extractor="pypinyin_initials_finals"
audio_extractor="Encodec"  # or Fbank
audio_feats_dir=/dev_huaying/zhijun/data/AISHELL/lhotse_data_deletekongge/data/data/tokenized


lhotse subset --first 400 \
    ${audio_feats_dir}/aishell_cuts_dev.jsonl.gz \
    ${audio_feats_dir}/cuts_dev.jsonl.gz

lhotse subset --last 13922 \
    ${audio_feats_dir}/aishell_cuts_dev.jsonl.gz \
    ${audio_feats_dir}/cuts_dev_others.jsonl.gz

# train
lhotse combine \
    ${audio_feats_dir}/cuts_dev_others.jsonl.gz \
    ${audio_feats_dir}/aishell_cuts_train.jsonl.gz \
    ${audio_feats_dir}/cuts_train.jsonl.gz

# test
lhotse copy \
    ${audio_feats_dir}/aishell_cuts_test.jsonl.gz \
    ${audio_feats_dir}/cuts_test.jsonl.gz

touch ${audio_feats_dir}/.aishell.train.done

# python3 ./bin/display_manifest_statistics.py --manifest-dir ${audio_feats_dir}