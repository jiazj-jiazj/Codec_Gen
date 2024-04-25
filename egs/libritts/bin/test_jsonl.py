from lhotse import CutSet, load_manifest_lazy



aa = load_manifest_lazy(
   "/mnt/zhijun/test_remote_json/cuts_train.jsonl.gz"
)

print(aa)