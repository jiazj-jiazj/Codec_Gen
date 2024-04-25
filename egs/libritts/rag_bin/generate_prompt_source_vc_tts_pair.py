import os
import shutil
prompt_list = []
source_list = []

prompt_dir = "/home/v-zhijunjia/data/data_update/benchmark_vc_tts_9s/prompt_9s_39spkers_3s"
source_dir = "/home/v-zhijunjia/data/data_update/benchmark_vc_tts_9s/vc_source"

tgt_dir = "/home/v-zhijunjia/data/data_update/benchmark_vc_tts_9s/vc_source_pair"
os.makedirs(tgt_dir, exist_ok=True)

for root, dirs, files in os.walk(prompt_dir):
    for file in files:
        file_path = os.path.join(root, file)
        prompt_list.append(file_path)

for root, dirs, files in os.walk(source_dir):
    for file in files:
        file_path = os.path.join(root, file)
        source_list.append(file_path)

print(len(prompt_list))
print(len(source_list))
for prompt_file, source_file in zip(prompt_list, source_list):
    basename_prompt = os.path.basename(prompt_file)
    no_houzui_basename_prompt = basename_prompt.split('.')[0]
    basename_source = os.path.basename(source_file)
    no_houzui_basename_source = basename_source.split('.')[0]

    tgt_file_path = os.path.join(tgt_dir, no_houzui_basename_prompt+ "_"+basename_source)
    print(tgt_file_path)
    shutil.copy(source_file, tgt_file_path)
