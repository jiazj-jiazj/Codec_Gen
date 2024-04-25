import sys
import os
import argparse
current_working_directory = os.getcwd()  
# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory) 
import h5py  

from lhotse import CutSet

# for i in range(0, 10, 1):
#     file_path = f'/home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token/merged_filter_Indic_sem_part_{i}.h5'
#     with h5py.File(file_path, 'r') as h5f: 
#         print(len(h5f.keys()))

# for i in range(0, 10, 1):
#     cuts = CutSet.from_file(f"/home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token/filter_Indic_TTScuts_all_{i}.jsonl.gz")
#     cuts = cuts.to_eager()
#     print(len(cuts))

cuts = CutSet.from_file("/home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token/add_sem_filter_Indic_TTScuts_all.jsonl.gz")
cuts = cuts.to_eager()
print(len(cuts))


longer_than_5s = cuts.filter(lambda c: "native_semantic_tokens" in c.supervisions[0].custom.keys())
print(len(longer_than_5s))

longer_than_5s.to_file("/home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token/filter_add_sem_filter_Indic_TTScuts_all.jsonl.gz")

quit()



cuts = CutSet.from_file("/home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token/filter_Indic_TTScuts_all.jsonl.gz")
cuts = cuts.to_eager()
print(len(cuts))
# for cut in cuts:
#     print(cut)
# quit()
file_path = '/home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token/merged_filter_Indic_sem.h5'  
#   data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token/filter_Indic_TTScuts_all_0.jsonl.gz
# 打开并读取HDF5文件 
i=0
with h5py.File(file_path, 'r') as h5f: 
    # 遍历文件中的所有键  
    for cut in cuts:
        i+=1
        if i%100==0:
            print(i)
        cut_id = cut.id
        key = cut_id+"_0"

        try:
            data = h5f[key][:] 
            data = data.tolist()
            cut.supervisions[0].custom["native_semantic_tokens"] = {}
            cut.supervisions[0].custom["native_semantic_tokens"]['speaker0'] = data
        except:
            print(f"{key} not exist")



cuts.to_file("/home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token/add_sem_filter_Indic_TTScuts_all.jsonl.gz")
for cut in cuts:
    print(cut)
    quit()

longer_than_5s = cuts.filter(lambda c: c.supervisions[0].custom["native_semantic_tokens"] > 5)


        # native_semantic = cut.supervisions[0].custom["native_semantic_tokens"]['speaker0']
        # indian_semantic = cut.supervisions[0].custom["tokens"]["semantic_tokens"]
        
        # print(native_semantic)
        # print(indian_semantic)
        # print(len(native_semantic))
        # print(len(indian_semantic))
        # quit()
        # quit()

# for key in h5f.keys():  
#     # 获取当前键对应的数据  
#     data = h5f[key][:] 
#     data = list(data) 
#     # 打印键和对应的数据  
#     print(f'{key}: {data}')  


    # quit()


# cuts_real = cuts.filter(lambda c: (c.custom['native_semantic_tokens']==None and c.custom['indian_semantic_tokens']==None))
# cuts_real.describe()
# for cut in cuts:
#     if cut.supervisions[0].custom['accent']=='Scottish':
#         print(cut.recording.id)
