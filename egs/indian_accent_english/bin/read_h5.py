import h5py  
  
# 定义文件路径  
file_path = '/home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token/merged_file_0_1_2_3_4.h5'  
  
# 打开并读取HDF5文件  
with h5py.File(file_path, 'r') as h5f:  
    # 遍历文件中的所有键  
    print(len(h5f.keys()))
    quit()
    for key in h5f.keys():  
        # 获取当前键对应的数据  
        data = h5f[key][:] 
        data = list(data) 
        # 打印键和对应的数据  
        print(f'{key}: {data}')  
