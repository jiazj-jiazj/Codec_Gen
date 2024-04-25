import os  
import h5py  
import numpy as np  
  
# 设置要搜索的文件夹和合并后的文件名  
for i in range(2,10,1):
    search_dir = "/home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/tokenized_tfnet_semantic_token"  # 假设这是你的文件目录  
    merged_file_name = f'merged_filter_Indic_sem_part_{i}.h5'  

    # 创建一个新的h5文件用于存放合并后的数据  
    with h5py.File(os.path.join(search_dir, merged_file_name), 'w') as merged_h5f:  
        # 遍历目录中的所有文件  
        for file_name in os.listdir(search_dir):  
            # 检查文件名是否符合条件  
            if file_name.startswith(f'filter_Indic_TTScuts_all_{i}') and file_name.endswith('.h5'):  
                print(f"file_name:{file_name}")
                # 打开符合条件的h5文件  
                with h5py.File(os.path.join(search_dir, file_name), 'r') as h5f:  
                    # 遍历文件中的所有数据集  
                    for key, value in h5f.items():
                        # 如果数据集已经在合并文件中，则追加数据，否则创建新数据集  
                        if key in merged_h5f:  
                            print("key in merged_h5")
                            # 读取现有数据并追加新数据  
                            existing_data = merged_h5f[key][()] 
                            new_data = np.concatenate((existing_data, value[()]), axis=0)  
                            # 重写数据集  
                            del merged_h5f[key]
                            merged_h5f.create_dataset(key, data=new_data, dtype=np.int32)  
                        else:  
                            # 创建新数据集  
                            merged_h5f.create_dataset(key, data=value[()], dtype=np.int32)  
