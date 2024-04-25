import os  
import pandas as pd  
from collections import Counter  
  
# 文件夹路径  
folder_path = '/home/v-zhijunjia/data/accent_iclr/iclr_final/ac_vctk_v2_1_dir_important_bef'  # 使用你的实际文件路径  
  
# 获取文件夹中的所有文件名  
file_names = os.listdir(folder_path)  
  
# 创建一个字典来存储每个index对应的speaker  
index_speaker_dict = {file_name.split('_')[0]: file_name.split('_')[2].split('p248')[0] for file_name in file_names if 'wav' in file_name}  
  
# 文件列表  
file_list = ['/home/v-zhijunjia/zhijundata_small/data_local/iclr_rebuttal/评测/accent/vctk_liuxy.xlsx',
'/home/v-zhijunjia/zhijundata_small/data_local/iclr_rebuttal/评测/accent/vctk_liuzy.xlsx',
'/home/v-zhijunjia/zhijundata_small/data_local/iclr_rebuttal/评测/accent/vctk_luoyx.xlsx',
'/home/v-zhijunjia/zhijundata_small/data_local/iclr_rebuttal/评测/accent/vctk_zhud.xlsx']  # 使用你的实际文件路径  
  
# 创建一个字典来存储每个speaker被选择的次数  
speaker_counts = Counter()  
  
# 遍历每个文件  
for file_path in file_list:
    print(file_path)
    # 读取xlsx文件  
    df = pd.read_excel(file_path)  
  
    # 选择被选择的行  
    selected_rows = df[df['choosed'].isin([1, 2])]  
  
    # 获取被选择的index  
    selected_indexes = selected_rows['Index']  
  
    # 根据index查找对应的speaker  
    selected_speakers = [index_speaker_dict[str(index)] for index in selected_indexes]  
  
    # 更新每个speaker被选择的次数  
    speaker_counts.update(selected_speakers)  
  
# 打印每个speaker被选择的总次数  
for speaker, count in speaker_counts.items():  
    print(f'Speaker: {speaker}, Total Count: {count}')  
