import pandas as pd  
  
# arctic_mos 
file_paths = ['/home/v-zhijunjia/zhijundata_small/data_local/iclr_rebuttal/评测/mos/l2-arctic语音质量测试_hexy.xlsx', 
'/home/v-zhijunjia/zhijundata_small/data_local/iclr_rebuttal/评测/mos/l2-arctic语音质量测试_peiyp.xlsx', 
'/home/v-zhijunjia/zhijundata_small/data_local/iclr_rebuttal/评测/mos/l2-arctic语音质量测试_shaozh.xlsx',
'/home/v-zhijunjia/zhijundata_small/data_local/iclr_rebuttal/评测/mos/l2-arctic语音质量测试_xiaofl.xlsx',
'/home/v-zhijunjia/zhijundata_small/data_local/iclr_rebuttal/评测/mos/l2-arctic语音质量测试_zuocx.xlsx']  # 用你的实际文件路径替换  

# # vctk_mos
# file_paths = ['/home/v-zhijunjia/zhijundata_small/data_local/iclr_rebuttal/评测/mos/vctk语音质量测试_hexy.xlsx', 
# '/home/v-zhijunjia/zhijundata_small/data_local/iclr_rebuttal/评测/mos/vctk语音质量测试_peiyp.xlsx', 
# '/home/v-zhijunjia/zhijundata_small/data_local/iclr_rebuttal/评测/mos/vctk语音质量测试_shaozh.xlsx',
# '/home/v-zhijunjia/zhijundata_small/data_local/iclr_rebuttal/评测/mos/vctk语音质量测试_zuocx.xlsx']  # 用你的实际文件路径替换

import pandas as pd  
  
# 多个文件路径的列表  
  
# 创建一个空的DataFrame来存储最终结果  
final_df = pd.DataFrame()  
  
for file_path in file_paths:  
    # 读取Excel文件  
    df = pd.read_excel(file_path)  
  
    # 计算每个model的平均mos得分  
    average_mos_scores = df.groupby('model')[['第一遍', '第二遍']].mean().mean(axis=1)  
  
    # 获取speaker的名称  
    speaker_name = file_path.split('_')[-1].replace('.xlsx', '')  
  
    # 将结果转化为DataFrame，以model为列名，以speaker为行索引  
    result = pd.DataFrame(average_mos_scores.values, columns=[speaker_name], index=average_mos_scores.index).transpose()  
  
    # 将结果添加到最终结果中  
    final_df = pd.concat([final_df, result])  
  
# 计算每个speaker的平均得分  
final_df['average'] = final_df.mean(axis=1)  
  
# 计算每个model的平均得分  
model_average = pd.DataFrame(final_df.mean()).transpose()  
model_average.index = ['model_average']  
final_df = pd.concat([final_df, model_average])  
  
# 输出最终结果  
print(final_df)  
  
# 将最终结果写入到Excel文件中  
final_df.to_excel('/home/v-zhijunjia/zhijundata_small/data_local/iclr_rebuttal/评测/mos/arctic_final.xlsx')