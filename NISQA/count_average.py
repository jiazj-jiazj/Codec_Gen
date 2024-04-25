# 将此路径替换为文件的实际路径  
file_path = "/home/v-zhijunjia/valle-4-23/NISQA/NISQA_results.csv"  
  
mos_pred_values = []  
  
with open(file_path, "r") as file:  
    # 跳过文件的第一行（标题行）  
    next(file)  
  
    # 逐行读取文件  
    for line in file:  
        # 用逗号分隔每行数据  
        columns = line.strip().split(",")  
  
        # 获取 mos_pred 值，并将其添加到列表中  
        mos_pred = float(columns[1])  
        print(mos_pred)
        mos_pred_values.append(mos_pred)  
  
# 计算 mos_pred 平均值  
mos_pred_average = sum(mos_pred_values) / len(mos_pred_values)  
print(len(mos_pred_values))
print(f"Average MOS_Pred value: {mos_pred_average:.4f}")  
