import os  
  
folder_path = "/scratch/data/l1_l2_arctic/combine_L1_L2/train_india/total/TNI/transcript"  
  
# 定义一个函数来删除逗号和句号  
def remove_comma_period(text):  
    return text.replace(",", "").replace(".", "")  
  
# 遍历文件夹中的所有文件  
for file in os.listdir(folder_path):  
    file_path = os.path.join(folder_path, file)  
      
    # 检查文件是否是文本文件（例如：以 .txt 结尾）  
    if file.endswith(".txt"):  
        # 读取文件内容  
        with open(file_path, "r") as f:  
            content = f.read()  
          
        # 转换为大写并删除逗号和句号  
        content_upper = content.upper()  
        content_no_comma_period = remove_comma_period(content_upper)  
          
        # 将更新后的内容写回文件  
        with open(file_path, "w") as f:  
            f.write(content_no_comma_period)  
          
        print(f"已更新文件：{file_path}")  
