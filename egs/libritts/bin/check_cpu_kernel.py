import os  
import multiprocessing  
  
def get_cpu_cores():  
    # 获取CPU核心数  
    cpu_cores = os.cpu_count() or multiprocessing.cpu_count()  
    return cpu_cores  
  
# 调用函数  
num_cores = get_cpu_cores()  
  
# 打印结果  
print(f"CPU核心数: {num_cores}")  
