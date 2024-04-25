import sys  
import os  
  
# 获取当前工作目录  
current_working_directory = os.getcwd()  
  
# 将当前工作目录添加到 sys.path 的首位  
sys.path.insert(0, current_working_directory)  

from icefall.utils import AttributeDict, MetricsTracker, setup_logger, str2bool
import icefall
import sys

print(sys.path)
print(icefall.__file__)  


print(MetricsTracker)