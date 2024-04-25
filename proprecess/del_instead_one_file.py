import os  
import shutil  
  
directory = "/home/v-zhijunjia/data/valle-tensorboard-models/pretrain_finetune/tfnet_sem/mode_5_mask_0_15"  
excluded_file = "pret-mode5-epoch-70.pt"  
  
for root, dirs, files in os.walk(directory, topdown=False):  
    for name in files:
        if name != excluded_file:  
            os.remove(os.path.join(root, name))  
    for name in dirs:  
        shutil.rmtree(os.path.join(root, name))  

