import random  
from collections import defaultdict  
  
def assign_targets(dataset, targets):  
    while targets:  
        # 为每个speaker分配一个随机的target  
        speaker_targets = defaultdict(lambda: random.choice(targets))  
          
        # 迭代数据集  
        for sample in dataset:  
            speaker = sample['speaker']  
            assigned_target = speaker_targets[speaker]  
            print(f"Speaker: {speaker}, Target: {assigned_target}")  
          
        # 从targets列表中删除已分配的target  
        for used_target in set(speaker_targets.values()):  
            targets.remove(used_target)  
  
# 假设数据集和targets列表如下  
dataset = [  
    {'speaker': 'A'},  
    {'speaker': 'B'},  
    {'speaker': 'C'},  
    {'speaker': 'A'},  
    {'speaker': 'C'},  
]  
  
targets = [1, 2, 3, 4, 5, 6]  
  
# 为数据集中的speaker分配targets  
assign_targets(dataset, targets)  
