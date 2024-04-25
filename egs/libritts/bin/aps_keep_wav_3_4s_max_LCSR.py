import os  
import librosa  
import sys
import itertools  

now_pwd = os.getcwd()
sys.path.append(now_pwd)
import random
# from egs.libritts.bin.test_semantic_max_path import compute_hubert, longest_common_subsequence 
path = "/scratch/data/l1_l2_arctic/combine_L1_L2/accented_speech_analysis/native_indian"  
  
speakers = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]  
  
# 初始化一个字典来存储每个speaker的音频文件名  
speaker_files = {}  
  
# 遍历每个speaker的文件夹  
for speaker in speakers:  
    speaker_path = os.path.join(path, speaker, 'wav')  
    # 找到所有3-5秒的音频文件  
    files = [f for f in os.listdir(speaker_path) if os.path.isfile(os.path.join(speaker_path, f)) and  3 <= librosa.get_duration(filename=os.path.join(speaker_path, f)) <= 5]  
    # 去除后缀，只保留文件名  
    files = [os.path.splitext(f)[0] for f in files]  
    speaker_files[speaker] = set(files)  

# print(speaker_files)
# 找到所有speaker都有的音频文件  
common_files = set.intersection(*speaker_files.values())  

# speaker列表  
speakers1 = ['cmu_us_bdl_arctic', 'cmu_us_clb_arctic', 'cmu_us_rms_arctic', 'cmu_us_slt_arctic']  
speakers2 = ['cmu_us_bdl_arctic', 'cmu_us_clb_arctic', 'cmu_us_rms_arctic', 'cmu_us_slt_arctic'] 


# 使用 itertools.product 生成所有可能的组合  
pairs = list(itertools.product(speakers1, speakers2))  
  
# 使用列表推导式过滤出两个 speaker 不同的配对  
pairs = [pair for pair in pairs if pair[0] != pair[1]]  
  
# 过滤掉重复的配对  
pairs = list(set(tuple(sorted(pair)) for pair in pairs))  
  
# 共同的音频文件  
common_files = list(common_files)  # 假设common_files是你已经获得的四个列表共同的音频文件名称  
random.shuffle(common_files)  # 打乱顺序  
  
# 结果列表  
result = []  
  
# 路径  
path = "/scratch/data/l1_l2_arctic/combine_L1_L2/accented_speech_analysis/native_indian"  


from valle.data import (
    ApplyKmeans,
    HubertFeatureReader
)

ckpt_path = "/home/v-zhijunjia/data/valle-tensorboard-models/other_models/hubert/hubert_base_ls960.pt"
layer = 9
km_path = "/home/v-zhijunjia/data/valle-tensorboard-models/other_models/hubert/hubert_base_ls960_L9_km500.bin"
reader = HubertFeatureReader(ckpt_path, layer)
apply_kmeans = ApplyKmeans(km_path)    

def compute_hubert(wav_path):
    feat = reader.get_feats(wav_path)
    lab = apply_kmeans(feat).tolist()
    return lab

def longest_common_subsequence(seq1, seq2):  
    m, n = len(seq1), len(seq2)  
    dp = [[0] * (n + 1) for _ in range(m + 1)]  
  
    for i in range(1, m + 1):  
        for j in range(1, n + 1):  
            if seq1[i - 1] == seq2[j - 1]:  
                dp[i][j] = dp[i - 1][j - 1] + 1  
            else:  
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  
  
    lcs = []  
    i, j = m, n  
    while i > 0 and j > 0:  
        if seq1[i - 1] == seq2[j - 1]:  
            lcs.append(seq1[i - 1])  
            i -= 1  
            j -= 1  
        elif dp[i - 1][j] > dp[i][j - 1]:  
            i -= 1  
        else:  
            j -= 1  
  
    return lcs[::-1]  

def depup(semantic_token):
    unique_tokens = []  
    for token in semantic_token:  
        if unique_tokens==[] or token != unique_tokens[-1]:  
            unique_tokens.append(token)
    return unique_tokens

# 遍历每个音频文件  

for filename in common_files:  
    # 从两个列表中各随机选择一个speaker  
    # speaker1 = random.choice(speakers1)  
    # speaker2 = random.choice(speakers2)  
      
    # 确保两个speaker不同
    for speaker_pair in pairs:
        speaker1, speaker2 = speaker_pair
        # 找到这两个speaker的wav_txt文件夹的文本文件  
        file1 = os.path.join(path, speaker1, 'wav_txt', filename + '.txt')  
        file2 = os.path.join(path, speaker2, 'wav_txt', filename + '.txt')  

        # 如果两个文本文件内容相同  
        if os.path.isfile(file1) and os.path.isfile(file2):  
            with open(file1, 'r') as f1, open(file2, 'r') as f2:  
                content1 = f1.read().strip()  
                content2 = f2.read().strip()
                if content1 == content2: 
                    # print(file1)
                    # print(file2)
                    # print(content1)
                    # print(content2)
                    # print("pass")
                    # 音频文件名称和两个speaker名称都加入到一个列表中  
                    result.append((filename, speaker1, speaker2))  
    

lcs_sio_sum = 0 
lcs_sio_num = 0
lcs_sio_list = []
with open("output_v3.txt", "w") as f:
    for item in result:
        file_name, speaker1, speaker2 = item
        # 找到这两个speaker的wav文件  
        file1 = os.path.join(path, speaker1, 'wav', file_name + '.wav')  
        file2 = os.path.join(path, speaker2, 'wav', file_name + '.wav')

        tokens1 = compute_hubert(file1)  
        tokens2 = compute_hubert(file2)  
        tokens1 = depup(tokens1)  
        tokens2 = depup(tokens2)
        
        lcs = longest_common_subsequence(tokens1, tokens2)
        # print(file1)
        # print(file2)
        f.write(str(tokens1))
        f.write(str(tokens2))
        f.write('\n')
        lcs_sio = len(lcs)/min(len(tokens1), len(tokens2))
        lcs_sio_list.append(lcs_sio)
        lcs_sio_sum += lcs_sio
        lcs_sio_num+=1

avg_lcs = lcs_sio_sum / lcs_sio_num

print(f"common_files:{len(common_files)}")
print(f"result:{len(result)}")
print(f"lcs_sio_num:{lcs_sio_num}", lcs_sio_num)
print(f"avg_lcs:{avg_lcs}")


import numpy as np  
import random  
def bootstrap(data, num_samples, alpha):  
    """生成自助法置信区间。  
  
    参数：  
    data -- 数据列表  
    num_samples -- 自助法样本数量  
    alpha -- 置信度水平  
  
    返回：  
    置信区间（元组形式：(下限, 上限)）  
    """  
    # 生成自助法样本  
    samples = [random.choices(data, k=len(data)) for _ in range(num_samples)]  
    # 计算样本均值  
    means = [np.mean(s) for s in samples]  
    # 计算置信区间  
    lower = np.percentile(means, 100*(alpha/2))  
    upper = np.percentile(means, 100*(1-alpha/2))  
    return (lower, upper)  

import numpy as np  
import scipy.stats as stats  
  
def compute_mean_and_confidence_interval(data):  
    # 计算平均值和标准差  
    mean = np.mean(data)  
    std_dev = np.std(data, ddof=1)  
  
    # 计算自由度和标准误差  
    degrees_of_freedom = len(data) - 1  
    std_error = std_dev / np.sqrt(len(data))  
  
    # 查找 t 值  
    t_value = stats.t.ppf(1-0.05/2, degrees_of_freedom)  # 两侧的 t 值  
  
    # 计算置信区间  
    confidence_interval = t_value * std_error  
    lower_bound = mean - confidence_interval  
    upper_bound = mean + confidence_interval  
  
    return mean, lower_bound, upper_bound  

  
# 给定的最大公共子序列列表  
lcs_list = lcs_sio_list  # 用实际的数据替换这里  

mean, lower_bound, upper_bound = compute_mean_and_confidence_interval(lcs_list)

print(f"mean:{mean}")
print(f"lower_bound:{lower_bound}")
print(f"upper_bound:{upper_bound}")
print(f"mean-lower_bound:{mean-lower_bound}")

  
# # 计算0.05的置信度结果（即95%的置信区间）  
# confidence_interval = bootstrap(lcs_list, 1000, 0.05)  
# print("95% confidence interval:", confidence_interval)  


