import json


from valle.data import (
    ApplyKmeans,
    HubertFeatureReader
)



ckpt_path = "/dev_huaying/zhijun/models/hubert/hubert_base_ls960.pt"
layer = 9
km_path = "/dev_huaying/zhijun/models/hubert/hubert_base_ls960_L9_km500.bin"
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

with open("/mnt/zhijun/Accents/combine_L1_L2/acoustic_tokens_dic/native_l1_l2_arctic_semantic_dic_v2.json", "r") as json_file_1:  
    semantic_loaded_dict = json.load(json_file_1)

# dic_keys = semantic_loaded_dict.keys()

# for key in dic_keys:
#     dicts = semantic_loaded_dict[key]
#     lists = []
#     for keyy , value in dicts.items():
#         lists.append(value)
    
#     a, b, c ,d = lists
#     pairs = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]  
#     for seq1, seq2 in pairs:  
#         lcs = longest_common_subsequence(seq1, seq2)  
#         # print(f"Longest common subsequence of {seq1} and {seq2}: {lcs}") 
#         print(len(seq1), len(seq2))
#         print(len(lcs))

list = []
for key ,value in semantic_loaded_dict["arctic_a0001.wav"].items():
    list.append(value)
a, b, c , d= list
print(a)
print(len(a), len(b))
lcs = longest_common_subsequence(a, b)
print(lcs)
print(len(lcs))
lcs = longest_common_subsequence(b, c)
print(lcs)
print(len(lcs))
lcs = longest_common_subsequence(c, d)
print(lcs)
print(len(lcs))


