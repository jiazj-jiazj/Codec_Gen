import torch
import random
def mask_in_group(mask, loc, audio_feature_len, audio_features, replace_prob=0.7, replace_all_prob=0.5):
    
    bef_audio_feature = audio_features[loc, :, :]

    print(f"bef_audio_feature:{bef_audio_feature}")
    print(bef_audio_feature.shape)

    batch_size, total_len, code_book = audio_features.shape  
    # 找到没有被mask的区域  
    unmasked_indices = torch.nonzero(mask == 0, as_tuple=False).squeeze(1)
    # print(f"unmasked_indices:{unmasked_indices}")
    # quit()
    # 计算要替换的token数量                                                                                                                         
    # print(f'unmasked_indices:{unmasked_indices}')
    num_to_replace = int(len(unmasked_indices) * replace_prob)  
    # print(f"num_to_replace:{num_to_replace}")
    # 随机选择一些token进行替换  
    indices_to_replace = unmasked_indices[torch.randperm(len(unmasked_indices))[:num_to_replace]]  
    # 使用[0,1024)之间的随机数替换选定的token  
    print(f'indices_to_replace:{indices_to_replace}')
    for idx in indices_to_replace:  
        num_to_replace_cb = torch.randint(1, 5, (1,)).item()  
        cb_indices_to_replace = torch.randperm(code_book)[:num_to_replace_cb]  
        # cb_indices_to_replace = min(cb_indices_to_replace, )
        # print(f"cb_indices_to_replace:{cb_indices_to_replace}")
        bef_audio_feature[idx.item(), cb_indices_to_replace] = torch.randint(0, 1024, (num_to_replace_cb,)).float()

    num_time_dim_to_replace = int(len(mask) * replace_all_prob)
    indices_to_replace_total = unmasked_indices[torch.randperm(len(unmasked_indices))[:num_time_dim_to_replace]]  
    # print(f'time_dim_to_replace: {time_dim_to_replace}') 

    indice_to_replace_cb = torch.randint(0, batch_size, (len(indices_to_replace_total),))
    indice_to_replace_cb_one_batch = torch.randint(0, len(mask), (len(indices_to_replace_total),))
    prob = random.random()
    if prob < 0.5:
        bef_audio_feature[indices_to_replace_total, :] = audio_features[indice_to_replace_cb, 0, :].squeeze(1)  # 假设aa_bb是预先定义好的 
    else:
        bef_audio_feature[indices_to_replace_total, :] = bef_audio_feature[indice_to_replace_cb_one_batch, :].squeeze(1)  # 假设aa_bb是预先定义好的 


    return bef_audio_feature


# 假设我们有一个形状为[10, 20, 30]的音频特征张量和一个相同形状的掩蔽张量  
audio_features = torch.randint(low=0, high=1024, size=(3, 10, 6)).float()
mask = torch.randint(0, 2, (8,))  

print(f'mask:{mask}')
# 我们想要对第一个batch的音频特征执行掩蔽操作  
loc = 0  
  
# 音频特征的长度为20  
audio_feature_len = 5  
  
# 执行掩蔽操作  
new_audio_features = mask_in_group(mask, loc, audio_feature_len, audio_features)  
  
# 输出的new_audio_features是一个形状为[10, 20, 30]的新的音频特征张量，其中第一个batch的一部分音频特征已被随机数替换  
print(f'new_audio_features:{new_audio_features}')  # 输出：torch.Size([10, 20, 30])  


