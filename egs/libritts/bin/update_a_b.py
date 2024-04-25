

from collections import Counter  

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

def mark_and_count_lcs_tokens(a, lcs_ab):  
    lcs_index = 0  
    lcs_token_counts = [0 for xx in range(len(lcs_ab))]

    ll = len(a)
    i=0
    while True:
        if i==ll:
            break
        if lcs_index < len(lcs_ab) and a[i] == lcs_ab[lcs_index]:  
            lcs_token_counts[lcs_index] += 1  
            i+=1
            while i<ll and a[i]==lcs_ab[lcs_index] :
                lcs_token_counts[lcs_index] += 1 
                i+=1
            lcs_index += 1

            i-=1   
        i+=1
    return lcs_token_counts 


def update_lcs_token(a, lcs_ab, lcs_token_counts_a, lcs_token_counts_b, update_nums, update_type=0):

    # update_type 0 del 1 add
    lcs_index = 0  
    lcs_token_counts = [0 for xx in range(len(lcs_ab))]
    updated_tokens = []
    ll = len(a)
    i=0
    while True:
        if i==ll:
            break
        if lcs_index < len(lcs_ab) and a[i] == lcs_ab[lcs_index]:  
            if lcs_token_counts_a[lcs_index] > lcs_token_counts_b[lcs_index] and update_type==0: 
                can_update_nums = lcs_token_counts_a[lcs_index]- lcs_token_counts_b[lcs_index]
                if can_update_nums <= update_nums:
                    nums = lcs_token_counts_b[lcs_index]
                    update_nums-=can_update_nums
                else:
                    nums = lcs_token_counts_a[lcs_index]
            elif update_type==0:
                nums = lcs_token_counts_a[lcs_index]

            if lcs_token_counts_a[lcs_index] < lcs_token_counts_b[lcs_index] and update_type==1:
                can_update_nums = lcs_token_counts_b[lcs_index]- lcs_token_counts_a[lcs_index]
                if can_update_nums <= update_nums:
                    nums = lcs_token_counts_b[lcs_index]
                    update_nums-=can_update_nums
                else:
                    nums = lcs_token_counts_a[lcs_index]
            elif update_type==1:
                nums = lcs_token_counts_a[lcs_index]

            updated_tokens+=[lcs_ab[lcs_index]]*nums
            i+=1
            while i<ll and a[i]==lcs_ab[lcs_index] :
                i+=1
            lcs_index += 1
            i-=1   
        else:
            updated_tokens+=[a[i]]
        i+=1
    return updated_tokens, update_nums


def update_non_lcs_token(a, lcs_ab, non_lcs_token_counts_a, update_nums, update_type=0):

    lcs_index = 0  
    lcs_token_counts = [0 for xx in range(len(lcs_ab))]
    if update_type==0:
        non_lcs_token_counts_a = del_count_elements(non_lcs_token_counts_a[:], update_nums)  # 使用切片创建一个副本，以免修改原始序列
    elif update_type==1:
        non_lcs_token_counts_a = add_count_elements(non_lcs_token_counts_a[:], update_nums)  # 使用切片创建一个副本，以免修改原始序列
   
    updated_tokens = []
    ll = len(a)
    i=0
    while True:
        if i==ll:
            break
        if lcs_index < len(lcs_ab) and a[i] == lcs_ab[lcs_index]:  
            nums = non_lcs_token_counts_a[lcs_index]

            updated_tokens+=[lcs_ab[lcs_index]]*nums
            i+=1
            while i<ll and a[i]==lcs_ab[lcs_index] :
                i+=1
            lcs_index += 1
            i-=1   
        else:
            updated_tokens+=[a[i]]
        i+=1
    return updated_tokens


# bug exists
def get_non_lcs_tokens(a, lcs_ab):  
    lcs_index = 0  
    non_lcs_tokens = []
    updated_tokens = []
    ll = len(a)
    i=0
    while True:
        if i==ll:
            break
        if lcs_index < len(lcs_ab) and a[i] == lcs_ab[lcs_index]:  
            i+=1
            while i<ll and a[i]==lcs_ab[lcs_index]:
                i+=1
            lcs_index += 1

            i-=1   
        else:
            if len(non_lcs_tokens) == 0:
                non_lcs_tokens+=[a[i]]
            elif a[i]!=non_lcs_tokens[-1] or a[i-1]!=non_lcs_tokens[-1]:
                non_lcs_tokens+=[a[i]]
        i+=1
    
    lcs_index = 0  

    depup_non_lcs_tokens = non_lcs_tokens

    # print(f"a:{a}")
    non_lcs_token_counts = mark_and_count_lcs_tokens(a, depup_non_lcs_tokens)
    # print(xx)
    # index = 0 
    # ll = len(non_lcs_tokens)
    # i=0
    # while True:
    #     if i==ll:
    #         break
    #     if index < len(depup_non_lcs_tokens) and non_lcs_tokens[i] == depup_non_lcs_tokens[index]:
    #         i+=1
    #         non_lcs_token_counts[index]+=1

    #         while i<ll and non_lcs_tokens[i]==depup_non_lcs_tokens[index] :
    #             non_lcs_token_counts[index]+=1
    #             i+=1
    #         index += 1
    #         i-=1   
    #     i+=1

    # print(depup_non_lcs_tokens)
    # print(non_lcs_token_counts)
    return depup_non_lcs_tokens, non_lcs_token_counts

from heapq import heapify, heappop, heappush  
def del_count_elements(sequence, total_to_subtract):
    # 创建一个索引堆，以便知道哪个元素被减去
    index_heap = [(-val, i) for i, val in enumerate(sequence)]
    heapify(index_heap)  # 建立最大堆

    # 从最大的数字开始逐一减去1
    for _ in range(total_to_subtract):
        if not index_heap:
            break  # 如果堆为空，则停止
        # 弹出最大的数字
        max_val, max_index = heappop(index_heap)
        if sequence[max_index] > 0:  # 如果该数字已经是0，则不再减去
            sequence[max_index] -= 1  # 减去1
        if sequence[max_index] > 0:  # 如果减去1后大于0，则放回堆中
            heappush(index_heap, (-sequence[max_index], max_index))

    return sequence

def add_count_elements(sequence, total_to_add):  
    # 创建一个索引堆，以便知道哪个元素被增加  
    index_heap = [(val, i) for i, val in enumerate(sequence)]  
    heapify(index_heap)  # 建立最小堆  
  
    # 从最小的数字开始逐一增加1  
    for _ in range(total_to_add):  
        if not index_heap:  
            break  # 如果堆为空，则停止  
        # 弹出最小的数字  
        min_val, min_index = heappop(index_heap)  
        sequence[min_index] += 1  # 增加1  
        # 增加1后，放回堆中  
        heappush(index_heap, (sequence[min_index], min_index))  
  
    return sequence

def get_a_larger_b(a, b):

    lcs_ab = longest_common_subsequence(a, b)  
    lcs_ab_depup = depup(lcs_ab)
    
    # 标记序列a中属于最大公共子序列的token，并统计数量  
    lcs_token_counts_a = mark_and_count_lcs_tokens(a, lcs_ab_depup)  

    lcs_token_count_b = mark_and_count_lcs_tokens(b, lcs_ab_depup)  

    updated_tokens, update_nums = update_lcs_token(a, lcs_ab_depup, lcs_token_counts_a, lcs_token_count_b, len(a)-len(b))

    non_lcs_tokens_a, non_lcs_token_counts_a = get_non_lcs_tokens(a, lcs_ab_depup)

    final_tokens = update_non_lcs_token(updated_tokens, non_lcs_tokens_a, non_lcs_token_counts_a, update_nums)
    return final_tokens

def get_a_smaller_b(a, b):

    lcs_ab = longest_common_subsequence(a, b)  
    lcs_ab_depup = depup(lcs_ab)
    
    # 标记序列a中属于最大公共子序列的token，并统计数量  
    lcs_token_counts_a = mark_and_count_lcs_tokens(a, lcs_ab_depup)  

    lcs_token_count_b = mark_and_count_lcs_tokens(b, lcs_ab_depup)  

    updated_tokens, update_nums = update_lcs_token(a, lcs_ab_depup, lcs_token_counts_a, lcs_token_count_b, len(b)-len(a), update_type=1)

    non_lcs_tokens_a, non_lcs_token_counts_a = get_non_lcs_tokens(a, lcs_ab_depup)

    # print(f"second_update_nums:{update_nums}")
    final_tokens = update_non_lcs_token(updated_tokens, non_lcs_tokens_a, non_lcs_token_counts_a, update_nums, update_type=1)
    return final_tokens

a = [17, 17, 17, 296, 17, 296, 363, 52, 408, 408, 51, 184, 184, 20, 184, 20, 184, 289, 289, 20, 320, 108, 377, 374, 374, 374, 153, 274, 274, 274, 43, 8, 109, 109, 109, 213, 213, 252, 143, 458, 96, 270, 270, 86, 142, 221, 336, 336, 336, 354, 354, 284, 311, 311, 311, 311, 311, 169, 169, 271, 150, 39, 86, 342, 224, 483, 483, 226, 209, 83, 55, 55, 446, 322, 94, 94, 199, 340, 340, 340, 340, 116, 33, 64, 394, 212, 384, 114, 222, 92, 92, 240, 313, 236, 35, 401, 401, 401, 401, 384, 180, 432, 290, 290, 290, 434, 434, 203, 203, 117, 117, 117, 225, 225, 465, 401, 75, 108, 119, 437, 437, 106, 481, 306, 206, 396, 396, 215, 215, 215, 35, 96, 26, 251, 251, 241, 241, 431, 443, 443, 443, 169, 173, 402, 402, 402, 6, 401, 75, 472, 472, 221, 458, 445, 445, 180, 365, 365, 365, 365, 282, 282, 203, 53, 381, 381, 243, 76, 401, 401, 20, 74, 419, 225, 225, 225, 80, 20, 80, 80, 401, 20, 75, 161, 161, 487, 487, 487, 213, 213, 252, 422, 143, 36, 36, 384, 371, 470, 432, 290, 290, 299, 299, 203, 303, 303, 471, 471, 185, 269, 323, 433, 160, 160, 18, 112, 439, 439, 439, 78, 421, 193, 193, 20, 17]
b = [17, 17, 296, 296, 296, 184, 140, 108, 108, 119, 119, 351, 374, 374, 374, 132, 132, 43, 43, 345, 109, 109, 213, 213, 213, 213, 143, 233, 458, 96, 270, 270, 342, 86, 221, 336, 82, 259, 74, 437, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 169, 169, 169, 150, 39, 86, 86, 86, 238, 6, 140, 227, 419, 483, 226, 226, 82, 140, 83, 55, 322, 322, 199, 199, 340, 340, 340, 340, 33, 466, 466, 479, 114, 92, 92, 92, 92, 92, 167, 457, 457, 465, 465, 119, 119, 103, 103, 103, 103, 103, 103, 103, 103, 85, 299, 299, 299, 203, 53, 195, 394, 394, 197, 197, 164, 164, 164, 164, 25, 106, 153, 153, 387, 372, 372, 245, 215, 259, 74, 472, 26, 26, 251, 241, 431, 443, 443, 443, 169, 349, 352, 402, 96, 401, 140, 472, 221, 144, 445, 445, 351, 351, 365, 365, 365, 365, 365, 282, 203, 215, 457, 233, 472, 472, 164, 164, 164, 164, 164, 487, 487, 487, 487, 337, 213, 324, 324, 422, 143, 36, 108, 119, 119, 119, 437, 103, 103, 103, 103, 103, 103, 85, 85, 85, 299, 299, 203, 381, 381, 471, 471, 185, 185, 269, 433, 433, 160, 18, 112, 112, 439, 439, 237, 82, 421, 193, 392, 392, 193, 193]

update_a_b = get_a_larger_b(a,b)

print(len(a))
print(a)
print(len(b))
print(b)
print(len(update_a_b))
print(update_a_b)



