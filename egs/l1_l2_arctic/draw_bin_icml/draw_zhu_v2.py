import matplotlib.pyplot as plt  
import numpy as np  
import os  
  
# 定义数据  
prompts = ['3s', '5s', '7s']  
languages = ['us', 'indian', 'england', 'newzealand', 'african', 'hongkong', 'canada', 'singapore', 'australia']  
colors = plt.cm.tab20.colors[:len(languages)]  # 创建颜色映射  
  
data_general = [  
    {'us': 72, 'indian': 4, 'england': 4, 'newzealand': 2, 'african': 4, 'hongkong': 3, 'canada': 6, 'singapore': 3, 'australia': 2},  
    {'us': 68, 'indian': 5, 'england': 5, 'newzealand': 3, 'african': 5, 'hongkong': 4, 'canada': 7, 'singapore': 2, 'australia': 1},  
    {'us': 75, 'indian': 3, 'england': 3, 'newzealand': 1, 'african': 3, 'hongkong': 2, 'canada': 5, 'singapore': 4, 'australia': 4}  
]  
data_indian = [  
    {'us': 65, 'indian': 10, 'england': 5, 'newzealand': 3, 'african': 5, 'hongkong': 4, 'canada': 4, 'singapore': 2, 'australia': 2},  
    {'us': 60, 'indian': 15, 'england': 4, 'newzealand': 4, 'african': 4, 'hongkong': 5, 'canada': 5, 'singapore': 2, 'australia': 1},  
    {'us': 70, 'indian': 8, 'england': 3, 'newzealand': 2, 'african': 3, 'hongkong': 3, 'canada': 6, 'singapore': 3, 'australia': 2}  
]  
  
fig, ax = plt.subplots()  
  
# 设置柱状图的位置和宽度  
bar_width = 0.4  # 减小柱子的宽度以留出空隙  
ind = np.arange(len(prompts)) * 1.2  # 增加不同prompt之间的距离  

# max_height = max([sum(data.values()) for data in data_general] + [sum(data.values()) for data in data_indian])  
# ax.set_ylim(0, max_height + 10)  # 这里+10是为了在顶部留出空间显示标注 
for i, prompt in enumerate(prompts):  
    bottom_general = 0  
    bottom_indian = 0  
    # 调整相同prompt内部的柱状图之间的间隔  
    general_position = ind[i] - bar_width / 2 - 0.01  # 向左微调位置  
    indian_position = ind[i] + bar_width / 2 + 0.01  # 向右微调位置  
      
    for j, lang in enumerate(languages):  
        general_val = data_general[i][lang]  
        indian_val = data_indian[i][lang]  
        # 绘制General English的柱状图  
        ax.bar(general_position, general_val, bar_width, bottom=bottom_general, color=colors[j])  
        bottom_general += general_val  
  
        # 绘制Indian的柱状图  
        ax.bar(indian_position, indian_val, bar_width, bottom=bottom_indian, color=colors[j])  
        bottom_indian += indian_val  
  
    # 在每组柱状图上方添加标注  
    ax.text(general_position, max_height, "General English", ha='center', va='bottom', fontsize=8, rotation=90)  
    ax.text(indian_position, max_height, "Indian", ha='center', va='bottom', fontsize=8, rotation=90)
  
# 添加每种语言的颜色标记  
for i, lang in enumerate(languages):  
    ax.bar(0, 0, color=colors[i], label=lang)  


# 设置图例  
ax.legend(title="Language", bbox_to_anchor=(1.05, 1), loc='upper left')  

# 设置轴标签和标题  
ax.set_xlabel('Prompt Duration')  
ax.set_ylabel('Percentage')  
ax.set_title('Percentage by prompt duration and speaker accent')  
ax.set_xticks(ind)  
ax.set_xticklabels(prompts)  

# 调整图形布局  
plt.tight_layout()  

# 确定保存图片的路径  
save_path = "/home/v-zhijunjia/CodecGen/egs/l1_l2_arctic/pics"  
os.makedirs(save_path, exist_ok=True)  # 如果路径不存在，创建它  
file_name = "stacked_bar_chart_with_labels.png"  
full_path = os.path.join(save_path, file_name)  

# 保存图形  
plt.savefig(full_path)  

# 显示图形  
plt.show()  
