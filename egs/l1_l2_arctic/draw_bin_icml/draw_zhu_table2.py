import matplotlib.pyplot as plt  
import numpy as np  
import os  
import matplotlib.patches as mpatches  # 导入patches模块  

# 定义数据  
prompts = ['3', '5', '7']  
languages = ['general American-English accent', 'Indian-English accent', 'African-English accent', 'Hong Kong-English accent', 'Singapore-English accent', 'Australian-English accent']  
colors = plt.cm.tab20.colors[:len(languages)]  # 创建颜色映射  
  
data_general = [  
    {'general American-English accent': 98, 'Indian-English accent': 1, 'Australian-English accent': 1},  
    {'general American-English accent': 96, 'African-English accent': 1,'Australian-English accent': 3},  
    {'general American-English accent': 97, 'Indian-English accent': 2, 'Australian-English accent': 1}  
]  
data_indian = [  
    {'general American-English accent': 84, 'Indian-English accent': 4, 'African-English accent': 4, 'Hong Kong-English accent': 3, 'Singapore-English accent': 3, 'Australian-English accent': 2},  
    {'general American-English accent': 84, 'Indian-English accent': 6, 'Australian-English accent': 2, 'Singapore-English accent': 1, 'African-English accent': 6, 'Hong Kong-English accent': 1},
    {'general American-English accent': 73, 'African-English accent': 9, 'Indian-English accent': 7, 'Singapore-English accent': 4, 'Australian-English accent': 6, 'Hong Kong-English accent': 1}  
]  
  
fig, ax = plt.subplots()  
  
# 设置柱状图的位置和宽度  
bar_width = 0.35  # 减小柱子的宽度以留出空隙  
ind = np.arange(len(prompts)) * 1.1  # 增加不同prompt之间的距离  
ax.set_ylim(0, 100)  

max_height = max([sum(data.values()) for data in data_general] + [sum(data.values()) for data in data_indian])  
# ax.set_ylim(0, max_height + 10)  # 这里+10是为了在顶部留出空间显示标注 
# 定义纹理图案  
hatch_pattern = '//////'  

for i, prompt in enumerate(prompts):  
    bottom_general = 0  
    bottom_indian = 0  
    # 调整相同prompt内部的柱状图之间的间隔  
    general_position = ind[i] - bar_width / 2 - 0.015  # 向左微调位置  
    indian_position = ind[i] + bar_width / 2 + 0.015  # 向右微调位置  
    for j, lang in enumerate(languages):  
        general_val = data_general[i].get(lang, 0)  
        indian_val = data_indian[i].get(lang, 0)  
        # 绘制general American English的柱状图  
        ax.bar(general_position, general_val, bar_width, bottom=bottom_general, color=colors[j], linewidth=0)  
        # 在柱状图上添加文本  
        if general_val > 0:  
            if general_val >= 3:
                ax.text(general_position, bottom_general + general_val/2, f"{general_val}%", ha='center', va='center', fontsize=8)  
        bottom_general += general_val  
          
        # 绘制Indian的柱状图，纹理使用不同颜色，其他部分透明  
        ax.bar(indian_position, indian_val, bar_width, bottom=bottom_indian, hatch=hatch_pattern, edgecolor=colors[j], facecolor='none', linewidth=0)  
        # 在柱状图上添加文本  
        if indian_val > 0: 
            if indian_val >= 3: 
                ax.text(indian_position, bottom_indian + indian_val/2, f"{indian_val}%", ha='center', va='center', fontsize=8)  
        bottom_indian += indian_val  
    # ax.text(general_position, max_height, "Native", ha='center', va='bottom', fontsize=7, rotation=0)  
    # ax.text(indian_position, max_height, "Non-native", ha='center', va='bottom', fontsize=7, rotation=0)  
  
legend_handles = []  
# 首先添加"Language"小标题占位符  

# 添加"Prompt type"小标题占位符  
legend_handles.append(mpatches.Patch(color='none', label='Prompt type'))  
legend_handles.append(mpatches.Patch(color='grey', label='general American-English accent'))  
legend_handles.append(mpatches.Patch(color='none', label='Accent type'))  

legend_handles.append(mpatches.Patch(color=colors[0], label=languages[0])  ) 
legend_handles.append(mpatches.Patch(color=colors[1], label=languages[1])  ) 
legend_handles.append(mpatches.Patch(color=colors[2], label=languages[2])  ) 
legend_handles.append(mpatches.Patch(color='none', label=''))  # 透明占位符  
legend_handles.append(mpatches.Patch(facecolor='none', edgecolor='grey', hatch=hatch_pattern, label='Indian-English accent'))  
legend_handles.append(mpatches.Patch(color='none', label=''))  # 透明占位符  
legend_handles.append(mpatches.Patch(color=colors[3], label=languages[3])  ) 
legend_handles.append(mpatches.Patch(color=colors[4], label=languages[4])  ) 
legend_handles.append(mpatches.Patch(color=colors[5], label=languages[5])  ) 

# 添加提示类型的图例项  

# 添加所有语言的图例项  
 
# 创建图例  
legend = ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.47), ncol=2, borderaxespad=0., handletextpad=0.5) # 使用图例文本列表中的索引来访问特定的图例文本  
# 并将第一个小标题设置为粗体  
texts = legend.get_texts()  
texts[0].set_weight('bold')  
texts[2].set_weight('bold')  # 第二个小标题在所有语言项之后

# 设置轴标签和标题  
ax.set_xlabel('Prompt length(s)')  
ax.set_ylabel('Percentage of predicted accent types(%)')  
ax.set_xticks(ind)  
ax.set_xticklabels(prompts)  
plt.subplots_adjust(top=0.58)
# 调整图形布局  
plt.tight_layout(rect=[0, 0, 1, 0.97])  

# 确定保存图片的路径  
save_path = "/home/v-zhijunjia/CodecGen/egs/l1_l2_arctic/pics"  
os.makedirs(save_path, exist_ok=True)  # 如果路径不存在，创建它  
file_name = "table2_v2.png"  
full_path = os.path.join(save_path, file_name)  

# 保存图形  
plt.savefig(full_path)  

# 显示图形  
plt.show()  
