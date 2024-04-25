import matplotlib.pyplot as plt  
import numpy as np  
import os  
import matplotlib.patches as mpatches  
from matplotlib.ticker import MaxNLocator  
 
# 定义数据  
prompts = ['Accent source', 'Referenced ground truth', 'Generative model(EnCodec)', 'Generative model(TF-Codec) ', 'Proposed']  
# prompts = ['Ac', 'Ref', 'Liu ', 'Gene', 'Gener', 'Pro']  

languages = ['general American-English accent', 'Indian-English accent', 'African-English accent', 'Singapore-English accent', 'Australian-English accent', 'Hong Kong-English accent']  
colors = plt.cm.tab20.colors[:len(languages)]  



data_general = [  
    {'general American-English accent':23, 'Indian-English accent': 75, 'Australian-English accent': 1, 'Singapore-English accent': 1},
    {'general American-English accent':100},
    {'general American-English accent':23, 'Indian-English accent': 75, 'Australian-English accent': 1, 'Singapore-English accent': 1},
    {'general American-English accent':23, 'Indian-English accent': 75, 'Australian-English accent': 1, 'Singapore-English accent': 1},
    {'general American-English accent':87, 'Indian-English accent': 11, 'Hong Kong-English accent': 1, 'African-English accent': 1}
]  
  
fig, ax = plt.subplots()  
ax.yaxis.set_major_locator(MaxNLocator(integer=True))  

# 设置柱状图的位置和宽度  
bar_width = 0.55
ind = np.arange(len(prompts))  
  
for i, prompt in enumerate(prompts):  
    bottom_general = 0  
    general_position = ind[i]  
    for j, lang in enumerate(languages):  
        general_val = data_general[i].get(lang, 0)  
        ax.bar(general_position, general_val, bar_width, bottom=bottom_general, color=colors[j], linewidth=0)  
        if general_val > 0: 
            if general_val >= 2:  
                ax.text(general_position, bottom_general + general_val / 2, f"{general_val}%", ha='center', va='center', fontsize=8)  
        bottom_general += general_val  
  
legend_handles = []
legend_handles.append(mpatches.Patch(color='none', label='Accent type'))  

legend_handles.append(mpatches.Patch(color=colors[0], label=languages[0]))
legend_handles.append(mpatches.Patch(color=colors[1], label=languages[1]))
legend_handles.append(mpatches.Patch(color=colors[2], label=languages[2]))
legend_handles.append(mpatches.Patch(color='none', label=''))  # 透明占位符  

legend_handles.append(mpatches.Patch(color=colors[3], label=languages[3]))
legend_handles.append(mpatches.Patch(color=colors[4], label=languages[4]))
legend_handles.append(mpatches.Patch(color=colors[5], label=languages[5]))
legend = ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2, borderaxespad=0., handletextpad=0.5) # 使用图例文本列表中的索引来访问特定的图例文本  

legend.get_title().set_weight('bold')  # 设置图例标题为粗体  
texts = legend.get_texts()  
texts[0].set_weight('bold')  
# 设置轴标签和标题  
ax.set_xlabel('Framework')  
ax.set_ylabel('Percentage of predicted accent types(%)')  
# 设置刻度位置和标签  
# 设置刻度位置和标签  
# 设置刻度位置和标签  
ax.set_xticks(ind)  
ax.set_xticklabels(prompts, fontsize=6, rotation=20,ha='right')  # ha='right'可以使标签斜着对齐  
  
# 获取当前的xtick标签并向右移动  
xticks = ax.xaxis.get_major_ticks()  
for tick in xticks:  
    tick.set_pad(tick.get_pad() + 0)  # 在原有的基础上增加5点的间距   调整图形布局  
plt.tight_layout()  
plt.subplots_adjust(top=0.73)

# 确定保存图片的路径  
save_path = "/home/v-zhijunjia/CodecGen/egs/l1_l2_arctic/pics"  
os.makedirs(save_path, exist_ok=True)  
file_name = "table4.png"  
full_path = os.path.join(save_path, file_name)  
  
# 保存图形  
plt.savefig(full_path)  
  
# 显示图形  
plt.show()  
