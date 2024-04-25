import matplotlib.pyplot as plt  
  
# 数据  
accent_source = [0.11, 0.42, 0.28, 0.16, 0.4, 0.58, 0.4, 0.3, 0.08, 0.24, 0.38, 0.38, 0.24, 0.12, 0.34]  
Liu_et_al = [0.08, 0.32, 0.26, 0.1, 0.22, 0.48, 0.3, 0.12, 0.06, 0.24, 0.26, 0.28, 0.18, 0.12, 0.3]  
proposed = [0.09, 0.39, 0.33, 0.12, 0.4, 0.56, 0.46, 0.58, 0.08, 0.26, 0.4, 0.43, 0.22, 0.12, 0.3]  
  
word_timeline = list(range(1, len(accent_source)+1))  
  
plt.plot(word_timeline, accent_source, label='Accent Source', linestyle='-', marker='o')  
plt.plot(word_timeline, Liu_et_al, label='Liu et al.', linestyle='--', marker='x')  
plt.plot(word_timeline, proposed, label='Proposed', linestyle='-.', marker='s')  
  
plt.title('Duration by Model')  
plt.xlabel('Word Timeline (s)')  
plt.ylabel('Duration (s)')  
  
plt.legend()  
  
save_path = '/home/v-zhijunjia/CodecGen/egs/l1_l2_arctic/pics/duration_by_model.png'  
plt.savefig(save_path, dpi=300, bbox_inches='tight')  
  
# 显示图表  
plt.show()  
