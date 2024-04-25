import librosa  
import librosa.display  
import matplotlib.pyplot as plt  
import numpy as np
# 1. 加载音频文件  
audio_file1 = '/home/v-zhijunjia/data/accent_icml/cases_analysis/source_p248_004_trimmed.wav'  
audio_file2 = '/home/v-zhijunjia/data/accent_iclr/ac_baseline_20cases/p248_004.wav'  
audio_file3 = '/home/v-zhijunjia/data/accent_icml/cases_analysis/proposed_p248_004_trimmed.wav'  

y1, sr1 = librosa.load(audio_file1)  
y2, sr2 = librosa.load(audio_file2)  
y3, sr3 = librosa.load(audio_file3)  

max_duration = max(librosa.get_duration(y=y1, sr=sr1), librosa.get_duration(y=y2, sr=sr2), librosa.get_duration(y=y3, sr=sr3))  

# 2. 创建两个子图  
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 6))  
  
# 3. 绘制第一个波形图  
librosa.display.waveshow(y1, sr1, ax=ax1)  
  
# 4. 移除第一张图的"Time"属性  
ax1.set_xlabel("")  
ax1.set_ylabel("Accent source", fontsize=13) 
ax1.set_xlim(0, max_duration)  # 设置x轴的范围  

# 5. 在第一个波形图框的上方添加transcript文本  
transcript1 = [{'start_time': 0.12, 'end_time': 0.28, 'text': 'We'}, 
                {'start_time': 0.38, 'end_time': 0.62, 'text': 'also'}, 
                {'start_time': 0.72, 'end_time': 0.9, 'text': 'need'}, 
                {'start_time': 0.95, 'end_time': 1.10, 'text': 'a'}, 
                {'start_time': 1.12, 'end_time': 1.46, 'text': 'small'}, 
                {'start_time': 1.58, 'end_time': 2.04, 'text': 'plastic'}, 
                {'start_time': 2.12, 'end_time': 2.44, 'text': 'snake'}, 
                {'start_time': 2.66, 'end_time': 2.74, 'text': 'and'}, 
                {'start_time': 2.75, 'end_time': 2.86, 'text': 'a'}, 
                {'start_time': 2.88, 'end_time': 3.06, 'text': 'big'}, 
                {'start_time': 3.18, 'end_time': 3.44, 'text': 'toy'}, 
                {'start_time': 3.54, 'end_time': 3.82, 'text': 'frog'}, 
                {'start_time': 3.96, 'end_time': 4.06, 'text': 'for'}, 
                {'start_time': 4.12, 'end_time': 4.18, 'text': 'the'}, 
                {'start_time': 4.24, 'end_time': 4.52, 'text': 'kids.'}]
  
for item in transcript1:  
    start = item['start_time']  
    end = item['end_time']  
    middle = (start + end) / 2  
    text = item['text']  
    ax1.axvline(x=start, ymin=0, ymax=1, color='r', linestyle='--', linewidth=1)  
    ax1.axvline(x=end, ymin=0, ymax=1, color='r', linestyle='--', linewidth=1)  
    ax1.text(middle, 1.05, text, fontsize=12.5, fontweight=('bold' if item.get('bold') else 'normal'), color=item.get('color', 'black'), transform=ax1.get_xaxis_transform(), ha='center')  
  
# 6. 绘制第二个波形图  
librosa.display.waveshow(y2, sr2, ax=ax2)  
  
# 7. 在第二个波形图框的上方添加transcript文本  
transcript2 = [{'start_time': 0.0, 'end_time': 0.18, 'text': 'We'}, 
{'start_time': 0.18, 'end_time': 0.4, 'text': 'also'}, 
{'start_time': 0.52, 'end_time': 0.66, 'text': 'need'}, 
{'start_time': 0.68, 'end_time': 0.76, 'text': 'a'}, 
{'start_time': 0.78, 'end_time': 0.98, 'text': 'small'}, 
{'start_time': 1.06, 'end_time': 1.46, 'text': 'plastic'}, 
{'start_time': 1.54, 'end_time': 1.76, 'text': 'snake'}, 
{'start_time': 1.82, 'end_time': 1.88, 'text': 'and'}, 
{'start_time': 1.91, 'end_time': 1.99, 'text': 'a'}, 
{'start_time': 2.02, 'end_time': 2.18, 'text': 'big'}, 
{'start_time': 2.28, 'end_time': 2.44, 'text': 'toy'}, 
{'start_time': 2.54, 'end_time': 2.72, 'text': 'frog'}, 
{'start_time': 2.8, 'end_time': 2.9, 'text': 'for'}, 
{'start_time': 2.96, 'end_time': 3.02, 'text': 'the'}, 
{'start_time': 3.06, 'end_time': 3.32, 'text': 'kids.'}]
  
  
for item in transcript2:  
    start = item['start_time']  
    end = item['end_time']  
    middle = (start + end) / 2  
    text = item['text']  
    ax2.axvline(x=start, ymin=0, ymax=1, color='r', linestyle='--', linewidth=1)  
    ax2.axvline(x=end, ymin=0, ymax=1, color='r', linestyle='--', linewidth=1)  

    ax2.text(middle, 1.05, text, fontsize=12.5, fontweight=('bold' if item.get('bold') else 'normal'), color=item.get('color', 'black'), transform=ax2.get_xaxis_transform(), ha='center')
ax2.set_xlabel("")  
ax2.set_ylabel("Liu. et al", fontsize=13)
# ax2.set_yticks(np.arange(-1, 0.5, 2))  
ax2.set_ylim(-1.0, 1.0)  # 设置y轴的范围为-1到1  

ax2.set_xlim(0, max_duration)  # 设置x轴的范围  

librosa.display.waveshow(y3, sr3, ax=ax3)  

# 4. 移除第一张图的"Time"属性  
ax3.set_xlabel("Time(s)", fontsize=18)  
ax3.set_ylabel("Proposed", fontsize=13) 
ax3.set_xlim(0, max_duration)  # 设置x轴的范围  

# 5. 在第一个波形图框的上方添加transcript文本  
transcript3 = [{'start_time': 0.0, 'end_time': 0.16, 'text': 'We'},
 {'start_time': 0.22, 'end_time': 0.5, 'text': 'also'}, 
 {'start_time': 0.64, 'end_time': 0.82, 'text': 'need'}, 
 {'start_time': 0.85, 'end_time': 0.99, 'text': 'a'}, 
 {'start_time': 1.06, 'end_time': 1.36, 'text': 'small'}, 
 {'start_time': 1.44, 'end_time': 1.96, 'text': 'plastic'}, 
 {'start_time': 2.08, 'end_time': 2.50, 'text': 'snake'}, 
  {'start_time': 2.52, 'end_time': 2.76, 'text': 'and'}, 
 {'start_time': 2.77, 'end_time': 2.97, 'text': 'a'}, 
 {'start_time': 2.98, 'end_time': 3.18, 'text': 'big'}, 
 {'start_time': 3.28, 'end_time': 3.56, 'text': 'toy'}, 
 {'start_time': 3.68, 'end_time': 4.02, 'text': 'frog'}, 
 {'start_time': 4.16, 'end_time': 4.28, 'text': 'for'}, 
 {'start_time': 4.38, 'end_time': 4.46, 'text': 'the'}, 
 {'start_time': 4.52, 'end_time': 4.82, 'text': 'kids.'}]
  
  
for item in transcript3:  
    start = item['start_time']  
    end = item['end_time']  
    middle = (start + end) / 2  
    text = item['text']  
    ax3.axvline(x=start, ymin=0, ymax=1, color='r', linestyle='--', linewidth=1)  
    ax3.axvline(x=end, ymin=0, ymax=1, color='r', linestyle='--', linewidth=1)  

    ax3.text(middle, 1.05, text, fontsize=12.5, fontweight=('bold' if item.get('bold') else 'normal'), color=item.get('color', 'black'), transform=ax3.get_xaxis_transform(), ha='center')
# 8. 显示图形  
plt.subplots_adjust(hspace=0.5)  # hspace参数控制子图之间的高度间距  

# 设置通用的y轴标签  
fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical', fontsize=18) 
plt.show()  
  
# 9. 保存到指定文件夹  
output_image_file = 'your_output_image_file.png'  # 替换为您的输出图像文件路径  
plt.savefig(output_image_file, dpi=300)
# 5. 保存到当前文件夹  
plt.savefig('/home/v-zhijunjia/data/plots_pictures/audio_waveform_with_transcript_v3.png', dpi=300)  
