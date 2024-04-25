import os  
import matplotlib.pyplot as plt  
import librosa  
import librosa.display  
  
def plot_time_domain(audio_file, save_path):  
    y, sr = librosa.load(audio_file)  
    plt.figure(figsize=(12, 4))  # 设置图像的宽度和高度  
    librosa.display.waveshow(y, sr)  
    plt.gca().axis('off')  # 关闭坐标轴  
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # 保存时去除边框和留白  
    plt.close()  
  
def process_directory(directory):  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if file.endswith(('.wav', '.mp3')):  
                audio_file = os.path.join(root, file)  
                save_path = os.path.join(root, f'{os.path.splitext(file)[0]}_time_domain_no_others.png')  
                plot_time_domain(audio_file, save_path)  
  
directory = '/home/v-zhijunjia/zhijundata_small/data_local/others/others'  
process_directory(directory)  
