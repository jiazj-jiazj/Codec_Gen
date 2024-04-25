
import os  
import random  
import shutil  
  
# 设置源文件夹和目标文件夹  
source_folder = "/dev_huaying/zhijun/data/LibriSpeech/filtered_4s_10s-test-spk-wer"  # 原始文件夹路径  
target_folder_source = "/dev_huaying/zhijun/data/test_vc/benchmark_librispeech_10speakers/source"  # 目标文件夹路径（源）  
target_folder_prompt = "/dev_huaying/zhijun/data/test_vc/benchmark_librispeech_10speakers/prompt"  # 目标文件夹路径（提示）  
target_folder_txt = "/dev_huaying/zhijun/data/test_vc/benchmark_librispeech_10speakers/asr_txt"  # 目标文件夹路径（文本）  

os.makedirs(target_folder_source, exist_ok=True)
os.makedirs(target_folder_prompt, exist_ok=True)
os.makedirs(target_folder_txt, exist_ok=True)

def change_gt_to_asr(filename):  
    return filename.replace("gt_", "asr_", 1)  
  
# 随机抽取30个speaker  
speakers = os.listdir(source_folder)  
if len(speakers) > 10:  
    speakers = random.sample(speakers, 10)  
  
# 从每个选定的speaker中随机抽取一个章节  
for selected_speaker in speakers:  
    speaker_folder = os.path.join(source_folder, selected_speaker)  
  
    # 确保这是一个文件夹  
    if os.path.isdir(speaker_folder):  
        gt_files = []  
        prompt_files = []  
  
        # 随机抽取一个章节  
        chapters = [chapter for chapter in os.listdir(speaker_folder) if os.path.isdir(os.path.join(speaker_folder, chapter))]  
        selected_chapter = random.choice(chapters)  
        chapter_folder = os.path.join(speaker_folder, selected_chapter)  
  
        # 遍历章节文件夹中的.flac文件和.txt文件  
        for file in os.listdir(chapter_folder):  
            if file.endswith(".flac"):  
                if file.startswith("gt_"):  
                    gt_files.append(os.path.join(chapter_folder, file))  
                elif file.startswith("prompt_"):  
                    prompt_files.append(os.path.join(chapter_folder, file))  
  
        # 从gt_files列表中随机抽取四个文件  
        selected_gt_files = random.sample(gt_files, 4)  
  
        # 将选定的gt文件及其相应的文本文件复制到目标文件夹（源）  
        for file in selected_gt_files:  
            shutil.copy(file, os.path.join(target_folder_source, os.path.basename(file)))  
            txt_file = os.path.splitext(file)[0] + ".txt"  
            ss ='/'.join(txt_file.split('/')[:-1])
            txt_basename = change_gt_to_asr(os.path.basename(txt_file))  
            txt_file = ss + '/' +txt_basename
            # txt_basename = change_gt_to_asr(os.path.basename(txt_file))  
            shutil.copy(txt_file, os.path.join(target_folder_txt, txt_basename))  
  
            # 修改源文件夹中的原始文本文件名称  
            os.rename(txt_file, os.path.join(os.path.dirname(txt_file), txt_basename))  
  
        # 从prompt_files列表中随机抽取一个文件  
        selected_prompt_file = random.choice(prompt_files)  
  
        # 将选定的prompt文件及其相应的文本文件复制到目标文件夹（提示）  
        os.makedirs(target_folder_prompt, exist_ok=True)
        shutil.copy(selected_prompt_file, os.path.join(target_folder_prompt, os.path.basename(selected_prompt_file)))  
        txt_file = os.path.splitext(selected_prompt_file)[0] + ".txt"  
        shutil.copy(txt_file, os.path.join(target_folder_txt, os.path.basename(txt_file)))  
  
print("随机抽取的文件已复制到目标文件夹，原始文本文件名已修改。")  


