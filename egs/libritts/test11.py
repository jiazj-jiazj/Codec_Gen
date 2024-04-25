import os  
  
folder_A = "/dev_huaying/zhijun/data/LibriSpeech/filtered_4s_10s-test-spk-wer"  # 请将此路径替换为实际文件夹A的路径  
  
text_files = []  
reference_files = []  

for root, dirs, files in os.walk(folder_A):  
    for file in files:  
        print(file)
        if file.startswith("gen_") and file.endswith(".txt"):  
            print(file)
            text_files.append(os.path.join(root, file))  
            reference_file = file.replace("gen_", "prompt_").replace(".txt", ".flac")  
            reference_files.append(os.path.join(root, reference_file))  
  
print("Text files:", text_files)  
print("Reference files:", reference_files)  