import os
import fire  
  
def process_trans_file(filepath):  
    with open(filepath, 'r', encoding='utf-8') as f:  
        lines = f.readlines()  
  
    for line in lines:  
        audio_name, text = line.split(maxsplit=1)  
        text = text.strip()  
        txt_filename = audio_name + '.txt'  
  
        with open(os.path.join(os.path.dirname(filepath), txt_filename), 'w', encoding='utf-8') as txt_file:  
            txt_file.write(text)  
  
def find_files_and_process(path, ext=".trans.txt"):  
    for root, dirs, files in os.walk(path):  
        for file in files:  
            if file.endswith(ext):  
                trans_filepath = os.path.join(root, file)  
                process_trans_file(trans_filepath)  
  
if __name__ == "__main__":
    fire.Fire(find_files_and_process)
