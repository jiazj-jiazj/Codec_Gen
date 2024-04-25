import os  
import re  
  
src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/arctic"  
  
for speaker in os.listdir(src_dir):  
    speaker_dir = os.path.join(src_dir, speaker)  
    if os.path.isdir(speaker_dir):  
        for audio_file in os.listdir(speaker_dir):  
            src_file = os.path.join(speaker_dir, audio_file)  
            if os.path.isfile(src_file) and audio_file.endswith('.wav'):  
                file_name, file_ext = os.path.splitext(audio_file)  
                match = re.search(r'(\w+?)_arctic_b(\d{4})', file_name) 
                 
                if match:  
                    new_file_name = match.group(1)  # Keep only the first matched string
                    dd = match.group(2) 
                    new_file_name = f"{new_file_name}_arctic_b{str(dd)}"
                    dest_file = os.path.join(speaker_dir, f"{new_file_name}{file_ext}")  
                    os.rename(src_file, dest_file)  
