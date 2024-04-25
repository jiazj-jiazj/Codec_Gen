import os  
import re  
  
src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/vctk_iclr_test_mos_ac"  
  
for audio_file in os.listdir(src_dir):  
    src_file = os.path.join(src_dir, audio_file)  
    if os.path.isfile(src_file) and audio_file.endswith('.wav'):  
        file_name, file_ext = os.path.splitext(audio_file)  
        match = re.search(r'_(\d+)$', file_name)  
        if match:  
            new_file_name = match.group(1)  # Only keep the digits  
            dest_file = os.path.join(src_dir, f"{new_file_name}{file_ext}")  
            os.rename(src_file, dest_file)  
