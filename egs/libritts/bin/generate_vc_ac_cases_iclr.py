import os  
import shutil  
  
src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/vctk_iclr"  
dest_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/vctk_iclr_test_mos_ac"
os.makedirs(dest_dir, exist_ok=True)  
patterns = ["003", "004", "041", "106", "142", "195", "233", "260", "353", "358"]  
  
for speaker in os.listdir(src_dir):  
    speaker_dir = os.path.join(src_dir, speaker)  
    if os.path.isdir(speaker_dir):  
        for pattern in patterns:  
            for audio_file in os.listdir(speaker_dir):  
                if pattern in audio_file:  
                    src_file = os.path.join(speaker_dir, audio_file)  
                    dest_file = os.path.join(dest_dir, f"{speaker}_p248_{pattern}.wav")  
                    shutil.copy(src_file, dest_file)  
