import os  
import torchaudio  
import shutil  
import fire
  
def get_audio_duration(audio_path):  
    waveform, sr = torchaudio.load(audio_path)  
    return waveform.shape[1] / sr  
  
def find_files_and_move(path, folder_B, ext_audio=".flac", ext_txt=".txt", min_duration=4, max_duration=10):  
    for root, dirs, files in os.walk(path):  
        for file in files:  
            if file.endswith(ext_audio):  
                audio_path = os.path.join(root, file)  
                duration = get_audio_duration(audio_path)  
  
                if min_duration <= duration <= max_duration:  
                    txt_file = file.replace(ext_audio, ext_txt)  
                    if txt_file in files:  
                        subfolder = os.path.relpath(root, path)  
                        os.makedirs(os.path.join(folder_B, subfolder), exist_ok=True)  
  
                        shutil.copy(audio_path, os.path.join(folder_B, subfolder, file))  
                        shutil.copy(os.path.join(root, txt_file), os.path.join(folder_B, subfolder, txt_file))
  

if __name__ == "__main__":
    fire.Fire(find_files_and_move)
