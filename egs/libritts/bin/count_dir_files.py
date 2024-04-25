import os  
  
def count_audio_files(folder_path):  
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}  
    audio_file_count = 0  
  
    for file in os.listdir(folder_path):  
        file_path = os.path.join(folder_path, file)  
        if os.path.isfile(file_path) and os.path.splitext(file)[-1].lower() in audio_extensions:  
            audio_file_count += 1  
  
    return audio_file_count  
  
folder_path = "/home/v-zhijunjia/data/data_update/benchmark_librispeech_noise_10speakers"  # Replace with the path to your folder  
audio_file_count = count_audio_files(folder_path)  
  
print(f"Number of audio files in the folder: {audio_file_count}")  
