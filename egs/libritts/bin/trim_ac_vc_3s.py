import os  
import librosa  
import soundfile as sf  
  
prompt_folder = "/home/v-zhijunjia/data/data_update/benchmark_librispeech_noise_10speakers/prompt"  
duration_s = 3  # 3 seconds  
  
# Iterate through all audio files in the folder  
for root, dirs, files in os.walk(prompt_folder):  
    for file in files:  
        if file.endswith(".wav"):  
            file_path = os.path.join(root, file)  
  
            # Load the audio file  
            audio, sample_rate = librosa.load(file_path, sr=None)  
  
            # Trim the audio to the desired duration  
            num_samples = duration_s * sample_rate  
            trimmed_audio = audio[:num_samples]  
  
            # Save the trimmed audio using soundfile  
            sf.write(file_path, trimmed_audio, sample_rate)  
            print(f"Trimmed {file_path} to {duration_s} seconds")  
