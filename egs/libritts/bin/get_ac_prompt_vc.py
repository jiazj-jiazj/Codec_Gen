import os  
import shutil  
import itertools  
  
src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/vctk_iclr_test_mos_ac_bef"  
dest_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/ac_vctk"  
  
if not os.path.exists(dest_dir):  
    os.makedirs(dest_dir)  
  
audio_files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]  
  
speakers = set()  
  
# Extract speaker names from audio file names  
for audio_file in audio_files:  
    speaker = audio_file.split('_')[0]  
    speakers.add(speaker)  
  
# Create pairs of speakers  
speaker_pairs = list(itertools.combinations(speakers, 2))  
  
# Move audio files of each speaker pair to a new folder  
for speaker_pair in speaker_pairs:  
    new_folder_name = f"{speaker_pair[0]}_{speaker_pair[1]}"  
    new_folder_path = os.path.join(dest_dir, new_folder_name)  
  
    if not os.path.exists(new_folder_path):  
        os.makedirs(new_folder_path)  
  
    for audio_file in audio_files:  
        speaker = audio_file.split('_')[0]  
        if speaker in speaker_pair:  
            src_file = os.path.join(src_dir, audio_file)  
            dest_file = os.path.join(new_folder_path, audio_file)  
            shutil.copy(src_file, dest_file)  

# import os  
  
# src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/arctic_mos_accent_onedir"  
  
# audio_files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]  
  
# for audio_file in audio_files:  
#     src_file = os.path.join(src_dir, audio_file)  
#     file_basename, file_ext = os.path.splitext(audio_file)  
      
#     new_file_basename = file_basename.replace('_', '', 1)  
#     dest_file = os.path.join(src_dir, f"{new_file_basename}{file_ext}")  
#     os.rename(src_file, dest_file)  

# import os  
# import re  
  
# src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/arctic_mos_accent_onedir"  
  
# audio_files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]  
  
# for audio_file in audio_files:  
#     src_file = os.path.join(src_dir, audio_file)  
#     file_basename, file_ext = os.path.splitext(audio_file)  
      
#     match = re.search(r'_(\d+)$', file_basename)  
#     if match:  
#         new_file_basename = match.group(0)  
#         dest_file = os.path.join(src_dir, f"{new_file_basename}{file_ext}")  
#         os.rename(src_file, dest_file)  

# import os  
# import random  
# import shutil  
  
# src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/arctic_mos_accent_onedir"  
# temp_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/arctic_mos_accent_temp"  
  
# if not os.path.exists(temp_dir):  
#     os.makedirs(temp_dir)  
  
# audio_files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]  
  
# random.shuffle(audio_files)  
  
# for i, audio_file in enumerate(audio_files, start=1):  
#     src_file = os.path.join(src_dir, audio_file)  
#     file_basename, file_ext = os.path.splitext(audio_file)  
#     dest_file = os.path.join(temp_dir, f"{file_basename}_{i}{file_ext}")  
#     shutil.move(src_file, dest_file)  
  
# # Remove the original directory  
# shutil.rmtree(src_dir)  
  
# # Rename the temp directory to the original directory  
# shutil.move(temp_dir, src_dir)  


# import os  
# import shutil  
# import random  
# import wave  
# from collections import defaultdict  
  
# base_folder = "/scratch/data/l2_test"  
# prompt_output_folder = "/home/v-zhijunjia/data/data_update/benchmark_l1l2_vc_india/prompt"  
# os.makedirs(prompt_output_folder, exist_ok=True)  
  
# min_duration = 3  # Minimum duration in seconds  
  
# # Group files by speaker and case number  
# file_group = defaultdict(lambda: defaultdict(list))  
  
# for root, dirs, files in os.walk(base_folder):  
#     for file in files:  
#         if file.endswith(".wav"):  
#             speaker = os.path.basename(os.path.dirname(root))  
#             case_number = int(file.split("_")[-1][1:].split(".")[0])  
#             file_path = os.path.join(root, file)  
#             file_group[speaker][case_number].append(file_path)  
  
# # Randomly select a case longer than 3 seconds from each speaker  
# for speaker, case_files in file_group.items():  
#     valid_files = []  
#     for case_number, file_paths in case_files.items():  
#         for file_path in file_paths:  
#             with wave.open(file_path, "rb") as wav_file:  
#                 duration = wav_file.getnframes() / wav_file.getframerate()  
#                 if duration > min_duration:  
#                     valid_files.append(file_path)  
  
#     if valid_files:  
#         selected_file = random.choice(valid_files)  
#         output_file_path = os.path.join(prompt_output_folder, os.path.basename(selected_file))  
#         shutil.copy(selected_file, output_file_path)  
#         print(f"Copied {selected_file} to {output_file_path}")  
#     else:  
#         print(f"No valid files found for speaker {speaker}")  
