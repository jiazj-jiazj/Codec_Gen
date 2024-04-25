import os  
  
src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/ac_vctk_v2_1_dir_important"  
  
audio_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.wav')], key=lambda x: int(os.path.splitext(x)[0]))  
  
for index, audio_file in enumerate(audio_files):  
    src_file = os.path.join(src_dir, audio_file)  
    suffix = index % 2  
    file_name_without_ext = os.path.splitext(audio_file)[0]  
    dest_file = os.path.join(src_dir, f"{file_name_without_ext}_{suffix}.wav")  
    os.rename(src_file, dest_file)  

# import os  
# import random  
# import re  
  
# src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/arctic_mos_accent_onedir_bef_v2_ac_one_dir"  
  
# audio_files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]  
# random.shuffle(audio_files)  
  
# num = 0  
# handled_files = set()  
  
# for audio_file in audio_files:  
#     if audio_file not in handled_files:  
#         src_file = os.path.join(src_dir, audio_file)  
#         file_name_without_ext = os.path.splitext(audio_file)[0]  
#         dest_file = os.path.join(src_dir, f"{file_name_without_ext}_{num}.wav")  
#         os.rename(src_file, dest_file)  
#         handled_files.add(audio_file)  
  
#         arctic_bxxxx = re.search("arctic_b\d+", audio_file).group()  
#         matched_files = [f for f in audio_files if f.startswith(audio_file[0]) and arctic_bxxxx in f]  

#         for matched_file in matched_files:  
#             if matched_file not in handled_files:  
#                 src_file = os.path.join(src_dir, matched_file)  
#                 file_name_without_ext = os.path.splitext(matched_file)[0]  
#                 dest_file = os.path.join(src_dir, f"{file_name_without_ext}_{num + 1}.wav")  
#                 os.rename(src_file, dest_file)  
#                 handled_files.add(matched_file)  
#                 break  
  
#         num += 2  


# import os  
  
# src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/arctic_mos_accent_onedir_bef_v2"  
  
# files = [f for f in os.listdir(src_dir) if f.startswith('arctic')]  
  
# for file in files:  
#     src_file = os.path.join(src_dir, file)  
#     dest_file = os.path.join(src_dir, f"native_bdl_{file}")  
#     os.rename(src_file, dest_file)  


# import os  
  
# src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/ac_vctk_v2_1_dir_important"  # 将此路径更改为要遍历的文件夹路径  
  
# audio_files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]  
  
# for audio_file in audio_files:  
#     src_file = os.path.join(src_dir, audio_file)  
#     new_file_name = "_".join(audio_file.split('_')[1:])  
#     dest_file = os.path.join(src_dir, new_file_name)  
#     os.rename(src_file, dest_file)  




# import os  
  
# src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/ac_vctk_v2_1_dir_important"  # 将此路径更改为要遍历的文件夹路径  
  
# audio_files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]  
  
# for audio_file in audio_files:  
#     src_file = os.path.join(src_dir, audio_file)  
#     new_file_name = f"{audio_file.split('_')[0]}.wav"  
#     dest_file = os.path.join(src_dir, new_file_name)  
#     os.rename(src_file, dest_file)  



# import os  
# import random  
  
# src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/ac_vctk_v2_1_dir"  
  
# audio_files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]  
# random.shuffle(audio_files)  
  
# num = 0  
  
# while audio_files:  
#     audio_file = audio_files.pop()  
#     src_file = os.path.join(src_dir, audio_file)  
#     dest_file = os.path.join(src_dir, f"{num}_{audio_file}")  
#     os.rename(src_file, dest_file)  
      
#     matching_file = None  
#     prefix = audio_file[0]  
#     suffix = audio_file[-8:-4]  
      
#     for f in audio_files:  
#         if f.startswith(prefix) and f.endswith(suffix + '.wav'):  
#             matching_file = f  
#             break  
  
#     if matching_file:  
#         audio_files.remove(matching_file)  
#         src_file = os.path.join(src_dir, matching_file)  
#         dest_file = os.path.join(src_dir, f"{num+1}_{matching_file}")  
#         os.rename(src_file, dest_file)  
  
#     num += 2  

# import os  
# import shutil  
  
# src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/ac_vctk_v2"  
# dest_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final_onedir"  
  
# if not os.path.exists(dest_dir):  
#     os.makedirs(dest_dir)  
  
# for root, dirs, files in os.walk(src_dir):  
#     for file in files:  
#         if file.endswith('.wav'):  
#             src_file = os.path.join(root, file)  
#             dest_file = os.path.join(dest_dir, file)  
#             shutil.copy(src_file, dest_file)  

# import os  
  
# src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/ac_vctk_v2"  
  
# subfolders = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]  
  
# for index, subfolder in enumerate(subfolders, start=1):  
#     subfolder_path = os.path.join(src_dir, subfolder)  
#     audio_files = [f for f in os.listdir(subfolder_path) if f.endswith('.wav')]  
      
#     for audio_file in audio_files:  
#         src_file = os.path.join(subfolder_path, audio_file)  
#         dest_file = os.path.join(subfolder_path, f"{index}_{audio_file}")  
#         os.rename(src_file, dest_file)  



# import os  
# import shutil  
# import itertools  
  
# src_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/vctk_iclr_test_mos_ac_bef"  
# dest_dir = "/home/v-zhijunjia/data/accent_iclr/iclr_final/ac_vctk_v2"  
  
# if not os.path.exists(dest_dir):  
#     os.makedirs(dest_dir)  
  
# audio_files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]  
  
# speakers = set()  
  
# # Extract speaker names from audio file names  
# for audio_file in audio_files:  
#     speaker = audio_file.split('_p248')[0]  
#     speakers.add(speaker)  
  
# # Create pairs of speakers  
# speaker_pairs = list(itertools.combinations(speakers, 2))  
  
# # Move audio files of each speaker pair to a new folder  
# for speaker_pair in speaker_pairs:  
#     new_folder_name = f"{speaker_pair[0]}_{speaker_pair[1]}"  
#     new_folder_path = os.path.join(dest_dir, new_folder_name)  
  
#     if not os.path.exists(new_folder_path):  
#         os.makedirs(new_folder_path)  
  
#     for audio_file in audio_files:  
#         speaker = audio_file.split('_p248')[0]  
#         if speaker in speaker_pair:  
#             src_file = os.path.join(src_dir, audio_file)  
#             dest_file = os.path.join(new_folder_path, audio_file)  
#             shutil.copy(src_file, dest_file)  
