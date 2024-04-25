import os  
import shutil  
from collections import defaultdict  
  
base_folder = "/scratch/data/l2_test"  
output_folder = "/home/v-zhijunjia/data/data_update/benchmark_l1l2_vc_india/source"  
os.makedirs(output_folder, exist_ok=True)  
  
cases_per_speaker = 4

# Group files by speaker and case number  
file_group = defaultdict(lambda: defaultdict(list))  
  
for root, dirs, files in os.walk(base_folder):  
    for file in files:  
        if file.endswith(".wav"):  
            speaker = os.path.basename(os.path.dirname(root))  
            case_number = int(file.split("_")[-1][1:].split(".")[0])  
            # print(case_number)
            file_path = os.path.join(root, file)  
            file_group[speaker][case_number].append(file_path)  

# Copy 4 cases from each speaker  
for speaker, case_files in file_group.items():  
    print(speaker)

    case_count = 0  
    for case_number, file_paths in case_files.items():  
        # print(case_number)
        # print(file_paths)
        for file_path in file_paths:  
            if case_count < cases_per_speaker:  
                output_file_path = os.path.join(output_folder, os.path.basename(file_path))  
                print(file_path)
                print(output_file_path)
                shutil.copy(file_path, output_file_path)  
                # print(f"Copied {file_path} to {output_file_path}")
                case_count += 1  
            else:  
                break  
