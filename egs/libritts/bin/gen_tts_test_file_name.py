import os  
  
input_dir = "/home/v-zhijunjia/data/test_vc/test_real_cases/3s_prompt"  
  
for filename in os.listdir(input_dir):  
    if filename.endswith(".wav"):  
        base_name, _ = os.path.splitext(filename)  
        new_name = f"{base_name}"  
        # os.rename(os.path.join(input_dir, filename), os.path.join(input_dir, f"{new_name}.wav"))  
        with open(os.path.join(input_dir, f"{base_name}.txt"), "w") as txt_file:  
            pass  
