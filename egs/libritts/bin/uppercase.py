import os  
  
txt_folder = "/home/v-zhijunjia/data/data_update/benchmark_l1l2_vc_12speakers/txt"  
  
# Iterate through all text files in the folder  
for root, dirs, files in os.walk(txt_folder):  
    for file in files:  
        if file.endswith(".txt"):  
            file_path = os.path.join(root, file)  
  
            # Read the original content of the file  
            with open(file_path, "r") as txt_file:  
                content = txt_file.read()  
  
            # Convert the content to uppercase  
            uppercase_content = content.upper()  
  
            # Write the uppercase content back to the file  
            with open(file_path, "w") as txt_file:  
                txt_file.write(uppercase_content)  
  
            print(f"Converted {file_path} to uppercase")  
