filename = "/mnt/zhijun/LibriTTS/data/vc_tokenized/unique_semantic_tokens.k2symbols"  
  
with open(filename, "w") as file:  
    for i in range(1, 501):  
        line = f"{i} {i}\n"  
        file.write(line)  
  
print(f"Token mapping file '{filename}' has been generated.")  
