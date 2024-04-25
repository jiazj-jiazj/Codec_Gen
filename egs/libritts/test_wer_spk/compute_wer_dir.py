import os  
from jiwer import wer  
import re 
import Levenshtein  
import nltk  
nltk.download('punkt')  
import torch
torch.backends.cudnn.enabled = False  
import fire

  
def get_file_path(folder, speaker, chapter, file_name):  
    return os.path.join(folder, speaker, chapter, file_name)  
  
def find_matching_gen_file_names(gen_folder, speaker, chapter, gt_file_name, prefix="begin1", is_gt=False):  
    original_file_name = gt_file_name[len("gen_"):].split(".")[0]  
    if is_gt is True:
        pattern = f"{prefix}_{original_file_name}.*\\.txt$"
    else:
        pattern = f"{prefix}_{original_file_name}_.*\\.txt$"  
      
    gen_speaker_chapter_folder = os.path.join(gen_folder, speaker, chapter)  
    matching_files = [f for f in os.listdir(gen_speaker_chapter_folder) if re.match(pattern, f)]  
      
    return matching_files  

def compute_wer(gt_folder, gen_folder, is_gt, prefix):
    total_S = 0  
    total_D = 0  
    total_I = 0  
    total_C = 0  
    compute_ave = False
    compute_tot = True

    for root, dirs, files in os.walk(gt_folder):  
        for file_name in files:  
            if file_name.startswith("gen_") and file_name.endswith(".txt"):  
                relative_path = os.path.relpath(root, gt_folder)  
                speaker, chapter = relative_path.split(os.sep)  
                gt_file_path = get_file_path(gt_folder, speaker, chapter, file_name)

                gen_file_names = find_matching_gen_file_names(gen_folder, speaker, chapter, file_name, prefix, is_gt)  
                print(gen_file_names)
                if gen_file_names == []:
                    continue

                with open(gt_file_path, "r") as gt_file:  
                    gt_text = gt_file.read()
    
                    for gen_file_name in gen_file_names:
                        gen_file_path = get_file_path(gen_folder, speaker, chapter, gen_file_name)  
    
                        with open(gen_file_path, "r") as gen_file:
                            gen_text = gen_file.read()  
                            if compute_tot is True and compute_ave is False :
                                gt_words = nltk.word_tokenize(gt_text)  
                                gen_words = nltk.word_tokenize(gen_text)  
                                edit_ops = Levenshtein.editops(gt_words, gen_words)  
                                insertions = sum(1 for op in edit_ops if op[0] == 'insert')  
                                deletions = sum(1 for op in edit_ops if op[0] == 'delete')  
                                substitutions = sum(1 for op in edit_ops if op[0] == 'replace')  
                                correct = len(gt_words) - (deletions + substitutions)
                                
                                total_S += substitutions  
                                total_D += deletions  
                                total_I += insertions  
                                total_C += correct  

    if compute_tot is True and compute_ave is False :
        total_wer = (total_S + total_D + total_I) / (total_S + total_D + total_C)
        print(f"total_S, total_D, total_I, total_C:{total_S, total_D, total_I, total_C}")
        print("总WER：", total_wer)

if __name__ == "__main__":
    
    fire.Fire(compute_wer)

    
