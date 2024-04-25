import os  
import nltk  
import Levenshtein  
  
def best_wer_in_group(group):  
    best_wer = float('inf')  
    best_result = None  
  
    for gen_text in group:  
        gt_text = group[gen_text]  
        gt_words = nltk.word_tokenize(gt_text)  
        gen_words = nltk.word_tokenize(gen_text)  
  
        edit_ops = Levenshtein.editops(gt_words, gen_words)  
        insertions = sum(1 for op in edit_ops if op[0] == 'insert')  
        deletions = sum(1 for op in edit_ops if op[0] == 'delete')  
        substitutions = sum(1 for op in edit_ops if op[0] == 'replace')  
        correct = len(gt_words) - (deletions + substitutions)  
  
        wer = (substitutions + deletions + insertions) / len(gt_words)  
  
        if wer < best_wer:  
            best_wer = wer  
            best_result = (substitutions, deletions, insertions, correct)  
  
    return best_result  
  
gen_folder = "/home/v-zhijunjia/data/accent_iclr/ASI/mode_5_mask015_source1spker_300cases_tgt1spker_lr_0005_source-topk-2-epoch14_2023-09-25_21:04:48_txt"  
gt_folder = "/home/v-zhijunjia/data/accent_iclr/ASI/wav_txt"  
  
grouped_files = {}  

i=0
for root, dirs, files in os.walk(gen_folder):  
    for file_name in files:  
        if file_name.endswith(".txt"):  
            gen_file_path = os.path.join(root, file_name)  
  
            with open(gen_file_path, "r") as gen_file:  
                gen_text = gen_file.read()  
  
            gt_file_name = file_name[:-6] + ".txt"  

            gt_file_path = os.path.join(gt_folder, gt_file_name)  
  
            with open(gt_file_path, "r") as gt_file:  
                gt_text = gt_file.read()  
  
            gt_text = gt_text.upper()  
            gt_text = "".join(c for c in gt_text if c.isalnum() or c.isspace())  
            i+=1
            base_name = file_name[:-6]
            if base_name not in grouped_files:  
                grouped_files[base_name] = {}  
            grouped_files[base_name][gen_text] = gt_text  

print(f"cases is {i}")
total_S = 0  
total_D = 0  
total_I = 0  
total_C = 0  
  
for group in grouped_files.values():  
    best_result = best_wer_in_group(group)  
    total_S += best_result[0]  
    total_D += best_result[1]  
    total_I += best_result[2]  
    total_C += best_result[3]  
  
print(total_S, total_D, total_I, total_C)  
total_wer = (total_S + total_D + total_I) / (total_S + total_D + total_C)  
print("总WER：", total_wer)
