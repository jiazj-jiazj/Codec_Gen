import os  
from jiwer import wer  
import re 
import Levenshtein  
import nltk  
nltk.download('punkt')  
import torch
torch.backends.cudnn.enabled = False  
import fire



def compute_wer(gt_folder, gen_folder, is_gt, dataset, prefix):
    total_S = 0  
    total_D = 0  
    total_I = 0  
    total_C = 0  
    compute_ave = False
    compute_tot = True
    nums = 0 
    
    for root, dirs, files in os.walk(gen_folder):  
        for file_name in files:  
            if file_name.endswith(".txt"):
                gen_file_path = os.path.join(root, file_name)

                # print(f"gen_file_path:{gen_file_path}")
                nums+=1
                with open(gen_file_path, "r") as gen_file:  
                    gen_text = gen_file.read() 
                    print(gen_text) 
                    if is_gt is True:
                        if file_name[:4]=="asr_":
                            gt_file_name = file_name[4:] # delete asr_
                        else:
                            gt_file_name = file_name
                    else:
                        # vctk dataset [:9] reference name
                        # gt_file_name = file_name[9:]
                        #yourtts
                        if dataset == "yourtts":
                            gt_file_name = file_name.split("yourtts_")[1].split("_2023")[0]
                            gt_file_name = "asr_"+gt_file_name + ".txt"
                        elif dataset == "valle_ours":
                            gt_file_name = file_name.split("valle_ours_")[1].split("_2023")[0]
                            gt_file_name = "gen_"+gt_file_name + ".txt"
                            print(gt_file_name)
                        elif dataset == "asr":
                            gt_file_name = file_name.split("asr_")[1]
                            gt_file_name = "gen_"+gt_file_name
                            print(gt_file_name)
                        elif dataset == "styletts":
                            gt_file_name = file_name.split("styletts_")[1].split("_2023")[0]
                            gt_file_name = "gen_"+gt_file_name + ".txt"
                            print(gt_file_name)
                        elif dataset == "encodec" or dataset == "tfcodec":
                            parts = file_name.split("gt_", 1)
                            parts = parts[1]
                            gt_file_name = parts.split("_ar", 1)[0]
                            gt_file_name = gt_file_name + ".txt"
                            print(gt_file_name)
                        elif dataset == "p248":
                            gt_file_name = file_name[:12]+".txt"
                            gt_file_name = gt_file_name.replace("p225", "p248")
                        elif dataset == "ac_benchmark_l1l2":
                            pattern = r"(arctic_.\d{4})"  
                            gt_file_name = re.search(pattern, file_name).group(1)
                            gt_file_name = gt_file_name + ".txt"
                        elif dataset == "val_ac_encoder_decoder_model":
                            gt_file_name = file_name.split("sys2_")[1].split("_model3")[0]
                            gt_file_name = gt_file_name[-12:]+".txt"
                        elif dataset == "tts_benchmark_one_dir":
                            gt_file_name = file_name.split("sys2_")[1].split("_model3")[0]
                            gt_file_name = gt_file_name + ".txt"
                        elif dataset == "native_val_ac_encoder_decoder_model":
                            gt_file_name = file_name[4:]
                            print(gt_file_name)

                        elif dataset == "vc_benchmark":
                            gt_file_name = file_name.split("sys2_")[1].split("_model3")[0]
                            gt_file_name = gt_file_name[3:]
                            gt_file_name = gt_file_name + ".txt"

                    # spk = file_name[9:13]
                    gt_file_path = os.path.join(gt_folder, gt_file_name)
                    print(f"gt_file_path:{gt_file_path}")
                    with open(gt_file_path, "r") as gt_file:  
                        gt_text = gt_file.read()
                    gt_text = gt_text.upper()  
                    gt_text = "".join(c for c in gt_text if c.isalnum() or c.isspace())  
                    
                    print(gt_text)  
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
                        print(total_S, total_D, total_I, total_C)

    if compute_tot is True and compute_ave is False :
        total_wer = (total_S + total_D + total_I) / (total_S + total_D + total_C)  
        print("总WER：", total_wer)
        print(f"总数:{nums}")

if __name__ == "__main__":
    
    fire.Fire(compute_wer)

    
