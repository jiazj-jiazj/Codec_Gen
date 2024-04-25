
from lhotse import CutSet, load_manifest_lazy
import torch
from tqdm import tqdm
if __name__ == "__main__":
    cuts = load_manifest_lazy("/dev_huaying/zhijun/data/AISHELL1_2_3/lhotse_data_v2/data/tokenized/cuts_train.jsonl.gz")
    cuts = CutSet.from_cuts(cuts)  

    cut = cuts["BAC009S0737W0408-4920"]
    feature = torch.from_numpy(cut.load_features())  
    for cut in tqdm(cuts, desc="处理 cuts"):  
        try:  
            print(cut.id)
            feature = torch.from_numpy(cut.load_features())  
        except Exception as e:  
            print(f"异常：{cut}")
            break