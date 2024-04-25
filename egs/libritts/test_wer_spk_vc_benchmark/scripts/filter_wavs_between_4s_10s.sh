folder_A = "/dev_huaying/zhijun/data/LibriSpeech/test_clean_add_txt"  # 请将此路径替换为实际文件夹A的路径  
folder_B = "/dev_huaying/zhijun/data/LibriSpeech/filtered_4s_10s_test_clean_add_txt"  # 请将此路径替换为实际文件夹B的路径  

CUDA_VISIBLE_DEVICES=1 nohup python -u /dev_huaying/zhijun/valle_23_4_22/egs/libritts/test_wer_spk/filter_4s_10s_libritts_test.py \
    --folder_A ${folder_A}
    --folder_B ${folder_B}
