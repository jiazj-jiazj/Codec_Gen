basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S")  
nohup python -u /dev_huaying/zhijun/valle_23_4_22/wavlm/compute_cosine_dir.py > /dev_huaying/zhijun/valle_23_4_22/egs/libritts/log/compute_cos_dir_${basestr}.txt 2>&1 &