basestr=$(TZ='Asia/Shanghai' date "+%Y-%m-%d_%H:%M:%S") 

nohup python -u /home/v-zhijunjia/valle-4-23/egs/mls/bin/tokenize_semantic_tfnet.py --input_file /scratch/data/Libritts/tokenized_tfnet_semantic_token/cuts_test.jsonl.gz \
>/home/v-zhijunjia/data/log/tokenize_semantic_tfnet_${basestr}.txt 2>&1 &