# mls
## data location
- about 6000 hours ready: /raid/dataset/lhotse_dataset/mls_train_lhotse_dataset azure:data/mls/mls_train_lhotse_dataset/
- all mls splited into 20 files. 3files above about 6000 hours have been processed.: /raid/dataset/lhotse_dataset/mls/cut_lhotse
- raw files:/raid/dataset/mls_files/mls_english azureloc: data_local/dataset/mls_files/mls_english/

## code
### train
#### local debug
egs/libritts/scripts_infer_train/train_ar_direct_local_no_mulprocess_vc_libritts_tfcodec_tts.sh
#### azureml train
egs/libritts/bin_azure/azure_32g_ar_libritts_tts_tfcodec_only_ar.py   
- 主要需要改下下面的变量
    manifest_dir = ws.datastores[datastore_name].path("data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes").as_mount()
    text_tokens = ws.datastores[datastore_name].path("data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_text_tokens.k2symbols").as_mount()
    semantic_tokens = ws.datastores[datastore_name].path("data/LibriTTS/lhotse_vc/vc_tokenized_16k_tfcodec_16codes/unique_semantic_tokens.k2symbols").as_mount()


#### azureml train
### preprocess

#### transfer raw_files into recordings and supervisions  
    python /home/v-zhijunjia/CodecGen/egs/mls/bin/prepare_mls.py  

#### process dev and test with acoustic tokens and txt
    bash egs/libritts/scripts_data_process/tokenized_16k_mls_tfcodec_dev_test.sh

#### process train
##### process train without acoustic tokens and txt
    bash /home/v-zhijunjia/CodecGen/egs/libritts/scripts_data_process/  
    bash tokenized_16k_mls_tfcodec_train_total.sh
##### split train
    python egs/libritts/bin/split_cutset.py #记得传上一步处理好的文件
##### process part train file with acoustic tokens and txt:
    bash /home/v-zhijunjia/CodecGen/egs/libritts/scripts_data_process/  
    bash tokenized_16k_mls_tfcodec_train_each_part.sh # 每次执行我处理一个train文件，还剩下17个,注意传每次待处理文件的地址
##### 说明 两个shell用的python文件代码差不多，输入上有差别

#### combile 
##### combile cuts_json
##### symblo_merge
    python valle-4-23/egs/libritts/bin/symbol_merge.py
##### train_cuts_{i} combine
    python egs/libritts/bin/combine_train_cuts.py



