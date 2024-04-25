# Train
## 参数解析
--model-name
--input-semantic
--semantic-depup
--semantic-remove
--semantic-type 0->hubert, 1->tfnet_256bps
--is-pretrain
--nar-mask-type  0-> soundstorm random 1-> group time generate
--nar-mask-ratio

##  LOCAL
### AR
* VC
    * Hubert as semantic
    --input-semantic True
    egs/libritts/scrips/valle_vallfe/scripts_infer_train/train_ar_direct_local_no_mulprocess_vc_libritts_tfcodec.sh
    * TFCodec as semantic
    --input-semantic True
    --semantic-type 1
    egs/libritts/scrips/valle_vallfe/scripts_infer_train/train_ar_direct_local_no_mulprocess_vc_libritts_tfcodec_sem_tfnet.sh
* Text2Acoustic
egs/libritts/scrips/valle_vallfe/scripts_infer_train/train_ar_direct_local_no_mulprocess_vc_libritts_tfcodec_tts.sh
* Text2Semantic
encoder-decoder vallfe系列
egs/libritts/scrips/valle_vallfe/scripts_infer_train/VALLFE_azure_debug_pretrain_finetune_libritts_txt2semantic.sh
// tune encoder
egs/libritts/scrips/valle_vallfe/scripts_infer_train/VALLFE_azure_debug_pretrain_finetune_libritts_l1l2_1spker_1spker_lr_change_tune_encoder.sh
decoder-only valle 系列
egs/libritts/scrips/valle_vallfe/scripts_infer_train/azure_debug_pretrain_finetune_libritts_l1l2.sh
egs/libritts/scrips/valle_vallfe/scripts_infer_train/azure_debug_pretrain_finetune_libritts_l1l2_4spker_4spker.sh
* SemanticPretrain 
    * --pret-mode
    --pret-prob
    --pret-lam
   egs/libritts/scrips/valle_vallfe/scripts_infer_train/azure_debug_pretrain_mode5_vallf.sh
* Accent_Semantic_Finetune
    * l1-l2
    egs/libritts/scrips/valle_vallfe/scripts_infer_train/train_ar_direct_local_no_mulprocess_vc_l1_l2_tfcodec_semantic.sh

### NAR Soundstorm
* VC
egs/libritts/scrips/soundstorm/scripts_infer_train/train_tfcodec_libritts_group_ar.sh
egs/libritts/scrips/soundstorm/scripts_infer_train/train_tfcodec_libritts.sh

## Azureml



# Infer
## model_path
    VC  depup
    /home/v-zhijunjia/zhijundata_small_v2/data/LibriTTS/lhotse_vc/output_vc/vc_Name_VALLE_m-dur_50_dty_float32_b-lr_0.05_ech_70_s_echo_1_n_quan_16_s_stps_5000_s_epo_4_a_g_steps_4_s_depup_True_2023_11_08_13_00_14/epoch-70.pt
    
    VC 
    /home/v-zhijunjia/data/valle-tensorboard-models/vc/only_ar/epoch-40.pt
    VC TFNet
    data/LibriTTS/lhotse_vc/output_vc/sem_tp_1_Name_VALLE_max-duration_50_dtype_float32_base-lr_0.05_train-stage_1_echo_70_start_echo_1_prefix_mode_1_input_semantic_True_only_ar_True_num_quantizers_16_sheduler_steps_5000_sheduler_epochs_4_accumulate_grad_steps_4_2023_10_31_07_45_20/epoch-70.pt
        
    VC TFNet depup
    data/LibriTTS/lhotse_vc/output_vc/vc_Name_VALLE_m-dur_50_dty_float32_b-lr_0.05_ech_70_s_echo_1_n_quan_16_s_stps_5000_s_epo_4_a_g_steps_4_s_depup_True_2023_11_08_13_00_14/epoch-18.pt

    ICLR accent convert
    data_local/valle-tensorboard-models/pretrain_finetune/mode_5_mask015_source1spker_600cases_tgt1spker_lr_0005/epoch-6.pt

    ICML accent convert

    /home/v-zhijunjia/zhijundata_small_v3/data_local/valle-tensorboard-models/pretrain_finetune/VALLE_hubert_sem/indiantts_l1l2_indain2native_4speakers/epoch-10.pt

    pretraining model
    data_local/valle-tensorboard-models/pretrain/trained_base_model/pret-epoch-70.pt
    
## parametter explain
- accent_remove:  prompt is myself is True
- input_semantic: vc_ac or tts
- is_pretrain: no use
## shell_path
    Codec gt
    egs/libritts/scripts_infer_train/codec_gt.sh
    tts one-stage
    egs/libritts/scripts_infer_train/infer_dir_valle_ar_nar_combine_dir_tts_v2.sh
    egs/libritts/scripts_infer_train/infer_dir_valle_ar_nar_combine_dir_mls_v2.sh  # mls infer
    ac
    # iclr
    egs/libritts/scripts_infer_train/infer_dir_valle_ar_nar_combine_vc_dir_libritts_tfcodec_onlyar_semantic_benchmark_iclr_ac.sh

    # icml
    egs/l1_l2_arctic/scripts_infer_train/infer_dir_valle_ar_nar_combine_vc_dir_libritts_tfcodec_onlyar_semantic_benchmark_iclr_ac.sh
    vc
    egs/libritts/scripts_infer_train/
    infer_dir_valle_ar_nar_combine_vc_dir_libritts_tfcodec_onlyar_vc_myself_depup_hubert.sh

    /home/v-zhijunjia/CodecGen/egs/libritts/scripts_infer_train/infer_dir_valle_ar_nar_combine_vc_dir_libritts_tfcodec_onlyar_vc_hubert.sh
    vc_tfnet
    egs/libritts/scripts_infer_train/infer_dir_valle_ar_nar_combine_vc_dir_libritts_tfcodec_onlyar_vc_tfnet.sh
    vc_tfnet depup
    egs/libritts/scripts_infer_train/infer_dir_valle_ar_nar_combine_vc_dir_libritts_tfcodec_onlyar_vc_tfnet_depup.sh


# WER
## get transcript
    egs/libritts/test_wer_spk_vc_benchmark/scripts/hubert_asr_dir_ac_vc.sh
## computer WER
    egs/libritts/test_wer_spk_vc_benchmark/scripts/compute_wer_dir.sh

# NISQA
   NISQA/scripts/predict_single_wav_dir.sh

# SPK
    // 注意使用unispeech的dockers container
    UniSpeech/downstreams/speaker_verification/script/inference_dir_benchmark_ac_vc.sh

## azureml PATH

# prepare dataset
## prepare mls
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

# test benchmark
* tts 
data_local/data/tts_test_librispeech/LibriSpeech/filtered_4s_10s-test-spk-wer
* vc
data_local/data/data_update/benchmark_librispeech_10speakers/source
data_local/data/data_update/benchmark_librispeech_10speakers/prompt
* ac
data_local/accent_wer/val_ac_models/indian_accent_test_arctics_15cases
data_local/accent_wer/val_ac_models/indian_accent_test_arctics_50cases