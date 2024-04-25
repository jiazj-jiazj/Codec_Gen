CUDA_VISIBLE_DEVICES=1 python run_predict.py --mode predict_file --pretrained_model weights/nisqa_tts.tar \
--deg /mnt/users/jiazhijun/data/Accents/arctic/cmu_us_aew_arctic/wav/arctic_a0001.wav --output_dir /mnt/users/jiazhijun/attack_speech/NISQA
