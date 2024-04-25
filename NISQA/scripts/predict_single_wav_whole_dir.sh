# tts
  
# Define the root directory containing subfolders  
ROOT_DIR="/home/v-zhijunjia/data/tts_test_librispeech/nar_test/converted_can_del_tts_first_top2_fixed_seed_v2/ours/v17_update_gp_i_mask_True_g_in_m_rep_p_0_g_in_m_rep_al_p_0.3/"  
  
# Define the output directory  
OUTPUT_DIR="/home/v-zhijunjia/CodecGen/NISQA/mos_files/tts/librispeech_test_clean_all_fixed_sem_v2/ours/v17_update_gp_i_mask_True_g_in_m_rep_p_0_g_in_m_rep_al_p_0.3"  
# Path to the pretrained model  
PRETRAINED_MODEL="weights/nisqa_tts.tar"  
  
# Iterate over each subfolder in the root directory  
for SUBDIR in "$ROOT_DIR"/*; do  
    if [ -d "$SUBDIR" ]; then  
        # Get the base name of the subdirectory  
        BASENAME=$(basename "$SUBDIR")  
  
        # Construct the file name for the CSV  
        FILE_NAME="${BASENAME}.csv"  
  
        # Run the prediction command for the current subfolder  
        CUDA_VISIBLE_DEVICES=0 python run_predict.py --mode predict_dir --pretrained_model $PRETRAINED_MODEL \
        --file_name "$FILE_NAME" --data_dir "$SUBDIR" --output_dir "$OUTPUT_DIR" --file_mode 0  
    fi  
done  

# #vc
# #!/bin/bash  
  
# # Define the root directory containing subfolders  
# ROOT_DIR="/home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/converted_test_can_del"  
  
# # Define the output directory  
# OUTPUT_DIR="/home/v-zhijunjia/CodecGen/NISQA/mos_files/vc/benchmark_librispeech_10speakers/baseline_train/baseline_infer"  
  
# # Path to the pretrained model  
# PRETRAINED_MODEL="weights/nisqa_tts.tar"  
  
# # Iterate over each subfolder in the root directory  
# for SUBDIR in "$ROOT_DIR"/*; do  
#     if [ -d "$SUBDIR" ]; then  
#         # Get the base name of the subdirectory  
#         BASENAME=$(basename "$SUBDIR")  
  
#         # Construct the file name for the CSV  
#         FILE_NAME="${BASENAME}.csv"  
  
#         # Run the prediction command for the current subfolder  
#         CUDA_VISIBLE_DEVICES=0 python run_predict.py --mode predict_dir --pretrained_model $PRETRAINED_MODEL \
#         --file_name "$FILE_NAME" --data_dir "$SUBDIR" --output_dir "$OUTPUT_DIR" --file_mode 0  
#     fi  
# done  
