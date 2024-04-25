import os
import argparse
from pathlib import Path 

def str2bool(value):  
    if isinstance(value, bool):  
        return value  
    if value.lower() in ('yes', 'true', 't', 'y', '1'):  
        return True  
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):  
        return False  
    else:  
        raise argparse.ArgumentTypeError('Boolean value expected.')  
def parser():
    import argparse  
    
    parser = argparse.ArgumentParser()  
    parser.add_argument('--nproc-per-node', type=int, default=2)  
    parser.add_argument('--nnodes', type=int, default=1)  
    parser.add_argument('--max-duration', type=int, default=20)  
    parser.add_argument('--filter-min-duration', type=float, default=0.5)  
    parser.add_argument('--filter-max-duration', type=float, default=14)  
    parser.add_argument('--train-stage', type=int, default=1)  
    parser.add_argument('--num-buckets', type=int, default=6)  
    parser.add_argument('--dtype', type=str, default="float16")  
    parser.add_argument('--save-every-n', type=int, default=100)  
    parser.add_argument('--valid-interval', type=int, default=200)
    parser.add_argument('--log-interval', type=int, default=100) 
    parser.add_argument('--model-name', type=str, default="valle")  
    parser.add_argument('--share-embedding', type=str2bool, default=True)  
    parser.add_argument('--norm-first', type=str2bool, default=True)  
    parser.add_argument('--add-prenet', type=str2bool, default=False)  
    parser.add_argument('--decoder-dim', type=int, default=1024)  
    parser.add_argument('--nhead', type=int, default=16)  
    parser.add_argument('--num-decoder-layers', type=int, default=12)  
    parser.add_argument('--prefix-mode', type=int, default=1)  
    parser.add_argument('--base-lr', type=float, default=0.01)  
    parser.add_argument('--warmup-steps', type=int, default=200)  
    parser.add_argument('--average-period', type=int, default=0)  
    parser.add_argument('--num-epochs', type=int, default=20)  
    parser.add_argument('--start-epoch', type=int, default=1)  
    parser.add_argument('--num-quantizers', type=int, default=8)
    parser.add_argument('--sheduler-steps', type=int, default=5000)
    parser.add_argument('--sheduler-epochs', type=int, default=4)

    parser.add_argument('--start-batch', type=int, default=0)  
    parser.add_argument('--accumulate-grad-steps', type=int, default=4)  
    parser.add_argument('--manifest-dir', type=str)  
    parser.add_argument('--text-tokens', type=str)  
    parser.add_argument('--newfile-suffix', type=str, default="test2")  
    parser.add_argument('--world-size', type=int, default=2)  
    parser.add_argument('--exp-dir', type=str)  
    parser.add_argument('--is-local', type=str2bool, default=False)  
    parser.add_argument(
        "--checkpoint-ar",
        type=str,
        default="/dev_huaying/zhijun/data/valle-tensorboard-models/ar/Name_VALLE_max-duration_80_dtype_float32_base-lr_0.01_world-size_8_train-stage_1_echo_150_start_echo_1_2023_05_29_03_00_16/best-valid-loss.pt",
        help="path of ar_model checkpoint",
    )
    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )
    parser.add_argument(
        "--checkpoint-nar",
        type=str,
        default="/dev_huaying/zhijun/data/valle-tensorboard-models/nar/Name_VALLE_max-duration_70_dtype_float32_base-lr_0.01_world-size_8_train-stage_2_echo_150_start_echo_1_2023_05_29_01_26_40/best-valid-loss.pt",
        help="path of nar_model checkpoint",
    )
    parser.add_argument(
        "--text-prompts",
        nargs="+",
        type=str,
        default=["looked out and tens the fives.", "Windows。The woman shout across over that."],
        help="Text prompts which are separated by |.",
    )
    parser.add_argument(
        "--audio-prompts",
        nargs="+",
        type=str,
        default=["/dev_huaying/zhijun/data/test_valle_naturalspeech2_yourtts_styleTTS/test1/reference_LibriSpeech_1st_txt_looktown.wav", "/dev_huaying/zhijun/data/test_valle_naturalspeech2_yourtts_styleTTS/test1/reference_LibriSpeech_2nd_txt_windows.wav"],
        help="Audio prompts which are separated by | and should be aligned with --text-prompts.",
    )
    parser.add_argument(
        "--text",
        nargs="+",
        type=str,
        default=["To get up and running quickly just follow the steps below.", "say i believe in you, and you also believe in me"],
        help="Text to be synthesized.",
    )
    parser.add_argument(
        "--min-test-epoch",
        type=int,
        default=10,
        help = "test from this epoch"
    )

    parser.add_argument(
        "--restore-file-name",
        type=str,
        default="/dev_huaying/zhijun/data/valle-tensorboard-models/nar/Name_VALLE_max-duration_70_dtype_float32_base-lr_0.01_world-size_8_train-stage_2_echo_150_start_echo_1_2023_05_29_01_26_40/best-valid-loss.pt",
        help="path of restore model checkpoint",
    )
    parser.add_argument(
        "--restore",
        type=str2bool,
        default=False,
        help="restore.",
    )
    parser.add_argument(
        "--input-semantic",
        type=str2bool,
        default=False,
        help="input-semantic",
    )
    parser.add_argument(
        "--semantic-remove",
        type=str2bool,
        default=False,
        help="semantic-remove",
    )
    parser.add_argument(
        "--only-autoregressive",
        type=str2bool,
        default=False,
        help="only-autoregressive",
    )
    parser.add_argument(
        "--semantic-depup",
        type=str2bool,
        default=False,
        help="path of semantic-tokens",
    )
    parser.add_argument(
        "--shared-linear",
        type=str2bool,
        default=False,
        help="shared-linear",
    )
    parser.add_argument(
        "--random-tgt-spk",
        type=str2bool,
        default=False,
        help="ac semantic target spks is random ",
    )
    parser.add_argument(
        "--tgt-spk-name",
        type=str,
        default="cmu_us_bdl_arctic",
        help="path of semantic-tokens",
    )

    parser.add_argument(
        "--semantic-tokens",
        type=str,
        help="path of semantic-tokens",
    )
    parser.add_argument('--test-demo', type=str2bool, default=False)  
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/mnt/zhijun/data/LibriTTS",
        help="audio-source",
    )   
    parser.add_argument(
        "--input-codec",
        type=int,
        default="0",
        help="0->encodec, 1->tfcodec",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/dev_huaying/zhijun/valle_23_4_22/exp"),
        help="Path to the tokenized files.",
    )
    parser.add_argument(
        "--ref-tokens-path",
        type=str,
        default="/mnt/zhijun/Accents/combine_L1_L2/acoustic_tokens_dic/libritts_clean_100_acoustic_semantic.json",
        help="audio-source",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="tfnet/config_6k_tfnetv2_20msvq_hop5_combine4_rd_multi_lingual.yaml",
        help="audio-source",
    )  
    parser.add_argument(
        "--tfnet-ckpt",
        type=str,
        default="/dev_huaying/zhijun/models/tfcodec/890hrs16khz_tfnet_v2i_vqvae_20msVQ_hop5_combine4_rd1_6kbps/tfnet_v2i_vqvae_combineVQ-iter-514000-val1-0.348327-entropy-118.968-mel-0.067548-recon-0.012822.ckpt",
        help="audio-source",
    )  
    args = parser.parse_args()  
    return args


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    args = parser()

    entry_cmd = "python3 -m torch.distributed.run"
    nproc_per_node = "--nproc_per_node={}".format(args.nproc_per_node)
    nnodes = "--nnodes={}".format(args.nnodes)
    entry_py = "egs/libritts/bin/generate_dataset.py"

    entry_cmd = entry_cmd + ' ' + nproc_per_node + ' ' + nnodes + ' ' + entry_py + ' '  + ' ' + nproc_per_node

    
    if args.max_duration is not None:  
        max_duration = "--max-duration={}".format(args.max_duration)  
        entry_cmd += ' ' + max_duration  
    
    if args.filter_min_duration is not None:  
        filter_min_duration = "--filter-min-duration={}".format(args.filter_min_duration)  
        entry_cmd += ' ' + filter_min_duration  
    
    if args.filter_max_duration is not None:  
        filter_max_duration = "--filter-max-duration={}".format(args.filter_max_duration)  
        entry_cmd += ' ' + filter_max_duration  
    
    if args.train_stage is not None:  
        train_stage = "--train-stage={}".format(args.train_stage)  
        entry_cmd += ' ' + train_stage  
    
    if args.num_buckets is not None:  
        num_buckets = "--num-buckets={}".format(args.num_buckets)  
        entry_cmd += ' ' + num_buckets  
    
    if args.dtype is not None:  
        dtype = "--dtype={}".format(args.dtype)  
        entry_cmd += ' ' + dtype  
    
    if args.save_every_n is not None:  
        save_every_n = "--save-every-n={}".format(args.save_every_n)  
        entry_cmd += ' ' + save_every_n  
    
    if args.valid_interval is not None:  
        valid_interval = "--valid-interval={}".format(args.valid_interval)  
        entry_cmd += ' ' + valid_interval  
    if args.log_interval is not None:  
        log_interval = "--log-interval={}".format(args.log_interval)  
        entry_cmd += ' ' + log_interval
    
    if args.model_name is not None:  
        model_name = "--model-name={}".format(args.model_name)  
        entry_cmd += ' ' + model_name  
    
    if args.share_embedding is not None:  
        share_embedding = "--share-embedding={}".format(args.share_embedding)  
        entry_cmd += ' ' + share_embedding  
    
    if args.norm_first is not None:  
        norm_first = "--norm-first={}".format(args.norm_first)  
        entry_cmd += ' ' + norm_first  
    
    if args.add_prenet is not None:  
        add_prenet = "--add-prenet={}".format(args.add_prenet)  
        entry_cmd += ' ' + add_prenet  
    
    if args.decoder_dim is not None:  
        decoder_dim = "--decoder-dim={}".format(args.decoder_dim)  
        entry_cmd += ' ' + decoder_dim  
    
    if args.nhead is not None:  
        nhead = "--nhead={}".format(args.nhead)  
        entry_cmd += ' ' + nhead  
    
    if args.num_decoder_layers is not None:  
        num_decoder_layers = "--num-decoder-layers={}".format(args.num_decoder_layers)  
        entry_cmd += ' ' + num_decoder_layers  
    
    if args.prefix_mode is not None:  
        prefix_mode = "--prefix-mode={}".format(args.prefix_mode)  
        entry_cmd += ' ' + prefix_mode  
    
    if args.base_lr is not None:  
        base_lr = "--base-lr={}".format(args.base_lr)  
        entry_cmd += ' ' + base_lr  
    
    if args.warmup_steps is not None:  
        warmup_steps = "--warmup-steps={}".format(args.warmup_steps)  
        entry_cmd += ' ' + warmup_steps  
    
    if args.average_period is not None:  
        average_period = "--average-period={}".format(args.average_period)  
        entry_cmd += ' ' + average_period  
    
    if args.num_epochs is not None:  
        num_epochs = "--num-epochs={}".format(args.num_epochs)  
        entry_cmd += ' ' + num_epochs  
    
    if args.start_epoch is not None:  
        start_epoch = "--start-epoch={}".format(args.start_epoch)  
        entry_cmd += ' ' + start_epoch  
    if args.num_quantizers is not None:  
        num_quantizers = "--num-quantizers={}".format(args.num_quantizers)  
        entry_cmd += ' ' + num_quantizers  
    if args.sheduler_steps is not None:  
        sheduler_steps = "--sheduler-steps={}".format(args.sheduler_steps)  
        entry_cmd += ' ' + sheduler_steps 
    if args.sheduler_epochs is not None:  
        sheduler_epochs = "--sheduler-epochs={}".format(args.sheduler_epochs)  
        entry_cmd += ' ' + sheduler_epochs 
    if args.start_batch is not None:  
        start_batch = "--start-batch={}".format(args.start_batch)  
        entry_cmd += ' ' + start_batch  
    
    if args.accumulate_grad_steps is not None:  
        accumulate_grad_steps = "--accumulate-grad-steps={}".format(args.accumulate_grad_steps)  
        entry_cmd += ' ' + accumulate_grad_steps  
    
    if args.manifest_dir is not None:  
        manifest_dir = "--manifest-dir={}".format(args.manifest_dir)  
        entry_cmd += ' ' + manifest_dir  
    
    if args.text_tokens is not None:  
        text_tokens = "--text-tokens={}".format(args.text_tokens)  
        entry_cmd += ' ' + text_tokens  
    
    if args.newfile_suffix is not None:  
        newfile_suffix = "--newfile-suffix={}".format(args.newfile_suffix)  
        entry_cmd += ' ' + newfile_suffix  
    
    if args.world_size is not None:  
        world_size = "--world-size={}".format(args.world_size)  
        entry_cmd += ' ' + world_size  
    
    if args.exp_dir is not None:  
        exp_dir = "--exp-dir={}".format(args.exp_dir)  
        entry_cmd += ' ' + exp_dir  

    if args.is_local is not None:  
        exp_dir = "--is-local={}".format(args.is_local)  
        entry_cmd += ' ' + exp_dir 
        
    if args.checkpoint_ar is not None:  
        checkpoint_ar = "--checkpoint-ar={}".format(args.checkpoint_ar)  
        entry_cmd += ' ' + checkpoint_ar  
    
    if args.text_extractor is not None:  
        text_extractor = "--text-extractor={}".format(args.text_extractor)  
        entry_cmd += ' ' + text_extractor  
    
    if args.checkpoint_nar is not None:  
        checkpoint_nar = "--checkpoint-nar={}".format(args.checkpoint_nar)  
        entry_cmd += ' ' + checkpoint_nar  
    
    if args.text_prompts is not None:  
        text_prompts = "--text-prompts {}".format(" ".join(f'"{x}"' for x in args.text_prompts))  
        entry_cmd += ' ' + text_prompts  
    
    if args.audio_prompts is not None:  
        audio_prompts = "--audio-prompts {}".format(" ".join(f'"{x}"' for x in args.audio_prompts))  
        entry_cmd += ' ' + audio_prompts  
    
    if args.text is not None:  
        text = "--text {}".format(" ".join(f'"{x}"' for x in args.text))  
        entry_cmd += ' ' + text  

    
    if args.min_test_epoch is not None:  
        min_test_epoch = "--min-test-epoch={}".format(args.min_test_epoch)  
        entry_cmd += ' ' + min_test_epoch  
    
    if args.restore_file_name is not None:
        restore_file_name = "--restore-file-name={}".format(args.restore_file_name)  
        entry_cmd += ' ' + restore_file_name
    if args.restore is not None:
        restore = "--restore={}".format(args.restore)  
        entry_cmd += ' ' + restore
    if args.input_semantic is not None:
        restore = "--input-semantic={}".format(args.input_semantic)  
        entry_cmd += ' ' + restore
    if args.semantic_remove is not None:
        restore = "--semantic-remove={}".format(args.semantic_remove)  
        entry_cmd += ' ' + restore   

    if args.only_autoregressive is not None:
        restore = "--only-autoregressive={}".format(args.only_autoregressive)  
        entry_cmd += ' ' + restore
    if args.semantic_depup is not None:
        restore = "--semantic-depup={}".format(args.semantic_depup)  
        entry_cmd += ' ' + restore
    if args.semantic_tokens is not None:
        restore = "--semantic-tokens={}".format(args.semantic_tokens)  
        entry_cmd += ' ' + restore    
    if args.semantic_tokens is not None:
        restore = "--shared-linear={}".format(args.shared_linear)  
        entry_cmd += ' ' + restore 
    if args.random_tgt_spk is not None:
        restore = "--random-tgt-spk={}".format(args.random_tgt_spk)  
        entry_cmd += ' ' + restore 
    if args.tgt_spk_name is not None:
        restore = "--tgt-spk-name={}".format(args.tgt_spk_name)  
        entry_cmd += ' ' + restore 
    if args.top_k is not None:
        restore = "--top-k={}".format(args.top_k)  
        entry_cmd += ' ' + restore 
    if args.checkpoint is not None:
        restore = "--checkpoint={}".format(args.checkpoint)  
        entry_cmd += ' ' + restore 
    if args.input_codec is not None:
        restore = "--input-codec={}".format(args.input_codec)  
        entry_cmd += ' ' + restore 
    if args.output_dir is not None:
        restore = "--output-dir={}".format(args.output_dir)  
        entry_cmd += ' ' + restore 
    if args.ref_tokens_path is not None:
        restore = "--ref-tokens-path={}".format(args.ref_tokens_path)  
        entry_cmd += ' ' + restore 
    if args.config_path is not None:
        restore = "--config-path={}".format(args.config_path)  
        entry_cmd += ' ' + restore 
    if args.tfnet_ckpt is not None:
        restore = "--tfnet-ckpt={}".format(args.tfnet_ckpt)  
        entry_cmd += ' ' + restore 


    print(f"entry_cmd : {entry_cmd}")
    os.system(entry_cmd)  



    