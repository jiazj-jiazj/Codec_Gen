import os
import argparse

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
    parser.add_argument('--nproc-per-node', type=int, default=1)  
    parser.add_argument('--nnodes', type=int, default=1)  
    parser.add_argument('--max-duration', type=int, default=20)  
    parser.add_argument('--filter-min-duration', type=float, default=0.5)  
    parser.add_argument('--filter-max-duration', type=float, default=14)  
    parser.add_argument('--train-stage', type=int, default=1)  
    parser.add_argument('--num-buckets', type=int, default=6)  
    parser.add_argument('--dtype', type=str, default="float16")  
    parser.add_argument('--save-every-n', type=int, default=100)  
    parser.add_argument('--model-name', type=str, default="valle")  
    parser.add_argument('--share-embedding', type=str2bool, default=True)  
    parser.add_argument('--norm-first', type=str2bool, default=True)  
    parser.add_argument('--add-prenet', type=str2bool, default=False)  
    parser.add_argument('--decoder-dim', type=int, default=1024)  
    parser.add_argument('--nhead', type=int, default=16)  
    parser.add_argument('--num-decoder-layers', type=int, default=12)  
    parser.add_argument('--prefix-mode', type=int, default=1)  
    parser.add_argument('--num-quantizers', type=int, default=8)
    parser.add_argument('--semantic-type', type=int, default=0)
    parser.add_argument('--manifest-dir', type=str)  
    parser.add_argument('--text-tokens', type=str)  
    parser.add_argument('--world-size', type=int, default=2)  
    parser.add_argument('--exp-dir', type=str)  
    parser.add_argument('--is-local', type=str2bool, default=False)  
    parser.add_argument('--num-workers', type=int, default=20)  
    parser.add_argument('--encoder-num-layers', type=int, default=6)  
    parser.add_argument('--decoder-num-layers', type=int, default=6)  

    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/dev_huaying/zhijun/data/valle-tensorboard-models/nar/Name_VALLE_max-duration_70_dtype_float32_base-lr_0.01_world-size_8_train-stage_2_echo_150_start_echo_1_2023_05_29_01_26_40/best-valid-loss.pt",
        help="path of nar_model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/dev_huaying/zhijun/data/valle-tensorboard-models/nar/Name_VALLE_max-duration_70_dtype_float32_base-lr_0.01_world-size_8_train-stage_2_echo_150_start_echo_1_2023_05_29_01_26_40/best-valid-loss.pt",
        help="path of nar_model checkpoint",
    )
    parser.add_argument(
        "--train-dir-name",
        type=str,
        default="cuts_train.jsonl.gz",
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
        "--task-id",
        type=int,
        default=0
    )
    parser.add_argument(
        "--text",
        nargs="+",
        type=str,
        default=["To get up and running quickly just follow the steps below.", "say i believe in you, and you also believe in me"],
        help="Text to be synthesized.",
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
        "--prepend-bos",
        type=str2bool,
        default=False,
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
        "--semantic-tokens",
        type=str,
        help="path of semantic-tokens",
    )
    parser.add_argument(
        "--sec-dataset",
        type=str2bool,
        default=False,
        help="only one linear no matter how num-quantizers",
    )
    parser.add_argument(
        "--manifest-dir-sec",
        type=str,
        default="/scratch/data/LibriTTS/vc_tokenized_16k_tfcodec_16codes",
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
    entry_py = "egs/indian_accent_english/bin_commit/generate_tokens.py"

    entry_cmd = entry_cmd + ' ' + nproc_per_node + ' ' + nnodes + ' ' + entry_py + ' '  + ' ' 

    
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
    
    if args.num_quantizers is not None:  
        num_quantizers = "--num-quantizers={}".format(args.num_quantizers)  
        entry_cmd += ' ' + num_quantizers  
    if args.manifest_dir is not None:  
        manifest_dir = "--manifest-dir={}".format(args.manifest_dir)  
        entry_cmd += ' ' + manifest_dir  
    
    if args.text_tokens is not None:  
        text_tokens = "--text-tokens={}".format(args.text_tokens)  
        entry_cmd += ' ' + text_tokens  
     
    if args.world_size is not None:  
        world_size = "--world-size={}".format(args.world_size)  
        entry_cmd += ' ' + world_size  

    if args.is_local is not None:  
        exp_dir = "--is-local={}".format(args.is_local)  
        entry_cmd += ' ' + exp_dir 
    
    if args.text_extractor is not None:  
        text_extractor = "--text-extractor={}".format(args.text_extractor)  
        entry_cmd += ' ' + text_extractor  
    
    if args.checkpoint is not None:  
        checkpoint = "--checkpoint={}".format(args.checkpoint)  
        entry_cmd += ' ' + checkpoint
    
    if args.input_semantic is not None:
        restore = "--input-semantic={}".format(args.input_semantic)  
        entry_cmd += ' ' + restore

    if args.only_autoregressive is not None:
        restore = "--only-autoregressive={}".format(args.only_autoregressive)  
        entry_cmd += ' ' + restore

    if args.semantic_tokens is not None:
        restore = "--semantic-tokens={}".format(args.semantic_tokens)  
        entry_cmd += ' ' + restore    
    if args.shared_linear is not None:
        restore = "--shared-linear={}".format(args.shared_linear)  
        entry_cmd += ' ' + restore 

    if args.sec_dataset is not None:
        restore = "--sec-dataset={}".format(args.sec_dataset)  
        entry_cmd += ' ' + restore 
    if args.manifest_dir_sec is not None:
        restore = "--manifest-dir-sec={}".format(args.manifest_dir_sec)  
        entry_cmd += ' ' + restore 
    if args.semantic_type is not None:
        restore = "--semantic-type={}".format(args.semantic_type)  
        entry_cmd += ' ' + restore 
    if args.train_dir_name is not None:
        restore = "--train-dir-name={}".format(args.train_dir_name)  
        entry_cmd += ' ' + restore 
    if args.num_workers is not None:
        restore = "--num-workers={}".format(args.num_workers)  
        entry_cmd += ' ' + restore    
    if args.prepend_bos is not None:
        restore = "--prepend-bos={}".format(args.prepend_bos)  
        entry_cmd += ' ' + restore  
    if args.output_dir is not None:
        restore = "--output-dir={}".format(args.output_dir)  
        entry_cmd += ' ' + restore 

    os.system(entry_cmd)  



    