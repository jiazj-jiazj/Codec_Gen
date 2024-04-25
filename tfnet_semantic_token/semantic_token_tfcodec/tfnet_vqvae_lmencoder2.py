import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
# from tfnet_models_mp.tfnet_v2i_vqvae_disentangle import TFNet as TFCodec_Disen
from tfnet_semantic_token.tfnet_models_mp.tfnet_v2i_vqvae_disentangle_org import TFNet as TFCodec_Disen_org
# from tfnet_models_mp.tfnet_v2i_vqvae import TFNet as TFCodec
from tfnet_semantic_token.tfnet_models_mp.layers.vq_layer_gumbel import GumbelVectorQuantizer
from tfnet_semantic_token.utils.tools import *
from .unit_lm import Transformer
from .audiolm_module import LayerNorm
# from typing import Dict
import yaml

def is_disentangle(config):
    if ('disentangle' in config['model_type']) or (
            ('_lm' in config['model_type']) and ('disentangle' in config['lm_encoder_type'])):
        return True
    return False


def exists(val):
    return val is not None
TRANSFORMER = {'LARGE': {'depth': 24, 'atten_head': 16, 'dim': 1024},
               'BASE': {'depth': 12, 'atten_head': 8, 'dim': 768},
               'SMALL': {'depth': 6, 'atten_head': 6, 'dim': 512},
               }


def load_pretrained_TFCodec(model, ckpt_path):
    new_dict = {}
    if ckpt_path is not None:
        tmp_dict = torch.load(ckpt_path, map_location='cpu')
        tmp_dict2 = tmp_dict["gen"] if 'gen' in tmp_dict.keys() else tmp_dict
        print('keys to load:{}'.format(len(tmp_dict2.keys())))
        for key in tmp_dict2.keys():
            new_key = key.split('module.')[-1]
            if 'generator.' in new_key:
                new_key = new_key.split('generator.')[-1]
            new_dict[new_key] = tmp_dict2[key]

            # model.load_state_dict(new_dict, strict=True)
    new_model_dict = model.state_dict()
    print('current model keys:{}'.format(len(new_model_dict.keys())))
    # filter out unnecessary keys
    new_dict_opt = {k: v for k, v in new_dict.items() if k in new_model_dict}
    print('keys loaded :{}'.format(len(new_dict_opt.keys())))
    new_model_dict.update(new_dict_opt)
    model.load_state_dict(new_model_dict)  # , strict=False)


def set_disen_model(config, bn):
    if config['disen_model_type'] == 'org':  # 'org', 'v1', 'transformer'
        model = TFCodec_Disen_org(config=config, bn=bn)
    else:
        model = TFCodec_Disen(config=config, bn=bn)
    return model

class TFNet_lmencoder(nn.Module):
    def __init__(self, config=None, bn=True, args=None):
        super(TFNet_lmencoder, self).__init__()
        self.config = config
        encoder_model_dir = os.path.join(config['pretrained_model_root'],os.path.dirname(config['lm_enc_pretrain_path']))
        print("Using config from ", os.path.join(encoder_model_dir, 'config.yaml'))
        with open(os.path.join(encoder_model_dir, 'config.yaml')) as f:
            encoder_config = yaml.safe_load(f)
        encoder_config['pretrained_model_root'] = config['pretrained_model_root']
        self.use_semantic_encoder = False
        self.use_folded_input = 1
        if config['lm_encoder_type'] == 'tfnet_v2i_vqvae_disentangle':
            self.combine_frames = config['combineVQ_frames']
            self.encoder_model = set_disen_model(encoder_config,bn=bn)
            # self.encoder_model = TFCodec_Disen_org(encoder_config, bn=bn) if encoder_config['disen_model_type'] == 'org' else TFCodec_Disen(
            #     encoder_config, bn=bn)
            load_pretrained_TFCodec(self.encoder_model,os.path.join(config['pretrained_model_root'], config['lm_enc_pretrain_path']))
            self.hop_len = self.encoder_model.hop_len
            self.config['fold_mode'] = '2' ##?
            num_frames_vq_in = encoder_config['combineVQ_frames'] if self.use_folded_input else 1
            self.feat_dim = self.set_fold_mode(self.config,num_frames_vq_in) ## fold phn feature  10ms->40ms
            self.set_fusion_mode(config, num_frames_vq_in)
            self.transformer_dim = self.feat_dim
        elif config['lm_encoder_type'] == 'tfnet_v2i_vqvae':
            self.encoder_model = TFCodec(encoder_config, bn=bn)
            self.condition_feat_dim = self.encoder_model.feat_dim  # after bottleneck 1x1
            load_pretrained_TFCodec(self.encoder_model,os.path.join(config['pretrained_model_root'], config['lm_enc_pretrain_path']))
            self.transformer_dim = self.encoder_model.feat_dim

        elif config['lm_encoder_type'] in ('tfnet_v2i_vqvae_lm','tfnet_v2i_vqvae_lm2','tfnet_v2i_vqvae_lm3'):  ## transformer code from audiolm/fairseq/huggingface
            self.use_semantic_encoder = True
            self.combine_frames = config['semantic_combineVQ_frames']
            self.encoder_model = set_disen_model(encoder_config, bn=bn)
            self.hop_len = self.encoder_model.hop_len
            # load_pretrained_TFCodec(self.encoder_model,os.path.join(config['pretrained_model_root'], config['lm_enc_pretrain_path']))
            load_pretrained_TFCodec(self.encoder_model, config['pretrained_model_ckpt_path'])

            self.feat_dim = self.set_fold_mode(config, self.combine_frames) ## fold phn feature  10ms->40ms
            self.set_fusion_mode(config, self.combine_frames)
            self.transformer_enc, self.transformer_dim = self.set_TransformerEnc(config)

            self.enc_feature_projection1 = nn.Linear(self.feat_dim, self.transformer_dim)
            nn.init.xavier_normal_(self.enc_feature_projection1.weight)
            if config['lm_decoder_type'] in ('LM+TFNet','TFNet'):
                self.enc_feature_projection2 = nn.Linear(self.transformer_dim, self.feat_dim)
                nn.init.xavier_normal_(self.enc_feature_projection2.weight)


        if config['src_ASR']:
            # vocab_size = 33 if not config['use_langIDs'] else 136
            if config['s2s_data'] == 'timit':
                vocab_size = 42
            elif 'librispeech' in config['s2s_data']:
                vocab_size = 33
            if config['use_lstm_for_asr']:
                hidden_size = 1024
                self.bilstm = nn.LSTM(self.transformer_dim, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
                self.lm_head_enc = nn.Linear(hidden_size*2, vocab_size)
            else:
                self.lm_head_enc = nn.Linear(self.transformer_dim, vocab_size)


        self.init_codebook_size(config)

        if config['semantic_tokenize']:
            if config['fold_mode'] == '0':
                self.semantic_codebook_dim = int(self.feat_dim * self.combine_frames // self.semantic_codebook_num)
            else:
                self.semantic_codebook_dim = int(self.feat_dim// self.semantic_codebook_num)

            self.semantic_vq_layer_list = nn.ModuleList(
                GumbelVectorQuantizer(config, input_dim=self.semantic_codebook_dim, n_embeddings=i,
                                      groups=config["groups"], combine_groups=config["combine_groups"], ) for i in self.semantic_codebook_size)

        if config["use_entropy_loss"]:
            if config['semantic_tokenize']:
                self.set_network_entropy_target(config["bitrate"], config["entropy_fuzz"], config['vq_in_dur'])
            else:
                self.encoder_model.set_network_entropy_target(encoder_config["bitrate"], encoder_config["entropy_fuzz"],encoder_config["sampling_rate"],self.encoder_model.hop_len)

        self.freeze_module(config)

    def AdaIN_merge(self, phn_feat, spk_feat):  # B,T,C
        phn_feat = self.phn_norm(self.pre_merge(phn_feat))
        scale = self.scale_layers[1](self.ada_norm(self.scale_layers[0](spk_feat)))
        bias = self.bias_layers[1](self.ada_norm(self.bias_layers[0](spk_feat)))
        return phn_feat * scale + bias

    def set_fold_mode(self,config,num_frames_vq_in):
        if config['fold_mode'] == '0':
            feat_dim = self.encoder_model.phn_latent_dim
        elif config['fold_mode'] == '4':
            feat_dim = self.encoder_model.phn_latent_dim
            self.temporal_downsample1 = nn.Conv1d(in_channels=self.encoder_model.phn_latent_dim,
                                                  out_channels=self.encoder_model.phn_latent_dim, kernel_size=4,
                                                  stride=4)
            # self.temporal_downsample2 = nn.Conv1d(in_channels=self.encoder_model.phn_latent_dim, out_channels=self.encoder_model.phn_latent_dim, kernel_size=2, stride=1)
            nn.init.xavier_normal_(self.temporal_downsample1.weight)
            # nn.init.xavier_normal_(self.temporal_downsample2.weight)
        else:
            feat_dim = self.encoder_model.phn_latent_dim * num_frames_vq_in
        return feat_dim

    def set_fusion_mode(self,config,num_frames_vq_in):
        # self.fusion_type = config['fusion_type'] if 'fusion_type' in config.keys() else 'Concat'
        self.fusion_type = config.get('fusion_type','Concat')
        if self.fusion_type == 'AdaIN':
            if config['lm_encoder_type'] == 'text_embs_with_spk':
                phn_in_c = 512  # t5 embs dimension
            else:
                phn_in_c = self.encoder_model.phn_latent_dim *num_frames_vq_in
            phn_out_c = 768
            self.pre_merge = nn.Linear(phn_in_c, phn_out_c)
            self.phn_norm = LayerNorm(phn_out_c)
            self.scale_layers = nn.ModuleList(
                (nn.Linear(self.encoder_model.spk_embedding_dim, phn_in_c), nn.Linear(phn_in_c, phn_out_c)))
            self.bias_layers = nn.ModuleList(
                (nn.Linear(self.encoder_model.spk_embedding_dim, phn_in_c), nn.Linear(phn_in_c, phn_out_c)))
            self.ada_norm = nn.LeakyReLU(0.1, inplace=True)
            self.condition_feat_dim = phn_out_c
        elif self.fusion_type == 'Concat':
            self.condition_feat_dim = self.feat_dim + self.encoder_model.spk_embedding_dim
        else:
            self.condition_feat_dim = self.encoder_model.phn_latent_dim *num_frames_vq_in


    def set_TransformerEnc(self, config):
        if config['lm_encoder_type'] == 'tfnet_v2i_vqvae_lm':
            dim = TRANSFORMER[config['transformer_enc']]['dim']
            transformer_enc = Transformer(
                dim=TRANSFORMER[config['transformer_enc']]['dim'],
                depth=TRANSFORMER[config['transformer_enc']]['depth'],
                heads=TRANSFORMER[config['transformer_enc']]['atten_head'],
                attn_dropout=self.config['attn_dropout'],
                cross_attend=False,
                cond_as_self_attn_prefix=False,
                codec_config=self.config,
            )
        elif config['lm_encoder_type'] == 'tfnet_v2i_vqvae_lm2':
            from .transformer_module import TransformerEncoder
            from fairseq.models.hubert.hubert import HubertConfig
            enc_cfg = HubertConfig()
            enc_cfg_overides = {}
            enc_type = config['transformer_enc']  ## SMALL/BASE/LARGE
            for key, value in config['transformer_enc_cfg'][enc_type].items():
                enc_cfg_overides[key] = value
            for key, value in enc_cfg_overides.items():
                if hasattr(enc_cfg, key):
                    setattr(enc_cfg, key, value)
            transformer_enc = TransformerEncoder(enc_cfg)
            dim = enc_cfg.encoder_embed_dim
            self.output_layer = None

        elif config['lm_encoder_type'] == 'tfnet_v2i_vqvae_lm3':
            from transformers import AutoProcessor, HubertForCTC
            model = HubertForCTC.from_pretrained("facebook/hubert-base-ls960")
            transformer_enc = model.hubert.encoder
            self.output_layer = None
            dim = 768
        return transformer_enc, dim
    def freeze_module(self,config):
        if config['lm_encoder_type'] in ('tfnet_v2i_vqvae_lm','tfnet_v2i_vqvae_lm2','tfnet_v2i_vqvae_lm3'):
            self.freeze_codec(self.encoder_model)  ##
            if config['freeze_lmencoder']:
                self.freeze_codec(self.enc_feature_projection1)
                self.freeze_codec(self.transformer_enc)
                if 'tune_TransEnc_layer' in config and config['tune_TransEnc_layer']:
                    # for param in self.transformer_enc.layers[-2:].parameters():
                    #     param.requires_grad = True
                    self.unfreeze_codec(self.transformer_enc.layers[-2:])
            if config['freeze_semantic_tokenizer']:
                self.freeze_codec(self.enc_feature_projection2)  ##
                self.freeze_codec(self.semantic_vq_layer_list)  ##

        if config['lm_encoder_type'] in ('tfnet_v2i_vqvae_disentangle','tfnet_v2i_vqvae'):
            if config['freeze_lmencoder']:
                # self.freeze_codec(self.enc_feature_projection1)
                self.freeze_codec(self.encoder_model)

    def init_codebook_size(self, config):
        if config['bitrate'] == '0.128k':
            if config["semantic_combineVQ_frames"] == 4:  # 5.12 bit per 40ms
                self.semantic_codebook_num = 1
                self.semantic_codebook_size = [256,]  # redundant: 8 bit per 40ms
        elif config['bitrate'] == '0.25k':
            if config["semantic_combineVQ_frames"] == 4:  # 10 bit per 40ms
                self.semantic_codebook_num = 1
                self.semantic_codebook_size = [1024,]  # complete: 10 bit per 40ms
        elif config['bitrate'] == '0.256k':
            if config["semantic_combineVQ_frames"] == 4:  # 10.24 bit per 40ms
                if self.config.get("semantic_1codebook",False):
                    self.semantic_codebook_num = 1
                    self.semantic_codebook_size = [4096]  # redundant: 12 bit per 40ms
                else:
                    self.semantic_codebook_num = 2
                    self.semantic_codebook_size = [256, 256]  # redundant: 16 bit per 40ms
            elif config["semantic_combineVQ_frames"] == 2:  # 5.12 bit per 20ms
                self.semantic_codebook_num = 1
                self.semantic_codebook_size = [256,]  # redundant: 8 bit per 20ms

    def freeze_codec(self, model):
        for name, module in model.named_children():
            for k, para in module.named_parameters():
                para.requires_grad = False
        for name, param in model.named_parameters():
            param.requires_grad = False

    def unfreeze_codec(self, model):
        for name, module in model.named_children():
            for k, para in module.named_parameters():
                para.requires_grad = True
        for name, param in model.named_parameters():
            param.requires_grad = True

    def load_input(self, input_list):
        self.source_signal = None
        if isinstance(input_list,dict):
            if 'source_signal' in input_list.keys():
                self.source_signal = input_list['source_signal']
            elif 'signal_clean' in input_list.keys():
                self.source_signal = input_list['signal_clean']
            self.signal = self.source_signal
            # self.source_signal = input_list['source_signal'] if 'source_signal' in input_list.keys() else None
            self.target_signal = input_list['target_signal'] if 'target_signal' in input_list.keys() else None
            self.src_seq_len = input_list['source_lens'] if 'source_lens' in input_list.keys() else None
            self.tgt_seq_len = input_list['target_lens'] if 'target_lens' in input_list.keys() else None
            self.transcript = input_list['transcript'] if 'transcript' in input_list.keys() else None
            self.trans_len = input_list['trans_len'] if 'trans_len' in input_list.keys() else None
            self.bitrate = input_list['bitrate'] if 'bitrate' in input_list.keys() else self.config['bitrate']
            if 'source_mask' in input_list.keys():
                self.source_mask = input_list['source_mask']
            elif exists(self.source_signal):
                self.source_mask = torch.ones_like(self.source_signal).bool()
            else:
                self.source_mask = None
            self.target_mask = input_list['target_mask'] if 'target_mask' in input_list.keys() else None
            self.source_text = input_list['source_text'] if 'source_text' in input_list.keys() else None
            self.target_text = input_list['target_text'] if 'target_text' in input_list.keys() else None
            self.target_lang = input_list['tgt_lang'] if 'tgt_lang' in input_list.keys() else None
            self.source_lang = input_list['src_lang'] if 'src_lang' in input_list.keys() else None
            # self.sour
        else:
            self.source_signal = input_list
            self.source_text = None
            self.signal = self.source_signal
            b, t1 = self.source_signal.size()
            self.source_mask = torch.full((b, t1),1).to(self.source_signal.device)

    def forward_padding_mask(self,features: torch.Tensor,padding_mask: torch.Tensor,) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        ones = torch.sum(padding_mask,dim=-1)
        last_valid_index = ones - 1
        return padding_mask, last_valid_index

    def ctc_loss(self,logits,text,mask):
        labels = text
        # retrieve loss input_lengths from attention_mask
        attention_mask = (
            mask if mask is not None else torch.ones_like(logits, dtype=torch.long)
        )
        # input_lengths = self.wav2vec2._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long) ## ?
        input_lengths = (attention_mask > 0).sum(-1)
        # assuming that padded tokens are filled with -100
        # when not being attended to
        labels_mask = labels > 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = labels.masked_select(labels_mask)

        # ctc_loss doesn't support fp16
        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = nn.functional.ctc_loss(log_probs,flattened_targets,input_lengths,target_lengths,blank=0,reduction='mean',zero_infinity=True,)
        return loss


    def fold_enc_feat(self, feat, num_frames_vq_in):
        if self.config['fold_mode'] == '0': #do not fold
            phn_vq_in = feat
        elif self.config['fold_mode'] == '1':
            B, T, channels = feat.shape
            phn_vq_in = feat.reshape(B, T // num_frames_vq_in, num_frames_vq_in, channels)
            phn_vq_in = phn_vq_in.permute(0, 1, 3, 2).reshape(B, T // num_frames_vq_in, -1)
        elif self.config['fold_mode'] == '2':
            B, T = feat.shape[0], feat.shape[1]  # B,T, C
            phn_vq_in = feat.reshape(B, T // num_frames_vq_in, -1)  # B,T,C
        elif self.config['fold_mode'] == '3':
            B, T, channels =feat.shape  # B,T, C
            phn_vq_in = feat.reshape(B, T // num_frames_vq_in, num_frames_vq_in, -1)  # B,T,C
            enc_feat_combine = torch.split(phn_vq_in, channels // self.semantic_codebook_num,dim=-1)  # codebook_num,B,T/2,2,C'//codebook_num
            enc_feat_combine = torch.stack(enc_feat_combine)
            enc_feat_combine = enc_feat_combine.reshape(self.semantic_codebook_num, B, T // num_frames_vq_in, -1)
            enc_feat_combine = enc_feat_combine.permute(1, 2, 0, 3)
            phn_vq_in = enc_feat_combine.reshape(B, T // num_frames_vq_in, -1)
        elif self.config['fold_mode'] == '4':
            phn_vq_in = feat.permute(0, 2, 1)
            phn_vq_in = self.temporal_downsample1(phn_vq_in).permute(0, 2, 1)
            # phn_vq_in = self.temporal_downsample1(phn_vq_in).permute(0,2,1)
        return phn_vq_in


    def forward(self, input_list, enc_only=False):
        result = {}
        result["bitrate"] = self.config['bitrate']
        self.load_input(input_list)

        if self.config['lm_encoder_type'] == 'tfnet_v2i_vqvae_disentangle':
            if self.config['freeze_lmencoder']:
                with torch.no_grad():
                    self.encoder_model.eval()
                    vq_out = self.encoder_model.encode_to_token_idx(self.signal)

            else:
                vq_out = self.encoder_model.encode_to_token_idx(self.signal)
            self.pad_len = self.encoder_model.pad_len
            phn_feat_vq_out = self.fold_enc_feat(vq_out["phn_feat_vq"].permute(0, 2, 1),self.encoder_model.combine_frames)
            if self.config['src_ASR']:
                if self.config['use_lstm_for_asr']:
                    semantic_feat2, _ = self.bilstm(phn_feat_vq_out)
                logits = self.lm_head_enc(semantic_feat2)
                result.update({'asr_logits': logits})
                if self.source_text is not None:
                    self.source_mask, _ = self.forward_padding_mask(semantic_feat2, self.source_mask)  ## todo: align bug
                    ctcloss = self.ctc_loss(logits, self.source_text, self.source_mask)
                    result['src_ctc_loss'] = ctcloss

            vq_out_feat = self.fuse_semantic_and_speaker(phn_feat_vq_out, vq_out['spk_feat_vq'])
            spk_vq_out = vq_out['spk_feat_vq'].permute(0, 2, 1)
            result.update(vq_out)
            result.update({'lmencoder_vq_in':None})
            result.update({'lmencoder_vq_feat':vq_out_feat})
            result.update({'spk_feat':vq_out["spk_feat_vq"]}) #todo
            return result
        elif self.config['lm_encoder_type'] == 'tfnet_v2i_vqvae':
            with torch.no_grad():
                self.encoder_model.eval()
                vq_out = self.encoder_model.encode_to_token_idx(self.signal)
            self.pad_len = self.encoder_model.pad_len
            vq_out_feat = vq_out["vq_feat"].unsqueeze(-1).permute(0, 2, 1)  # B, C, T, 1 -> B, T, C
            num_frames_in_group = self.encoder_config['combineVQ_frames']
            B, T = vq_out_feat.shape[0], vq_out_feat.shape[1]
            vq_out_feat = vq_out_feat.reshape(B, T // num_frames_in_group, -1)
            result.update(vq_out)
            result.update({'lmencoder_vq_in':None})
            result.update({'lmencoder_vq_feat':vq_out_feat})
            result.update({'spk_feat':vq_out["spk_feat_vq"]}) #todo

        elif self.config['lm_encoder_type'] in ('tfnet_v2i_vqvae_lm','tfnet_v2i_vqvae_lm2','tfnet_v2i_vqvae_lm3'):
            with torch.no_grad():
                self.encoder_model.eval()
                vq_out = self.encoder_model.encode_to_token_idx(self.signal)
                result.update(vq_out)
            self.pad_len = self.encoder_model.pad_len
            phn_vq_in = self.fold_enc_feat(vq_out["phn_feat_vq_in"].permute(0,2,1),self.combine_frames)
            feat = self.enc_feature_projection1(phn_vq_in)
            unmasked_features = feat.clone() ## for contrastive learning

            if self.source_mask is None:
                b, t1 = self.signal.size()
                self.source_mask = torch.full((b, t1),1).to(self.signal.device)

            self.source_mask, _ = self.forward_padding_mask(feat, self.source_mask)  ## todo: align bug

            if self.config['freeze_lmencoder']:
                self.transformer_enc.eval()
                if self.training:
                    if 'tune_TransEnc_layer' in self.config and self.config['tune_TransEnc_layer']:
                        for name, module in self.transformer_enc.layers[-2:].named_children():
                                module.train()
            #here 
            # self.transformer_enc.load_state_dict(torch.load("/home/v-zhijunjia/data/valle-tensorboard-models/other_models/tfnet_semantic_tokens/semantic_token_resynt/tfcodec_256bps_disen/tfnetv2_vqvae_lm2-val-1-loss-4.646021-vq-0.199693-iter-716000.ckpt")) 
            if self.config['lm_encoder_type'] == 'tfnet_v2i_vqvae_lm':
                semantic_feat = self.transformer_enc(x=feat, self_attn_mask=self.source_mask)
            elif self.config['lm_encoder_type'] == 'tfnet_v2i_vqvae_lm2':
                semantic_feat, _ = self.transformer_enc(x=feat,padding_mask=~self.source_mask,layer=None if self.output_layer is None else self.output_layer - 1,)
            elif self.config['lm_encoder_type'] == 'tfnet_v2i_vqvae_lm3':
                semantic_feat = self.transformer_enc(feat, attention_mask=self.source_mask, output_attentions=False,output_hidden_states=False, return_dict=True, )[0]


            if self.config['src_ASR']:
                if self.config['use_lstm_for_asr']:
                    semantic_feat2, _ = self.bilstm(semantic_feat)
                logits = self.lm_head_enc(semantic_feat2)
                result.update({'asr_logits': logits})
                if self.source_text is not None:
                    ctcloss = self.ctc_loss(logits, self.source_text, self.source_mask)
                    result['src_ctc_loss'] = ctcloss


            vq_out_feat = None
            semantic_vq_out_feat = None
            if self.config['lm_decoder_type'] in ('LM+TFNet','TFNet'): ## use decoder
                semantic_feat = self.enc_feature_projection2(semantic_feat)
                if self.config['semantic_tokenize']:
                    semantic_vq_out = self.semantic_quantize(semantic_feat.permute(0, 2, 1))
                    B, T = semantic_vq_out["feat_vq"].shape[0], semantic_vq_out["feat_vq"].shape[2]  # B, C, T
                    semantic_vq_out_feat = semantic_vq_out['feat_vq'].permute(0, 2, 1)
                    result['quantization_inds'] = semantic_vq_out['quantization_inds']
                    result.update(semantic_vq_out)
                else:
                    semantic_vq_out_feat = semantic_feat
                    B, T = semantic_vq_out_feat.shape[0], semantic_vq_out_feat.shape[1]  # B, C, T
                vq_out_feat = self.fuse_semantic_and_speaker(semantic_vq_out_feat, vq_out['spk_feat_vq'])
            result.update({'semantic_vq_feat':semantic_vq_out_feat})

            result.update({'lmencoder_vq_feat':vq_out_feat})
            result.update({'spk_feat':vq_out['spk_feat_vq']}) #todo
            return result

    def fuse_semantic_and_speaker(self,semantic_feat,speaker_feat):
        #speaker_feat b,c,t
        B, T = semantic_feat.shape[0], semantic_feat.shape[1]  # B, C, T
        speaker_feat_expand = speaker_feat.permute(0, 2, 1).expand(B, T, -1)  # B,T,C
        if self.fusion_type == 'AdaIN':
            vq_out_feat = self.AdaIN_merge(semantic_feat,speaker_feat.permute(0, 2, 1))
        elif self.fusion_type == 'Concat':
            vq_out_feat = torch.cat((semantic_feat, speaker_feat_expand), dim=-1)
        else:
            vq_out_feat = semantic_feat
        return vq_out_feat

    def encode_to_token_idx(self,signal,mask=None):
        assert self.config['lm_encoder_type'] == 'tfnet_v2i_vqvae_lm2'
        result = {}
        with torch.no_grad():
            self.encoder_model.eval()
            vq_out = self.encoder_model.encode_to_token_idx(signal)
        self.pad_len = self.encoder_model.pad_len
        phn_vq_in = self.fold_enc_feat(vq_out["phn_feat_vq_in"].permute(0,2,1), self.combine_frames)
        feat = self.enc_feature_projection1(phn_vq_in)
        if mask is None:
            b, t1 = signal.size()
            mask = torch.full((b, t1), 1).to(signal.device)
        mask, _ = self.forward_padding_mask(feat, mask)  ## todo: align bug
        semantic_feat, _ = self.transformer_enc(x=feat, padding_mask=~mask,layer=None)
        semantic_feat = self.enc_feature_projection2(semantic_feat)
        semantic_vq_out = self.semantic_quantize(semantic_feat.permute(0, 2, 1))
        result.update(semantic_vq_out)
        result['spk_feat_vq'] = vq_out['spk_feat_vq']
        return result

    def set_network_entropy_target(self, bitrate, fuzz, hop_dur):
        bitrate_dict = {'0.128k':128,'0.2k':200, '0.25k':250 ,'0.256k': 256,'0.5k': 500, '0.512k': 512, '1k': 1000, '3k': 3000, '6k': 6000, '9k': 9000, '12k': 12000,
                        '24k': 24000}
        bitrate_per_vq_layer = bitrate_dict[bitrate] * self.combine_frames / self.semantic_codebook_num
        self.target_entropy = 0
        self.entropy_fuzz = 0
        fuzz_per_vq_layer = fuzz / self.semantic_codebook_num
        for ii in range(self.semantic_codebook_num):
            self.target_entropy += bitrate_to_entropy_2(bitrate_per_vq_layer, hop_dur)
            self.entropy_fuzz += bitrate_to_entropy_2(fuzz_per_vq_layer, hop_dur)


    def semantic_quantize(self, feat_in):
        # input shape [B,C,T]
        result = {}
        vq_layer_out, vq_inds_out, prob_perplexity_list, code_perplexity_list = [], [], [], []
        codebook_penalty, encoder_penalty, codebook_usage, rate, entropy_loss, entropy, commitment_loss = 0, 0, 0, 0, 0, 0, 0
        cb_num, cb_dim = self.semantic_codebook_num, self.semantic_codebook_dim
        vq_layer_list = self.semantic_vq_layer_list
        if self.config["use_entropy_loss"]:
            target_entropy_per_vqlayer = torch.tensor(self.target_entropy/self.semantic_codebook_num).to(feat_in)
            fuzz_entropy_per_vqlayer = torch.tensor(self.entropy_fuzz/self.semantic_codebook_num).to(feat_in)

        feat_in = feat_in.permute(0, 2, 1)
        _combine_frames = self.combine_frames if self.config['fold_mode'] == '0' else 1
        if _combine_frames > 1:
            B, T, channels = feat_in.shape  # B,T,C'
            feat_in_combine = feat_in.reshape(B, T // _combine_frames, _combine_frames, channels)
            # result["feat_vq_in"] = feat_in_combine.permute(0, 1, 3, 2).reshape(B, T // _combine_frames, -1)  # B,T,C_spk
            enc_feat_combine = torch.split(feat_in_combine, channels // cb_num, dim=-1)

        for layer_i in range(len(vq_layer_list)):
            vq_layer = vq_layer_list[layer_i]
            if _combine_frames > 1:
                vq_in = enc_feat_combine[layer_i].reshape(B, T // _combine_frames, -1)
            else:
                vq_in = feat_in[:, :, layer_i * cb_dim:(layer_i + 1) * cb_dim]

            if self.config["use_entropy_loss"]:
                vq_out = vq_layer(vq_in, target_entropy_per_vqlayer, fuzz_entropy_per_vqlayer)
            else:
                vq_out = vq_layer(vq_in)

            if _combine_frames > 1:
                vq_layer_out.append(
                    vq_out["quantized_feature"].reshape(B, T // _combine_frames, _combine_frames, channels // cb_num))
            else:
                vq_layer_out.append(vq_out["quantized_feature"])  # [B,T,C'//codebook_num]

            vq_inds_out.append(vq_out["quantization_inds"])  # [B,T,1]
            if 'prob_perplexity' in vq_out.keys():  # softmax prob
                prob_perplexity_list.append(vq_out["prob_perplexity"])
            if 'code_perplexity' in vq_out.keys():  # hard prob
                code_perplexity_list.append(vq_out["code_perplexity"])
            if 'codebook_usage' in vq_out.keys():
                codebook_usage += vq_out["codebook_usage"]  #
            if 'codebook_penalty' in vq_out.keys():
                codebook_penalty += vq_out["codebook_penalty"]  # vq loss
            if 'encoder_penalty' in vq_out.keys():
                encoder_penalty += vq_out["encoder_penalty"]  # commitment loss
            if 'entropy_loss' in vq_out.keys():
                entropy_loss += vq_out["entropy_loss"]
            if 'entropy' in vq_out.keys():
                entropy += vq_out["entropy"]
            if 'commitment_loss' in vq_out.keys():
                commitment_loss += vq_out["commitment_loss"]


        if _combine_frames > 1:
            feat_vq_out = torch.cat(vq_layer_out, dim=-1)
            result["feat_vq_out"] = feat_vq_out.permute(0, 1, 3, 2).reshape(B, T // _combine_frames, -1)  # B,T,C_spk
            feat_vq_out = feat_vq_out.reshape(B, T, channels)
        else:
            feat_vq_out = torch.cat(vq_layer_out, dim=-1)  # B,T,320

        codebook_usage /= len(vq_layer_list)
        codebook_penalty /= len(vq_layer_list)
        encoder_penalty /= len(vq_layer_list)
        commitment_loss /= len(vq_layer_list)

        result["feat_vq"] = feat_vq_out.permute(0, 2, 1)  # B,C_spk,T
        result["feat_vq_in"] = feat_in.permute(0, 2, 1)  # B,C_spk,T
        result["quantization_inds"] = torch.cat(vq_inds_out, dim=-1)  # B,T,spk_codebook_num
        result["prob_perplexity_list"] = prob_perplexity_list  # [spk_codebook_num]
        result["code_perplexity_list"] = code_perplexity_list  # [spk_codebook_num]
        result["codebook_usage"] = codebook_usage  # [1,]  avg_codebook_usage
        result["codebook_penalty"] = codebook_penalty  # [1,]  avg_codebook_penalty
        result["encoder_penalty"] = encoder_penalty  # [1,]  avg_encoder_penalty
        result["semantic_entropy_loss"] = entropy_loss  #
        result["entropy"] = entropy
        result["commitment_loss"] = commitment_loss

        return result

    def dequantize(self, vq_inds):
        assert self.config['lm_encoder_type'] == 'tfnet_v2i_vqvae_lm2'
        vq_layer_out = []
        _combine_frames = self.combine_frames if self.config['fold_mode'] == '0' else 1
        if self.combine_frames > 1:
            B, T, _ = vq_inds.shape  # B,T,C'
        for layer_i in range(len(self.semantic_vq_layer_list)):
            vq_layer = self.semantic_vq_layer_list[layer_i]
            vq_out = vq_layer.dequantize(vq_inds[:, :, layer_i])
            if _combine_frames > 1:
                vq_layer_out.append(vq_out.reshape(B, T, _combine_frames, -1))  # [B,T/4,4,C'//codebook_num]
            else:
                vq_layer_out.append(vq_out)  # [B,T,C'//codebook_num]
        vq_feat = torch.cat(vq_layer_out, dim=-1)  # B,T,C'
        # vq_feat_combine = vq_feat.permute(0, 1, 3, 2).reshape(B, T, -1)  # B,T,C_spk
        if _combine_frames > 1:
            vq_feat = vq_feat.reshape(B, T * _combine_frames, -1)
        return vq_feat

    def decode_from_token_idx(self, vq_inds):
        ## spk/phn feature merge todo
        phn_vq_out, _ = self.dequantize(vq_inds)
        phn_vq_out = phn_vq_out.permute(0, 2, 1)  # [B, C, T]
        phn_vq_out = self.conv1x1_dec(phn_vq_out.unsqueeze(-1)).squeeze(-1)
        merge_feat, condition = self.merge(spk_embed, phn_vq_out)
        pred_1d, _ = self.decoder(merge_feat, condition)
        return pred_1d


    def reset_entropy_hists_eval(self):
        if self.config['semantic_tokenize']:
            for vq_layer in self.semantic_vq_layer_list:
                vq_layer.entropy_avg_eval.reset()
        else:
            self.encoder_model.reset_entropy_hists_eval()


    def reset_entropy_hists_train(self):
        if self.config['semantic_tokenize']:
            for vq_layer in self.semantic_vq_layer_list:
                vq_layer.entropy_avg_train.reset()
        else:
            self.encoder_model.reset_entropy_hists_train()

    def get_overall_entropy_avg_train(self):
        semantic_avgs = []
        if self.config['semantic_tokenize']:
            for vq_layer in self.semantic_vq_layer_list:
                semantic_avgs.append(vq_layer.entropy_avg_train.avg)
            return torch.sum(torch.Tensor(semantic_avgs))
        else:
            return self.encoder_model.get_overall_entropy_avg_train()


    def get_overall_entropy_avg_eval(self):
        semantic_avgs = []
        if self.config['semantic_tokenize']:
            for vq_layer in self.semantic_vq_layer_list:
                semantic_avgs.append(vq_layer.entropy_avg_eval.avg)
            return torch.sum(torch.Tensor(semantic_avgs))
        else:
            return self.encoder_model.get_overall_entropy_avg_eval()

    def update_temperature_gumbel(self, cur_iter):
        if self.config['semantic_tokenize']:
            for vq_layer in self.semantic_vq_layer_list:
                vq_layer.temp_updates(cur_iter)
        else:
            self.encoder_model.update_temperature_gumbel(cur_iter)




