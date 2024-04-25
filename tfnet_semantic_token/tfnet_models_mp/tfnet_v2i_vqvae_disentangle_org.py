import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.time2freq2 import Time2Freq2
from .layers.conv_layer import ConvLayer
from .layers.deconv_layer import DeconvLayer
from .layers.tcm_layer import TCMLayer
from .layers.vq_layer_ema import VQEmbeddingEMA_DDP
from .layers.freq2timecodec import Freq2TimeCodec
from .layers.vq_layer_gumbel import GumbelVectorQuantizer
from .layers.temporal_pooling import TemporalPooling_spk, TemporalPooling_phn
from .layers.classifier import SpeakerClassifier, PhonemeClassifier
from .layers.gradient_reversal import GradientReversal
from tfnet_semantic_token.utils.tools import *
from typing import Optional


class TFNet(nn.Module):
    def __init__(self, config=None, bn=True, ):
        super(TFNet, self).__init__()

        frm_size = config["dft_size"]
        shift_size = int(config["dft_size"] * config["hop_vqvae"])
        self.win_len = n_fft = frm_size
        self.hop_len = shift_size

        self.in_channels = 2
        activation_functions = {'PRELU': nn.PReLU, 'ELU': nn.ELU}
        activation = activation_functions[config['activation']]
        self.config = config

        if config['train_downstream_model_noVQ']:
            assert (config['disable_spk_vq'] == True), 'disable vq when training VC model'
            assert (config['disable_phn_vq'] == True), 'disable vq when training VC model'
            assert (config['freeze_encoder'] == True), 'freeze encoder when training VC model'

        if config["use_learnable_compression"]:
            # self.power_cprs_spk = nn.Parameter(torch.FloatTensor(1))
            # nn.init.constant_(self.power_cprs_spk, config["power"])
            self.power_cprs_phn = nn.Parameter(torch.FloatTensor(1))
            nn.init.constant_(self.power_cprs_phn, config["power"])
        ### input layer
        self.time2freq_spk = Time2Freq2(frm_size, shift_size, self.win_len, self.hop_len, n_fft, config=config,
                                        power=self.power_cprs_phn if config["use_learnable_compression"] else None)
        self.time2freq_phn = Time2Freq2(frm_size, shift_size, self.win_len, self.hop_len, n_fft, config=config,
                                        power=self.power_cprs_phn if config["use_learnable_compression"] else None)

        ### encoder
        enc_kernel_size = (2, 5)
        self.enc_time_kernel_len = enc_kernel_size[0]
        if frm_size == 320:  # 20ms
            enc_kernel_size = (2, 5)
            enc_channels = (16, 32, 64, 64)
            enc_strides = ((1, 1), (1, 4), (1, 4), (1, 2))  # 161, 41, 11, 5
            self.frm_pad = ((2, 2), (2, 2), (2, 2), (1, 1))
            self.last_layer_pad = (2, 2)
            enc_feat_dims = (161 * 16, 41 * 32, 11 * 64, 5 * 64)
            self.enc_out_frm_size = 5
            self.enc_out_channels = enc_channels[-1]
            self.tcm_mid_channels = 512  # enlarge from 256
            self.tcm_repeat1, self.tcm_block1 = 1, 4
            self.gru_num = 1
            self.tcm_repeat2, self.tcm_block2 = 1, 4
            self.depth_ker_size = 5
            self.enc_time_kernel_len = enc_kernel_size[0]
            is_last = [False, False, False, True]
            self.pad_list = [None, None, None, (self.frm_pad[0], (self.enc_time_kernel_len - 1, 0))]
            self.pad_list_frmwise = [None, None, None, (self.frm_pad[0], (0, 0))]
        elif frm_size == 640:  # 40ms
            enc_kernel_size = (2, 5)
            enc_channels = (16, 32, 32, 64, 64)
            enc_strides = ((1, 1), (1, 2), (1, 4), (1, 4), (1, 2))  # # 321, 161, 41, 11, 5
            enc_feat_dims = (321 * 16, 161 * 32, 41 * 32, 11 * 64, 5 * 64)  # 5136,5152,1312,704,320
            self.frm_pad = ((2, 2), (2, 2), (2, 2), (2, 2), (1, 1))
            self.last_layer_pad = (2, 2)
            self.enc_out_frm_size = 5
            self.enc_out_channels = enc_channels[-1]
            self.tcm_mid_channels = 512  # enlarge from 256
            self.tcm_repeat1, self.tcm_block1 = 1, 4
            self.gru_num = 1
            self.tcm_repeat2, self.tcm_block2 = 1, 4
            self.depth_ker_size = 3
            self.enc_time_kernel_len = enc_kernel_size[0]
            is_last = [False, False, False, False, True]
            self.pad_list = [None, None, None, None, (self.frm_pad[0], (self.enc_time_kernel_len - 1, 0))]
            self.pad_list_frmwise = [None, None, None, None, (self.frm_pad[0], (0, 0))]

        self.feat_dim = self.enc_out_frm_size * self.enc_out_channels
        assert (self.feat_dim == 320)  # used for VQ module initialization

        # self.enc_frm_size = (161, 81, 41, 21, 11, 5)
        Norm_enc = 'IN' if self.config['use_InsNorm_enc'] else 'BN'
        useXuebias = True if self.config['use_xuemodel_bias'] else False
        self.phn_enc_list = nn.ModuleList(
            ConvLayer(in_c, out_c, enc_kernel_size, enc_strides[i], nonlinearity=activation, Norm=Norm_enc,
                      useXuebias=useXuebias)
            for i, (in_c, out_c) in enumerate(
                zip((self.in_channels,) + enc_channels[:-1], enc_channels)))

        self.spk_enc_list = nn.ModuleList(
            ConvLayer(in_c, out_c, enc_kernel_size, enc_strides[i], nonlinearity=activation, Norm='BN',
                      useXuebias=useXuebias)
            for i, (in_c, out_c) in enumerate(
                zip((self.in_channels,) + enc_channels[:-1], enc_channels)))

        ### temporal convolutional module (tcm) and gru in encoder
        if config["use_encoder_tcm"]:
            self.spk_tcm_list = nn.ModuleList(
                TCMLayer(self.feat_dim, self.tcm_mid_channels, self.depth_ker_size, 2 ** i, nonlinearity=activation,
                         Norm='BN', useXuebias=useXuebias)
                for _ in range(self.tcm_repeat1) for i in range(self.tcm_block1))
            self.phn_tcm_list = nn.ModuleList(
                TCMLayer(self.feat_dim, self.tcm_mid_channels, self.depth_ker_size, 2 ** i, nonlinearity=activation,
                         Norm=Norm_enc, useXuebias=useXuebias)
                for _ in range(self.tcm_repeat1) for i in range(self.tcm_block1))
        if config["use_encoder_gru"]:
            self.spk_gru = nn.GRU(self.feat_dim, self.feat_dim, num_layers=self.gru_num, bias=True, batch_first=True)
            self.spk_h0 = nn.Parameter(torch.FloatTensor(self.gru_num, 1, self.feat_dim).zero_())
            self.phn_gru = nn.GRU(self.feat_dim, self.feat_dim, num_layers=self.gru_num, bias=True, batch_first=True)
            self.phn_h0 = nn.Parameter(torch.FloatTensor(self.gru_num, 1, self.feat_dim).zero_())

        ## VQ bottleneck initial
        bitrate_dict = {'0.25k': 250,'0.256k': 256, '0.512k': 512, '1k': 1000, '3k': 3000, '6k': 6000, '9k': 9000, '12k': 12000,
                        '24k': 24000}
        bit_per_frame = int(shift_size / config["sampling_rate"] * bitrate_dict[config['bitrate']])
        self.init_codebook_size(config, bit_per_frame)
        ## 1x1 conv change latent dim
        self.combine_frames = config["combineVQ_frames"]
        self.conv1x1_spk_enc = nn.Conv2d(self.feat_dim, self.spk_latent_dim, kernel_size=1, stride=1, bias=False)
        nn.init.xavier_normal_(self.conv1x1_spk_enc.weight)
        self.conv1x1_phn_enc = nn.Conv2d(self.feat_dim, self.phn_latent_dim, kernel_size=1, stride=1, bias=False)
        nn.init.xavier_normal_(self.conv1x1_phn_enc.weight)
        if config['disen_scheme'] == 'global_spk':
            self.spk_embedding_dim = 256  ## [global_spk] do not need to combine frames
            self.spk_codebook_dim = self.spk_embedding_dim // self.spk_codebook_num
        else:
            self.spk_codebook_dim = self.spk_latent_dim * self.combine_frames // self.spk_codebook_num
            self.spk_embedding_dim = 256
        self.phn_codebook_dim = self.phn_latent_dim * self.combine_frames // self.phn_codebook_num

        if config["vq_type"] == 'Gumbel':
            if not config["disable_spk_vq"]:
                self.spk_vq_layer_list = nn.ModuleList(
                    GumbelVectorQuantizer(config, input_dim=self.spk_codebook_dim, n_embeddings=i,
                                          groups=config["groups"], combine_groups=config["combine_groups"], ) for i in
                    self.spk_codebook_size)
            if not config["disable_phn_vq"]:
                self.phn_vq_layer_list = nn.ModuleList(
                    GumbelVectorQuantizer(config, input_dim=self.phn_codebook_dim, n_embeddings=i,
                                          groups=config["groups"], combine_groups=config["combine_groups"], ) for i in
                    self.phn_codebook_size)

        ### disentangle related
        nspeakers = 251  # librispeech100
        nphonemes = 71  # librispeech100
        spk_mid_channels = [self.spk_embedding_dim, self.spk_embedding_dim]
        if self.config['use_temporal_aggregator']:  ## causal spk embedding pooling
            self.tap_spk = TemporalPooling_spk(self.spk_latent_dim, spk_mid_channels,
                                               pooling_kernel=self.config['tap_rate'],
                                               max_pooling_len=self.config['max_pooling_len'],
                                               transmit_rate=self.config['transmit_rate'])
        else:  ## non-causal spk embedding pooling
            self.tap_spk = TemporalPooling_spk(self.spk_latent_dim, spk_mid_channels, out_dim=1)
        ## classification supervision
        if not self.config["disable_spk_classify"]:
            self.speaker_classifier1 = SpeakerClassifier(self.spk_embedding_dim, nspeakers)
        if not self.config["disable_phn_classify"]:
            self.phoneme_classifier1 = PhonemeClassifier(self.phn_latent_dim, nphonemes)

        ## disentangle
        if not self.config["disable_disen"]:
            # type1: gradient reversal
            if self.config['disen_type'] == 'grl':
                if self.config["phn_grl"]:
                    phn_mid_channels = [256, 256, 128, 128]
                    self.tap_phn = TemporalPooling_phn(self.phn_latent_dim, phn_mid_channels, out_dim=1)
                    self.speaker_classifier2 = SpeakerClassifier(phn_mid_channels[-1], nspeakers)
                    self.grl_phn = GradientReversal(scale=1.0)
                if self.config["spk_grl"]:
                    self.phoneme_classifier2 = PhonemeClassifier(self.spk_latent_dim, nphonemes)
                    self.grl_spk = GradientReversal(scale=1.0)
            # type2: instance normalization for de-identify
            elif self.config['disen_type'] == 'ins':
                self.norm_layer1 = nn.InstanceNorm1d(self.phn_latent_dim, affine=False)
                self.norm_layer2 = nn.InstanceNorm1d(self.feat_dim, affine=False)
                if self.config['merge_type'] == 'multicondition2':
                    self.norm_layers = nn.ModuleList(
                        [nn.InstanceNorm1d(ch, affine=False) for ch in enc_feat_dims[::-1]])

        ### decoder
        ### spk and phn feature merge
        if config["merge_type"] == 'concat':
            if config["disen_scheme"] == 'global_spk':
                merge_ch = self.phn_latent_dim + self.spk_embedding_dim
            else:
                merge_ch = self.phn_latent_dim + self.spk_latent_dim
            self.conv1x1_dec = nn.Conv2d(merge_ch, self.feat_dim, kernel_size=1, stride=1, bias=False)
            nn.init.xavier_normal_(self.conv1x1_dec.weight)
        elif 'condition' in config['merge_type']:
            if config["merge_type"] == 'condition':
                self.cond_positions = 1  # condition(*)   (*merge) -> 1x1 -> tcm -> gru -> tcm
                hidden_size = self.spk_embedding_dim
                cond_chns = ((hidden_size, self.feat_dim),)
                self.flag = ((True, False),)
            elif config["merge_type"] == 'multicondition1':
                self.cond_positions = 3  # condition(*)   (*merge) -> 1x1 -> tcm (*merge) -> gru (*merge) -> tcm
                hidden_size = self.spk_embedding_dim
                cond_chns = ((hidden_size, self.feat_dim),
                             (hidden_size, self.feat_dim),
                             (hidden_size, self.feat_dim))
                self.flag = ((True, False), (True, False), (True, False))
            elif config["merge_type"] == 'multicondition2':
                self.cond_positions = 8  # condition(*)   (*merge)->1x1->tcm(*merge)gru(*merge)tcm(*merge)dec_0(*merge)dec_1(*merge)dec_2(*merge)dec_3(*merge)dec_4
                cond_chns = ((self.feat_dim,), (self.feat_dim,), (self.feat_dim,),) + tuple(
                    [(ch,) for ch in enc_feat_dims[::-1]])
                self.flag = ((False,), (False,), (False,),) + tuple([(False,) for ch in enc_feat_dims[::-1]])

            scale_list = [nn.ModuleList() for _ in range(self.cond_positions)]
            bias_list = [nn.ModuleList() for _ in range(self.cond_positions)]
            spk_cond_ch = self.spk_embedding_dim if config["disen_scheme"] == 'global_spk' else self.spk_latent_dim
            for pos_i in range(self.cond_positions):
                for i, (in_c, out_c) in enumerate(
                        zip((spk_cond_ch,) + cond_chns[pos_i][:-1], cond_chns[pos_i])):  # todo
                    scale_list[pos_i].append(nn.Linear(in_c, out_c))
                    bias_list[pos_i].append(nn.Linear(in_c, out_c))
            self.scale_list = nn.ModuleList(scale_list)
            self.bias_list = nn.ModuleList(bias_list)
            self.conv1x1_dec = nn.Conv2d(self.phn_latent_dim, self.feat_dim, kernel_size=1, stride=1, bias=False)
            nn.init.xavier_normal_(self.conv1x1_dec.weight)

        out_dim = 2
        ### interleave structure in decoder(tcm-gru-tcm)
        Norm_dec = 'IN' if self.config['use_InsNorm_dec'] else False
        self.tcm_list1 = nn.ModuleList(
            TCMLayer(self.feat_dim, self.tcm_mid_channels, self.depth_ker_size, 2 ** i, nonlinearity=activation,
                     Norm=Norm_dec, useXuebias=useXuebias)
            for _ in range(self.tcm_repeat1) for i in range(self.tcm_block1))
        self.gru = nn.GRU(self.feat_dim, self.feat_dim, num_layers=self.gru_num, bias=True, batch_first=True)
        self.h0 = nn.Parameter(torch.FloatTensor(self.gru_num, 1, self.feat_dim).zero_())
        self.tcm_list2 = nn.ModuleList(
            TCMLayer(self.feat_dim, self.tcm_mid_channels, self.depth_ker_size, 2 ** i, nonlinearity=activation,
                     Norm=Norm_dec, useXuebias=useXuebias)
            for _ in range(self.tcm_repeat2) for i in range(self.tcm_block2))

        self.dec_list = nn.ModuleList(
            DeconvLayer(in_c, out_c, enc_kernel_size, enc_strides[len(self.phn_enc_list) - 1 - i],
                        nonlinearity=activation, Norm=Norm_dec, useXuebias=useXuebias, is_last=is_last[i])
            for i, (in_c, out_c) in enumerate(
                zip(enc_channels[::-1], enc_channels[::-1][1:] + (out_dim,)))
        )
        # self.dec_last = ConvLayer(enc_channels[0], 2, enc_kernel_size, enc_strides[0], nonlinearity=nn.PReLU, bn=bn)
        self.out_conv = nn.Conv2d(out_dim, out_dim, enc_kernel_size, (1, 1), bias=False)
        nn.init.xavier_normal_(self.out_conv.weight)
        self.freq2time = Freq2TimeCodec(frm_size, shift_size, self.win_len, self.hop_len, n_fft,
                                        power=self.power_cprs_phn if config["use_learnable_compression"] else None,
                                        config=config, )

        if config["use_entropy_loss"]:
            self.set_network_entropy_target(config["bitrate"], config["entropy_fuzz"], config["sampling_rate"],
                                            self.hop_len)

    def load_input(self, input_list):
        if isinstance(input_list, dict):
            self.signal = input_list['signal_clean'] if 'signal_clean' in input_list.keys() else None
            self.signal_aug = input_list['signal_aug'] if 'signal_aug' in input_list.keys() else None
            self.signal_noisy = input_list['signal_noisy'] if 'signal_noisy' in input_list.keys() else None
            if 'spk_id' in input_list.keys():
                self.spk_id = input_list['spk_id']
            self.source_signal = input_list['source_signal'] if 'source_signal' in input_list.keys() else None
            self.target_signal = input_list['target_signal'] if 'target_signal' in input_list.keys() else None
            self.signal = self.source_signal if self.source_signal is not None else self.signal
            if 'bitrate' in input_list.keys():
                self.bitrate = input_list['bitrate']
        else:
            self.signal = input_list
            self.signal_aug = None
            self.signal_noisy = None
            self.source_signal = None
            self.target_signal = None
            self.bitrate = self.config['bitrate']

    def forward(self, input_list):
        result = {}
        self.load_input(input_list)
        self.signal_len = self.signal.shape[1] if self.signal is not None else self.signal_noisy.shape[1]
        result["bitrate"] = self.bitrate

        ## Input shape: (B, L)
        num_frames = int(self.signal_len / self.hop_len) + 1
        self.pad_len = 0
        if num_frames % self.combine_frames:
            padded_frames = self.combine_frames - num_frames % self.combine_frames
            self.pad_len = int(padded_frames * self.hop_len)
            if self.signal is not None:
                self.signal = F.pad(self.signal, (0, self.pad_len))
            if self.signal_noisy is not None:
                self.signal_noisy = F.pad(self.signal_noisy, (0, self.pad_len))
            if self.source_signal is not None:
                self.source_signal = F.pad(self.source_signal, (0, self.pad_len))
            if self.target_signal is not None:
                self.target_signal = F.pad(self.target_signal, (0, self.pad_len))
            if self.signal_aug is not None:
                self.signal_aug = F.pad(self.signal_aug, (0, self.pad_len))

        if self.signal_noisy is not None:
            in_feat_spk, self.input_stft_r_spk = self.time2freq_spk(self.signal_noisy)
            in_feat_phn, self.input_stft_r_phn = self.time2freq_phn(self.signal_noisy)
        else:
            in_feat_spk, self.input_stft_r_spk = self.time2freq_spk(self.signal)
            in_feat_phn, self.input_stft_r_phn = self.time2freq_phn(self.signal)

        ## Encoder
        if self.target_signal is not None:
            in_feat_T, in_stft_T = self.time2freq_spk(self.target_signal)
            spk_in = in_feat_T
        else:
            spk_in = in_feat_spk

        if self.source_signal is not None:
            in_feat_S, in_stft_S = self.time2freq_phn(self.source_signal)
            phn_in = in_feat_S  ## voice conversion
        elif self.config['data_aug'] and self.signal_aug is not None:
            in_feat_phn_aug, self.input_stft_r_phn_aug = self.time2freq_phn(self.signal_aug)
            phn_in = in_feat_phn_aug  ## aug
        else:
            phn_in = in_feat_phn  ##orig

        spk_enc_feat = self.spk_encoder(spk_in)  # B,C_spk,T
        if self.config['use_contrastive_loss']:
            _, phn_enc_feat_aug = self.phn_encoder(in_feat_phn_aug)  ## augment signal
            phn_enc_feat, phn_enc_feat_orig = self.phn_encoder(in_feat_phn)  ## orig signal
        else:
            phn_enc_feat, _ = self.phn_encoder(phn_in)  # B,C_phn,T

        ## content contrastive learning
        if self.config['use_contrastive_loss']:
            logits1 = self.compute_contrastive_loss(feats_pos=phn_enc_feat_aug, feats_neg=phn_enc_feat_orig,
                                                    neg_num=self.config['neg_num'])
            logits2 = self.compute_contrastive_loss(feats_pos=phn_enc_feat_orig, feats_neg=phn_enc_feat_aug,
                                                    neg_num=self.config['neg_num'])
            result["logits"] = [logits1, logits2]

        ## spk feat temporal pooling
        if self.config["disen_scheme"] == 'global_spk':
            spk_enc_feat, spk_visual = self.tap_spk(spk_enc_feat)  # B,C,T/10 or B,C,1

        ## VQ bottleneck
        vq_out = self.vq_bottleneck(spk_enc_feat, phn_enc_feat)
        spk_vq_out = vq_out["spk_feat_vq"]  # B,C,1
        phn_vq_out = vq_out["phn_feat_vq"]  # B,C,T
        result.update(vq_out)

        if self.config["suprv_position"] == 'vq_out':
            class_result = self.classify(spk_vq_out, phn_vq_out)
            result.update(class_result)
        elif self.config["suprv_position"] == 'vq_in':
            class_result = self.classify(spk_enc_feat, phn_enc_feat)
            result.update(class_result)

        ## spk/phn feature merge
        phn_vq_out = self.conv1x1_dec(phn_vq_out.unsqueeze(-1)).squeeze(-1)
        merge_feat, condition = self.merge(spk_vq_out, phn_vq_out)
        pred_1d, pred_2d = self.decoder(merge_feat, condition)
        if self.pad_len > 0:
            pred_1d = pred_1d[:, :-self.pad_len]
        for key in vq_out.keys():
            result[key] = vq_out[key]

        result["x_hat"] = pred_1d
        result["pred_freq"] = pred_2d  # pred_2d for supervision before convert back to time domain
        result["spk_embedding"] = spk_vq_out
        result["phn_embedding"] = phn_vq_out
        new_result = {self.bitrate: {}}
        for key in result.keys():
            new_result[self.bitrate].update({key: result[key]})
        return new_result

    def encode_to_token_idx(self, signal):
        result = {}
        self.load_input(signal)
        self.signal_len = self.signal.shape[1] if self.signal is not None else self.signal_noisy.shape[1]
        result["bitrate"] = self.bitrate

        ## Input shape: (B, L)
        num_frames = int(self.signal_len / self.hop_len) + 1
        self.pad_len = 0
        if num_frames % self.combine_frames:
            padded_frames = self.combine_frames - num_frames % self.combine_frames
            self.pad_len = int(padded_frames * self.hop_len)
            if self.signal is not None:
                self.signal = F.pad(self.signal, (0,self.pad_len))
            if self.signal_noisy is not None:
                self.signal_noisy = F.pad(self.signal_noisy,(0,self.pad_len))
            if self.source_signal is not None:
                self.source_signal = F.pad(self.source_signal,(0,self.pad_len))
            if self.target_signal is not None:
                self.target_signal = F.pad(self.target_signal, (0,self.pad_len))
            if self.signal_aug is not None:
                self.signal_aug = F.pad(self.signal_aug, (0,self.pad_len))

        if self.signal_noisy is not None:
            in_feat_spk, self.input_stft_r_spk = self.time2freq_spk(self.signal_noisy)
            in_feat_phn, self.input_stft_r_phn = self.time2freq_phn(self.signal_noisy)
        else:
            in_feat_spk, self.input_stft_r_spk = self.time2freq_spk(self.signal)
            in_feat_phn, self.input_stft_r_phn = self.time2freq_phn(self.signal)

        ## Encoder
        if self.target_signal is not None:
            in_feat_T, in_stft_T = self.time2freq_spk(self.target_signal)
            spk_in = in_feat_T
        else:
            spk_in = in_feat_spk

        if self.source_signal is not None:
            in_feat_S, in_stft_S = self.time2freq_phn(self.source_signal)
            phn_in = in_feat_S  ## voice conversion
        elif self.config['data_aug'] and self.signal_aug is not None:
            in_feat_phn_aug, self.input_stft_r_phn_aug = self.time2freq_phn(self.signal_aug)
            phn_in = in_feat_phn_aug  ## aug
        else:
            phn_in = in_feat_phn  ##orig

        spk_enc_feat = self.spk_encoder(spk_in)  # B,C_spk,T
        if self.config['use_contrastive_loss']:
            _, phn_enc_feat_aug = self.phn_encoder(in_feat_phn_aug)  ## augment signal
            phn_enc_feat, phn_enc_feat_orig = self.phn_encoder(in_feat_phn)  ## orig signal
        else:
            phn_enc_feat, _ = self.phn_encoder(phn_in)  # B,C_phn,T

        ## content contrastive learning
        if self.config['use_contrastive_loss']:
            logits1 = self.compute_contrastive_loss(feats_pos=phn_enc_feat_aug, feats_neg=phn_enc_feat_orig,
                                                    neg_num=self.config['neg_num'])
            logits2 = self.compute_contrastive_loss(feats_pos=phn_enc_feat_orig, feats_neg=phn_enc_feat_aug,
                                                    neg_num=self.config['neg_num'])
            result["logits"] = [logits1, logits2]

        ## spk feat temporal pooling
        if self.config["disen_scheme"] == 'global_spk':
            spk_enc_feat, spk_visual = self.tap_spk(spk_enc_feat)  # B,C,T/10 or B,C,1

        ## VQ bottleneck
        vq_out = self.vq_bottleneck(spk_enc_feat, phn_enc_feat)
        result.update(vq_out)
        return result

    def decode_from_token_idx(self, phn_vq_inds, spk_embed):
        ## spk/phn feature merge
        phn_vq_out,_ = self.dequantize(phn_vq_inds)
        phn_vq_out = phn_vq_out.permute(0, 2, 1)  # [B, C, T]
        phn_vq_out = self.conv1x1_dec(phn_vq_out.unsqueeze(-1)).squeeze(-1)
        merge_feat, condition = self.merge(spk_embed, phn_vq_out)
        pred_1d, _ = self.decoder(merge_feat, condition)
        return pred_1d

    def phn_encoder(self, in_feat):
        x_conv = in_feat  # B,2,T,161
        ### encoder
        self.phn_enc_out_list = [x_conv]
        enc_tcm_feats = []
        for layer_i in range(len(self.phn_enc_list)):
            enc_layer = self.phn_enc_list[layer_i]
            enc_out = enc_layer(self.phn_enc_out_list[-1], (self.frm_pad[layer_i], (1, 0)))
            self.phn_enc_out_list.append(enc_out)

        enc_out = self.phn_enc_out_list[-1]  ##B,64,T,5
        if self.config["use_encoder_tcm"]:
            batch_size, nb_frames = enc_out.shape[0], enc_out.shape[2]
            tcm_feat = enc_out.permute(0, 2, 1, 3).reshape(batch_size, nb_frames, -1).permute(0, 2, 1).unsqueeze(
                -1)  # (B,C,T,1)
            enc_tcm_feats.append(tcm_feat)
            for tcm in self.phn_tcm_list:
                tcm_feat, _ = tcm(tcm_feat)
                enc_tcm_feats.append(tcm_feat)
            enc_out = tcm_feat.squeeze(-1).reshape(batch_size, self.enc_out_channels, self.enc_out_frm_size,
                                                   nb_frames).permute(0, 1, 3, 2)
            self.phn_enc_tcm_feats = enc_tcm_feats
        if self.config["use_encoder_gru"]:
            rnn_in = enc_out.permute(0, 2, 1, 3).reshape(batch_size, nb_frames, -1).permute(0, 2, 1).unsqueeze(
                -1)  # (B,C,T,1)
            rnn_in = rnn_in.squeeze(-1).permute(0, 2, 1)  # (B,T,C)
            rnn_out, h_n_enc = self.phn_gru(rnn_in, self._phn_gru_init_state(batch_size))
            rnn_out = rnn_out.permute(0, 2, 1).unsqueeze(-1)  # (B,C,T,1)
            enc_out = rnn_out.squeeze(-1).reshape(batch_size, self.enc_out_channels, self.enc_out_frm_size,
                                                  nb_frames).permute(0, 1, 3, 2)
            self.phn_enc_gru_feat = rnn_out

        enc_out = enc_out.permute(0, 2, 1, 3).reshape(batch_size, nb_frames, -1).permute(0, 2, 1).unsqueeze(
            -1)  # (B,C,T,1)

        phn_feat = self.conv1x1_phn_enc(enc_out).squeeze(-1)
        self.phn_enc_1x1_feat = phn_feat
        return phn_feat, enc_out.squeeze(-1)

    def spk_encoder(self, in_feat):
        x_conv = in_feat  # B,2,T,161
        ### encoder
        self.spk_enc_out_list = [x_conv]
        enc_tcm_feats = []
        for layer_i in range(len(self.spk_enc_list)):
            enc_layer = self.spk_enc_list[layer_i]
            enc_out = enc_layer(self.spk_enc_out_list[-1], (self.frm_pad[layer_i], (1, 0)))
            self.spk_enc_out_list.append(enc_out)

        enc_out = self.spk_enc_out_list[-1]  ##B,64,T,5
        if self.config["use_encoder_tcm"]:
            batch_size, nb_frames = enc_out.shape[0], enc_out.shape[2]
            tcm_feat = enc_out.permute(0, 2, 1, 3).reshape(batch_size, nb_frames, -1).permute(0, 2, 1).unsqueeze(
                -1)  # (B,C,T,1)
            enc_tcm_feats.append(tcm_feat)
            for tcm in self.spk_tcm_list:
                tcm_feat, _ = tcm(tcm_feat)
                enc_tcm_feats.append(tcm_feat)
            enc_out = tcm_feat.squeeze(-1).reshape(batch_size, self.enc_out_channels, self.enc_out_frm_size,
                                                   nb_frames).permute(0, 1, 3, 2)
            self.spk_enc_tcm_feats = enc_tcm_feats
        if self.config["use_encoder_gru"]:
            rnn_in = enc_out.permute(0, 2, 1, 3).reshape(batch_size, nb_frames, -1).permute(0, 2, 1).unsqueeze(
                -1)  # (B,C,T,1)
            rnn_in = rnn_in.squeeze(-1).permute(0, 2, 1)  # (B,T,C)
            rnn_out, h_n_enc = self.spk_gru(rnn_in, self._spk_gru_init_state(batch_size))
            rnn_out = rnn_out.permute(0, 2, 1).unsqueeze(-1)  # (B,C,T,1)
            enc_out = rnn_out.squeeze(-1).reshape(batch_size, self.enc_out_channels, self.enc_out_frm_size,
                                                  nb_frames).permute(0, 1, 3, 2)
            self.spk_enc_gru_feat = rnn_out
        enc_out = enc_out.permute(0, 2, 1, 3).reshape(batch_size, nb_frames, -1).permute(0, 2, 1).unsqueeze(
            -1)  # (B,C,T)
        spk_feat = self.conv1x1_spk_enc(enc_out).squeeze(-1)
        self.spk_enc_1x1_feat = spk_feat

        return spk_feat

    def append_cond(self, feat, cond, norm_layer):
        if self.config['disen_type'] == 'ins' and self.config['use_InsNorm_enc']:
            feat = norm_layer(feat)  # B,C,T
        # cond[0]=torch.ones_like(cond[0])
        # cond[1]=torch.zeros_like(cond[1])
        T1, T2 = feat.shape[2], cond[0].shape[2]
        if T2 != T1 and self.config['use_temporal_aggregator']:
            pad_scale = torch.repeat_interleave(cond[0][:, :, -1].unsqueeze(-1), (T1 - T2), dim=-1)
            scale = torch.cat((cond[0], pad_scale), dim=-1)
            pad_bias = torch.repeat_interleave(cond[1][:, :, -1].unsqueeze(-1), (T1 - T2), dim=-1)
            bias = torch.cat((cond[1], pad_bias), dim=-1)
        else:
            scale, bias = cond[0], cond[1]

        feat = scale * feat + bias
        return feat

    def decoder(self, vq_feat, condition=None, signal=None, inference=False):
        orig_signal = signal  # for autoregressive teacher forcing

        ### interleave structure
        batch_size, nb_frames = vq_feat.shape[0], vq_feat.shape[2]
        tcm_feat = vq_feat.permute(0, 2, 1).reshape(batch_size, nb_frames, -1).permute(0, 2, 1).unsqueeze(
            -1)  # (B,C,T,1)

        for tcm in self.tcm_list1:
            tcm_feat, _ = tcm(tcm_feat)

        if self.config["merge_type"] in ('multicondition1', 'multicondition2'):
            tcm_feat = self.append_cond(tcm_feat.squeeze(-1), [condition[0][1], condition[1][1]], self.norm_layer2)
            tcm_feat = tcm_feat.permute(0, 2, 1)
        else:
            tcm_feat = tcm_feat.squeeze(-1).permute(0, 2, 1)  # (B,T,C)

        tcm_feat, h_n = self.gru(tcm_feat, self._gru_init_state(batch_size))

        if self.config["merge_type"] in ('multicondition1', 'multicondition2'):
            tcm_feat = self.append_cond(tcm_feat.permute(0, 2, 1), [condition[0][2], condition[1][2]], self.norm_layer2)
            tcm_feat = tcm_feat.unsqueeze(-1)
        else:
            tcm_feat = tcm_feat.permute(0, 2, 1).unsqueeze(-1)  # (B,C,T,1)

        for tcm in self.tcm_list2:
            tcm_feat, _ = tcm(tcm_feat)

        dec_input = tcm_feat.squeeze(-1).reshape(batch_size, self.enc_out_channels, self.enc_out_frm_size,
                                                 nb_frames).permute(0, 1, 3, 2)
        self.dec_out_list = [dec_input]

        for layer_i, dec_layer in enumerate(self.dec_list):
            if self.config['merge_type'] == 'multicondition2':
                _, c, _, f = dec_input.shape
                dec_input = dec_input.permute(0, 2, 1, 3).reshape(batch_size, nb_frames, -1).permute(0, 2, 1)
                dec_input = self.append_cond(dec_input, [condition[0][layer_i + 3], condition[1][layer_i + 3]],
                                             self.norm_layers[layer_i])
                dec_input = dec_input.reshape(batch_size, c, f, nb_frames).permute(0, 1, 3, 2)
                dec_input = dec_layer(dec_input, None, self.frm_pad[::-1][layer_i], pad_size=self.pad_list[layer_i])
            else:
                dec_input = dec_layer(dec_input, None, self.frm_pad[::-1][layer_i], pad_size=self.pad_list[layer_i])

            self.dec_out_list.append(dec_input)

        dec_out = dec_input  # (B,2,T,161)
        ### prediction layer
        dec_out = F.pad(dec_out, (self.last_layer_pad[0], self.last_layer_pad[1], self.enc_time_kernel_len - 1, 0))
        x_out = self.out_conv(dec_out)
        self.dec_out_list.append(x_out)
        pred_1d = self.freq2time(x_out)
        return pred_1d, x_out

    def classify(self, spk_feat, phn_feat):
        result = {}
        if not self.config["disable_spk_classify"]:
            result['spk_predict_spk'] = self.speaker_classifier1(spk_feat.squeeze(-1))
        if not self.config["disable_phn_classify"]:
            result['phn_predict_phn'], phn_visual_list = self.phoneme_classifier1(phn_feat)
        # gradient reversal layer
        if self.config['disen_type'] == 'grl':
            if not self.config["disable_disen"]:
                if self.config["phn_grl"]:
                    phn_enc_feat_GRL = self.grl_phn(phn_feat)
                    phn_enc_feat_tap, phn_visual = self.tap_phn(phn_enc_feat_GRL)  # B,C,1
                    result['phn_predict_spk'] = self.speaker_classifier2(phn_enc_feat_tap.squeeze(-1))
                if self.config["spk_grl"]:
                    spk_enc_feat_GRL = self.grl_spk(spk_feat)
                    result['spk_predict_phn'], _ = self.phoneme_classifier2(spk_enc_feat_GRL)

        return result

    def merge(self, spk_feat, phn_feat):
        if self.config["use_temporal_aggregator"]:
            step = self.config['transmit_rate']
            T = spk_feat.shape[2]
            part = int(T // step)
            feature_transmit = torch.zeros_like(spk_feat).to(spk_feat)
            for part_id in range(part):
                start = part_id * step
                end = start + step if part_id < part - 1 else T
                feature_transmit[:, :, start:end] = torch.repeat_interleave(spk_feat[:, :, start].unsqueeze(-1),
                                                                            (end - start), dim=-1)  ##  error ?

            spk_feat_r = torch.repeat_interleave(feature_transmit, 10, dim=-1)  # B,T/10,C
            spk_feat = spk_feat_r

        if self.config["merge_type"] == 'concat':
            B, C_spk, _ = spk_feat.shape  #
            B, C_phn, T = phn_feat.shape  #
            spk_feat_repeat = spk_feat.expand(B, C_spk, T)
            merge_feat = torch.cat((spk_feat_repeat, phn_feat), dim=1)  # [B,C_spk+C_phn,T]
            # merge_feat = self.conv1x1_dec(merge_feat.unsqueeze(-1))  # [B,320,T,1]
            return merge_feat
        elif 'condition' in self.config["merge_type"]:
            condition = [[], []]  # scale,bias

            for pos_i in range(self.cond_positions):
                scale_in = spk_feat.permute(0, 2, 1)
                bias_in = spk_feat.permute(0, 2, 1)
                for scale_layer, bias_layer, flag in zip(self.scale_list[pos_i], self.bias_list[pos_i],
                                                         self.flag[pos_i]):
                    scale_in = scale_layer(scale_in)
                    bias_in = bias_layer(bias_in)
                    if flag:
                        scale_in = F.leaky_relu(scale_in, 0.1, inplace=True)
                        bias_in = F.leaky_relu(bias_in, 0.1, inplace=True)
                condition[0].append(scale_in.permute(0, 2, 1))
                condition[1].append(bias_in.permute(0, 2, 1))

            merge_feat = self.append_cond(phn_feat, [condition[0][0], condition[1][0]], self.norm_layer1)
            return merge_feat, condition

    def vq_bottleneck(self, spk_feat, phn_feat):
        result = {}
        bitrate = self.config['bitrate']
        if not self.config["disable_spk_vq"]:
            spk_result = self.quantize(spk_feat, type='spk')
            for k, v in spk_result.items():
                result['{}_{}'.format('spk', k)] = v
        else:
            result["spk_feat_vq"] = spk_feat

        if not self.config["disable_phn_vq"]:
            phn_result = self.quantize(phn_feat, type='phn')
            for k, v in phn_result.items():
                result['{}_{}'.format('phn', k)] = v
        else:
            result["phn_feat_vq"] = phn_feat

        return result

    def quantize(self, feat_in, type='spk'):
        # input shape [B,T,C_spk]
        result = {}
        vq_layer_out, vq_inds_out, prob_perplexity_list, code_perplexity_list = [], [], [], []
        codebook_penalty, encoder_penalty, codebook_usage, rate, entropy_loss, entropy, commitment_loss = 0, 0, 0, 0, 0, 0, 0
        if type == 'spk':
            cb_num, cb_dim = self.spk_codebook_num, self.spk_codebook_dim
            vq_layer_list = self.spk_vq_layer_list
            if self.config["use_entropy_loss"]:
                target_entropy_per_vqlayer = self.target_entropy_spk
                fuzz_entropy_per_vqlayer = self.fuzz_entropy_spk
        if type == 'phn':
            cb_num, cb_dim = self.phn_codebook_num, self.phn_codebook_dim
            vq_layer_list = self.phn_vq_layer_list
            if self.config["use_entropy_loss"]:
                target_entropy_per_vqlayer = self.target_entropy_phn
                fuzz_entropy_per_vqlayer = self.fuzz_entropy_phn

        feat_in = feat_in.permute(0, 2, 1)
        # result["feat_vq_in_1frame"] = feat_in  # B,T,C_spk

        if (not self.config['disable_spk_vq']) and type == 'spk':
            _combine_frames = 1
        else:
            _combine_frames = self.combine_frames
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
                vq_out = vq_layer(vq_in, target_entropy_per_vqlayer[layer_i], fuzz_entropy_per_vqlayer[layer_i])
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
                commitment_loss += vq_out["commitment_loss"] # commitment loss

        if _combine_frames > 1:
            feat_vq_out = torch.cat(vq_layer_out, dim=-1)
            # result["feat_vq_out"] = feat_vq_out.permute(0, 1, 3, 2).reshape(B, T // _combine_frames, -1)  # B,T,C_spk
            feat_vq_out = feat_vq_out.reshape(B, T, channels)
        else:
            feat_vq_out = torch.cat(vq_layer_out, dim=-1)  # B,T,320

        codebook_usage /= len(vq_layer_list)
        codebook_penalty /= len(vq_layer_list)
        encoder_penalty /= len(vq_layer_list)
        commitment_loss /= len(vq_layer_list)

        result["feat_vq"] = feat_vq_out.permute(0, 2, 1)  # B,C,T
        result["feat_vq_in"] = feat_in.permute(0, 2, 1)
        result["quantization_inds"] = torch.cat(vq_inds_out, dim=-1)  # B,T,spk_codebook_num
        result["prob_perplexity_list"] = prob_perplexity_list  # [spk_codebook_num]
        result["code_perplexity_list"] = code_perplexity_list  # [spk_codebook_num]
        result["codebook_usage"] = codebook_usage  # [1,]  avg_codebook_usage
        result["codebook_penalty"] = codebook_penalty  # [1,]  avg_codebook_penalty
        result["encoder_penalty"] = encoder_penalty  # [1,]  avg_encoder_penalty
        result["entropy_loss"] = entropy_loss  #
        result["entropy"] = entropy
        result["commitment_loss"] = commitment_loss

        return result

    def dequantize(self, phn_vq_inds):
        vq_layer_out = []
        if self.combine_frames > 1:
            B, T, _ = phn_vq_inds.shape  # B,T,C'
        for layer_i in range(len(self.phn_vq_layer_list)):
            vq_layer = self.phn_vq_layer_list[layer_i]
            vq_out = vq_layer.dequantize(phn_vq_inds[:, :, layer_i])
            if self.combine_frames > 1:
                vq_layer_out.append(vq_out.reshape(B, T, self.combine_frames, -1))  # [B,T/4,4,C'//codebook_num]
            else:
                vq_layer_out.append(vq_out)  # [B,T,C'//codebook_num]
        vq_feat = torch.cat(vq_layer_out, dim=-1)  # B,T,C'
        vq_feat_combine =vq_feat.permute(0, 1, 3, 2).reshape(B, T , -1) # B,T,C_spk
        if self.combine_frames > 1:
            vq_feat = vq_feat.reshape(B, T * self.combine_frames, -1)
        return vq_feat,vq_feat_combine
    def compute_contrastive_loss(self, feats_pos, feats_neg, neg_num):
        # features B,T,C
        feats_pos = feats_pos.permute(0, 2, 1)
        feats_neg = feats_neg.permute(0, 2, 1)
        negative_features = self._sample_negatives(feats_neg, neg_num)  # neg_num, B,T,C
        positive_features = feats_pos
        logits = self.compute_preds(feats_neg, positive_features, negative_features)  ## todo check
        return logits

    def _sample_negatives(self, features: torch.FloatTensor, num_negatives: int,
                          attention_mask: Optional[torch.LongTensor] = None):
        """
        Sample `num_negatives` vectors from feature vectors.
        """
        batch_size, sequence_length, hidden_size = features.shape
        if sequence_length <= 1:
            raise ValueError(
                f"`features should have `sequence_length` > 1, but are of shape (batch_size, sequence_length, hidden_size) = ({batch_size, sequence_length, hidden_size})."
            )

        features = features.contiguous().view(-1, hidden_size)  # BTC => (BxT)C

        with torch.no_grad():
            # get `num_negatives` random vector indices from the same utterance
            sampled_negative_indices = []
            for batch_idx in range(batch_size):
                high = attention_mask[batch_idx].sum() - 1 if attention_mask is not None else sequence_length - 1
                sampled_indices_slice = torch.randint(
                    0, high, size=(num_negatives * sequence_length,), device=features.device
                )
                sampled_negative_indices.append(sampled_indices_slice)

            sampled_negative_indices = torch.stack(sampled_negative_indices)

            # generate indices of the positive vectors themselves, repeat them `num_negatives` times
            feature_indices = (torch.arange(sequence_length, device=features.device)[:, None].expand(sequence_length,
                                                                                                     num_negatives).flatten())

            # avoid sampling the same positive vector, but keep the distribution uniform
            sampled_negative_indices[sampled_negative_indices >= feature_indices] += 1

        # correct for batch size
        for batch_idx in range(1, batch_size):
            sampled_negative_indices[batch_idx] += batch_idx * sequence_length

        # take negative vectors from sampled indices
        sampled_negatives = features[sampled_negative_indices.view(-1)]
        sampled_negatives = sampled_negatives.view(batch_size, sequence_length, num_negatives, hidden_size).permute(2,
                                                                                                                    0,
                                                                                                                    1,
                                                                                                                    3)

        return sampled_negatives

    def compute_preds(self, x, y, negatives):

        def is_xla_tensor(tensor):
            return torch.is_tensor(tensor) and tensor.device.type == "xla"

        def index_put(tensor, indices, value):
            if is_xla_tensor(tensor):
                for _ in range(indices.dim(), tensor.dim()):
                    indices = indices.unsqueeze(-1)
                if indices.size(-1) < tensor.size(-1):
                    indices = indices.expand_as(tensor)
                tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)
            else:
                tensor[indices] = value
            return tensor

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)
        self.logit_temp = 0.1
        logits = logits / self.logit_temp
        logits = logits.type_as(x)

        if is_xla_tensor(logits) or neg_is_pos.any():
            if not hasattr(self, "_inftensor"):
                fillval = -float(2 ** 30)
                self._inftensor = (
                    torch.tensor(fillval).to(x.device)
                    if is_xla_tensor(logits)
                    else float("-inf")
                )
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

        # if neg_is_pos.any():
        #     logits[1:][neg_is_pos] = float("-inf")
        # logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def get_logits(self, net_output):
        net_output = net_output[self.config['bitrate']]
        logits1, logits2 = net_output["logits"][0], net_output["logits"][1]
        logits1 = logits1.transpose(0, 2)
        logits1 = logits1.reshape(-1, logits1.size(-1))
        logits2 = logits2.transpose(0, 2)
        logits2 = logits2.reshape(-1, logits2.size(-1))
        return logits1, logits2

    def get_targets(self, net_output):
        net_output = net_output[self.config['bitrate']]
        x = net_output["logits"][0]
        target = x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)
        return target, target

    def _gru_init_state(self, n):
        if not torch._C._get_tracing_state():
            return self.h0.expand(-1, n, -1).contiguous()
        else:
            return self.h0.expand(self.h0.size(0), n, self.h0.size(2)).contiguous()

    def _spk_gru_init_state(self, n):
        # for encoder GRU
        if not torch._C._get_tracing_state():
            return self.spk_h0.expand(-1, n, -1).contiguous()
        else:
            return self.spk_h0.expand(self.spk_h0.size(0), n, self.spk_h0.size(2)).contiguous()

    def _phn_gru_init_state(self, n):
        # for encoder GRU
        if not torch._C._get_tracing_state():
            return self.phn_h0.expand(-1, n, -1).contiguous()
        else:
            return self.phn_h0.expand(self.phn_h0.size(0), n, self.phn_h0.size(2)).contiguous()

    def _norm_const(self, feat, const=20.):
        return feat / const

    def set_network_entropy_target(self, bitrate, fuzz, sample_rate, hop_len):
        bitrate_dict = {'0.25k': 250,'0.256k': 256, '0.512k': 512, '1k': 1000, '3k': 3000, '6k': 6000, '9k': 9000, '12k': 12000,
                        '24k': 24000}
        self.target_entropy = 0
        self.entropy_fuzz = 0
        total_bitrate = bitrate_dict[bitrate] * self.combine_frames
        self.target_entropy = bitrate_to_entropy(total_bitrate, sample_rate, hop_len)
        if self.config["disen_scheme"] == 'global_spk':
            total_bitrate_phn = bitrate_dict[bitrate] * self.combine_frames / self.phn_codebook_num
            self.target_entropy_phn = [bitrate_to_entropy(total_bitrate_phn, sample_rate,
                                                          hop_len)] * self.phn_codebook_num
            self.fuzz_entropy_phn = [0.] * self.phn_codebook_num
            self.target_entropy_spk = [10] * self.spk_codebook_num  ## global_spk 10bit todo
            self.fuzz_entropy_spk = [0.] * self.spk_codebook_num

    def reset_entropy_hists_train(self):
        if not self.config["disable_spk_vq"]:
            for vq_layer in self.spk_vq_layer_list:
                vq_layer.entropy_avg_train.reset()
        if not self.config["disable_phn_vq"]:
            for vq_layer in self.phn_vq_layer_list:
                vq_layer.entropy_avg_train.reset()

    def reset_entropy_hists_eval(self):
        if not self.config["disable_spk_vq"]:
            for vq_layer in self.spk_vq_layer_list:
                vq_layer.entropy_avg_eval.reset()
        if not self.config["disable_phn_vq"]:
            for vq_layer in self.phn_vq_layer_list:
                vq_layer.entropy_avg_eval.reset()

    def get_overall_entropy_avg_train(self):
        spk_avgs = []
        phn_avgs = []
        if not self.config["disable_spk_vq"]:
            for vq_layer in self.spk_vq_layer_list:
                spk_avgs.append(vq_layer.entropy_avg_train.avg)
        if not self.config["disable_phn_vq"]:
            for vq_layer in self.phn_vq_layer_list:
                phn_avgs.append(vq_layer.entropy_avg_train.avg)
        return torch.sum(torch.tensor(spk_avgs)), torch.sum(torch.tensor(phn_avgs))

    def get_overall_entropy_avg_eval(self):
        spk_avgs = []
        phn_avgs = []

        if not self.config["disable_spk_vq"]:
            for vq_layer in self.spk_vq_layer_list:
                spk_avgs.append(vq_layer.entropy_avg_eval.avg)
        if not self.config["disable_phn_vq"]:
            for vq_layer in self.phn_vq_layer_list:
                phn_avgs.append(vq_layer.entropy_avg_eval.avg)
        return torch.sum(torch.tensor(spk_avgs)), torch.sum(torch.tensor(phn_avgs))

    def init_codebook_size(self, config, bit_per_frame):
        if config['bitrate'] == '0.25k':
            if config["combineVQ_frames"] == 4:  # 40ms  reduntant: 16bit
                if config["disen_scheme"] == 'global_spk':
                    self.spk_latent_dim, self.phn_latent_dim = 80, 80
                    self.spk_codebook_num = 4  # 1
                    self.spk_codebook_size = [1024 for i in range(self.spk_codebook_num)]  # per 1s
                    self.phn_codebook_num = 1
                    self.phn_codebook_size = [1024 for i in range(self.phn_codebook_num)]  # 12bit per 40ms
        if config['bitrate'] == '0.256k':
            if (bit_per_frame < 5) and (bit_per_frame > 1):  # 10ms  2.5bit
                if config["combineVQ_frames"] == 4:  # 40ms  reduntant: 16bit
                    if config["disen_scheme"] == 'global_spk':
                        self.spk_latent_dim, self.phn_latent_dim = 80, 80
                        self.spk_codebook_num = 4  # 1
                        self.spk_codebook_size = [1024 for i in range(self.spk_codebook_num)]  # per 1s
                        if config['vq_rate'] == 'redundant2':
                            self.phn_codebook_num = 1
                            self.phn_codebook_size = [4096 for i in range(self.phn_codebook_num)]  # 12bit per 40ms
                        else:
                            self.phn_codebook_num = 2
                            self.phn_codebook_size = [256 for i in range(self.phn_codebook_num)]
                elif config["combineVQ_frames"] == 1:  # 10ms  reduntant: 5bit
                    if config["disen_scheme"] == 'global_spk':
                        self.spk_codebook_num, self.phn_codebook_num = 1, 1
                        self.spk_latent_dim, self.phn_latent_dim = 80, 80
                        self.spk_codebook_size = [32 for i in range(self.spk_codebook_num)]
                        self.phn_codebook_size = [32 for i in range(self.phn_codebook_num)]
        elif config['bitrate'] == '0.512k':
            if bit_per_frame == 5:  # 10ms  5bit
                if config["combineVQ_frames"] == 4:  # 40ms  reduntant: 27bit
                    if config["disen_scheme"] == 'global_spk':
                        self.spk_codebook_num, self.phn_codebook_num = 1, 3
                        self.spk_latent_dim, self.phn_latent_dim = 81, 81
                        self.spk_codebook_size = [512 for i in range(self.spk_codebook_num)]
                        self.phn_codebook_size = [512 for i in range(self.phn_codebook_num)]
                elif config["combineVQ_frames"] == 1:  # 10ms  reduntant: 8bit
                    if config["disen_scheme"] == 'global_spk':
                        self.spk_codebook_num, self.phn_codebook_num = 1, 1
                        self.spk_latent_dim, self.phn_latent_dim = 81, 81
                        self.spk_codebook_size = [256 for i in range(self.spk_codebook_num)]
                        self.phn_codebook_size = [256 for i in range(self.phn_codebook_num)]
        elif config['bitrate'] == '1k':
            if bit_per_frame == 10:  # 10ms  10bit
                if config["combineVQ_frames"] == 4:  # 40ms  complete:40bit,reduntant:60bit
                    if config["disen_scheme"] == 'global_spk':
                        self.spk_latent_dim, self.phn_latent_dim = 288, 288
                        self.spk_codebook_num = 16  # 1
                        self.spk_codebook_size = [1024 for i in range(self.spk_codebook_num)]  # per 1s
                        self.phn_codebook_num = 6
                        self.phn_codebook_size = [1024 for i in range(self.phn_codebook_num)]
                elif config['combineVQ_frames'] == 1:  # 10ms  complete:10bit,reduntant:16bit
                    if config["disen_scheme"] == 'global_spk':
                        self.spk_codebook_num, self.phn_codebook_num = 1, 2
                        self.spk_latent_dim, self.phn_latent_dim = 192, 192
                        self.spk_codebook_size = [1024 for i in range(self.spk_codebook_num)]
                        self.phn_codebook_size = [256 for i in range(self.phn_codebook_num)]
        elif config['bitrate'] == '3k':
            if bit_per_frame == 30:  # 10ms  30bit
                if config["combineVQ_frames"] == 4:  # 40ms  complete:120bit,reduntant:160bit
                    if config["disen_scheme"] == 'global_spk':
                        self.spk_codebook_num, self.phn_codebook_num = 1, 16
                        self.spk_latent_dim, self.phn_latent_dim = 288, 288
                        self.spk_codebook_size = [1024 for i in range(self.spk_codebook_num)]
                        self.phn_codebook_size = [1024 for i in range(self.phn_codebook_num)]
                elif config['combineVQ_frames'] == 1:  # 10ms  complete:30bit,reduntant:40bit
                    if config["disen_scheme"] == 'global_spk':
                        self.spk_codebook_num, self.phn_codebook_num = 1, 4
                        self.spk_latent_dim, self.phn_latent_dim = 192, 192
                        self.spk_codebook_size = [1024 for i in range(self.spk_codebook_num)]
                        self.phn_codebook_size = [1024 for i in range(self.phn_codebook_num)]
        elif config["bitrate"] == '6k':
            if bit_per_frame == 60:  # 10ms  60bit
                if config["combineVQ_frames"] == 4:  # 40ms  complete:240bit,reduntant:320bit
                    if config["disen_scheme"] == 'global_spk':
                        self.spk_codebook_num, self.phn_codebook_num = 1, 32
                        self.spk_latent_dim, self.phn_latent_dim = 288, 288
                        self.spk_codebook_size = [1024 for i in range(self.spk_codebook_num)]
                        self.phn_codebook_size = [1024 for i in range(self.phn_codebook_num)]
                    if config["disen_scheme"] == 'rate1':
                        self.spk_codebook_num, self.spk_latent_dim = 8, 24
                        self.spk_latent_dim, self.phn_latent_dim = 288, 288
                        self.spk_codebook_size = [1024 for i in range(self.spk_codebook_num)]
                        self.phn_codebook_size = [1024 for i in range(self.phn_codebook_num)]
                elif config["combineVQ_frames"] == 1:
                    if config["disen_scheme"] == 'global_spk':
                        self.spk_codebook_num, self.phn_codebook_num = 1, 8
                        self.spk_latent_dim, self.phn_latent_dim = 192, 192
                        self.spk_codebook_size = [1024 for i in range(self.spk_codebook_num)]
                        self.phn_codebook_size = [1024 for i in range(self.phn_codebook_num)]

    def update_temperature_gumbel(self, cur_iter):
        if 'Gumbel' in self.config["vq_type"]:
            if not self.config["disable_spk_vq"]:
                for vq_layer in self.spk_vq_layer_list:
                    vq_layer.temp_updates(cur_iter)
            if not self.config["disable_phn_vq"]:
                for vq_layer in self.phn_vq_layer_list:
                    vq_layer.temp_updates(cur_iter)

