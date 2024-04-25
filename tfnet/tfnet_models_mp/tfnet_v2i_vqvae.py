import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .layers.time2freq2 import Time2Freq2
from .layers.conv_layer import ConvLayer
from .layers.deconv_layer import DeconvLayer
from .layers.tcm_layer import TCMLayer, TCMLayerWithMask
from .layers.vq_layer_ema import VQEmbeddingEMA_DDP
from .layers.vq_layer_predictive_ema import VQEmbeddingEMA_Predictive_DDP
from .layers.vq_layer_gumbel import GumbelVectorQuantizer
from .layers.vq_layer_predictive_gumbel import GumbelVectorQuantizer_Predictive
from .layers.freq2timecodec import Freq2TimeCodec
from .layers.temporal_sa import TemporalSA as TemporalSA
from tfnet.utils.tools import *
from .multiframe_vq_bottleneck import MultiFrmVQBottleNeck

def time2featMask(in_x):
    frm_size = 320
    frm_half = 160
    frm_shift = 80    
    
    loss_mask = F.pad(in_x.to(torch.float32), (0, frm_shift), "constant", 0)   # 1d
    loss_mask = loss_mask.unfold(-1, frm_shift, frm_shift)  # each 20ms loss corresponds to two 10ms feature loss
    loss_mask = torch.unsqueeze(loss_mask, 1) # (B, 1, T, 80)
    loss_mask = torch.clamp(torch.sum(loss_mask, dim=-1, keepdim=True), min=0.0, max=1.0) # (B, 1, T, 1)
    return loss_mask.to(torch.float32)
    
class TFNet(MultiFrmVQBottleNeck):
    def __init__(self, config=None, bn=True, enableBPstft=False, center=True):
        #super(TFNet, self).__init__()
        super().__init__(config=config, bitrate=config['bitrate'], feat_dim=320)   

        frm_size = config["dft_size"]
        shift_size = int(config["dft_size"] * config["hop_vqvae"])
        self.win_len = n_fft = frm_size
        self.hop_len = shift_size
        self.frm_size = frm_size
        self.pad_len = 0

        self.in_channels = 2
        bn = config["bn"]
        activation_functions = {'PRELU':nn.PReLU,'ELU':nn.ELU}
        activation = activation_functions[config['activation']]
        self.config = config
        
        if config["use_learnable_compression"]:
            self.power_cprs = nn.Parameter(torch.FloatTensor(1))
            nn.init.constant_(self.power_cprs, 0.4)#config["power"])

        ### input layer
        self.time2freq = Time2Freq2(frm_size, shift_size, self.win_len, self.hop_len, n_fft, config=config, power=self.power_cprs if config["use_learnable_compression"] else None, enableBP=enableBPstft, center=center)

        ### encoder
        enc_kernel_size = (2, 5)            
        enc_channels = (16, 32, 64, 64)
        enc_strides = ((1, 1), (1, 4), (1, 4), (1, 2)) # 161, 41, 11, 5
        self.frm_pad = ((2, 2), (2, 2), (2, 2), (1, 1))
        self.last_layer_pad = (2, 2)        

        self.enc_out_frm_size = 5
        self.enc_out_channels = enc_channels[-1]
        self.tcm_mid_channels = 512 # enlarge from 256
        self.tcm_repeat1, self.tcm_block1 = 1, 4
        self.gru_num = 1
        self.tcm_repeat2, self.tcm_block2 = 1, 4
        self.depth_ker_size = 5
        
        self.enc_time_kernel_len = enc_kernel_size[0]
        is_last = [False, False, False, True]
        self.pad_list = [None, None, None, (self.frm_pad[0], (self.enc_time_kernel_len - 1, 0))]
        self.pad_list_frmwise = [None, None, None, (self.frm_pad[0], (0, 0))]       
        self.feat_dim = self.enc_out_frm_size * self.enc_out_channels
        assert(self.feat_dim == 320)  # used for VQ module initialization
        # encoder
        self.enc_list = nn.ModuleList(
            ConvLayer(in_c, out_c, enc_kernel_size, enc_strides[i], nonlinearity=activation, bn=bn)
            for i, (in_c, out_c) in enumerate(
                zip((self.in_channels,) + enc_channels[:-1], enc_channels)))
                
        ## encoder tcm
        if config["use_encoder_tcm"]:
            self.tcm_list_enc = nn.ModuleList(TCMLayer(
                self.feat_dim, self.tcm_mid_channels, self.depth_ker_size, 2 ** i if not config["use_less_dilations"] else 1, nonlinearity=activation, bn=bn, tcm_prelu_fix=config["tcm_prelu_fix"])
                                           for _ in range(self.tcm_repeat1) for i in range(self.tcm_block1))
        ## encoder gru
        if config["use_encoder_gru"]:
            self.gru_enc = nn.GRU(self.feat_dim, self.feat_dim, num_layers=self.gru_num, bias=True, batch_first=True)
            self.h0_enc = nn.Parameter(torch.FloatTensor(self.gru_num, 1, self.feat_dim).zero_())
        # decoder
        ## tcm-gru-tcm
        self.tcm_list1 = nn.ModuleList(TCMLayer(self.feat_dim, self.tcm_mid_channels, self.depth_ker_size, 2 ** i if not config["use_less_dilations"] else 1, nonlinearity=activation, bn=bn, tcm_prelu_fix=config["tcm_prelu_fix"])
                                    for _ in range(self.tcm_repeat1) for i in range(self.tcm_block1))
        self.gru = nn.GRU(self.feat_dim, self.feat_dim, num_layers=self.gru_num, bias=True, batch_first=True)        
        self.h0 = nn.Parameter(torch.FloatTensor(self.gru_num, 1, self.feat_dim).zero_())
        self.tcm_list2 = nn.ModuleList(TCMLayer(
            self.feat_dim, self.tcm_mid_channels, self.depth_ker_size, 2 ** i if not config["use_less_dilations"] else 1, nonlinearity=activation, bn=bn, tcm_prelu_fix=config["tcm_prelu_fix"])
                                        for _ in range(self.tcm_repeat2) for i in range(self.tcm_block2))

        out_dim = 2
        self.dec_list = nn.ModuleList(
            DeconvLayer(in_c, out_c, enc_kernel_size,enc_strides[len(self.enc_list) - 1 - i], nonlinearity=activation, bn=bn, is_last=is_last[i])
            for i, (in_c, out_c) in enumerate(
                zip(enc_channels[::-1], enc_channels[::-1][1:] + (out_dim,)))
        )

        ### last layer
        self.out_conv = nn.Conv2d(out_dim, out_dim, enc_kernel_size, (1, 1), bias=False)
        nn.init.xavier_normal_(self.out_conv.weight)
        self.freq2time = Freq2TimeCodec(frm_size, shift_size, self.win_len, self.hop_len, n_fft, power=self.power_cprs if config["use_learnable_compression"] else None, config=config, center=center)

    def forward(self, inputs, ar_signal=None, epo=None):
        self.signal = inputs
        self.signal_len = self.signal.shape[1]

        num_frames = int(self.signal_len / self.hop_len) + 1
        self.pad_len = 0
        if num_frames % self.combine_frames:
            padded_frames = self.combine_frames - num_frames % self.combine_frames
            self.pad_len = int(padded_frames * self.hop_len)
            self.signal = F.pad(self.signal, (self.pad_len, 0))
            if ar_signal is not None:
                ar_signal = F.pad(ar_signal, (self.pad_len, 0))
            if self.add_packet_loss:
                self.pad_frm = padded_frames
                loss_mask = F.pad(loss_mask, (0, 0, self.pad_frm, 0))

        in_feat, self.input_stft_r = self.time2freq(self.signal)        
        enc_feat = self.encoder(in_feat)    

        vq_result = self.vq_bottleneck(enc_feat, loss_mask=loss_mask if self.add_packet_loss else None, epo=epo)
        return vq_result["vq_feat"], vq_result["quantization_inds"], vq_result["commitment_loss"]

    def encoder(self, in_feat):
        x_conv = in_feat  # B,2,T,161
        # print(x_conv.shape)
        # quit()
        ### encoder
        self.enc_out_list = [x_conv]
        for layer_i in range(len(self.enc_list)):
            enc_layer = self.enc_list[layer_i]
            enc_out = enc_layer(self.enc_out_list[-1], (self.frm_pad[layer_i],(self.enc_time_kernel_len-1,0)))
            self.enc_out_list.append(enc_out)

        enc_out = self.enc_out_list[-1]  ##B,64,T,5
        if self.config["use_encoder_tcm"]:
            batch_size, nb_frames = enc_out.shape[0], enc_out.shape[2]
            tcm_feat = enc_out.permute(0, 2, 1, 3).reshape(batch_size, nb_frames, -1).permute(0, 2, 1).unsqueeze(-1)  # (B,C,T,1)
            for tcm in self.tcm_list_enc:
                tcm_feat ,_= tcm(tcm_feat)
            enc_out = tcm_feat.squeeze(-1).reshape(batch_size, self.enc_out_channels, self.enc_out_frm_size,
                                                   nb_frames).permute(0, 1, 3, 2)
        if self.config["use_encoder_gru"]:
            rnn_in = enc_out.permute(0, 2, 1, 3).reshape(batch_size, nb_frames, -1).permute(0, 2, 1).unsqueeze(
                -1)  # (B,C,T,1)
            rnn_in = rnn_in.squeeze(-1).permute(0, 2, 1)  # (B,T,C)
            rnn_out, h_n_enc = self.gru_enc(rnn_in, self._gru_init_state_enc(batch_size))
            rnn_out = rnn_out.permute(0, 2, 1).unsqueeze(-1)  # (B,C,T,1)
            enc_out = rnn_out
        return enc_out

    def decoder(self,vq_feat,plmask=None,ar_signal=None,pad_len=None):
        if pad_len is None:
            pad_len = self.pad_len
        ### interleave structure
        batch_size, nb_frames = vq_feat.shape[0], vq_feat.shape[2]
        tcm_feat = vq_feat.permute(0, 2, 1, 3).reshape(batch_size, nb_frames, -1).permute(0, 2, 1).unsqueeze(-1)  # (B,C,T,1)
        if self.add_packet_loss and (not (self.plc_unit_type=='none')):
            size_feat = tcm_feat.size()
            plmask2 = plmask.to(torch.float32).repeat(1, size_feat[1], 1, size_feat[-1]) # (B, C, T, 1)
            idx = 0            
            for tcm in self.tcm_list1:
                if (self.plc_unit_type=='full') or (0 == idx):
                    tcm_feat,_ = tcm([tcm_feat, plmask2])
                else:
                    tcm_feat,_ = tcm(tcm_feat) 
                idx += 1                    
        else:
            for tcm in self.tcm_list1:
                tcm_feat,_ = tcm(tcm_feat)
        
        if self.config["autoregressive"]:
            if ar_signal is None:
                return self.decoder_ar(in_feat=tcm_feat, plmask=plmask)
            else:
                ar_signal = ar_signal.detach()
                ar_latency = self.frm_size + (self.combine_frames-1)*self.hop_len
                ar_pad = torch.zeros((batch_size, ar_latency)).to(vq_feat) # 35ms latency when hop=5ms
                ar_input = torch.cat((ar_pad, ar_signal[:, :-ar_latency]), dim=1)               
                in_feat, _ = self.time2freq(ar_input)                
                ar_feat = self.encoder_ar(in_feat) # (B, C, T, 1)
                #ar_pad = torch.zeros((batch_size, self.feat_dim, 7, 1)).to(ar_feat) # 7 overlapped frames before are fully obtained when hop=5ms
                #ar_feat = torch.cat((ar_pad, ar_feat[:, :, :-7, :]), dim=2)
                if self.config["autoregressive_merge"] == 'concat':
                    tcm_feat = torch.cat((tcm_feat, ar_feat), dim=1) # (B, C, T, 1)
                else:
                    tcm_feat, _ = self.tsa_merge(tcm_feat, y=ar_feat)
 
        tcm_feat = tcm_feat.squeeze(-1).permute(0, 2, 1)  # (B,T,C)
        if self.add_packet_loss and (not (self.plc_unit_type=='none')):
            if self.plc_unit_type=='full':
                tcm_feat_mask = torch.cat((tcm_feat, plmask2.squeeze(-1).permute(0, 2, 1)),-1)
            else:
                tcm_feat_mask = torch.cat((tcm_feat, plmask.squeeze(-1).permute(0, 2, 1)),-1) 
            tcm_feat, h_n = self.gru(tcm_feat_mask, self._gru_init_state(batch_size))
        else:
            tcm_feat, h_n = self.gru(tcm_feat, self._gru_init_state(batch_size))
        tcm_feat = tcm_feat.permute(0, 2, 1).unsqueeze(-1)  # (B,C,T,1)

        if self.add_packet_loss and self.plc_unit_type=='full':
            for tcm in self.tcm_list2:
                tcm_feat,_ = tcm([tcm_feat, plmask2])
        else:
            for tcm in self.tcm_list2:
                tcm_feat,_ = tcm(tcm_feat)

        dec_input = tcm_feat.squeeze(-1).reshape(batch_size, self.enc_out_channels, self.enc_out_frm_size, nb_frames).permute(0, 1, 3, 2)        
        self.dec_out_list = [dec_input]
        for layer_i, dec_layer in enumerate(self.dec_list):
            dec_input = dec_layer(dec_input, None, self.frm_pad[::-1][layer_i], pad_size=self.pad_list[layer_i])
            self.dec_out_list.append(dec_input)

        dec_out = dec_input  # (B,2,T,161)
        ### prediction layer
        dec_out = F.pad(dec_out, (self.last_layer_pad[0], self.last_layer_pad[1], self.enc_time_kernel_len - 1, 0))
        x_out = self.out_conv(dec_out)
        pred_1d = self.freq2time(x_out)
        if self.pad_len > 0:
            pred_1d = pred_1d[:,pad_len:]
            
        return pred_1d, x_out

    def get_gru_cell(self,gru):
        # for teacher forcing frame by frame inference
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def _gru_init_state(self, n):
        if not torch._C._get_tracing_state():
            return self.h0.expand(-1, n, -1).contiguous()
        else:
            return self.h0.expand(self.h0.size(0), n, self.h0.size(2)).contiguous()

    def _gru_init_state_enc(self, n):
        # for encoder GRU
        if not torch._C._get_tracing_state():
            return self.h0_enc.expand(-1, n, -1).contiguous()
        else:
            return self.h0_enc.expand(self.h0_enc.size(0), n, self.h0_enc.size(2)).contiguous()

    def _norm_const(self, feat, const=20.):
        return feat / const

    def load(self, path, strict=True,):

        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location='cpu')
        pkg = pkg["gen"] if "gen" in pkg.keys() else pkg
        new_pkg = dict()
        for k, v in pkg.items():
            if "module" in k:
                new_pkg[k.split('module.')[-1]] = v
            else:
                new_pkg[k] = v
        
        self.load_state_dict(new_pkg, strict=strict)

    def freeze_encoder(self,):
        encoder_layers = ['enc_list', 'tcm_list_enc', 'gru_enc', 'conv1x1_1',  'time2freq']
        encoder_params = ['power_cprs', 'h0_enc']
        freeze_or_unfreeze_module(self, encoder_layers, freeze=True)
        freeze_or_unfreeze_para(self, encoder_params, freeze=True)

    def freeze_codebook(self,):
        codebook_layers = ['vq_layer_list', 'vq_layer']
        freeze_or_unfreeze_module(self, codebook_layers, freeze=True)

    def freeze_decoder(self,):
        decoder_layers = ['conv1x1_2', 'tcm_list1', 'tcm_list2', 'gru', 'dec_list', 'out_conv', 'freq2time', 'decoders']
        decoder_params = ['h0']
        freeze_or_unfreeze_module(self, decoder_layers, freeze=True)
        freeze_or_unfreeze_para(self, decoder_params, freeze=True)

    def decode_from_codebook_indices(self, quantized_indices, pad_len=None):
        B = quantized_indices.shape[0]
        codes = self.dequantize(quantized_indices)
        codes = codes.permute(0, 2, 1).unsqueeze(-1)  # [B,360,T,1]
        x = self.conv1x1_2(codes)
        return self.decoder(x, pad_len=pad_len)

    def decode_from_codebook_logits(self, logits, pad_len=None):
        B = logits.shape[0]

        if self.training:
            # print(x.view(bsz * tsz * self.groups, -1).argmax(dim=-1))
            logits = F.gumbel_softmax(logits, tau=1, hard=True).type_as(logits)
        else:
            softmax = nn.Softmax(dim=1)
            logits = torch.argmax(softmax(logits), 1)
        codes = self.dequantize_from_logits(logits)
        nb_frames = codes.shape[2]
        x = codes.squeeze(-1).reshape(B, self.enc_out_channels, self.enc_out_frm_size, nb_frames).permute(0, 1, 3, 2)
        return self.decoder(x, pad_len=pad_len)
