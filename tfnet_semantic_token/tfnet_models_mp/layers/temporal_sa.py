import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_layer import ConvLayer

class TemporalSA(nn.Module):
    def __init__(self, in_channels, num_channels, num_heads=1, temporal_depth=None, ds_rate=4, res_skip=False, nonlinearity=nn.ReLU, bn=True, causal=True):
        super(TemporalSA, self).__init__()
        self.num_channels = num_channels
        self.temporal_depth = temporal_depth
        self.num_heads = num_heads
        self.ds_rate = ds_rate
        self.res_skip = res_skip
        self.tsa_scale = num_channels // ds_rate
        self.causal = causal
        self.key_conv = ConvLayer(in_channels, self.tsa_scale, (1, 1), (1, 1), bn=bn, nonlinearity=nonlinearity)
        self.query_conv = ConvLayer(in_channels, self.tsa_scale, (1, 1), (1, 1), bn=bn, nonlinearity=nonlinearity)
        self.value_conv = ConvLayer(in_channels, num_channels, (1, 1), (1, 1), bn=bn, nonlinearity=nonlinearity)
        if not res_skip:
            self.merge = ConvLayer(2*in_channels, num_channels, (1, 1), (1, 1), bn=bn, nonlinearity=nonlinearity)
        self.softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self, x, y=None, previous_frame_features=None):
        # input shape: (B,CF,T,1)
        if y is not None:
            key = self.key_conv(y).squeeze(-1)  # (B,CF,T)
            query = self.query_conv(x).permute(0,2,3,1)  # (B,T,1,CF)
            value = self.value_conv(y).squeeze(-1)  # (B,CF,T)
        else:
            key = self.key_conv(x).squeeze(-1)  # (B,CF,T)
            query = self.query_conv(x).permute(0,2,3,1)  # (B,T,1,CF)
            value = self.value_conv(x).squeeze(-1)  # (B,CF,T)
        nb_frames = x.shape[2]

        # for causal
        if previous_frame_features is not None:
            key_context, value_context = torch.split(previous_frame_features, [self.tsa_scale,self.num_channels], dim=1)                
            key = torch.cat((key_context, key), dim=-1)
            value = torch.cat((value_context, value), dim=-1)
            key_input = key
            value_input = value
        else:            
            key = F.pad(key, (self.temporal_depth-1, 0), "constant", 0) if self.causal else F.pad(key, (int((self.temporal_depth-1)//2), int((self.temporal_depth-1)//2)), "constant", 0)
            value = F.pad(value, (self.temporal_depth-1, 0), "constant", 0) if self.causal else F.pad(value, (int((self.temporal_depth-1)//2), int((self.temporal_depth-1)//2)), "constant", 0)
            
        if self.num_heads > 1:            
            key_list = torch.chunk(key, self.num_heads, dim=1)
            value_list = torch.chunk(value, self.num_heads, dim=1)
            query_list = torch.chunk(query, self.num_heads, dim=-1)
            att_output_list = [None for i in range(self.num_heads)]            
            for i in range(self.num_heads):
                framed_key = key_list[i].unfold(-1, self.temporal_depth, 1).permute(0,2,1,3)  # [B, T, CF, K]
                framed_value = value_list[i].unfold(-1, self.temporal_depth, 1).permute(0,2,3,1)  # [B, T, K, CF]
                ssa_mat = torch.matmul(query_list[i], framed_key) # [B, T, 1, K]
                ssa_scale = ssa_mat / (self.tsa_scale**0.5)
                ssa_softmax = self.softmax(ssa_scale) # [B, T, 1, K]
                att_output_list[i] = torch.matmul(ssa_softmax, framed_value)  # [B, T, 1, CF]                
            att_output = torch.cat(att_output_list, dim=-1)
            att_output = att_output.permute(0,3,1,2) # [B, CF, T, 1]
        else:                         
            framed_key = key.unfold(-1, self.temporal_depth, 1).permute(0,2,1,3)  # [B, T, CF, K]
            framed_value = value.unfold(-1, self.temporal_depth, 1).permute(0,2,3,1)  # [B, T, K, CF]
            ssa_mat = torch.matmul(query, framed_key) # [B, T, 1, K]
            ssa_scale = ssa_mat / (self.tsa_scale**0.5)
            ssa_softmax = self.softmax(ssa_scale) # [B, T, 1, K]            
            att_output = torch.matmul(ssa_softmax, framed_value)  # [B, T, 1, CF]
            att_output = att_output.permute(0,3,1,2) # [B, CF, T, 1]
            
        if self.res_skip:        
            outputs = x + att_output
        else:
            outputs = torch.cat((x, att_output), 1)  # (B,2CF,T,1)
            outputs = self.merge(output)  # (B,CF,T,1)
            
        if previous_frame_features is not None:
            current_feature = torch.cat((key_input, value_input), dim=1) # (B,C,T)
        else:
            current_feature = None
        
        return outputs, current_feature