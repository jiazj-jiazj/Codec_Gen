import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_layer import ConvLayer

class ChannelSA3dCausal(nn.Module):
    def __init__(self, num_channels, local_temporal_depth=32, ds_rate=1, res_skip=True, nonlinearity=nn.ReLU, bn=True):
        super(ChannelSA3dCausal, self).__init__()
        self.num_channels = num_channels
        self.local_temporal_depth = local_temporal_depth
        self.ds_rate = ds_rate
        self.res_skip = res_skip
        self.csa_scale = local_temporal_depth // ds_rate
        self.key_conv = ConvLayer(local_temporal_depth, self.csa_scale, (1, 1), (1, 1), bn=bn, nonlinearity=nonlinearity)
        self.query_conv = ConvLayer(local_temporal_depth, self.csa_scale, (1, 1), (1, 1), bn=bn, nonlinearity=nonlinearity)
        #self.value_conv = ConvLayer(local_temporal_depth, local_temporal_depth, (1, 1), (1, 1), bn=bn, nonlinearity=nonlinearity)
        if not res_skip:
            self.merge = ConvLayer(2*num_channels, num_channels, (1, 1), (1, 1), bn=bn, nonlinearity=nonlinearity)
        self.softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self, x):
        # input shape: (B,CF,T,1)
        x_pad = F.pad(x.squeeze(-1), (self.local_temporal_depth-1, 0), "constant", 0) # [B, C, T+K-1]
        framed_x = x_pad.unfold(-1, self.local_temporal_depth, 1).permute(0,3,2,1)  # [B, K, T, C]

        key = self.key_conv(framed_x).permute(0,2,1,3) # [B, T, K, C]
        query = self.query_conv(framed_x).permute(0,2,3,1)# [B, T, C, K]
        value = x.permute(0,2,1,3) # [B, T, C, 1]
        ssa_mat = torch.matmul(query, key) / (self.csa_scale**0.5) # [B, T, C, C]
        ssa_softmax = self.softmax(ssa_mat) # [B, T, C, C]            
        att_output = torch.matmul(ssa_softmax, value)  # [B, T, C, 1]
        att_output = att_output.permute(0,2,1,3) # [B, C, T, 1]            
            
        if self.res_skip:        
            outputs = x + att_output
        else:
            outputs = torch.cat((x, att_output), 1)  # (B,2CF,T,1)
            outputs = self.merge(output)  # (B,CF,T,1)
        
        return outputs
        
        
class ChannelSA3dCausal_temporalDS(ChannelSA3dCausal):
    def __init__(self, num_channels, local_temporal_depth=32, ds_rate=1, temporal_DS_rate=8, res_skip=True, nonlinearity=nn.ReLU, bn=True):
        super().__init__(num_channels=num_channels, local_temporal_depth=local_temporal_depth, ds_rate=ds_rate, res_skip=res_skip, nonlinearity=nonlinearity, bn=bn)   
        self.temporal_DS_rate = temporal_DS_rate
        
    def forward(self, x):
        # input shape: (B,CF,T,1)
        num_frames = x.size()[2]
        x_pad = F.pad(x.squeeze(-1), (self.local_temporal_depth-1, 0), "constant", 0) # [B, C, T+K-1]
        framed_x = x_pad.unfold(-1, self.local_temporal_depth, self.temporal_DS_rate).permute(0,3,2,1)  # [B, K, T/4, C]

        key = self.key_conv(framed_x).permute(0,2,1,3) # [B, T/4, K, C]
        query = self.query_conv(framed_x).permute(0,2,3,1)# [B, T/4, C, K]
        value = x.permute(0,2,1,3) # [B, T, C, 1]
        ssa_mat = torch.matmul(query, key) / (self.csa_scale**0.5) # [B, T/4, C, C]
        ssa_mat = torch.repeat_interleave(self.softmax(ssa_mat), self.temporal_DS_rate, dim=1) # [B, T, C, C]
        att_output = torch.matmul(ssa_mat[:,0:num_frames,:,:], value)  # [B, T, C, 1]
        att_output = att_output.permute(0,2,1,3) # [B, C, T, 1]            
            
        if self.res_skip:        
            outputs = x + att_output
        else:
            outputs = torch.cat((x, att_output), 1)  # (B,2CF,T,1)
            outputs = self.merge(output)  # (B,CF,T,1)
        
        return outputs