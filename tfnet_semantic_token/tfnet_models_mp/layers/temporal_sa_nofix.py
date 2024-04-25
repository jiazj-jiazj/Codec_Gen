import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_layer import ConvLayer


class TemporalSA(nn.Module):
    def __init__(self, in_channels, num_channels, temporal_depth=None, ds_rate=4, res_skip=False, nonlinearity=nn.ReLU, bn=True):
        super(TemporalSA, self).__init__()
        self.num_channels = num_channels
        self.temporal_depth = temporal_depth
        self.ds_rate = ds_rate
        self.res_skip = res_skip
        self.tsa_scale = num_channels // ds_rate
        self.key_conv = ConvLayer(in_channels, self.tsa_scale, (1, 1), (1, 1), bn=bn, nonlinearity=nonlinearity)
        self.query_conv = ConvLayer(in_channels, self.tsa_scale, (1, 1), (1, 1), bn=bn, nonlinearity=nonlinearity)
        self.value_conv = ConvLayer(in_channels, num_channels, (1, 1), (1, 1), bn=bn, nonlinearity=nonlinearity)
        if not res_skip:
            self.merge = ConvLayer(num_channels, num_channels, (1, 1), (1, 1), bn=bn, nonlinearity=nonlinearity)
        # self.softmax = torch.nn.Softmax(dim=2)
        
    def forward(self, x):
        # input shape: (B,CF,T,1)
        key = self.key_conv(x).squeeze(-1)                     # (B,CF,T)
        query = self.query_conv(x).squeeze(-1).permute(0,2,1)  # (B,T,CF)
        value = self.value_conv(x).squeeze(-1).permute(0,2,1)  # (B,T,CF)
        nb_frames = x.shape[2]

        # for causal, to make a mask
        diag_mask = torch.tril(torch.ones(nb_frames, nb_frames), diagonal=0).unsqueeze(0)  # lower diagonal
        if self.temporal_depth is not None:
            diag_small_lower = torch.triu(torch.ones_like(diag_mask), diagonal=-(self.temporal_depth-1))
            diag_mask = diag_mask*diag_small_lower
        tsa_mat = torch.matmul(query, key) * diag_mask.to(key) / (self.tsa_scale**0.5)  # (B,T,T)
        tsa_mat = torch.exp(tsa_mat - torch.max(tsa_mat, 2, keepdim=True)[0]) * diag_mask.to(key)
        tsa_mat = tsa_mat / (torch.sum(tsa_mat, 2, keepdim=True)[0] + torch.tensor(1e-30).to(key))
        tsa_out = torch.matmul(tsa_mat, value)  # (B,T,CF)
        tsa_out = tsa_out.permute(0,2,1).unsqueeze(-1)  # (B,CF,T,1)

        if self.res_skip:
            output = x + tsa_out
        else:
            output = torch.cat((x, tsa_out), 1)  # (B,2CF,T,1)
            output = self.merge(output)  # (B,CF,T,1)
        
        return output