import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal_sa import TemporalSA


class Conv1dGroupTSALayer(nn.Module):
    def __init__(self, in_channels, mid_channels, num_groups=4, temporal_depth_tsa=50, nonlinearity=nn.ReLU, bn=True, causal=True):
        super(Conv1dGroupTSALayer, self).__init__()
        self.num_groups = num_groups
        self.bn = bn
        self.temporal_depth_tsa = temporal_depth_tsa
        use_bias = True if (not bn) else False
        # conv-1
        self.conv_first = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=use_bias)
        self.bn_first = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.99)
        # TSA layers
        self.ssa_nodes = int(mid_channels / num_groups)
        self.ssa_layers = nn.ModuleList(TemporalSA(
            self.ssa_nodes, self.ssa_nodes, temporal_depth=self.temporal_depth_tsa, ds_rate=1, res_skip=True, nonlinearity=nonlinearity, bn=self.bn, causal=causal)
            for _ in range(num_groups))
        # conv-2
        self.conv_second = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, bias=use_bias)
        self.nonlinearity = nonlinearity(init=0.5) \
            if nonlinearity == nn.PReLU else nonlinearity()

        # initial
        nn.init.xavier_normal_(self.conv_first.weight)
        nn.init.xavier_normal_(self.conv_second.weight)

    def forward(self, x, previous_frame_features=None):
        # conv-1
        x_conv_first = self.conv_first(x)
        if self.bn: 
            x_conv_first = self.bn_first(x_conv_first)
        x_conv_first = self.nonlinearity(x_conv_first)

        # group TSA
        tsa_inputs = torch.split(x_conv_first, self.ssa_nodes, dim=1)  # split along the channel dim (N,C,...)
        tmp_list = [None for i in range(self.num_groups)]
        keyval_output = None
        if previous_frame_features is not None:
            prev_keyval_inputs =  torch.chunk(previous_frame_features, self.num_groups, dim=1)           
            tmp_list_keyval = [None for i in range(self.num_groups)]          
            for i in range(self.num_groups):
                self.ssa_layers[i].to(tsa_inputs[i].device)            
                tmp_list[i], tmp_list_keyval[i] = self.ssa_layers[i](tsa_inputs[i], previous_frame_features=prev_keyval_inputs[i])
            keyval_output = torch.cat(tmp_list_keyval, dim=1)           
        else:
            for i in range(self.num_groups):
                self.ssa_layers[i].to(tsa_inputs[i].device)
                tmp_list[i], _ = self.ssa_layers[i](tsa_inputs[i])
        tsa_output = torch.cat(tmp_list, dim=1) # concatenate along the channel dim (N,C,...)

        # conv-2
        return self.conv_second(tsa_output) + x, keyval_output