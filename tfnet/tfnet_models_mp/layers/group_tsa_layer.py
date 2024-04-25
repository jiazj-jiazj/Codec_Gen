import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal_sa import TemporalSA


class GroupTSA(nn.Module):
    def __init__(self, in_channels, mid_channels, num_groups=4, temporal_depth_tsa=50, nonlinearity=nn.ReLU, bn=True):
        super(GroupTSA, self).__init__()
        self.num_groups = num_groups
        self.bn = bn
        self.temporal_depth_tsa = temporal_depth_tsa
            
        # TSA layers
        self.ssa_nodes = int(mid_channels / num_groups)
        self.ssa_layers = nn.ModuleList(TemporalSA(
            self.ssa_nodes, self.ssa_nodes, self.temporal_depth_tsa, ds_rate=1, res_skip=True, nonlinearity=nonlinearity, bn=self.bn)
            for _ in range(num_groups))       

    def forward(self, x, y=None, previous_frame_features=None):        
        # group TSA
        tsa_inputs = torch.split(x, self.ssa_nodes, dim=1)  # split along the channel dim (N,C,...)
        tmp_list = [None for i in range(self.num_groups)]
        keyval_output = None # [B, C, T]
        if previous_frame_features is not None:            
            prev_keyval_inputs =  torch.chunk(previous_frame_features, self.num_groups, dim=1)            
            tmp_list_keyval = [None for i in range(self.num_groups)]          
            for i in range(self.num_groups):
                self.ssa_layers[i].to(tsa_inputs[i].device)            
                tmp_list[i], tmp_list_keyval[i] = self.ssa_layers[i](tsa_inputs[i], y=y, previous_frame_features=prev_keyval_inputs[i])
            keyval_output = torch.cat(tmp_list_keyval, dim=1)          
        else:
            for i in range(self.num_groups):
                self.ssa_layers[i].to(tsa_inputs[i].device)
                tmp_list[i] = self.ssa_layers[i](tsa_inputs[i], y=y)
        tsa_output = torch.cat(tmp_list, dim=1) # concatenate along the channel dim (N,C,...)

        return tsa_output + x, keyval_output