import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dGroupGRU(nn.Module):
    def __init__(self, in_channels, mid_channels, num_groups, nonlinearity=nn.PReLU, bn=True):
        super(Conv1dGroupGRU, self).__init__()
        self.num_groups = num_groups
        self.bn = bn
        use_bias = True if (not bn) else False
        
        # conv-1
        self.conv_first = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=use_bias)
        self.bn_first = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.99)
        # group GRU
        each_gru_channel = int(mid_channels / num_groups)
        self.each_gru_channel = each_gru_channel
        self.gru_list = nn.ModuleList(nn.GRU(each_gru_channel, each_gru_channel, num_layers=1, bias=True, batch_first=True) for i in range(num_groups))
        self.h0_list = [nn.Parameter(torch.FloatTensor(1, 1, each_gru_channel).zero_()) for i in range(num_groups)]
        # conv-2
        self.conv_second = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, bias=use_bias)

        self.nonlinearity = nonlinearity(init=0.5) \
            if nonlinearity == nn.PReLU else nonlinearity()

        # initial
        nn.init.xavier_normal_(self.conv_first.weight)
        nn.init.xavier_normal_(self.conv_second.weight)

    def forward(self, x, previous_frame_features=None):
        batch_size, nb_frames = x.shape[0], x.shape[2] # (B,C,T,1)
        # conv-1
        x_conv_first = self.conv_first(x)
        if self.bn: 
            x_conv_first = self.bn_first(x_conv_first)
        x_conv_first = self.nonlinearity(x_conv_first)  # (B,C,T,1)
        # depthwise conv
        x_conv_first = x_conv_first.squeeze(-1).permute(0,2,1) # (B,T,C)         
        input_list = torch.split(x_conv_first, self.each_gru_channel, dim=2)
        output_list = []
        h_n_list = []
        if previous_frame_features is not None:
            prev_h_n_list = torch.chunk(previous_frame_features, self.num_groups, dim=-1)
            for i, gru_layer in enumerate(self.gru_list):                
                gru_output, h_n = gru_layer(input_list[i], prev_h_n_list[i])
                gru_output = gru_output.permute(0,2,1).unsqueeze(-1)  # (B,C,T,1)
                output_list.append(gru_output)
                h_n_list.append(h_n)
        else:
            for i, gru_layer in enumerate(self.gru_list):
                initial_state = self._gru_init_state(batch_size, i)
                gru_output, h_n = gru_layer(input_list[i], initial_state.to(input_list[i].device))
                gru_output = gru_output.permute(0,2,1).unsqueeze(-1)  # (B,C,T,1)
                output_list.append(gru_output)
                h_n_list.append(h_n)         
        groupgru_output = torch.cat(output_list, 1) # (B,C,T,1)
        h_n_out = torch.cat(h_n_list, dim=-1)
        # conv-2
        return self.conv_second(groupgru_output) + x, h_n_out
        
    def _gru_init_state(self, n, i):
        if not torch._C._get_tracing_state():
            return self.h0_list[i].expand(-1, n, -1).contiguous()
        else:
            return self.h0_list[i].expand(self.h0_list[i].size(0), n, self.h0_list[i].size(2)).contiguous()
        
        
class Conv1dGroupGRUWithMask(nn.Module):
    def __init__(self, in_channels, mid_channels, depth_ker_size, rate, nonlinearity=nn.ReLU, bn=True):
        super(Conv1dGroupGRUWithMask, self).__init__()
        self.num_groups = num_groups
        self.bn = bn
        use_bias = True if (not bn) else False
        
        # conv-1
        self.conv_first = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=use_bias)
        self.bn_first = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.99)
        # group GRU
        each_gru_channel = mid_channels / num_groups
        self.gru_list = nn.ModuleList(nn.GRU(
            each_gru_channel, each_gru_channel, num_layers=1, bias=True, batch_first=True)
            for i in range(num_groups))
        self.h0_list = [nn.Parameter(torch.FloatTensor(1, 1, each_gru_channel).zero_()) for i in range(num_groups)]
        # conv-2
        self.conv_second = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, bias=use_bias)

        self.nonlinearity = nonlinearity(init=0.5) \
            if nonlinearity == nn.PReLU else nonlinearity()

        # initial
        nn.init.xavier_normal_(self.conv_first.weight)
        nn.init.xavier_normal_(self.conv_second.weight)

    def forward(self, x):
        feature, mask = x[0], x[1]
        feature_concat = torch.cat((feature, mask), 1)
        batch_size, nb_frames = feature.shape[0], feature.shape[2] # (B,C,T,1)
        # conv-1
        x_conv_first = self.conv_first(feature_concat)
        if self.bn: 
            x_conv_first = self.bn_first(x_conv_first)
        x_conv_first = self.nonlinearity(x_conv_first)  # (B,C,T,1)
        # depthwise conv
        x_conv_first = x_conv_first.squeeze(-1).permute(0,2,1) # (B,T,C)  
        input_list = torch.split(x_conv_first, self.num_groups, -1)
        output_list = []
        h_n_list = []
        for i in range(self.num_groups):
            gru_output, h_n = self.gru_list[i](input_list[i], self._gru_init_state(batch_size, i))
            gru_output = gru_output.permute(0,2,1).unsqueeze(-1)  # (B,C,T,1)
            output_list.append(gru_output)
            h_n_list.append(h_n)         
        groupgru_output = torch.cat(output_list, 1) # (B,C,T,1)
        # conv-2
        return self.conv_second(groupgru_output) + feature
        
    def _gru_init_state(self, n, i):
        if not torch._C._get_tracing_state():
            return self.h0_list[i].expand(-1, n, -1).contiguous()
        else:
            return self.h0_list[i].expand(self.h0_list[i].size(0), n, self.h0_list[i].size(2)).contiguous()
