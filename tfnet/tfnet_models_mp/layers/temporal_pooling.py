import torch
import torch.nn as nn

class TemporalPooling_phn(nn.Module):
    """
    Input: (N, channels, T) numeric tensor
    Output: (N, channels, 1) numeric tensor
    """

    def __init__(self, n_channels,mid_channels=None,out_dim=1):
        super().__init__()
        self.n_channels = n_channels
        self.avg = nn.AdaptiveAvgPool1d(out_dim)
        if mid_channels:
            self.fc1 = nn.Linear(n_channels, mid_channels[0])
            self.fc2 = nn.Linear(mid_channels[0], mid_channels[1])
            self.fc3 = nn.Linear(mid_channels[1], mid_channels[2])
            self.fc4 = nn.Linear(mid_channels[2], mid_channels[3])
        else:
            self.fc1 = nn.Linear(n_channels, n_channels)
            self.fc2 = nn.Linear(n_channels, n_channels)
            self.fc3 = nn.Linear(n_channels, n_channels)
            self.fc4 = nn.Linear(n_channels, n_channels)

        return

    def forward(self, x):
        visual_list = []
        T = x.shape[2]
        tap = self.avg(x)  # B,C,1
        visual_list.append(tap)
        tap = tap.squeeze(2)
        ff1 = self.fc1(tap)
        ff2 = self.fc2(ff1)
        ff3 = self.fc3(ff2)
        ff4 = self.fc4(ff3)
        ff4 = ff4.unsqueeze(2)
        ret = ff4
        visual_list.append(ff1.unsqueeze(2))
        visual_list.append(ff2.unsqueeze(2))
        visual_list.append(ff3.unsqueeze(2))
        visual_list.append(ret)
        return ret,visual_list

class TemporalPooling_spk(nn.Module):
    """
    Input: (B, channels, T) numeric tensor
    Output: (B, channels, T) numeric tensor
    """

    def __init__(self, n_channels, mid_channels=None,out_dim=1,pooling_kernel=None, max_pooling_len=-1, transmit_rate=None): # max_pooling_len=10: 1s pooling, max_pooling_len=-1: global causal pooling
        super().__init__()
        self.n_channels = n_channels
        # this is size of sample outputs from encoder
        if pooling_kernel is not None:
            self.avg = nn.AvgPool1d(kernel_size=pooling_kernel,stride=pooling_kernel)
        else:
            self.avg = nn.AdaptiveAvgPool1d(out_dim)
        if mid_channels:
            self.fc1 = nn.Linear(n_channels, mid_channels[0])
            self.fc2 = nn.Linear(mid_channels[0], mid_channels[1])
        else:
            self.fc1 = nn.Linear(n_channels, n_channels)
            self.fc2 = nn.Linear(n_channels, n_channels)
            
        self.pooling_kernel = pooling_kernel
        self.transmit_rate = transmit_rate
        self.max_pooling_len = max_pooling_len

        return

    def forward(self, x):
        ## in: B,C,T
        T = x.shape[2]
        tap = self.avg(x)  # B,C,1
        # tap = tap.squeeze(2)
        tap = tap.permute(0,2,1)
        ff1 = self.fc1(tap) # B,C
        ff2 = self.fc2(ff1) # B,C
        ff2 = ff2.permute(0, 2, 1)
        ret = ff2
        feat_list = [tap.permute(0, 2, 1), ff1.permute(0, 2, 1),ff2]
        
        if self.pooling_kernel is not None and (self.transmit_rate > 1):
            B,C,T = ff2.shape
            weight = torch.zeros(T, T).to(ff2)  # T,T
            ret = torch.zeros_like(ff2).to(ff2)
            if self.max_pooling_len < 0:
                for i in range(1, T + 1):
                    weight[i - 1, :i] = 1.0 / i
            else:
                for i in range(1, T + 1):
                    curr_len = self.max_pooling_len if i >= self.max_pooling_len else i
                    weight[i - 1, i-curr_len:i] = 1.0 / curr_len
            feature_ = ff2.unsqueeze(-2).repeat(1,1,T,1)  # B,C,T,T
            feature_avg = feature_ * weight
            feature_avg = feature_avg.sum(-1)  # B,C,T
            step = self.transmit_rate
            part = int(T//step)
            for part_id in range(part):
                start = part_id * step
                end = start + step if part_id < part - 1 else T
                ret[:,:,start:end] = torch.repeat_interleave(feature_avg[:,:,start].unsqueeze(-1),(end-start),dim=-1) ## todo inplace error ?

        return ret,feat_list

