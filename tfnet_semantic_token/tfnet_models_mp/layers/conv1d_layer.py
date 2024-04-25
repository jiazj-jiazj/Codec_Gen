import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
            strides, nonlinearity=nn.ReLU, bn=True):
        super(Conv1dLayer, self).__init__()

        self.bn = bn
        use_bias = True if (not bn) else False
        self.enc_time_kernel_len = kernel_size

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, strides, bias=use_bias)
        if self.bn:
            self.bn_conv = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1)#0.99)
        self.nonlinearity = nonlinearity(init=0.5) \
            if nonlinearity == nn.PReLU else nonlinearity()

        # initial
        nn.init.xavier_normal_(self.conv.weight)

    def _pad_time(self, x, pad_size):
        """one-sided padding at time dimension for causal convolutions. Expects last two dimensions as [time x F]"""
        # https://pytorch.org/docs/stable/nn.functional.html#pad
        if pad_size:
            return F.pad(x, (pad_size[0], pad_size[1]))
        # else:
            # return F.pad(x, (self.enc_time_kernel_len - 1, 0)) #todo
        return x

    def forward(self, x, pad_size=None):
        x = self.conv(self._pad_time(x, pad_size))
        if self.bn: 
            x = self.bn_conv(x)
        return self.nonlinearity(x)