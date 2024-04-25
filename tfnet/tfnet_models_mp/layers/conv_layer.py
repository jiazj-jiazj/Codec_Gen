import torch
import torch.nn as nn
import torch.nn.functional as F
from .adaIN import AdaIN1d, AdaIN2d, MyInstanceNorm

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
            strides, nonlinearity=nn.ReLU, bn=True, Norm='BN', useXuebias=False, causalNorm=False):
        super(ConvLayer, self).__init__()

        self.bn = bn
        self.Norm = Norm
        if useXuebias:
            use_bias = False if (bn and Norm) else True
        else:
            use_bias = True if (not bn and not Norm) else False
        norm_functions = {'BN':nn.BatchNorm2d,'IN':MyInstanceNorm}
        if bn and Norm:
            if Norm == 'BN':
                self.bn_conv = norm_functions[Norm](out_channels, eps=0.001, momentum=0.99)
            elif Norm == 'IN':
                self.bn_conv = norm_functions[Norm](out_channels, affine=False, causal=causalNorm)

        self.enc_time_kernel_len = kernel_size[0]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, bias=use_bias)
        self.nonlinearity = nonlinearity(init=0.5) \
            if nonlinearity == nn.PReLU else nonlinearity()

        # initial
        nn.init.xavier_normal_(self.conv.weight)

    def _pad_time(self, x, pad_size):
        """one-sided padding at time dimension for causal convolutions. Expects last two dimensions as [time x F]"""
        # https://pytorch.org/docs/stable/nn.functional.html#pad
        if pad_size:
            return F.pad(x, (pad_size[0][0], pad_size[0][1], pad_size[1][0], pad_size[1][1]))
        else:
            return F.pad(x, (0, 0, self.enc_time_kernel_len - 1, 0)) #todo

    def forward(self, x, pad_size=None):
        x = self.conv(self._pad_time(x, pad_size))
        if self.bn and self.Norm: x = self.bn_conv(x)
        return self.nonlinearity(x)
        
class ConvLayer_IN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, nonlinearity=nn.ReLU):
        super(ConvLayer_IN, self).__init__()
        
        self.enc_time_kernel_len = kernel_size[0]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, bias=False)
        self.bn_conv = MyInstanceNorm(out_channels, affine=False, causal=True)  #out_channels*out_freq actually
        self.nonlinearity = nonlinearity(init=0.5) \
            if nonlinearity == nn.PReLU else nonlinearity()

        # initial
        nn.init.xavier_normal_(self.conv.weight)

    def _pad_time(self, x, pad_size):
        """one-sided padding at time dimension for causal convolutions. Expects last two dimensions as [time x F]"""
        # https://pytorch.org/docs/stable/nn.functional.html#pad
        if pad_size:
            return F.pad(x, (pad_size[0][0], pad_size[0][1], pad_size[1][0], pad_size[1][1]))
        else:
            return F.pad(x, (0, 0, self.enc_time_kernel_len - 1, 0)) #todo

    def forward(self, x, pad_size=None):
        x = self.conv(self._pad_time(x, pad_size))
        x = self.bn_conv(x)
        return self.nonlinearity(x)
        
        
class ConvLayer_adaIN(nn.Module):
    def __init__(self, in_channels, out_channels, out_freq, kernel_size, strides, c_cond, nonlinearity=nn.ReLU):
        super(ConvLayer_adaIN, self).__init__()

        self.enc_time_kernel_len = kernel_size[0]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, bias=use_bias)
        self.adaIN = AdaIN2d(c_cond, out_channels*out_freq, causal=True)  
        self.nonlinearity = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()

        # initial
        nn.init.xavier_normal_(self.conv.weight)

    def _pad_time(self, x, pad_size):
        """one-sided padding at time dimension for causal convolutions. Expects last two dimensions as [time x F]"""
        # https://pytorch.org/docs/stable/nn.functional.html#pad
        if pad_size:
            return F.pad(x, (pad_size[0][0], pad_size[0][1], pad_size[1][0], pad_size[1][1]))
        else:
            return F.pad(x, (0, 0, self.enc_time_kernel_len - 1, 0)) #todo

    def forward(self, x, x_cond, pad_size=None):
        x = self.conv(self._pad_time(x, pad_size))
        x = self.adaIN(x, x_cond)
        return self.nonlinearity(x)