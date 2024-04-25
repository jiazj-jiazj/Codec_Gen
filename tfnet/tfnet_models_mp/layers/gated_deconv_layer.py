import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_layer import ConvLayer
from .deconv_layer import DeconvLayer

class GatedDeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, 
            gate_in_channels=None, nonlinearity=nn.ReLU, bn=True, causal=True, is_last=False):
        super(GatedDeconvLayer, self).__init__()

        self.out_channels = out_channels
        self.bn = bn
        use_bias = True if (not bn) else False
        self.kernel_size = kernel_size
        self.is_last = is_last

        if is_last:
            self.conv2d_dec_1 = ConvLayer(in_channels, out_channels, kernel_size, strides, nonlinearity=nonlinearity, bn=bn)
        else:
            # deconv block
            self.deconv = DeconvLayer(in_channels, out_channels, kernel_size, strides, nonlinearity=nonlinearity, bn=bn, causal=causal)

        # gates
        if gate_in_channels is None:
            gate_in_channels = out_channels*2
        self.conv_gate = ConvLayer(gate_in_channels, out_channels, (1, 1), (1, 1), nonlinearity=nonlinearity, bn=bn)
        self.conv_rd = ConvLayer(gate_in_channels, out_channels, (1, 1), (1, 1), nonlinearity=nonlinearity, bn=bn)


    def forward(self, deconv_x, conv_x, left_size=None, pad_size=None, only_current=False):
        """  """
        if self.is_last:
            deconv_out = self.conv2d_dec_1(deconv_x, pad_size=pad_size)
        else:
            # deconv block
            deconv_out = self.deconv(deconv_x, left_size=left_size, only_current=only_current)
            
        # multiplicative gate
        concat_x = torch.cat((deconv_out, conv_x), dim=1)
        gated_x = conv_x * self.conv_gate(concat_x)

        # additive gate 
        concat_x2 = torch.cat((deconv_out, gated_x), dim=1)
        return deconv_out + self.conv_rd(concat_x2)