import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv1d_layer import Conv1dLayer


class Deconv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides,
            gate_in_channels=None, nonlinearity=nn.ReLU, bn=True, is_last=False):
        super(Deconv1dLayer, self).__init__()
        self.out_channels = out_channels
        self.bn = bn
        use_bias = True if (not bn) else False
        self.kernel_size = kernel_size
        self.stride = strides
        self.is_last = is_last
        if is_last:
            self.conv2d_dec_1 = Conv1dLayer(in_channels, out_channels, kernel_size, strides, nonlinearity= nonlinearity, bn=self.bn)
        else:
            self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, strides, bias=use_bias)
            nn.init.xavier_normal_(self.deconv.weight)   # initial
            if self.bn:
                self.bn_deconv = nn.BatchNorm1d(out_channels)
            self.nonlinearity = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()

    def forward(self, deconv_x, pad_size=None, only_current=False):
        # print('===================conv_x shape:{}'.format(conv_x.size()))
        """  """
        if self.is_last:
            deconv_out = self.conv2d_dec_1(deconv_x, pad_size=pad_size)
        else:
            # deconv block           
            x = self.deconv(deconv_x)
            if pad_size is not None:
                x = F.pad(x, (pad_size[0], pad_size[1]))
            if self.bn: x = self.bn_deconv(x)
            deconv_out = self.nonlinearity(x)

        return deconv_out

