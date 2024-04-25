import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_layer import ConvLayer, ConvLayer_IN, ConvLayer_adaIN
from .adaIN import AdaIN1d, AdaIN2d, MyInstanceNorm


class DeconvLayer(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, strides,
            gate_in_channels=None, nonlinearity=nn.ReLU, bn=True, Norm='BN', useXuebias=False, causal=True, causalNorm=False, is_last=False):
        super(DeconvLayer, self).__init__()
        self.out_channels = out_channels
        self.bn = bn
        self.Norm = Norm
        self.causal = causal
        if useXuebias:
            use_bias = False if (bn and Norm) else True
        else:
            use_bias = True if (not bn and not Norm) else False
        self.kernel_size = kernel_size
        self.is_last = is_last
        if is_last:
            self.conv2d_dec_1 = ConvLayer(in_channels, out_channels, kernel_size, strides, nonlinearity= nonlinearity, bn=bn, Norm=Norm, useXuebias=useXuebias, causalNorm=causalNorm)
        else:
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, strides, bias=use_bias)
            nn.init.xavier_normal_(self.deconv.weight)   # initial
            if bn and Norm:
                norm_functions = {'BN': nn.BatchNorm2d, 'IN': MyInstanceNorm}
                if Norm == 'BN':
                    self.bn_deconv = norm_functions[Norm](out_channels, eps=0.001, momentum=0.1)#0.99)
                elif Norm == 'IN':
                    self.bn_deconv = norm_functions[Norm](out_channels, affine=False, causal=causalNorm)
            self.nonlinearity = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()

    def forward(self, deconv_x, conv_x=None,left_size=None, pad_size=None, only_current=False):
        # print('===================conv_x shape:{}'.format(conv_x.size()))
        """  """
        if self.is_last:
            deconv_out = self.conv2d_dec_1(deconv_x, pad_size=pad_size)
        else:
            # deconv block
            x = self.deconv(deconv_x)
            if left_size is not None:                
                if self.kernel_size[0] > 1:
                    if self.causal:
                        if only_current:
                            x = x[:, :, (self.kernel_size[0] - 1):-(self.kernel_size[0] - 1), left_size[0]:-left_size[1]] if left_size[1] else x[:, :, (self.kernel_size[0] - 1):-(self.kernel_size[0] - 1), left_size[0]:]
                        else:
                            x = x[:, :, :-(self.kernel_size[0] - 1), left_size[0]:-left_size[1]] if left_size[1] else x[:, :, :-(self.kernel_size[0] - 1), left_size[0]:]
                    else:
                        half_kernel = int((self.kernel_size[0] - 1) // 2)
                        x = x[:, :, half_kernel:-half_kernel, left_size[0]:-left_size[1]] if left_size[1] else x[:, :, half_kernel:-half_kernel, left_size[0]:]
                else:
                    x = x[:, :, :, left_size[0]:-left_size[1]] if left_size[1] else x[:, :, :, left_size[0]:]

            if self.bn and self.Norm: x = self.bn_deconv(x)
            deconv_out = self.nonlinearity(x)

        return deconv_out
        
class DeconvLayer_IN(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, strides, nonlinearity=nn.ReLU, is_last=False):
        super(DeconvLayer_IN, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.is_last = is_last
        if is_last:
            self.conv2d_dec_1 = ConvLayer_IN(in_channels, out_channels, kernel_size, strides, nonlinearity= nonlinearity) 
        else:
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, strides, bias=False)
            nn.init.xavier_normal_(self.deconv.weight)   # initial
            self.bn_deconv = MyInstanceNorm(out_channels, affine=False, causal=True)    #out_channels*out_freq actually   
            self.nonlinearity = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()

    def forward(self, deconv_x, conv_x=None,left_size=None, pad_size=None, only_current=False):
        # print('===================conv_x shape:{}'.format(conv_x.size()))
        """  """
        if self.is_last:
            deconv_out = self.conv2d_dec_1(deconv_x, pad_size=pad_size)
        else:
            # deconv block
            x = self.deconv(deconv_x)
            if left_size is not None:
                if self.kernel_size[0] > 1:
                    if only_current:
                        x = x[:, :, (self.kernel_size[0] - 1):-(self.kernel_size[0] - 1), left_size[0]:-left_size[1]] if left_size[1] else x[:, :, (self.kernel_size[0] - 1):-(self.kernel_size[0] - 1), left_size[0]:]
                    else:
                        x = x[:, :, :-(self.kernel_size[0] - 1), left_size[0]:-left_size[1]] if left_size[1] else x[:, :, :-(self.kernel_size[0] - 1), left_size[0]:]
                else:
                    x = x[:, :, :, left_size[0]:-left_size[1]] if left_size[1] else x[:, :, :, left_size[0]:]

            x = self.bn_deconv(x)
            deconv_out = self.nonlinearity(x)

        return deconv_out
        
        
class DeconvLayer_adaIN(nn.Module):
    def __init__(self,in_channels, out_channels, out_freq, kernel_size, strides, c_cond, nonlinearity=nn.ReLU, is_last=False):
        super(DeconvLayer_adaIN, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.is_last = is_last
        if is_last:
            self.conv2d_dec_1 = ConvLayer(in_channels, out_channels, kernel_size, strides, nonlinearity= nonlinearity, Norm='BN', useXuebias=True)
        else:
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, strides, bias=False)
            nn.init.xavier_normal_(self.deconv.weight)   # initial
            self.nonlinearity = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()
            self.adaIN = AdaIN2d(c_cond, out_channels*out_freq, causal=True)  

    def forward(self, deconv_x, x_cond, left_size=None, pad_size=None, only_current=False):
        # print('===================conv_x shape:{}'.format(conv_x.size()))
        """  """
        if self.is_last:
            deconv_out = self.conv2d_dec_1(deconv_x, pad_size=pad_size)
        else:
            # deconv block
            x = self.deconv(deconv_x)
            if left_size is not None:
                if self.kernel_size[0] > 1:
                    if only_current:
                        x = x[:, :, (self.kernel_size[0] - 1):-(self.kernel_size[0] - 1), left_size[0]:-left_size[1]] if left_size[1] else x[:, :, (self.kernel_size[0] - 1):-(self.kernel_size[0] - 1), left_size[0]:]
                    else:
                        x = x[:, :, :-(self.kernel_size[0] - 1), left_size[0]:-left_size[1]] if left_size[1] else x[:, :, :-(self.kernel_size[0] - 1), left_size[0]:]
                else:
                    x = x[:, :, :, left_size[0]:-left_size[1]] if left_size[1] else x[:, :, :, left_size[0]:]

            x = self.adaIN(x, x_cond)
            deconv_out = self.nonlinearity(x)

        return deconv_out

