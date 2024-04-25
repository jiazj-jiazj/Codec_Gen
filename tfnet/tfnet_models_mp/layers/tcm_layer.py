import torch
import torch.nn as nn
import torch.nn.functional as F
from .adaIN import AdaIN1d, AdaIN2d, MyInstanceNorm

class TCMLayer(nn.Module):
    def __init__(self, in_channels, mid_channels, depth_ker_size, rate, nonlinearity=nn.ReLU, bn=True, Norm='BN', useXuebias=False, tcm_prelu_fix=True, causal=True, causalNorm=False):
        super(TCMLayer, self).__init__()
        self.depth_ker_size = depth_ker_size
        self.rate = rate
        self.causal = causal
        self.bn = bn
        self.Norm = Norm
        if useXuebias:
            use_bias = False if (bn and Norm) else True
        else:
            use_bias = True if (not bn and not Norm) else False
        norm_functions = {'BN': nn.BatchNorm2d, 'IN': MyInstanceNorm} 
        if bn and Norm:
            if Norm == 'BN':
                self.bn_first = norm_functions[Norm](mid_channels, eps=0.001, momentum=0.99)
                self.bn_depthwise = norm_functions[Norm](mid_channels, eps=0.001, momentum=0.99)
            elif Norm == 'IN':
                self.bn_first = norm_functions[Norm](mid_channels, affine=False, causal=causalNorm)
                self.bn_depthwise = norm_functions[Norm](mid_channels, affine=False, causal=causalNorm)
        
        # conv-1
        self.conv_first = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=use_bias)
        # depthwise conv
        self.conv_depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size=(self.depth_ker_size, 1), 
            dilation=(self.rate, 1), groups=mid_channels, bias=use_bias)
       # conv-2
        self.conv_second = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, bias=use_bias)
        self.nonlinearity1 = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()
            
        if tcm_prelu_fix:            
            self.nonlinearity2 = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()
        else:
            self.nonlinearity2 = self.nonlinearity1

        # initial
        nn.init.xavier_normal_(self.conv_first.weight)
        nn.init.xavier_normal_(self.conv_depthwise.weight)
        nn.init.xavier_normal_(self.conv_second.weight)

    def forward(self, x, previous_frame_features=None):
        # conv-1
        x_conv_first = self.conv_first(x)
        if self.bn and self.Norm:
            x_conv_first = self.bn_first(x_conv_first)
        x_conv_first = self.nonlinearity1(x_conv_first)
        # depthwise conv
        if previous_frame_features is not None:
            inputs_pad = torch.cat((previous_frame_features,x_conv_first),dim=-2)
        else:
            inputs_pad = F.pad(x_conv_first, (0, 0, (self.depth_ker_size - 1)*self.rate, 0)) if self.causal else F.pad(x_conv_first, (0, 0, int((self.depth_ker_size-1)//2)*self.rate, int((self.depth_ker_size-1)//2)*self.rate))
        x_conv_dw = self.conv_depthwise(inputs_pad)
        if self.bn and self.Norm:
            x_conv_dw = self.bn_depthwise(x_conv_dw)
        x_conv_dw = self.nonlinearity2(x_conv_dw)
        # conv-2
        return self.conv_second(x_conv_dw) + x, inputs_pad
        
        
class TCMLayer_adaIN(nn.Module):
    def __init__(self, in_channels, mid_channels, depth_ker_size, rate, c_cond, nonlinearity=nn.ReLU, causal=True):
        super(TCMLayer_adaIN, self).__init__()
        self.depth_ker_size = depth_ker_size
        self.rate = rate
        
        self.bn_first = MyInstanceNorm(mid_channels, affine=False, causal=causal)       
        self.bn_depthwise = MyInstanceNorm(mid_channels, affine=False, causal=causal)    
        
        # conv-1
        self.conv_first = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        
        # depthwise conv
        self.conv_depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size=(self.depth_ker_size, 1), 
            dilation=(self.rate, 1), groups=mid_channels, bias=False)

       # conv-2
        self.conv_second = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.adaIN_last = AdaIN2d(c_cond, in_channels, causal=causal)
        
        self.nonlinearity1 = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()
            
        if tcm_prelu_fix:            
            self.nonlinearity2 = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()
        else:
            self.nonlinearity2 = self.nonlinearity1

        # initial
        nn.init.xavier_normal_(self.conv_first.weight)
        nn.init.xavier_normal_(self.conv_depthwise.weight)
        nn.init.xavier_normal_(self.conv_second.weight)

    def forward(self, x, x_cond):
        # conv-1
        x_conv_first = self.conv_first(x)
        x_conv_first = self.bn_first(x_conv_first)
        x_conv_first = self.nonlinearity1(x_conv_first)
        # depthwise conv
        inputs_pad = F.pad(x_conv_first, (0, 0, (self.depth_ker_size - 1)*self.rate, 0)) if self.causal else F.pad(x_conv_first, (0, 0, int((self.depth_ker_size-1)//2)*self.rate, int((self.depth_ker_size-1)//2)*self.rate))
        x_conv_dw = self.conv_depthwise(inputs_pad)
        x_conv_dw = self.bn_depthwise(x_conv_dw)
        x_conv_dw = self.nonlinearity2(x_conv_dw)
        # conv-2
        x_conv_second = self.conv_second(x_conv_dw)
        x_conv_second = self.adaIN_last(x_conv_second, x_cond)
        output = x_conv_second + x
        return output, inputs_pad
        
# use instance norm for all three convolutions
class TCMLayer_fullIN(nn.Module):
    def __init__(self, in_channels, mid_channels, depth_ker_size, rate, nonlinearity=nn.ReLU, tcm_prelu_fix=True, causal=True):
        super(TCMLayer_fullIN, self).__init__()
        self.depth_ker_size = depth_ker_size
        self.rate = rate
        self.causal = causal

        self.bn_first = MyInstanceNorm(mid_channels, affine=False, causal=causal)  
        self.bn_depthwise = MyInstanceNorm(mid_channels, affine=False, causal=causal)  
        self.bn_last = MyInstanceNorm(in_channels, affine=False, causal=causal)  
        
        # conv-1
        self.conv_first = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        # depthwise conv
        self.conv_depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size=(self.depth_ker_size, 1), 
            dilation=(self.rate, 1), groups=mid_channels, bias=False)
       # conv-2
        self.conv_second = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.nonlinearity1 = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()
            
        if tcm_prelu_fix:            
            self.nonlinearity2 = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()
        else:
            self.nonlinearity2 = self.nonlinearity1

        # initial
        nn.init.xavier_normal_(self.conv_first.weight)
        nn.init.xavier_normal_(self.conv_depthwise.weight)
        nn.init.xavier_normal_(self.conv_second.weight)

    def forward(self, x, previous_frame_features=None):
        # conv-1
        x_conv_first = self.conv_first(x)
        x_conv_first = self.bn_first(x_conv_first)
        x_conv_first = self.nonlinearity1(x_conv_first)
        # depthwise conv
        if previous_frame_features is not None:
            inputs_pad = torch.cat((previous_frame_features,x_conv_first),dim=-2)
        else:
            inputs_pad = F.pad(x_conv_first, (0, 0, (self.depth_ker_size - 1)*self.rate, 0)) if self.causal else F.pad(x_conv_first, (0, 0, int((self.depth_ker_size-1)//2)*self.rate, int((self.depth_ker_size-1)//2)*self.rate))
        x_conv_dw = self.conv_depthwise(inputs_pad)
        x_conv_dw = self.bn_depthwise(x_conv_dw)
        x_conv_dw = self.nonlinearity2(x_conv_dw)
        # conv-2
        x_conv_second = self.conv_second(x_conv_dw)
        x_conv_second = self.bn_last(x_conv_second)
        return x_conv_second + x, inputs_pad
        
        
class TCMLayerWithMask(nn.Module):
    def __init__(self, in_channels, mid_channels, depth_ker_size, rate, nonlinearity=nn.ReLU, bn=True, tcm_prelu_fix=True):
        super(TCMLayerWithMask, self).__init__()
        self.depth_ker_size = depth_ker_size
        self.rate = rate
        self.bn = bn
        use_bias = True if (not bn) else False
        # conv-1
        self.conv_first = nn.Conv2d(in_channels*2, mid_channels, kernel_size=1, stride=1, bias=use_bias)
        self.bn_first = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.99)
        # depthwise conv
        self.conv_depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size=(self.depth_ker_size, 1), 
            dilation=(self.rate, 1), groups=mid_channels, bias=use_bias)
        self.bn_depthwise = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.99)
        # conv-2
        self.conv_second = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, bias=use_bias)

        self.nonlinearity1 = nonlinearity(init=0.5) \
            if nonlinearity == nn.PReLU else nonlinearity()
            
        if tcm_prelu_fix:            
            self.nonlinearity2 = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()
        else:
            self.nonlinearity2 = self.nonlinearity1

        # initial
        nn.init.xavier_normal_(self.conv_first.weight)
        nn.init.xavier_normal_(self.conv_depthwise.weight)
        nn.init.xavier_normal_(self.conv_second.weight)

    def forward(self, x, previous_frame_features=None):
        # conv-1
        feature, mask = x[0], x[1]
        feature_concat = torch.cat((feature, mask), 1)
        x_conv_first = self.conv_first(feature_concat)
        if self.bn: 
            x_conv_first = self.bn_first(x_conv_first)
        x_conv_first = self.nonlinearity1(x_conv_first)
        # depthwise conv
        if previous_frame_features is not None:
            inputs_pad = torch.cat((previous_frame_features,x_conv_first),dim=-2)
        else:
            inputs_pad = F.pad(x_conv_first, (0, 0, (self.depth_ker_size - 1)*self.rate, 0))
        x_conv_dw = self.conv_depthwise(inputs_pad)
        if self.bn: 
            x_conv_dw = self.bn_depthwise(x_conv_dw)
        x_conv_dw = self.nonlinearity2(x_conv_dw)
        # conv-2
        return self.conv_second(x_conv_dw) + feature, inputs_pad