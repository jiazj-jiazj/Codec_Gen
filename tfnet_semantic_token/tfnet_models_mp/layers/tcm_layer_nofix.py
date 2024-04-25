import torch
import torch.nn as nn
import torch.nn.functional as F


class TCMLayer(nn.Module):
    def __init__(self, in_channels, mid_channels, depth_ker_size, rate, nonlinearity=nn.ReLU, bn=True):
        super(TCMLayer, self).__init__()
        self.depth_ker_size = depth_ker_size
        self.rate = rate
        self.bn = bn
        use_bias = True if (not bn) else False
        # conv-1
        self.conv_first = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=use_bias)
        if self.bn:
            self.bn_first = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.1)#0.99)
        # depthwise conv
        self.conv_depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size=(self.depth_ker_size, 1), 
            dilation=(self.rate, 1), groups=mid_channels, bias=use_bias)
        if self.bn:
            self.bn_depthwise = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.1)#0.99)
        # conv-2
        self.conv_second = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, bias=use_bias)

        self.nonlinearity = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()            

        # initial
        nn.init.xavier_normal_(self.conv_first.weight)
        nn.init.xavier_normal_(self.conv_depthwise.weight)
        nn.init.xavier_normal_(self.conv_second.weight)

    def forward(self, x, previous_frame_features=None):
        # conv-1
        x_conv_first = self.conv_first(x)
        if self.bn: 
            x_conv_first = self.bn_first(x_conv_first)
        x_conv_first = self.nonlinearity(x_conv_first)
        # depthwise conv
        if previous_frame_features is not None:
            inputs_pad = torch.cat((previous_frame_features,x_conv_first),dim=-2)
            x_conv_dw = self.conv_depthwise(inputs_pad)
        else:
            x_conv_dw = self.conv_depthwise(F.pad(x_conv_first, (0, 0, (self.depth_ker_size - 1)*self.rate, 0)))
        if self.bn: 
            x_conv_dw = self.bn_depthwise(x_conv_dw)
        x_conv_dw = self.nonlinearity(x_conv_dw)
        # conv-2
        return self.conv_second(x_conv_dw) + x, x_conv_first
        
        
class TCMLayerWithMask(nn.Module):
    def __init__(self, in_channels, mid_channels, depth_ker_size, rate, nonlinearity=nn.ReLU, bn=True):
        super(TCMLayerWithMask, self).__init__()
        self.depth_ker_size = depth_ker_size
        self.rate = rate
        self.bn = bn
        use_bias = True if (not bn) else False
        # conv-1
        self.conv_first = nn.Conv2d(in_channels*2, mid_channels, kernel_size=1, stride=1, bias=use_bias)
        self.bn_first = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.1)#0.99)
        # depthwise conv
        self.conv_depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size=(self.depth_ker_size, 1), 
            dilation=(self.rate, 1), groups=mid_channels, bias=use_bias)
        self.bn_depthwise = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.1)#0.99)
        # conv-2
        self.conv_second = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, bias=use_bias)

        self.nonlinearity = nonlinearity(init=0.5) \
            if nonlinearity == nn.PReLU else nonlinearity()            

        # initial
        nn.init.xavier_normal_(self.conv_first.weight)
        nn.init.xavier_normal_(self.conv_depthwise.weight)
        nn.init.xavier_normal_(self.conv_second.weight)

    def forward(self, x):
        # conv-1
        feature, mask = x[0], x[1]
        feature_concat = torch.cat((feature, mask), 1)
        x_conv_first = self.conv_first(feature_concat)
        if self.bn: 
            x_conv_first = self.bn_first(x_conv_first)
        x_conv_first = self.nonlinearity(x_conv_first)
        # depthwise conv
        x_conv_dw = self.conv_depthwise(F.pad(x_conv_first, (0, 0, (self.depth_ker_size - 1)*self.rate, 0)))
        if self.bn: 
            x_conv_dw = self.bn_depthwise(x_conv_dw)
        x_conv_dw = self.nonlinearity(x_conv_dw)
        # conv-2
        return self.conv_second(x_conv_dw) + feature, x_conv_first