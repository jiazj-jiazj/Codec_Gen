import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_layer import ConvLayer

class TransformerAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, nonlinearity=nn.ReLU, bn=True):
        super(TransformerAttentionLayer, self).__init__()
        self.bn = bn
        self._eps = torch.tensor(1e-7)
        self.cur_query_conv = ConvLayer(
            in_channels, out_channels, kernel_size, stride, nonlinearity=nonlinearity, bn=bn)
        self.other_key_conv = ConvLayer(
            in_channels, out_channels, kernel_size, stride, nonlinearity=nonlinearity, bn=bn)
        self.other_value_conv = ConvLayer(
            in_channels, out_channels, kernel_size, stride, nonlinearity=nonlinearity, bn=bn)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, cur_x, other_x):
        self._eps = self._eps.to(cur_x)
        # (B,32,T,5)
        cur_query = self.cur_query_conv(cur_x)
        other_key = self.other_key_conv(other_x)
        other_value = self.other_value_conv(other_x)
        mat_out = torch.matmul(other_key.permute(0,2,3,1), cur_query.permute(0,2,1,3))
        mat_out = mat_out/(cur_query.shape[1]+self._eps)
        mat_out = self.softmax(mat_out)  # (B,T,5,5)
        mat_out = torch.matmul(mat_out, other_value.permute(0,2,3,1))  # (B,T,5,32)
        return mat_out.permute(0,3,1,2)  # (B,32,T,5) 