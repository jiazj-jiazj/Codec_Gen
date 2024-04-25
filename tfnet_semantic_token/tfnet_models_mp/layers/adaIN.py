import torch
import torch.nn as nn
from .time2freq2 import unbiasedExponentialSmoother

class MyInstanceNorm(nn.Module):
    def __init__(self, channels: int, *,
                 eps: float = 1e-5, affine: bool = False, causal: bool = True):
        super().__init__()

        self.channels = channels
        self.eps = eps
        self.affine = affine
        self.causal = causal
        self.use_unbiasEMA = True
            
        if not self.causal:
            self.norm = nn.InstanceNorm2d(channels, affine=affine)
        else:            
            
            # Create parameters for $\gamma$ and $\beta$ for scale and shift
            if self.affine:
                self.scale = nn.Parameter(torch.ones(channels))
                self.shift = nn.Parameter(torch.zeros(channels))
            if self.causal:
                self.pooling_kernel = 10
                self.avg = nn.AvgPool1d(kernel_size=self.pooling_kernel,stride=self.pooling_kernel)  # 100ms latency
            if self.use_unbiasEMA:
                self.unbiasedExponentialSmoother = unbiasedExponentialSmoother()

    def forward(self, x: torch.Tensor): # B, C, T, F
        if not self.causal:
            return self.norm(x)
            
        if self.use_unbiasEMA:
            return self.unbiasedExponentialSmoother(x.permute(0,2,3,1)).permute(0,3,1,2)
        
        B,C,T,F = x.shape
        x = x.permute(0,1,3,2).reshape(B, -1, T) # B, CF, T
        #assert self.channels == x.shape[1]
        
        if not self.causal:            
            x = x.view(B, C*F, -1)
            mean = x.mean(dim=[-1], keepdim=True)
            mean_x2 = (x ** 2).mean(dim=[-1], keepdim=True)
            var = mean_x2 - mean ** 2
        else:            
            x_pool_mean = self.avg(x)  # B,C,T/10
            x2_pool_mean = self.avg(x ** 2)
            _,_,T1 = x_pool_mean.shape
            mask = torch.zeros(T1, T1).to(x)  # T/10,T/10
            for i in range(1, T1 + 1):
                mask[i-1, :i] = 1
            causal_pool_nfrms = mask.sum(-1) # T/10
            
            x_pool_mean_ = x_pool_mean.unsqueeze(-2).repeat(1,1,T1,1) * mask  # B,CF,T1,T1
            mean = x_pool_mean_.sum(-1) / causal_pool_nfrms # B,CF,T1
            x2_pool_mean_ = x2_pool_mean.unsqueeze(-2).repeat(1,1,T1,1) * mask  # B,CF,T1,T1
            mean_x2 = x2_pool_mean_.sum(-1)  / causal_pool_nfrms # B,CF,T1
            var = mean_x2 - mean ** 2 # B,CF,T1
            mean = torch.repeat_interleave(mean,self.pooling_kernel,dim=-1)  # B,CF,T
            var = torch.repeat_interleave(var,self.pooling_kernel,dim=-1)  # B,CF,T
            
            if T1*self.pooling_kernel < T:
                pad_mean = torch.repeat_interleave(mean[:,:,-1].unsqueeze(-1), int(T - T1*self.pooling_kernel), dim=-1)
                mean = torch.cat((mean, pad_mean), dim=-1)
                pad_var = torch.repeat_interleave(var[:,:,-1].unsqueeze(-1), int(T - T1*self.pooling_kernel), dim=-1)
                var = torch.cat((var, pad_var), dim=-1)
            
            # mask = torch.zeros(T, T).to(x)  # T,T
            # ret = torch.zeros_like(x).to(x)
            # for i in range(1, T + 1):
                # mask[i-1, :i] = 1
            # causal_nfrms = mask.sum(-1) # T
            
            # x_ = x.unsqueeze(-2).repeat(1,1,T,1) * mask  # B,CF,T,T
            # mean = x_.sum(-1) / causal_nfrms # B,CF,T
            # mean_x2 = (x_ ** 2).sum(-1)  / causal_nfrms # B,CF,T
            # var = mean_x2 - mean ** 2 # B,CF,T
            
            # mean = torch.zeros_like(x).to(x)
            # mean_x2 = torch.zeros_like(x).to(x)
            # for i in range(T):
                # x_ = x * mask[i,:]  # B,CF, T
                # mean[:,:,i] = x_.sum(-1) / causal_nfrms[i] # B,CF
                # mean_x2[:,:,i] = (x_ ** 2).sum(-1)  / causal_nfrms[i] # B,CF
            # var = mean_x2 - mean ** 2 # B,CF,T
            
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(B, C*F, -1)

        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        # Reshape to original and return
        return x_norm.reshape(B,C,F,T).permute(0,1,3,2)
        
class MyInstanceNorm1d(nn.Module):
    def __init__(self, channels: int, *,
                 eps: float = 1e-5, affine: bool = False, causal: bool = True):
        super().__init__()

        self.channels = channels
        self.eps = eps
        self.affine = affine
        self.causal = causal
        self.use_unbiasEMA = True
            
        if not self.causal:
            self.norm = nn.InstanceNorm1d(channels, affine=affine)
        else:            
            # Create parameters for $\gamma$ and $\beta$ for scale and shift
            if self.affine:
                self.scale = nn.Parameter(torch.ones(channels))
                self.shift = nn.Parameter(torch.zeros(channels))
            if self.causal:
                self.pooling_kernel = 10
                self.avg = nn.AvgPool1d(kernel_size=self.pooling_kernel,stride=self.pooling_kernel)  # 100ms latency
            if self.use_unbiasEMA:
                self.unbiasedExponentialSmoother = unbiasedExponentialSmoother()

    def forward(self, x: torch.Tensor): # B, C, T
        if not self.causal:
            return self.norm(x)
            
        if self.use_unbiasEMA:
            return self.unbiasedExponentialSmoother(x.permute(0,2,1).unsqueeze(-2)).squeeze(-2).permute(0,2,1)
        
        B,C,T = x.shape
        assert self.channels == x.shape[1]
        
        if not self.causal:            
            x = x.view(B, C, -1)
            mean = x.mean(dim=[-1], keepdim=True)
            mean_x2 = (x ** 2).mean(dim=[-1], keepdim=True)
            var = mean_x2 - mean ** 2
        else:
            x_pool_mean = self.avg(x)  # B,C,T/10
            x2_pool_mean = self.avg(x ** 2)
            _,_,T1 = x_pool_mean.shape
            mask = torch.zeros(T1, T1).to(x)  # T/10,T/10
            ret = torch.zeros_like(x).to(x)
            for i in range(1, T1 + 1):
                mask[i-1, :i] = 1
            causal_pool_nfrms = mask.sum(-1) # T
            
            x_pool_mean_ = x_pool_mean.unsqueeze(-2).repeat(1,1,T1,1) * mask  # B,CF,T1,T1
            mean = x_pool_mean_.sum(-1) / causal_pool_nfrms # B,CF,T1
            x2_pool_mean_ = x2_pool_mean.unsqueeze(-2).repeat(1,1,T1,1) * mask  # B,CF,T1,T1
            mean_x2 = x2_pool_mean_.sum(-1)  / causal_pool_nfrms # B,CF,T1
            var = mean_x2 - mean ** 2 # B,CF,T1
            mean = torch.repeat_interleave(mean,self.pooling_kernel,dim=-1)  # B,CF,T
            var = torch.repeat_interleave(var,self.pooling_kernel,dim=-1)  # B,CF,T
            
            if T1*self.pooling_kernel < T:
                pad_mean = torch.repeat_interleave(mean[:,:,-1].unsqueeze(-1), int(T - T1*self.pooling_kernel), dim=-1)
                mean = torch.cat((mean, pad_mean), dim=-1)
                pad_var = torch.repeat_interleave(var[:,:,-1].unsqueeze(-1), int(T - T1*self.pooling_kernel), dim=-1)
                var = torch.cat((var, pad_var), dim=-1)
                
            # mask = torch.zeros(T, T).to(x)  # T,T
            # ret = torch.zeros_like(x).to(x)
            # for i in range(1, T + 1):
                # mask[i-1, :i] = 1
            # causal_nfrms = mask.sum(-1) # T
            # x_ = x.unsqueeze(-2).repeat(1,1,T,1) * mask  # B,C,T,T
            # mean = x_.sum(-1) / causal_nfrms # B,C,T
            # mean_x2 = (x_ ** 2).sum(-1)  / causal_nfrms # B,C,T
            # var = mean_x2 - mean ** 2 # B,C,T
            
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(B, C, -1)

        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        # Reshape to original and return
        return x_norm
        
# class InsNormLayer1d(nn.Module):
    # def __init__(self, c_h, causal=True): 
        # super(InsNormLayer, self).__init__()
        # self.c_h = c_h
        # self.causal = causal
        # self.norm_layer = nn.InstanceNorm1d(c_h, affine=False) if not causal else nn.LayerNorm(c_h, elementwise_affine=False)

    # def forward(self, x):  # x (B, C, T)
        # x = self.norm_layer(x) if not causal else self.norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        # return x
        
# class InsNormLayer2d(nn.Module):
    # def __init__(self, c_h, causal=True): 
        # super(InsNormLayer, self).__init__()
        # self.c_h = c_h
        # self.causal = causal
        # self.norm_layer = nn.InstanceNorm2d(c_h, affine=False) if not causal else nn.LayerNorm(c_h, elementwise_affine=False)

    # def forward(self, x):  # x (B, C, T, F)  c_h=C
        # x = self.norm_layer(x) if not causal else self.norm_layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # return x

class AdaIN1d(nn.Module):
    def __init__(self, c_cond: int, c_h: int, causal: bool = False): 
        super(AdaIN1d, self).__init__()
        self.c_h = c_h
        self.norm_layer = MyInstanceNorm(c_h, affine=False, causal=causal)
        self.linear_layer = nn.Linear(c_cond, c_h * 2)

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:  # x (B, C, T), x_cond (B, T, C)
        x_cond = self.linear_layer(x_cond)
        mean, std = x_cond[:, :, : self.c_h], x_cond[:, :, self.c_h :]
        mean, std = mean.permute(0, 2, 1), std.permute(0, 2, 1) #(B, C, T)
        x = self.norm_layer(x.unsqueeze(-1)).squeeze(-1)
        x = x * std + mean
        return x
        
class AdaIN2d(nn.Module):
    def __init__(self, c_cond: int, c_h: int, causal: bool = False):
        super(AdaIN2d, self).__init__()
        self.c_h = c_h
        self.norm_layer = MyInstanceNorm(c_h, affine=False, causal=causal)
        self.linear_layer = nn.Linear(c_cond, c_h * 2)

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:  # x (B, C, T, F), x_cond (B, T, C)
        B,C,T,F = x.shape
        assert self.c_h == C*F
        x_cond = self.linear_layer(x_cond)
        mean, std = x_cond[:, :, : self.c_h], x_cond[:, :, self.c_h :]  # (B, T, CF)
        mean, std = mean.reshape(B, -1, C, F).permute(0,2,1,3), std.reshape(B, -1, C, F).permute(0,2,1,3) #(B, C, T, F)
        x = self.norm_layer(x)
        x = x * std + mean  # (B, C, T, F)
        return x