import torch
import torch.nn as nn
import torch.nn.functional as F

class BandCompress48khz(nn.Module):
    def __init__(self, num_freq_bins, in_channels=2, out_channels=2, kernel_size=5, config=None):
        super(BandCompress48khz, self).__init__()
        self.num_freq_bins = num_freq_bins
        self.config = config
        self.kernel_size = kernel_size
        self.use_pre_filter_inbandCprs = config["use_pre_filter_inbandCprs"] if config is not None else True
        
        if self.use_pre_filter_inbandCprs:
            self.prefiltering = nn.Conv2d(in_channels, out_channels, (1, kernel_size), (1, 1), bias=False)
            nn.init.xavier_normal_(self.prefiltering.weight)
            
        #self.ds_2x = nn.AvgPool2d((1, 2)) # [B, C, T, F]
        #self.ds_4x = nn.AvgPool2d((1, 4))

    def forward(self, x):
        if self.use_pre_filter_inbandCprs:
            pad_size = (self.kernel_size - 1) // 2
            inputs_filtered = self.prefiltering(nn.functional.pad(x, (pad_size, pad_size)))

        bins1 = (self.num_freq_bins - 1) // 4 + 1  # 6k
        bins2 = (self.num_freq_bins - 1) // 4      # 12k
        bins3 = (self.num_freq_bins - 1) // 2      # 24k
        
        if self.use_pre_filter_inbandCprs:
            band0to6k, band6kto12k, band12kto24k = torch.split(inputs_filtered, [bins1, bins2, bins3], dim=-1)
        else:
            band0to6k, band6kto12k, band12kto24k = torch.split(x, [bins1, bins2, bins3], dim=-1)           

        #band6kto12k_cprs = self.ds_2x(band6kto12k)
        #band12kto24k_cprs = self.ds_4x(band12kto24k)
        
        band6kto12k_cprs = nn.functional.interpolate(band6kto12k, scale_factor=(1, 0.5))
        band12kto24k_cprs = nn.functional.interpolate(band12kto24k, scale_factor=(1, 0.25))              
        
        output = torch.cat((band0to6k, band6kto12k_cprs, band12kto24k_cprs), dim=-1)    
        return output
        

class BandDecompress48khz(nn.Module):
    def __init__(self, num_freq_bins, in_channels=2, out_channels=2, kernel_size=5, config=None):
        super(BandDecompress48khz, self).__init__()
        self.num_freq_bins = num_freq_bins
        self.config = config
        self.kernel_size = kernel_size
        self.use_post_filter_inbandCprs = config["use_post_filter_inbandCprs"] if config is not None else True
        
        if self.use_post_filter_inbandCprs:
            self.postfiltering = nn.Conv2d(in_channels, out_channels, (1, kernel_size), (1, 1), bias=False)
            nn.init.xavier_normal_(self.postfiltering.weight)
            
        self.up_2x = nn.Upsample(scale_factor=(1, 2), mode='bilinear')
        self.up_4x = nn.Upsample(scale_factor=(1, 4), mode='bilinear')

    def forward(self, x):        
        bins1 = (self.num_freq_bins - 1) // 4 + 1
        bins2_cprs = (self.num_freq_bins - 1) // 8
        bins3_cprs = (self.num_freq_bins - 1) // 8
        
        band0to6k, band6kto12k_cprs, band12kto24k_cprs = torch.split(x, [bins1, bins2_cprs, bins3_cprs], dim=-1)

        band6kto12k = self.up_2x(band6kto12k_cprs)
        band12kto24k = self.up_4x(band12kto24k_cprs)
        inputs_uncprs = torch.cat((band0to6k, band6kto12k, band12kto24k), dim=-1)   

        if self.use_post_filter_inbandCprs:
            pad_size = (self.kernel_size - 1) // 2
            return self.postfiltering(F.pad(inputs_uncprs, (pad_size, pad_size)))
        else:
            return inputs_uncprs 