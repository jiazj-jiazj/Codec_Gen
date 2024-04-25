import torch
import torch.nn as nn
import torch.nn.functional as F


class Time2Freq2(nn.Module):
    def __init__(self, frm_size, shift, win_len, hop_len, n_fft, config=None, enableBP=False, power=None, center=True, use_compressed_input=False):
        super(Time2Freq2, self).__init__()

        self.frm_size = frm_size
        self.shift = shift
        self.win_len = win_len
        self.hop_len = hop_len
        self.n_fft = n_fft
        self.window_fn = torch.hann_window(win_len)
        self.center = center
        self.config = config
        if config is not None:
            self.use_compressed_input = config["use_compressed_input"]
            self.use_learnable_compression = config["use_learnable_compression"]
            self.use_online_feature_norm = config["use_online_feature_norm"]
            self.power = config["power"] if power is None else power
        else:
            self.power = 0.3
            self.use_compressed_input = use_compressed_input
            self.use_learnable_compression = False
            self.use_online_feature_norm = False

        self._eps = torch.tensor(1e-7).to(torch.float32)
        self.enableBP = enableBP
        
        if config is not None and config["use_online_feature_norm"]:
            self.unbiasedExponentialSmoother = unbiasedExponentialSmoother()

    def forward(self, input_1d, oneframe = False):
        self._eps = self._eps.to(input_1d)
        self.window_fn = self.window_fn.to(input_1d)
        if self.config is not None and self.config["use_learnable_compression"] and torch.is_tensor(self.power):
            power = self.power.to(input_1d)
        else:
            power = self.power
        # input shape: (B,T)
        input_1d = input_1d.to(torch.float32) # to avoid float64-input
        if oneframe:
            stft_r = torch.stft(input_1d, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window_fn,
                              pad_mode='constant', center=False, return_complex=False).permute(0,3,2,1)
        else:
            stft_r = torch.stft(input_1d, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window_fn,
                              pad_mode='constant', center=self.center, return_complex=False).permute(0,3,2,1)   # (B,2,T,F)

        if self.use_compressed_input:
            if self.enableBP or self.use_learnable_compression:
                mag_spec = (stft_r[:,0,:,:]**2 + stft_r[:,1,:,:]**2 + self._eps) ** 0.5
                mag_spec = mag_spec.unsqueeze(1)
                phs_ = stft_r / (mag_spec + self._eps)
                mag_spec_compressed = (mag_spec + self._eps) ** power
            else:
                mag_spec = (stft_r[:,0,:,:]**2 + stft_r[:,1,:,:]**2) ** 0.5
                mag_spec = mag_spec.unsqueeze(1)
                phs_ = stft_r / torch.maximum(mag_spec, self._eps)
                mag_spec_compressed = mag_spec ** power
            in_feature = mag_spec_compressed * phs_
        else:
            in_feature = stft_r
            
        if self.use_online_feature_norm:
            in_feature = self.unbiasedExponentialSmoother(in_feature.permute(0,2,3,1)).permute(0,3,1,2)

        return in_feature.to(torch.float32), stft_r.to(torch.float32)
        
class unbiasedExponentialSmoother(nn.Module):
    """
    Unbiased Exponentially Weighted Average/Variance
    https://mixedrealitywiki.com/display/AUDIODOCS/Feature+Normalization#FeatureNormalization-OnlineFeatureNormalization
    """
    def __init__(self, beta=0.99, eps=1e-5, norm_mean=True, norm_var=True):
        super(unbiasedExponentialSmoother, self).__init__()
        self.beta = beta
        self.beta_t = 1.0
        self.v1 = 0.0
        self.v2 = 0.0
        self.S = 0.0
        self.eps = eps

        self.norm_mean = norm_mean
        self.norm_var = norm_var

    def reset(self):
        self.beta_t = 1.0
        self.v1 = 0.0
        self.v2 = 0.0
        self.S = 0.0

    def forward(self, s, reset=True):
        """
        normalize complex STFT (real + imag)
        Args
            complex STFT: N x T x F x C
        Return
             normalized : N x T x F x C
        Note: in-place data update
        """

        # reset the statistics for each sequence
        if reset:
            self.reset()

        # iterative update over time axis
        batchsize, numFrames, freqBin, numChannel = s.shape
        # Tensor doesn't support modification based on index. So we have to do that in numpy array, and later on, convert back to Tensor
        # for custom runtime or Onnx, we don't need to do that
        #s = s.numpy()
        output = torch.zeros(s.shape).to(s)
        for t in range(numFrames):
            x = s[:, t]
            self.beta_t = self.beta * self.beta_t
            self.v1 = self.beta * self.v1 + (1 - self.beta) * x
            v2_prev = self.v2
            self.v2 = self.v1 / (1.0 - self.beta_t)
            self.S = (1 - (1 - self.beta)/(1 - self.beta_t)) * self.S + (1 - self.beta)/(1 - self.beta_t) * (x - self.v2) * (x - v2_prev)
            self.S = self.S * (self.S > 0)
            if self.norm_mean:
                x = x - self.v2
            if self.norm_var:
                x = x / ((self.S + self.eps) ** 0.5)
            output[:, t] = x

        return output


class Time2Freq_Disc(nn.Module):
    def __init__(self, frm_size, shift, win_len, hop_len, n_fft, config=None,):
        super(Time2Freq_Disc, self).__init__()

        self.frm_size = frm_size
        self.shift = shift
        self.win_len = win_len
        self.hop_len = hop_len
        self.n_fft = n_fft
        self.window_fn = torch.hann_window(win_len)

        if config is not None:
            self.use_compressed_input = config["use_compressed_input_adv"]
            self.use_compressed_mag_input = config["use_compressed_mag_input_adv"]
            self.use_complex_input = config["use_uncompressed_input_adv"] 
            self.power = config["power"]
        else:
            self.use_compressed_input = False
        self._eps = torch.tensor(1e-7)

    def forward(self, input_1d, oneframe = False):
        self._eps = self._eps.to(input_1d)
        self.window_fn = self.window_fn.to(input_1d)
        # input shape: (B,T)
        input_1d = input_1d.to(torch.float32) # to avoid float64-input
        if oneframe:
            stft_r = torch.stft(input_1d, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window_fn,
                              pad_mode='constant', center= False,return_complex=False).permute(0,3,2,1)
        else:
            stft_r = torch.stft(input_1d, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window_fn,
                              pad_mode='constant', return_complex=False).permute(0,3,2,1)   # (B,2,T,F)

        if self.use_compressed_input:
            mag_spec = (stft_r[:,0,:,:]**2 + stft_r[:,1,:,:]**2 + self._eps) ** 0.5
            mag_spec = mag_spec.unsqueeze(1)
            phs_ = stft_r / (mag_spec+self._eps)
            mag_spec_compressed = mag_spec ** self.power
            in_feature = mag_spec_compressed * phs_
        elif self.use_compressed_mag_input:
            mag_spec = (stft_r[:,0,:,:]**2 + stft_r[:,1,:,:]**2 + self._eps) ** 0.5
            mag_spec = mag_spec.unsqueeze(1)
            mag_spec_compressed = mag_spec ** self.power
            in_feature = mag_spec_compressed
        else:
            in_feature = stft_r

        return in_feature, stft_r