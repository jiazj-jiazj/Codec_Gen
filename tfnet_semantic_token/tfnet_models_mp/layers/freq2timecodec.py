import torch
import torch.nn as nn
import torch.nn.functional as F

from .istft import ISTFT


class Freq2TimeCodec(nn.Module):
    def __init__(self, frm_size, shift, win_len, hop_len, n_fft, config=None, power=None):
        super(Freq2TimeCodec, self).__init__()

        self.frm_size = frm_size
        self.shift = shift
        self.win_len = win_len
        self.hop_len = hop_len
        self.n_fft = n_fft
        self.config = config
        self._eps = torch.tensor(1e-7)
        self.istft = ISTFT(filter_length=win_len, hop_length=hop_len, window='hann',) #window='hanning', )
        if config is not None:
            self.learn_uncompressed_amp = config["learn_uncompressed_amp"]
            self.power = config["power"] if power is None else power
        else:
            self.learn_uncompressed_amp = False
        self.tanh = nn.Tanh()
        self.window_fn = torch.hann_window(win_len)

    def forward(self, dec_out, oneframe=False):
        self._eps = self._eps.to(dec_out)
        amp_output, phs_output = self._get_amp_and_phase(dec_out)
        self.window_fn = self.window_fn.to(dec_out)        
        
        if not self.learn_uncompressed_amp:
            if self.config is not None and self.config["use_learnable_compression"]:
                power = self.power.to(dec_out)
                amp_output = (amp_output + self._eps) ** (1/(power + self._eps))
            else:
                amp_output = amp_output**(1/self.power)
        output_stft_r = amp_output * phs_output
        if (self.n_fft != self.win_len) or ((self.win_len % self.hop_len) != 0): 
            if oneframe:            
                output_1d = torch.istft(output_stft_r.permute(0,3,2,1), n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window_fn, center=False)
            else:
                output_1d = torch.istft(output_stft_r.permute(0,3,2,1), n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window_fn, center=True)
        else:
            output_1d = self.istft(output_stft_r[:,0,:,:].permute(0,2,1), output_stft_r[:,1,:,:].permute(0,2,1),oneframe=oneframe)
        return output_1d.squeeze(1)

    def _get_amp_and_phase(self, x_out):
        x_out_real = x_out[:, 0, :, :]  # (B,T,F)
        x_out_imag = x_out[:, 1, :, :]
        amp = (x_out_real ** 2 + x_out_imag ** 2 + self._eps) ** 0.5
        amp = amp.unsqueeze(1) # (B,2,T,F)
        phase = x_out / (amp + self._eps)
        return amp, phase

    def _get_amp_and_phase_nogrd(self, x_out):
        x_out_real = x_out[:, 0, :, :]  # (B,T,F)
        x_out_imag = x_out[:, 1, :, :]
        amp = (x_out_real ** 2 + x_out_imag ** 2) ** 0.5
        amp = amp.unsqueeze(1)  # (B,2,T,F)
        phase = x_out / torch.maximum(amp, self._eps)
        return amp, phase