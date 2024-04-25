import torch
import torch.nn as nn
#import torch.nn.functional as F

from .istft import ISTFT


class Freq2Time2(nn.Module):
    def __init__(self, frm_size, shift, win_len, hop_len, n_fft, config=None, output_type='amp_gain_complex_ratio'):
        super(Freq2Time2, self).__init__()

        self.frm_size = frm_size
        self.shift = shift
        self.win_len = win_len
        self.hop_len = hop_len
        self.n_fft = n_fft

        self._eps = torch.tensor(1e-7)
        self.istft = ISTFT(filter_length=win_len, hop_length=hop_len, window='hanning', )
        self.tanh = nn.Tanh()  
        self.output_options = ['residue_bound10', 'residue_unbound', 'amp_gain_complex_ratio']
        self.output_type = output_type 
        self.window_fn = torch.hann_window(win_len)

    def forward(self, input_list, oneframe=False):  
        self._eps = self._eps.to(input_list[0])
        dec_out, input_stft_r = input_list
        B, C, T, F = input_stft_r.size()
        self.window_fn = self.window_fn.to(dec_out)
        
        if self.output_type == 'residue_bound10':
            complex_residue = 10*self.tanh(dec_out)
            output_stft_r = input_stft_r + complex_residue
        elif self.output_type == 'residue_unbound':
            output_stft_r = input_stft_r + dec_out
        elif self.output_type == 'amp_gain_complex_ratio':            
            amp_gain, phs_output = self._get_amp_and_phase(dec_out)
            amp_gain = self.tanh(amp_gain)  # uncompressed
            output_stft_r = self._get_output_stft(amp_gain, phs_output, input_stft_r)
            
        if (self.n_fft != self.win_len) or ((self.win_len % self.hop_len) != 0): 
            if oneframe:            
                output_1d = torch.istft(output_stft_r.permute(0,3,2,1), n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window_fn, center=False)
            else:
                output_1d = torch.istft(output_stft_r.permute(0,3,2,1), n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window_fn, center=True)
        else:
            output_1d = self.istft(output_stft_r[:,0,:,:].permute(0,2,1), output_stft_r[:,1,:,:].permute(0,2,1), oneframe=oneframe)
        #output_1d = self.istft(output_stft_r[:,0,:,:].permute(0,2,1), output_stft_r[:,1,:,:].permute(0,2,1), oneframe=oneframe)
        return output_1d.squeeze(1)


    def _get_amp_and_phase(self, x_out):
        x_out_real = x_out[:, 0, :, :]  # (B,T,F)
        x_out_imag = x_out[:, 1, :, :]
        amp = (x_out_real ** 2 + x_out_imag ** 2 + self._eps) ** 0.5
        amp = amp.unsqueeze(1) # (B,1,T,F)
        phase = x_out / (amp + self._eps) # (B,2,T,F)
        return amp, phase

    def _get_amp_and_phase_nogrd(self, x_out):
        x_out_real = x_out[:, 0, :, :]  # (B,T,F)
        x_out_imag = x_out[:, 1, :, :]
        amp = (x_out_real ** 2 + x_out_imag ** 2) ** 0.5
        amp = amp.unsqueeze(1)  # (B,2,T,F)
        phase = x_out / torch.maximum(amp, self._eps)
        return amp, phase

    def _get_output_stft(self, amp_gain, phs_output, input_stft_r):
        amp_input, phs_input = self._get_amp_and_phase_nogrd(input_stft_r)
        # for amplitude 
        amp_out = amp_input * amp_gain
        # for phase
        insig_r, insig_i = phs_input[:,0,:,:], phs_input[:,1,:,:]
        gain_r, gain_i = phs_output[:,0,:,:], phs_output[:,1,:,:]
        outsig_r = insig_r*gain_r - insig_i*gain_i
        outsig_i = insig_r*gain_i + insig_i*gain_r
        outsig_phs = torch.stack((outsig_r, outsig_i), dim=1)
        output = amp_out * outsig_phs
        return output  # (B,2,T,F)