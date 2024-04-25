import torch.nn as nn
class SpeakerClassifier(nn.Module):
    def __init__(self, indims, n_spk):
        super().__init__()
        self.fc = nn.Linear(indims, n_spk)
        return

    def forward(self, x):
        predicted_speaker = self.fc(x)
        return predicted_speaker


class PhonemeClassifier(nn.Module):
    def __init__(self, indims, n_phoneme):
        super().__init__()
        self.ff1 = nn.Linear(indims, indims)
        self.ff2 = nn.Linear(indims, n_phoneme)
        return

    def forward(self, x):
        # x [B,C_phn,T]
        B,C,T  = x.shape
        x_flat = x.permute(0,2,1).reshape(-1, C)
        x = self.ff1(x_flat)
        predicted_phoneme = self.ff2(x)
        predicted_phoneme = predicted_phoneme.reshape(B,T,-1).permute(0,2,1)
        feat_list = [x.reshape(B,T,-1).permute(0,2,1), predicted_phoneme]
        return predicted_phoneme,feat_list # [B,n_phoneme,T]

