import torch
import numpy as np
import shutil
import os
from scipy.io import wavfile
import librosa

def rpad_signal_for_codec(signal, combine_frames, hop_len):
    # pad for center stft
    signal_len = signal.shape[-1]
    pad_len = 0
    if signal_len % hop_len:
        pad_len = (int(signal_len / hop_len) + 1)*hop_len - signal_len        
    
    # pad for combine_frames   
    num_frames = int((signal_len + pad_len) / hop_len) + 1    
    if num_frames % combine_frames:
        padded_frames = combine_frames - num_frames % combine_frames
        pad_len += int(padded_frames * hop_len) 
        
    if pad_len > 0 and (signal is not None):
        signal = torch.nn.functional.pad(signal, (0, pad_len))
        
    return signal, pad_len

def read_feature(file,dtype):
    feature = []
    with open(file,'r') as f:
        lines = f.readlines()
    f.close()
    for line in lines:
        parts = line.split()
        feature.append(parts)
    return np.array(feature).astype(dtype)


def read_inds(file,dtype):
    inds = []
    with open(file, "r") as f:
        lines = f.readlines()
    f.close()
    for line in lines:
        items = line.strip(' \n').split(' ')
        inds.append(items)
    return np.array(inds).astype(dtype)

def write_inds(inds,out_path):
    with open(out_path, "w") as f:
        for line in inds:
            for item in line:
                f.write(str(item) + ' ')
            f.write('\n')
    f.close()

def create_folders(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def delete_folders(dir_path):
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

def samples_to_segment_len(sig_size, segment_size, shift):
    seq_size = 0 if sig_size < segment_size else (sig_size - segment_size + shift) // shift
    return seq_size

def segment_signal(signal, segment_size, shift_size):
    # compute how many segments that the signal has
    signal_size = len(signal)
    seq_size = samples_to_segment_len(signal_size, segment_size, shift_size)
    seg_signal = np.zeros((seq_size, segment_size), dtype=np.float32)
    # create the segment signal
    for i, j in enumerate(range(0, seq_size*shift_size, shift_size)):
        seg_signal[i] = signal[j:j+segment_size]
    return seg_signal

def read_wav(path, offset=0.0, mono=True, duration=None, samp_rate=16000):
    signal, sr = librosa.load(path, mono=mono, sr=samp_rate,
                              offset=offset, duration=duration)
    return signal.astype(np.float32)


def audiowrite(path, data, samp_rate=16000):
    amp_max = max(np.abs(data))
    if amp_max > 1:
        data = data / amp_max
    data = (data + 1) / 2 * 65535 - 32768
    data = data.astype(np.int16)
    wavfile.write(path, samp_rate, data)
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        # print(" n: {}   overall entropy : {}".format(self.count,self.avg))

def entropy_to_bitrate(entropy, sr, hop_size):
    return (sr / hop_size) * entropy

def bitrate_to_entropy(bitrate, sr, hop_size,):
    return bitrate * hop_size / sr
    
def bitrate_to_entropy_2(bitrate, hop_dur):
    return int(bitrate * hop_dur)

def overlap_and_add(sig_2d, shift):
    frame_num = sig_2d.shape[0]
    frame_size = sig_2d.shape[1]
    sig_1D = np.zeros((frame_num - 1) * shift + frame_size, np.float32)
    for i in range(frame_num):
        sig_1D[(i * shift):(i * shift + frame_size)] += sig_2d[i]
    return sig_1D