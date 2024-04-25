import numpy as np
import shutil
import os
from scipy.io import wavfile
import librosa
import math

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

def samples_to_segment_len(sig_size, segment_size, shift, estim_type="floor"):
    if sig_size < segment_size:
        seq_size = 0
    elif estim_type == "ceil":
        seq_size = math.ceil((sig_size - segment_size) / shift) + 1
    else:
        seq_size = (sig_size - segment_size + shift) // shift
    return seq_size

def segment_signal(signal, segment_size, shift_size, estim_type="floor"):
    # compute how many segments that the signal has
    signal_size = len(signal)
    if segment_size < 0:
        segment_size = signal_size
        shift_size = signal_size
    seq_size = samples_to_segment_len(signal_size, segment_size, shift_size, estim_type=estim_type)
    seg_signal = np.zeros((seq_size, segment_size), dtype=np.float32)
    # create the segment signal
    for i, j in enumerate(range(0, seq_size*shift_size, shift_size)):
        seg_signal[i][:min(signal_size - j, segment_size)] = signal[j:min(j+segment_size, signal_size)]
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

def is_vqvae(config):
    return True if "vqvae" in config["model_type"] else False
    
def is_controllable_dns_vqvae(config):
    return True if "controldns" in config["model_type"] else False
    
def is_multirate_vqvae(config):
    return True if config['model_type'] == 'tfnetv2_interleave_multiraterps_vqvae' and (config['bitrate'] == 'all4') else False

def overlap_and_add(sig_2d, shift):
    frame_num = sig_2d.shape[0]
    frame_size = sig_2d.shape[1]
    sig_1D = np.zeros((frame_num - 1) * shift + frame_size, np.float32)
    for i in range(frame_num):
        sig_1D[(i * shift):(i * shift + frame_size)] += sig_2d[i]
    return sig_1D

# return [(start_1, end_1), (start_2, end_2), ...]
def read_timestamp(file, samplerate, drop=0):
    timestamp_sample = []

    txt = open(file, "r")
    list_lines = txt.readlines()
    for line in list_lines:
        try:
            parts = line.split()
            start = int(float(parts[0]) * samplerate) - drop
            end = int(float(parts[1]) * samplerate) - drop
            timestamp_sample.append((start, end))
        except Exception as e:
            # print("[warning]{} at {}".format(e, file))
            continue
    return timestamp_sample


def segment_pl_record(timestamp, seq_size, segment_size, shift_size, log_writer=None):
    factor = segment_size // shift_size
    seg_pl = np.zeros(shape=(seq_size, factor), dtype=int)
    seq_pl_len = (seq_size - 1) + factor
    seq_pl = np.zeros(seq_pl_len, dtype=int)
    seq_period = ['None'] * seq_pl_len
    seg_period = [['None'] * factor] * seq_size
    count_positive_frames = 0

    for period in timestamp:
        start_frm_idx = max((period[0] - shift_size) // shift_size + 1, 0)
        end_frm_idx = int(np.ceil(period[1] / shift_size))
        end_frm_idx = min(end_frm_idx, seq_pl_len)
        start_frm_idx = min(start_frm_idx, end_frm_idx)
        seq_pl[start_frm_idx: end_frm_idx] = 1
        for i in range(start_frm_idx, end_frm_idx):
            seq_period[i] = '({},{})'.format(str(period[0]), str(period[1]))
    # create the segment pl
    for i in range(seq_size):
        seg_pl[i] = seq_pl[i:i + factor]
        seg_period[i] = seq_period[i:i + factor]
    # get all the postive frames(including half pl frame)
    positive_frames = [frame for frame in seg_pl if np.sum(frame) > 0]
    count_positive_frames += len(positive_frames)

    if log_writer:
        for frm_idx in range(seq_size):
            start_frm_samp = frm_idx * shift_size
            frame_pl = seg_pl[frm_idx]
            frame_period = seg_period[frm_idx]
            for i in range(factor):
                if frame_pl[i] == 1:
                    start_sample = start_frm_samp + i * shift_size
                    end_sample = start_sample + shift_size
                    log_writer.write(
                        "frame:{}\t start:{}\t end:{}\t pl_period:{}\n".format(frm_idx, start_sample, end_sample,
                                                                               frame_period[i]))

    return seg_pl, count_positive_frames


def freeze_or_unfreeze_module(model, module_name, freeze=True):
    grad = True if not freeze else False
    for name, module in model.named_children():
        if name in module_name:
            for k, para in module.named_parameters():
                para.requires_grad = grad

def freeze_or_unfreeze_para(model, para_name, freeze=True):
    grad = True if not freeze else False
    for name, param in model.named_parameters():
        if name in para_name:
            param.requires_grad = grad