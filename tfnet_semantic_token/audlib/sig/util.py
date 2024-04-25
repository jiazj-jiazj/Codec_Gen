"""Utility functions related to audio processing.
TODO: move  FREQZ_CEILING = 1e5
and DITHER_SCALE= 1e-6
to config
"""

import numpy as np
from numpy.fft import fft
import numpy.random as rand
from scipy.signal import lfilter, fftconvolve

FREQZ_CEILING = 1e5
DITHER_SCALE = 1e-12
#1e-6

class EnergyVAD(object):
    """An energy-based Voice Activity Detector.

    This is based on Sebastian Braun's MATLAB implementation.
    """

    def __init__(self, sr, nfft, fmin=300., fmax=4000.):
        """Instantiate a VAD class.

        Keyword Parameters
        ------------------
        fmin: float, 300
            Minimum frequency from which energy is collected.
        fmax: float, 4000
            Maximum frequency from which energy is collected.

        """
        assert (fmin > 0) and (fmin < fmax) and (fmax < sr/2)
        fintv = sr / nfft
        self.bmin = int(fmin//fintv)
        self.bmax = int(min(fmax//fintv, nfft//2))  # inclusive
        self.nfft = nfft

    def __call__(self, pspec, dbfloor=-30., smoothframes=0):
        """Detect speech-active frames from a power spectra.

        Keyword Parameters
        ------------------
        dbfloor: float, -30
            Energy floor for a frame below maximum energy to be considered speech.
        smoothframes: int, 0
            Number of frames to apply a moving-average filter on power contour.
        """
        assert pspec.shape[1] == (self.nfft//2+1), "Incompatible dimension."
        pspec = pspec[:, self.bmin:self.bmax+1].sum(axis=1)
        if smoothframes > 0:
            pspec = lfilter(1/smoothframes * np.ones(smoothframes), 1, pspec)
        return pspec > (10**(dbfloor/10))*pspec.max()


def pre_emphasis(sig, alpha):
    """First-order highpass filter signal."""
    return lfilter([1, -alpha], 1, sig)


def dither(sig, scale=DITHER_SCALE):
    """Dither signal by adding small amount of noise to signal.

    Parameters
    ----------
    sig: array_like
        Signal to be processed.
    norm: bool, optional
        Normalize signal amplitude to range [-1, 1] before dithering.
        Default to no.
    scale: float, optional
        Amplitude scale to be applied to Gaussian noise.

    """
    return sig + np.random.randn(*sig.shape) * scale


def clipcenter(sig, threshold):
    """Center clipping by a threshold."""
    if threshold == 0:
        return sig
    out = np.zeros_like(sig)

    threshold = np.abs(threshold)
    maskp = sig > threshold
    maskn = sig < -threshold
    out[maskp] = sig[maskp] - threshold
    out[maskn] = sig[maskn] + threshold
    return out


def clipcenter3lvl(sig, threshold):
    """Three-level center clipping by a threshold."""
    out = np.zeros_like(sig)
    threshold = np.abs(threshold)
    maskp = sig > threshold
    maskn = sig < -threshold
    out[maskp] = 1
    out[maskn] = -1
    return out


def firfreqz(h, nfft):
    """Compute frequency response of an FIR filter."""
    ww = np.linspace(0, 2, num=nfft, endpoint=False)
    hh = fft(h, n=nfft)
    return ww, hh


def iirfreqz(h, nfft, ceiling=FREQZ_CEILING):
    """Compute frequency response of an IIR filter.

    Parameters
    ----------
    h: array_like
        IIR filter coefficent array for denominator polynomial.
        e.g. y[n] = x[n] + a*y[n-1] + b*y[n-2]
             Y(z) = X(z) + a*z^-1*Y(z) + b*z^-2*Y(z)
                                  1
             H(z) = ---------------------------------
                           1 - a*z^-1 -b*z^-2
             h = [1, -a, -b]

    """
    ww = np.linspace(0, 2, num=nfft, endpoint=False)
    hh_inv = fft(h, n=nfft)
    hh = np.zeros_like(hh_inv)
    zeros = (hh_inv == 0)
    hh[~zeros] = 1 / hh_inv
    hh[zeros] = ceiling
    return ww, hh


def freqz(b, a, nfft, ceiling=FREQZ_CEILING):
    """Compute the frequency response of a z-transform polynomial."""
    ww, hh_numer = firfreqz(b, nfft)
    __, hh_denom = iirfreqz(a, nfft, ceiling=ceiling)
    return ww, hh_numer * hh_denom


def nextpow2(n):
    """Give next power of 2 bigger than n."""
    return 1 << (n - 1).bit_length()


def ispow2(n):
    """Check if n is an integer power of 2."""
    return ((n & (n - 1)) == 0) and n != 0


def add_noise(x, n, snr=None):
    """Add user provided noise n with SNR=snr and signal x."""
    noise = additive_noise(x, n, snr=snr)
    if snr == -np.inf:
        return noise
    else:
        return x + noise


def additive_noise(x, n, snr=None):
    """Make additive noise at specific SNR.

    SNR = 10log10(Signal Energy/Noise Energy)
    NE = SE/10**(SNR/10)
    """
    if snr == np.inf:
        return np.zeros_like(x)
    # Modify noise to have equal length as signal
    xlen, nlen = len(x), len(n)
    if xlen > nlen:  # need to append noise several times to cover x range
        nn = np.tile(n, int(np.ceil(xlen / nlen)))[:xlen]
    else:
        nn = n[:xlen]

    if snr == -np.inf:
        return nn
    if snr is None:
        snr = (rand.random() - 0.25) * 20

    xe = x.dot(x)  # signal energy
    ne = nn.dot(nn)  # noise energy
    nscale = np.sqrt(xe / (10 ** (snr / 10.0)) / ne)

    return nscale * nn


def add_white_noise(x, snr=None):
    """Add white noise with SNR=snr to signal x.

    SNR = 10log10(Signal Energy/Noise Energy) = 10log10(SE/var(noise))
    var(noise) = SE/10**(SNR/10)
    """
    n = rand.normal(0, 1, x.shape)
    return add_noise(x, n, snr)


def white_noise(x, snr=None):
    """Return the white noise array given signal and desired SNR."""
    n = rand.normal(0, 1, x.shape)
    if snr is None:
        snr = (rand.random() - 0.25) * 20
    xe = x.dot(x)  # signal energy
    ne = n.dot(n)  # noise power
    nscale = np.sqrt(xe / (10 ** (snr / 10.0)) / ne)  # scaling factor

    return nscale * n


def normalize(x):
    """Normalize signal amplitude to be in range [-1,1]."""
    return x / np.max(np.abs(x))


def add_white_noise_rand(x):
    """Add white noise with SNR in range [-10dB,10dB]."""
    return add_white_noise(x, (rand.random() - 0.25) * 20)


def quantize(x, n):
    """Apply n-bit quantization to signal."""
    x /= np.ma.max(np.abs(x))  # make sure x in [-1,1]
    bins = np.linspace(-1, 1, 2 ** n + 1, endpoint=True)  # [-1,1]
    qvals = (bins[:-1] + bins[1:]) / 2
    bins[-1] = 1.01  # Include 1 in case of clipping
    return qvals[np.digitize(x, bins) - 1]

def rolling_window(x, window_size, hop_size):
    nframes = (len(x)-window_size)//hop_size + 1
    assert nframes > 0
    shape = (nframes, window_size)
    strides = x.strides[0]
    strides = (strides*hop_size, strides)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

def find_delay_from_slice(src_sig, f_sig,
    f_start=3000, f_len=1000, s_start=0, s_end=10000):
    y1 = src_sig[s_start:s_end]
    y2 = f_sig[f_start:f_start + f_len]
    dots = fftconvolve(y1, y2[::-1], "valid")
    offset = np.argmax(dots) - f_start + s_start
    return offset

def find_delay(sig1, sig2):
    sig1 = sig1.copy()
    sig1 -= np.mean(sig1)
    sig2 = sig2.copy()
    sig2 -= np.mean(sig2)

    ssq1 = np.sqrt(np.sum(np.power(sig1, 2)))
    ssq2 = np.sqrt(np.sum(np.power(sig2, 2)))

    dots = fftconvolve(sig1, sig2[::-1], "full")
    offset = np.argmax(dots) - len(sig2) + 1
    corr = max(dots)/ssq1/ssq2

    return offset, corr