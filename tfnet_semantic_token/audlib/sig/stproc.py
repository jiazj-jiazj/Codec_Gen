"""Short-Time PROCessing of audio signals.

This module implements common audio analysis and synthesis techniques based on
short-time signal analysis. All short-time representations are returned as
GeneratorType.

See Also
--------
fbanks: Filterbank analysis and synthesis

"""

import math
from types import GeneratorType

import numpy as np
from numpy.lib.stride_tricks import as_strided

from .window import hop2hsize


def stcenters(sig, sr, wind, hop, synth=False, center=False):
    """Calculate window centers for each frame.

    See `numframes` for meaning of the parameters.
    """
    ssize = len(sig)
    fsize = len(wind)
    hsize = hop2hsize(wind, hop)
    if synth:
        sstart = hsize-fsize  # int(-fsize * (1-hfrac))
    elif center:
        sstart = -int(len(wind)/2)  # odd window centered at exactly n=0
    else:
        sstart = 0
    send = ssize

    return (np.arange(sstart, send, hsize) + (fsize-1)/2.) / sr


def numframes(sig, sr, wind, hop, synth=False, center=False):
    """Calculate total number of frames.

    Use this function to pre-determine the size of stft.

    Parameters
    ----------
    sig: array_like
        Signal to be analyzed.
    sr: int
        Sampling rate.
    wind: array_like
        Window function.
    hop: float or int
        Hop fraction in (0, 1) or hop size in integers.
    trange: tuple of float
        Starting and ending point in seconds.
        Default to (None, None), which computes a duration that enables
        perfect reconstruction.

    Returns
    -------
    out: int
        Number of frames to be computed by `stana`.

    See Also
    --------
    `stana`.

    """
    ssize = len(sig)
    fsize = len(wind)

    hsize = hop2hsize(wind, hop)
    if synth:
        sstart = hsize-fsize  # int(-fsize * (1-hfrac))
    elif center:
        sstart = -int(len(wind)/2)  # odd window centered at exactly n=0
    else:
        sstart = 0
    send = ssize

    return math.ceil((send-sstart)/hsize)


def stana(sig, sr, wind, hop, synth=False, center=False):
    """[S]hort-[t]ime [Ana]lysis of audio signal by windowing.

    Parameters
    ----------
    sig: array_like
        Time series to be analyzed.
    sr: int
        Sampling rate.
    wind: array_like
        Window function used for framing. See `window` for window functions.
    hop: float or int
        Hop fraction in (0, 1) or hop size in integers.
    synth: bool, optional
        Whether time-domain synthesis is the end goal.
        Default to false.
    center: bool, optional
        Shift the windowed signal by half a window length if true. This is
        useful for applications like speech activity detection.
        Default to false.

    Returns
    -------
    frames: numpy.ndarray
        Short-time signal after windowing.

    See Also
    --------
    window.hamming: used to construct a valid window for analysis(/synthesis).

    """
    ssize = len(sig)
    fsize = len(wind)
    hsize = hop2hsize(wind, hop)
    if synth:
        sstart = hsize-fsize  # int(-fsize * (1-hfrac))
    elif center:
        sstart = -int(len(wind)/2)  # odd window centered at exactly n=0
    else:
        sstart = 0
    send = ssize

    nframe = math.ceil((send-sstart)/hsize)
    # Calculate zero-padding sizes
    zpleft = -sstart
    zpright = (nframe-1)*hsize+fsize - zpleft - ssize
    if zpleft > 0 or zpright > 0:
        sigpad = np.zeros(ssize+zpleft+zpright)
        sigpad[zpleft:len(sigpad)-zpright] = sig
    else:
        sigpad = sig

    std = sigpad.strides[0]
    return as_strided(sigpad, shape=(nframe, fsize),
                      strides=(std*hsize, std)) * wind

# Below is equivalent and more readable code for reference
    # """
    # frames = np.empty((math.ceil((send-sstart)/hsize), fsize))

    # for ii, si in enumerate(range(sstart, send, hsize)):
    #     sj = si + fsize

    #     if si < 0:  # [0 0 ... x[0] x[1] ...]
    #         frames[ii] = np.pad(sig[:sj], (fsize-sj, 0), 'constant')
    #     elif sj > ssize:  # [... x[-2] x[-1] 0 0 ... 0]
    #         frames[ii] = np.pad(sig[si:], (0, fsize-ssize+si), 'constant')
    #     else:  # [x[..] ..... x[..]]
    #         frames[ii] = sig[si:sj]

    # return frames * wind
    # """

def ola(sframes, sr, wind, hop):
    """Short-time Synthesis by [O]ver[l]ap-[A]dd.

    Perform the Overlap-Add algorithm on an array of short-time analyzed
    frames. Arguments used to call `stana` should be used here for consistency.
    Assume stana is called with trange set to default (i.e., permits perfect
    reconstruction).

    Parameters
    ----------
    sframes: array_like or iterable
        Array of short-time frames.
    sr: int
        Sampling rate.
    wind: 1-D ndarray
        Window function.
    hop: int, float
        Hop size or hop fraction of window.

    Returns
    -------
    sout: ndarray
        Reconstructed time series.

    See Also
    --------
    stana

    """
    if type(sframes) is GeneratorType:
        sframes = np.asarray(list(sframes))

    nframe = len(sframes)
    fsize = len(wind)
    hsize = hop2hsize(wind, hop)
    hfrac = hsize*1. / fsize
    ssize = hsize*(nframe-1)+fsize  # total OLA size
    sstart = int(-fsize * (1-hfrac))  # OLA starting index
    send = ssize + sstart  # OLA ending index
    ii = sstart  # pointer to beginning of current time frame

    sout = np.zeros(send)
    for frame in sframes:
        frame = frame[:fsize]  # for cases like DFT
        if ii < 0:  # first (few) frames
            sout[:ii+fsize] += frame[-ii:]
        elif ii+fsize > send:  # last (few) frame
            sout[ii:] += frame[:(send-ii)]
        else:
            sout[ii:ii+fsize] += frame
        ii += hsize

    return sout


def frate2hsize(sr, frate):
    """Translate frame rate in Hz to hop size in integer."""
    return int(sr*1.0/frate)
