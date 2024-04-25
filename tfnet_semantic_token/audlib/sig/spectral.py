"""Frame-level frequency-domain processing."""
import math

import numpy as np
from numpy.fft import rfft, irfft

def onlineMVN_broadband(x, frameShift, tauFeat=3., tauFeatInit=0.1, t_init=0.1):
    """Online mean and variance normalization (broadband)"""
    nInitFrames = math.ceil(t_init/frameShift)
    alphaFeatInit = math.exp(-frameShift/tauFeatInit)
    alphaFeat = math.exp(-frameShift/tauFeat)
    (nFeat, nFrames) = x.shape
    # initialize with first frame
    mu = np.mean(x[:,0])
    sigmaSquare = np.mean(x[:,0]**2)
    y = x
    for nn in range(0, nFrames):
        x_n = x[:,nn]
        if nn < nInitFrames:
            alpha = alphaFeatInit
        else:
            alpha = alphaFeat
        mu = alpha*mu + (1-alpha)*np.mean(x_n)
        sigmaSquare = alpha*sigmaSquare + (1-alpha)*np.mean(x_n**2)
        sigma = np.sqrt(np.maximum(sigmaSquare - mu**2, 1e-12)) # limit for sqrt
        y[:,nn] = (x_n - mu) / sigma
    return y

def onlineMVN(x, frameShift, tauFeat=3., tauFeatInit=0.1, t_init=0.1):
    """Online mean and variance normalization (per frequency)"""
    nInitFrames = math.ceil(t_init/frameShift)
    alphaFeatInit = math.exp(-frameShift/tauFeatInit)
    alphaFeat = math.exp(-frameShift/tauFeat)
    (nFeat, nFrames) = x.shape
    # initialize with first frame
    mu = x[:,0]
    sigmaSquare = x[:,0]**2
    y = x
    for nn in range(0, nFrames):
        x_n = x[:,nn]
        if nn < nInitFrames:
            alpha = alphaFeatInit
        else:
            alpha = alphaFeat
        mu = alpha*mu + (1-alpha)*x_n
        sigmaSquare = alpha*sigmaSquare + (1-alpha)*x_n**2
        sigma = np.sqrt(np.maximum(sigmaSquare - mu**2, 1e-12)) # limit for sqrt
        y[:,nn] = (x_n - mu) / sigma
    return y
     
def onlineMVN_v2(powspec, frameShift, tauFeat=3., tauFeatInit=0.1, t_init=0.1):
    """Online mean and variance normalization (per frequency)
        x: numpy.ndarray
        Real-valued short-time power spectra with dimension (T,F).
        """
    x = np.transpose(powspec)
#    print(x.shape)
    nInitFrames = math.ceil(t_init/frameShift)
    alphaFeatInit = math.exp(-frameShift/tauFeatInit)
    alphaFeat = math.exp(-frameShift/tauFeat)
    (nFeat, nFrames) = x.shape
    # initialize with first frame
    mu = x[:,0]
    sigmaSquare = x[:,0]**2
    y = x
    for nn in range(0, nFrames):
        x_n = x[:,nn]
        if nn < nInitFrames:
            alpha = alphaFeatInit
        else:
            alpha = alphaFeat
        mu = alpha*mu + (1-alpha)*x_n
        sigmaSquare = alpha*sigmaSquare + (1-alpha)*x_n**2
        sigma = np.sqrt(np.maximum(sigmaSquare - mu**2, 1e-12)) # limit for sqrt
        y[:,nn] = (x_n - mu) / sigma
#        print(sigma.shape)
#        print(mu.shape)
#    print(y.shape)
    return np.transpose(y)
    
def mvnorm1(powspec, frameshift, tau=3., tau_init=.1, t_init=.2):
    """Online mean and variance normalization of a short-time power spectra.

    This function computes online mean/variance as a scalar instead of a vector
    in `mvnorm`.

    Parameters
    ----------
    powspec: numpy.ndarray
        Real-valued short-time power spectra with dimension (T,F).
    frameshift: float
        Number of seconds between adjacent frames.

    Keyword Parameters
    ------------------
    tau: float, 3.
        Time constant of the median-time recursive averaging function.
    tau_init: float, .1
        Initial time constant for fast adaptation.
    t_init: float, .2
        Amount of time in seconds from the beginning during which `tau_init` is applied.
        The rest of time will use `tau`.

    Returns
    -------
    powspec_norm: numpy.ndarray
        Normalized short-time power spectra with dimension (T,F).

    """
    alpha = np.exp(-frameshift / tau)
    alpha0 = np.exp(-frameshift / tau_init)  # fast adaptation
    init_frames = math.ceil(t_init / frameshift)

    mu = np.empty(len(powspec))
    var = np.empty(len(powspec))
    for ii, spec in enumerate(powspec):
        if ii == 0:  # very first frame
            mu[ii] = alpha0 * powspec.mean() + (1-alpha0)*spec.mean()
            var[ii] = alpha0 * (powspec**2).mean() + \
                (1-alpha0)*(spec**2).mean()
            continue

        if (ii+1) < init_frames:  # select adaptation speed
            alpha_ = alpha0
        else:
            alpha_ = alpha
        mu[ii] = alpha_*mu[ii-1] + (1-alpha_)*spec.mean()
        var[ii] = alpha_*var[ii-1] + (1-alpha_)*(spec**2).mean()

    return (powspec - mu[:, np.newaxis]) / np.maximum(
        np.sqrt(np.maximum(var[:, np.newaxis]-mu[:, np.newaxis]**2, 0)), 1e-12)


def mvnorm(powspec, frameshift, tau=3., tau_init=.1, t_init=.2):
    """Online mean and variance normalization of a short-time power spectra.

    This is a direct port of Sebastian Braun's MATLAB function inside `generateVADfeatures.m`.

    Parameters
    ----------
    powspec: numpy.ndarray
        Real-valued short-time power spectra with dimension (T,F).
    frameshift: float
        Number of seconds between adjacent frames.

    Keyword Parameters
    ------------------
    tau: float, 3.
        Time constant of the median-time recursive averaging function.
    tau_init: float, .1
        Initial time constant for fast adaptation.
    t_init: float, .2
        Amount of time in seconds from the beginning during which `tau_init` is applied.
        The rest of time will use `tau`.

    Returns
    -------
    powspec_norm: numpy.ndarray
        Normalized short-time power spectra with dimension (T,F).

    """
    alpha = np.exp(-frameshift / tau)
    alpha0 = np.exp(-frameshift / tau_init)  # fast adaptation
    init_frames = math.ceil(t_init / frameshift)

    mu = np.empty_like(powspec)
    var = np.empty_like(powspec)
    for ii, spec in enumerate(powspec):
        if ii == 0:  # very first frame
            mu[ii] = alpha0 * powspec.mean(axis=0) + (1-alpha0)*spec
            var[ii] = alpha0 * (powspec**2).mean(axis=0) + (1-alpha0)*(spec**2)
            continue

        if (ii+1) < init_frames:  # select adaptation speed
            alpha_ = alpha0
        else:
            alpha_ = alpha
        mu[ii] = alpha_*mu[ii-1] + (1-alpha_)*spec
        var[ii] = alpha_*var[ii-1] + (1-alpha_)*(spec**2)

    return (powspec - mu) / np.maximum(np.sqrt(np.maximum(var-mu**2, 0)), 1e-12)


def agc_simple(cspectrum, frameshift, nfft=None):
    """Simple automatic gain control by Sebastian Braun.

    This is a direct port Sebastian's MATLAB function AGCsimple.

    Parameters
    ----------
    cspectrum: numpy.ndarray
        Complex short-time spectrum with dimension (T,F).
    frameshift: float
        Number of seconds between adjacent frames.
    nfft: int, None
        N-point DFT used to compute cspectrum. Default to cspectrum.shape[1].
    Returns
    -------
    gains: numpy.ndarray
        Gain function per frame with dimension (T,).

    """
    def db2pow(db): return 10**(db/10.)
    def _powspec(cspec): return cspec.real**2 + cspec.imag**2
    alpha_x = np.exp(-frameshift / .3)
    alpha0 = np.exp(-frameshift / .8)
    beta_slow = db2pow(.01 * frameshift)
    targetlvl = db2pow(-25) * (nfft if nfft else cspectrum.shape[1])

    gains = np.empty(len(cspectrum))
    stpow = 0  # power per frame
    for ii, spectrum in enumerate(cspectrum):
        stpow = alpha_x * stpow + (1-alpha_x) * _powspec(spectrum).mean()
        if ii < 3:
            speechlvl = stpow
        if (ii < 9) or (stpow > speechlvl):
            # fast signal dependent update
            speechlvl = alpha0*speechlvl + (1-alpha0)*stpow
        else:
            # slow signal independent update
            speechlvl *= (beta_slow**np.sign(np.log(stpow/speechlvl)))
        gains[ii] = np.sqrt(targetlvl / speechlvl)

    return gains


def magphasor(complexspec):
    """Decompose a complex spectrogram into magnitude and unit phasor.

    m, p = magphasor(c) such that c == m * p.
    """
    mspec = np.abs(complexspec)
    pspec = np.empty_like(complexspec)
    zero_mag = (mspec == 0.)  # fix zero-magnitude
    pspec[zero_mag] = 1.
    pspec[~zero_mag] = complexspec[~zero_mag]/mspec[~zero_mag]
    return mspec, pspec


def magphase(cspectrum, unwrap=False):
    """Decompose complex spectrum into magnitude and phase."""
    mag = np.abs(cspectrum)
    phs = np.angle(cspectrum)
    if unwrap:
        phs = np.unwrap(phs)
    return mag, phs


def logmagphase(cspectrum, unwrap=False, floor=-10.):
    """Compute (log-magnitude, phase) of complex spectrum."""
    mag, phs = magphase(cspectrum, unwrap=unwrap)
    return logmag(mag, floor=floor), phs


def logmag(sig, floor=-10.):
    """Compute log magnitude of complex spectrum.

    Floor any -`np.inf` value to `floor` plus log minimum. If all values are
    0s, floor all values to floor*5.
    """
    mag = np.abs(sig)
    zeros = mag == 0
    logm = np.empty_like(mag)
    logm[~zeros] = np.log(mag[~zeros])
    logmin = np.log(mag[~zeros].min()) if np.any(~zeros) else floor*5
    logm[zeros] = floor + logmin

    return logm


def logpow_msrtc(sig, floor=None):
    pspec = np.maximum(sig**2, 1e-12)
    return np.log10(pspec)

def logpow_dns(sig, floor=-30.):
    """Compute log power of complex spectrum.

    Floor any -`np.inf` value to (nonzero minimum + `floor`) dB.
    If all values are 0s, floor all values to -80 dB.
    """
    log10e = np.log10(np.e)
    pspec = sig.real**2 + sig.imag**2
    zeros = pspec == 0
    logp = np.empty_like(pspec)
    if np.any(~zeros):
        logp[~zeros] = np.log(pspec[~zeros])
        logp[zeros] = np.log(pspec[~zeros].min()) + floor / 10 / log10e
    else:
        logp.fill(-80 / 10 / log10e)

    return logp

logpow = logpow_msrtc

def logpow2(sig, floor=None):
    """Compute log power of complex spectrum.
    floot argument is for compatibility with logpow
    """
    pspec = sig.real**2 + sig.imag**2
    return np.log10(pspec + 1e-10)


def phasor(mag, phase):
    """Compute complex spectrum given magnitude and phase."""
    return mag * np.exp(1j*phase)


def dct1(x, dft=False):
    """Perform Type-1 Discrete Cosine Transform (DCT-1) on input signal.

    Parameters
    ----------
    x: array_like
        Signal to be processed.
    dft: boolean
        Implement using dft?

    Returns
    -------
    X: array_like
        Type-1 DCT of x.

    """
    if len(x) == 1:
        return x.copy()
        
    ndct = len(x)

    if dft:  # implement using dft
        x_ext = np.concatenate((x, x[-2:0:-1]))  # create extended sequence
        return np.real(rfft(x_ext)[:ndct])

# otherwise, implement using definition
    xa = x * 1.
    xa[1:-1] *= 2.  # form x_a sequence
    X = np.zeros_like(xa)
    ns = np.arange(ndct)
    for k in range(ndct):
        cos = np.cos(np.pi*k*ns/(ndct-1))
        X[k] = cos.dot(xa)
    return X

def idct1(x_dct1, dft=False):
    """Perform inverse Type-1 Discrete Cosine Transform (iDCT-1) on spectrum.

    Parameters
    ----------
    x_dct1: array_like
        Input DCT spectrum.
    dft: boolean
        Implement using dft?

    Returns
    -------
    x: array_like
        Inverse Type-1 DCT of x_dct1.

    """
    if len(x_dct1) == 1:
        return x_dct1.copy()
    ndct = len(x_dct1)

    if dft:  # implement using dft
        x = irfft(x_dct1, n=2*(ndct-1))[:ndct]
    else:  # implement using definition
        Xb = x_dct1 / (ndct-1.)
        Xb[0] /= 2.
        Xb[-1] /= 2.
        x = np.zeros_like(Xb)
        ks = np.arange(ndct)
        for n in range(ndct):
            cos = np.cos(np.pi*n*ks/(ndct-1))
            x[n] = cos.dot(Xb)
    return x


def dct2(x, norm=True, dft=False):
    """Perform Type-2 Discrete Cosine Transform (DCT-2) on input signal.

    Parameters
    ----------
    x: array_like
        Input signal.
    norm: boolean, optional
        Normalize so that energy is preserved. Default to True.
    dft: boolean, optional
        Implement using dft? Default to False.

    Returns
    -------
    X: numpy array
        Type-2 DCT of x.

    """
    if len(x) == 1:
        return x.copy()
    ndct = len(x)

    if dft:  # implement using dft
        if norm:
            raise ValueError("DFT method does not support normalization!")
        Xk = rfft(x, 2*ndct)[:ndct]
        X = 2*np.real(Xk*np.exp(-1j*(np.pi*np.arange(ndct)/(2*ndct))))
    else:  # implement using definition
        if norm:
            xa = 1.*x
        else:
            xa = 2.*x
        X = np.zeros_like(xa)
        ns = np.arange(ndct)
        for k in range(ndct):
            cos = np.cos(np.pi*k*(2*ns+1)/(2*ndct))
            X[k] = cos.dot(xa)
            if norm:
                X[k] *= np.sqrt(2./ndct)
        if norm:
            X[0] /= np.sqrt(2)
    return X


def idct2(x_dct2, norm=True, dft=False):
    """Perform inverse Type-2 Discrete Cosine Transform (DCT-2) on spectrum.

    Parameters
    ----------
    x_dct2: array_like
        Input signal.
    norm: boolean, optional
        Normalize so that energy is preserved. Default to True.
    dft: boolean, optional
        Implement using dft? Default to False.

    Returns
    -------
    x: array_like
        Inverse Type-2 DCT of x_dct2.

    """
    if len(x_dct2) == 1:
        return x_dct2.copy()
    ndct = len(x_dct2)

    if dft:  # implement using dft
        if norm:
            raise ValueError("DFT method does not support normalization!")
        ks = np.arange(ndct)
        Xseg1 = x_dct2*np.exp(1j*np.pi*ks/(2*ndct))
        Xseg2 = -x_dct2[-1:0:-1]*np.exp(1j*np.pi*(ks[1:]+ndct)/(2*ndct))
        X_ext = np.concatenate((Xseg1, [0.], Xseg2))
        x = irfft(X_ext[:ndct+1])[:ndct]
    else:  # implement using definition
        if norm:
            Xb = x_dct2 * np.sqrt(2./ndct)
            Xb[0] /= np.sqrt(2.)
        else:
            Xb = x_dct2 / (ndct+0.0)
            Xb[0] /= 2.
        x = np.zeros_like(Xb)
        ks = np.arange(ndct)
        for n in range(ndct):
            cos = np.cos(np.pi*ks*(2*n+1)/(2*ndct))
            x[n] = cos.dot(Xb)
    return x


def realcep(frame, n, nfft=4096, floor=-10., comp=False, ztrans=False):
    """Compute real cepstrum of short-time signal `frame`.

    There are two modes for calculation:
        1. complex = False (default). This calculates c[n] using inverse DFT
        of the log magnitude spectrum.
        2. complex = True. This first calculates the complex cepstrum through
        the z-transform method (see `compcep`), and takes the even function to
        obtain c[n].
    In both cases, `len(cep) = nfft//2+1`.

    Parameters
    ----------
    frame: 1-D ndarray
        signal to be processed.
    nfft: non-negative int
        nfft//2+1 cepstrum in range [0, nfft//2] will be evaluated.
    floor: float [-10.]
        flooring for log(0). Ignored if complex=True.
    complex: boolean [False]
        Use mode 2 for calculation.

    Returns
    -------
    cep: 1-D ndarray
        Real ceptra of signal `frame` of:
        1. length `len(cep) = nfft//2+1`.
        2. quefrency index [0, nfft//2].

    """
    if comp:  # do complex method
        ccep = compcep(frame, n-1, ztrans=ztrans)
        rcep = .5*(ccep+ccep[::-1])
        return rcep[n-1:]  # only keep non-negative quefrency
    else:  # DFT method
        rcep = irfft(logmag(rfft(frame, nfft), floor=floor))
        return rcep[:n]


def compcep(frame, n, nfft=4096, floor=-10., ztrans=False):
    """Compute complex cepstrum of short-time signal using Z-transform.

    Compute the aliasing-free complex cepstrum using Z-transform and polynomial
    root finder. Implementation is based on RS eq 8.68 on page 436.

    Parameters
    ----------
    frame: 1-D ndarray
        signal to be processed.
    n: non-negative int
        index range [-n, n] in which complex cepstrum will be evaluated.

    Returns
    -------
    cep: 1-D ndarray
        complex ceptrum of length `2n+1`; quefrency index [-n, n].

    """
    if ztrans:
        frame = np.trim_zeros(frame)
        f0 = frame[0]
        roots = np.roots(frame/f0)
        rmag = np.abs(roots)
        assert 1 not in rmag
        ra, rb = roots[rmag < 1], roots[rmag > 1]
        amp = f0 * np.prod(rb)
        if len(rb) % 2:  # odd number of zeros outside UC
            amp = -amp
        # obtain complex cepstrum through eq (8.68) in RS, pp. 436
        cep = np.zeros(2*n+1)
        if rb.size > 0:
            for ii in range(-n, 0):
                cep[-n+ii] = np.real(np.sum(rb**ii))/ii
        cep[n] = np.log(np.abs(amp))
        if ra.size > 0:
            for ii in range(1, n+1):
                cep[n+ii] = -np.real(np.sum(ra**ii))/ii
    else:
        assert n <= nfft//2
        spec = rfft(frame, n=nfft)
        lmag, phase = logmagphase(spec, unwrap=True, floor=floor)
        cep = irfft(lmag+1j*phase, n=nfft)[:2*n+1]
        cep = np.roll(cep, n)
    return cep
