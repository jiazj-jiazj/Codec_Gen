"""Data encapsulation suitable for batch processing."""


class Audio(object):
    """A class for any processing that requires only signal and sr."""

    __slots__ = "signal", "samplerate"

    def __init__(self, signal=None, samplerate=None):
        self.signal = signal
        self.samplerate = samplerate


class EchoSpeech(object):
    """Data structure for echo speech resulted from (additive) noise."""

    __slots__ = "mic", "nearend", "target", "echo", "farend", "snr", "is_dt", "meta"

    def __init__(self, mic=None, nearend=None, target=None, echo=None, farend=None, snr=None, is_dt=None, meta=None):
        self.mic = mic
        self.nearend = nearend
        self.target = target
        self.echo = echo
        self.farend = farend
        self.snr = snr
        self.is_dt = is_dt
        self.meta = meta

class EchoSpeechVQE(object):
    """Data structure for echo speech resulted from vqe."""

    __slots__ = "mic", "nearend", "target", "echo", "farend", "snr", "is_dt", "vqe", "echo_vqe", "meta"

    def __init__(self, mic=None, nearend=None, target=None, echo=None, farend=None, snr=None, is_dt=None, vqe=None, echo_vqe=None, meta=None):
        self.mic = mic
        self.nearend = nearend
        self.target = target
        self.echo = echo
        self.farend = farend
        self.snr = snr
        self.is_dt = is_dt
        self.vqe = vqe
        self.echo_vqe = echo_vqe
        self.meta = meta
