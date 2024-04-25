"""Abstract dataset class.

This is a direct copy of PyTorch's dataset class:
https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset
with some omissions and additions.
"""
import os
import bisect

from torch.utils.data import Dataset

from ..io.batch import lsfiles
from ..io.audio import audioread
from .datatype import Audio




class AudioDataset(Dataset):
    """A dataset that gets all audio files from a directory."""

    @staticmethod
    def isaudio(path):
        "tries to find wav, flac or sph audio"
        return path.endswith((".wav", ".flac", ".sph"))

    @staticmethod
    def read(path, sr):
        """Read audio and put in an Audio object."""
        return Audio(*audioread(path, sr=sr))

    def __init__(self, root, sr=None, filt=None, read=None, transform=None):
        """Instantiate an audio dataset.

        Parameters
        ----------
        root: str
            Root directory of a dataset.
        sr: int
            Sampling rate in Hz.
        read: callable, optional
            Function to be called on each file path to get the signal.
            Default to `audioread`.
        filt: callable, optional
            Filter function to be applied on each file path.
            Default to `isaudio`, which accepts every file ended in .wav, .sph,
            or .flac.
        transform: callable, optional
            Transform to be applied on each sample after read in.
            Default to None.

        See Also
        --------
        datatype.Audio

        """
        super(AudioDataset).__init__()
        self.root = root
        self.sr = sr
        self._filepaths = lsfiles(
            root, filt=filt if filt else self.isaudio, relpath=True
        )
        self._read = read if read else lambda f: AudioDataset.read(f, sr)
        self.transform = transform

    @property
    def filepaths(self):
        """Return all valid file paths in a list."""
        return self._filepaths

    def __len__(self):
        """Return number of valid audio files."""
        return len(self._filepaths)

    def __getitem__(self, idx):
        """Get i-th valid item after reading in and transform."""
        sample = self._read(os.path.join(self.root, self._filepaths[idx]))
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __str__(self):
        """Print out a summary of instantiated dataset."""
        report = """
            +++++ Summary for [{}] +++++
            Total [{}] valid files to be processed.
        """.format(
            self.__class__.__name__, len(self._filepaths)
        )

        return report