"""Utility functions for batch processing."""

import os

def genfiles(root, filt=None, relpath=False):
    """Recursively find all files within a directory and return a generator.

    Parameters
    ----------
    root: str
        Directory to look into.
    filt: callable, optional
        Filter function applied to each file path (to keep or discard).
        Default to None, which accepts all files.
    relpath: bool, optional
        Return relative path to `root` instead of full path.
        Default to False.

    Returns
    -------
    type: generator
        A valid file path is returned on each yield.

    """
    root = os.path.abspath(root)
    if os.path.isfile(root):  # single file mode
        if (filt is None) or filt(root):
            yield root
    else:
        for rr, dirs, files in os.walk(root):
            for fname in files:
                fullpath = os.path.join(rr, fname)
                if (filt is None) or filt(fullpath):
                    if relpath:
                        yield os.path.relpath(fullpath, root)
                    else:
                        yield fullpath


def lsfiles(root, filt=None, relpath=False):
    """Recursively find all files within a directory and return a list.

    See Also
    --------
    genfiles

    """
    return list(genfiles(root, filt=filt, relpath=relpath))
