"""Utility functions."""
import configparser as CP


def cfgread(path, section):
    """Read a configuration section from path."""
    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(path)
    return cfg._sections[section]
