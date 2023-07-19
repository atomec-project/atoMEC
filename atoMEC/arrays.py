import numpy as np
from . import config


def array(*args, **kwargs):
    if config.float_precision == 32:
        kwargs.setdefault("dtype", np.float32)
    elif config.float_precision == 64:
        kwargs.setdefault("dtype", np.float64)
    return np.array(*args, **kwargs)


def zeros(*args, **kwargs):
    if config.float_precision == 32:
        kwargs.setdefault("dtype", np.float32)
    elif config.float_precision == 64:
        kwargs.setdefault("dtype", np.float64)
    return np.zeros(*args, **kwargs)


def ones(*args, **kwargs):
    if config.float_precision == 32:
        kwargs.setdefault("dtype", np.float32)
    elif config.float_precision == 64:
        kwargs.setdefault("dtype", np.float64)
    return np.ones(*args, **kwargs)


def eye(*args, **kwargs):
    if config.float_precision == 32:
        kwargs.setdefault("dtype", np.float32)
    elif config.float_precision == 64:
        kwargs.setdefault("dtype", np.float64)
    return np.eye(*args, **kwargs)
