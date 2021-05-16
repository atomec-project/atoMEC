"""
Contains unit conversions
"""

import config


def cm_to_bohr(x):
    y = x * 1.889716165e8
    return y


def bohr_to_cm(x):
    y = x / 1.889716165e8
    return y
