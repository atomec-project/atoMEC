"""
Low-level module containing various mathematical functions
"""

# standard libraries
from math import sqrt, pi

# external libraries
import numpy as np

# internal libraries
import config


def normalize_orbs(eigfuncs_x):
    """
    Normalizes the KS orbitals within the Voronoi sphere

    Inputs:
    - eigfuncs_x (np array)      : eigenfunctions on the log grid
    Returns:
    - eigfuncs_x_norm (np array) : normalized eigenfunctions on the log grid
    """

    # initialize the normalized eigenfunctions
    eigfuncs_x_norm = eigfuncs_x

    # loop over the eigenfunctions
    for n in range(np.shape(eigfuncs_x)[0]):
        # compute the mod squared eigenvalues
        eigfuncs_sq = eigfuncs_x[n].real ** 2 + eigfuncs_x[n].imag ** 2
        # compute the intergal ampsq=4*pi*\int_dr r^2 |R(r)|^2
        exp_x = np.exp(config.xgrid)
        ampsq = int_sphere(exp_x * eigfuncs_sq)
        # normalize eigenfunctions
        eigfuncs_x_norm[n] = eigfuncs_x[n] / sqrt(ampsq)

    return eigfuncs_x_norm


def int_sphere(fx):
    """
    Computes spherical integrals 4*pi*\int_dr r^2 f(r)
    Integral is done on the logarithmic grid:
    I = 4*pi*int_dx exp(3x) f(x)

    Inputs:
    fx (numpy array)   : function to be integrated
    Outputs:
    I_sph (float)      : value of spherical integral
    """

    func_int = 4 * pi * np.exp(3 * config.xgrid) * fx
    I_sph = np.trapz(func_int, config.xgrid)

    return I_sph
