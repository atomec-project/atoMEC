"""
Low-level module containing various mathematical functions
"""

# standard libraries
from math import sqrt, pi, exp

# external libraries
import numpy as np
import scipy.integrate

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


def fermi_dirac(eps, mu, beta, n=0):
    """
    Computes the Fermi-Dirac function
    f_fd = (eps)^(n/2) / (1 + exp(beta(eps-mu))

    Inputs:
    - mu   (float)  : chemical potential
    - beta (float)  : inverse temperature
    - eps  (float)  : energy
    - n    (int)    : power to which energy is raised (opt)
    Returns:
    - f_fd (float)  : fermi_dirac occupation
    """

    # dfn the exponential function
    fn_exp = exp(beta * (eps - mu))

    # fermi_dirac dist
    f_fd = (eps) ** (n / 2.0) / (1 + fn_exp)

    return f_fd


def fd_int_complete(mu, beta, n):
    """
    Computes complete Fermi-Dirac integrals of the form
    I_(n/2)(mu, beta) = \int_0^inf \de e^(n/2) f_fd(mu, e, beta)

    Inputs:
    - mu (float)    : chemical potential
    - beta (float)  : inverse temperature
    - n (int)       : order of integral I_(n/2)
    Returns
    - I_n (float)   : fd integral
    """

    return integrate.quad(fermi_dirac, 0, np.inf, args=(mu, beta, n))
