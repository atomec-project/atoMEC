"""
Low-level module containing various mathematical functions
"""

# standard libraries
from math import sqrt, pi, exp

# external libraries
import numpy as np
from scipy import optimize, integrate

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
    - mu   (float)     : chemical potential
    - beta (float)     : inverse temperature
    - eps  (np array)  : energy
    - n    (int)       : power to which energy is raised (opt)
    Returns:
    - f_fd (float)     : fermi_dirac occupation
    """

    # dfn the exponential function
    fn_exp = np.minimum(np.exp(beta * (eps - mu)), 1e12)

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

    # use scipy quad integration routine
    limup = mu + 10
    I_n, err = integrate.quad(fermi_dirac, 0, limup, args=(mu, beta, n))

    return I_n


def chem_pot(orbs):
    """
    Determines the chemical potential by enforcing charge neutrality
    Finds the roots of the eqn:
    \sum_{nl} (2l+1) f_fd(e_nl,beta,mu) + N_ub(beta,mu) - N_e = 0

    Inputs:
    - orbs (object)           : the orbitals object
    Returns:
    - mu (list of floats)         : chem pot for each spin
    """

    mu = config.mu
    mu0 = mu  # set initial guess to existing value of chem pot

    # so far only the ideal treatment for unbound electrons is implemented
    if config.unbound == "ideal":
        for i in range(config.spindims):
            try:
                soln = optimize.root(
                    f_root_id,
                    mu0[i],
                    args=(orbs.eigvals[i], orbs.lbound[i], config.nele[i]),
                    method="broyden1",
                )
                mu[i] = soln.x
            # this handles the case when there are no electrons in one spin channel
            except TypeError:
                mu[i] = -np.inf

    return mu


def f_root_id(mu, eigvals, lbound, nele):

    # caluclate the contribution from the bound electrons
    occnums = lbound * fermi_dirac(eigvals, mu, config.beta)
    contrib_bound = occnums.sum()

    # now compute the contribution from the unbound electrons
    # this function uses the ideal approximation

    prefac = nele * config.sph_vol / (sqrt(2) * pi ** 2)
    contrib_unbound = prefac * fd_int_complete(mu, config.beta, 0.5)

    # return the function whose roots are to be found
    return contrib_bound + contrib_unbound - nele