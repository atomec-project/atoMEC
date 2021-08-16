r"""
Low-level module containing miscalleneous mathematical functions.

Functions
---------
* :func:`normalize_orbs`: normalize KS orbitals within defined sphere
* :func:`int_sphere`: integral :math:`4\pi \int \mathrm{d}r r^2 f(r)`
* :func:`laplace`: compute the second-order derivative :math:`d^2 y(x) / dx^2`
* :func:`fermi_dirac`: compute the Fermi-Dirac occupation function for given order `n`
* :func:`ideal_entropy`: Define the integrand to be used in :func:`ideal_entropy_int`.
* :func:`ideal_entropy_int`: Compute the entropy for the ideal electron gas (no prefac).
* :func:`fd_int_complete`: compute the complete Fermi-Dirac integral for given order `n`
* :func:`chem_pot`: compute the chemical potential by enforcing charge neutrality
* :func:`f_root_id`: make root input fn for chem_pot with ideal apprx for free electrons
"""

# standard libraries
from math import sqrt, pi
import warnings

# external libraries
import numpy as np
from scipy import optimize, integrate

# internal libraries
from . import config


def normalize_orbs(eigfuncs_x, xgrid):
    r"""
    Normalize the KS orbitals within the chosen sphere.

    Parameters
    ----------
    eigfuncs : ndarray
        The radial KS eigenfunctions :math:`X_{nl}^{\sigma}(x)`
    xgrid : ndarray
        The logarithmic grid over which normalization is performed

    Returns
    -------
    eigfuncs_x_norm : ndarray
        The radial KS eigenfunctions normalized over the chosen sphere
    """
    # initialize the normalized eigenfunctions
    eigfuncs_x_norm = eigfuncs_x

    # loop over the eigenfunctions
    for n in range(np.shape(eigfuncs_x)[0]):
        # compute the mod squared eigenvalues
        eigfuncs_sq = eigfuncs_x[n].real ** 2 + eigfuncs_x[n].imag ** 2
        # compute the intergal ampsq=4*pi*\int_dr r^2 |R(r)|^2
        exp_x = np.exp(-xgrid)
        ampsq = int_sphere(exp_x * eigfuncs_sq, xgrid)
        # normalize eigenfunctions
        eigfuncs_x_norm[n] = eigfuncs_x[n] / sqrt(ampsq)

    return eigfuncs_x_norm


def int_sphere(fx, xgrid):
    r"""
    Compute integral over sphere defined by input grid.

    The integral is performed on the logarithmic grid (see notes).

    Parameters
    ----------
    fx : array_like
        The function (array) to be integrated
    xgrid : ndarray
        The logarithmic radial grid

    Returns
    -------
    I_sph : float
        The value of the integrand

    Notes
    -----
    The integral formula is given by

    .. math:: I = 4 \pi \int \mathrm{d}x\ e^{3x} f(x)
    """
    func_int = 4.0 * pi * np.exp(3.0 * xgrid) * fx
    I_sph = np.trapz(func_int, xgrid)

    return I_sph


def laplace(y, x, axis=-1):
    r"""
    Compute the second-order derivative :math:`d^2 y(x) / dx^2`.

    Derivative can be computed over any given axis.

    Parameters
    ----------
    y : ndarray
        array y(x) on which laplacian is computed
    x : ndarray
        x array
    axis: int, optional
        axis over which derivatives are taken
        default : -1

    Returns
    -------
    grad2_y : ndarray
        the laplacian of y
    """
    # first compute the first-order gradient
    grad1_y = np.gradient(y, x, edge_order=2, axis=axis)

    # now compute the second-order gradient
    grad2_y = np.gradient(grad1_y, x, edge_order=2, axis=axis)

    return grad2_y


def fermi_dirac(eps, mu, beta, n=0):
    r"""
    Compute the Fermi-Dirac function, see notes for functional form.

    Parameters
    ----------
    mu : array_like
        the chemical potential
    beta : float
        the inverse potential
    eps : array_like
        the energies
    n : int
        energy is raised to power n/2 in the numerator (see notes)

    Returns
    -------
    f_fd : array_like
        the fermi dirac occupation(s)

    Notes
    -----
    The FD function is defined as:

    .. math:: f^{(n)}_{fd}(\epsilon, \mu, \beta) = \frac{\epsilon^{n/2}}{1+\exp(1+
        \beta(\epsilon - \mu))}
    """
    # dfn the exponential function
    # ignore warnings here
    with np.errstate(over="ignore"):
        fn_exp = np.minimum(np.exp(beta * (eps - mu)), 1e12)

    # fermi_dirac dist
    f_fd = (eps) ** (n / 2.0) / (1 + fn_exp)

    return f_fd


def ideal_entropy(eps, mu, beta, n=0):
    r"""
    Define the integrand to be used in :func:`ideal_entropy_int` (see notes).

    Parameters
    ----------
    eps : array_like
        the energies
    mu : array_like
        the chemical potential
    beta : float
        the inverse potential
    n : int
        energy is raised to power n/2 in the numerator (see notes)

    Returns
    -------
    f_ent : array_like
        the entropy integrand function

    Notes
    -----
    The ideal entropy integrand is defined as

    .. math::
        f_n(\epsilon,\mu,\beta) = \epsilon^{n/2} (f_\mathrm{fd}\log{f_\mathrm{fd}}
        + (1-f_\mathrm{fd}) \log(1-f_\mathrm{fd}) ),

    where :math:`f_\mathrm{fd}=f_\mathrm{fd}(\epsilon,\mu,\beta)` is the Fermi-Dirac
    distribution.
    """
    # dfn the exponential function
    # ignore warnings here
    with np.errstate(over="ignore"):
        fn_exp = np.minimum(np.exp(beta * (eps - mu)), 1e12)

    # the 'raw' Fermi-Dirac distribution
    f_fd_raw = 1 / (1 + fn_exp)

    # define high and low tolerances for the log function (to avoid nans)
    tol_l = 1e-8
    tol_h = 1.0 - 1e-8

    # first replace the zero values
    f_fd_mod = np.where(f_fd_raw > tol_l, f_fd_raw, tol_l)
    # now replace the one values
    f_fd = np.where(f_fd_mod < tol_h, f_fd_mod, tol_h)

    # fermi_dirac dist
    f_ent = (eps) ** (n / 2.0) * (f_fd * np.log(f_fd) + (1 - f_fd) * np.log(1 - f_fd))

    return f_ent


def fd_int_complete(mu, beta, n):
    r"""
    Compute complete Fermi-Dirac integral for given order (see notes for function form).

    Parameters
    ----------
    mu : float
        chemical potential
    beta: float
        inverse temperature
    n : int
        order of Fermi-Dirac integral (see notes)

    Returns
    -------
    I_n : float
        the complete fermi-dirac integral

    Notes
    -----
    Complete Fermi-Dirac integrals are of the form

    .. math::

        I_{n}(\mu,\beta)=\int_0^\infty\mathrm{d}\epsilon\ \epsilon^{n/2}f_{fd}
        (\mu,\epsilon,\beta)

    where n is the order of the integral
    """
    # use scipy quad integration routine
    limup = np.inf

    # ignore integration warnings (omnipresent because of inf upper limit)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        I_n, err = integrate.quad(fermi_dirac, 0, limup, args=(mu, beta, n))

    return I_n


def ideal_entropy_int(mu, beta, n):
    r"""
    Compute the entropy for the ideal electron gas (without prefactor) - see notes.

    Parameters
    ----------
    mu : float
        chemical potential
    beta: float
        inverse temperature
    n : int
        order of Fermi-Dirac integral (see notes)

    Returns
    -------
    I_n : float
        the complete fermi-dirac integral

    Notes
    -----
    The entropy of an ideal electron gas is defined as

    .. math::
        I_n(\mu,\beta) = \int_0^\infty \mathrm{d}\epsilon\ \epsilon^{n/2}
        (f_\mathrm{fd}\log{f_\mathrm{fd}} + (1-f_\mathrm{fd}) \log(1-f_\mathrm{fd}) ),

    where :math:`f_\mathrm{fd}=f_\mathrm{fd}(\epsilon,\mu,\beta)` is the Fermi-Dirac
    distribution.
    """
    # use scipy quad integration routine
    limup = np.inf

    # ignore integration warnings (omnipresent because of inf upper limit)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        I_n, err = integrate.quad(ideal_entropy, 0, limup, args=(mu, beta, n))

    return I_n


def chem_pot(orbs):
    r"""
    Determine the chemical potential by enforcing charge neutrality (see notes).

    Uses scipy.optimize.root_scalar with brentq implementation.

    Parameters
    ----------
    orbs : staticKS.Orbitals
        the orbitals object

    Returns
    -------
    mu : array_like
        chemical potential (spin-dependent)

    Notes
    -----
    Finds the roots of equation

    .. math:: \sum_{nl} (2l+1) f_{fd}(\epsilon_{nl},\beta,\mu) +
        N_{ub}(\beta,\mu) - N_e = 0.

    The number of unbound electrons :math:`N_{ub}` depends on the implementation choice.
    """
    mu = config.mu
    mu0 = mu  # set initial guess to existing value of chem pot

    # so far only the ideal treatment for unbound electrons is implemented
    if config.unbound == "ideal":
        for i in range(config.spindims):
            if config.nele[i] != 0:
                soln = optimize.root_scalar(
                    f_root_id,
                    x0=mu0[i],
                    args=(orbs.eigvals[i], orbs.lbound[i], config.nele[i]),
                    method="brentq",
                    bracket=[-100, 100],
                    options={"maxiter": 100},
                )
                mu[i] = soln.root
            # in case there are no electrons in one spin channel
            else:
                mu[i] = -np.inf

    return mu


def f_root_id(mu, eigvals, lbound, nele):
    r"""
    Functional input for the chemical potential root finding function (ideal approx).

    See notes for function returned, the ideal approximation is used for free electrons.

    Parameters
    ----------
    mu : array_like
        chemical potential
    eigvals : ndarray
        the energy eigenvalues
    lbound : ndarray
        the lbound matrix :math:`(2l+1)\Theta(\epsilon_{nl}^\sigma)`
    nele : union(int, float)
        the number of electrons for given spin

    Returns
    -------
    f_root : float
       the difference of the predicted electron number with given mu
       and the actual electron number

    Notes
    -----
    The returned function is

    .. math:: f = \sum_{nl} (2l+1) f_{fd}(\epsilon_{nl},\beta,\mu) +
        N_{ub}(\beta,\mu) - N_e
    """
    # caluclate the contribution from the bound electrons
    occnums = lbound * fermi_dirac(eigvals, mu, config.beta)
    contrib_bound = occnums.sum()

    # now compute the contribution from the unbound electrons
    # this function uses the ideal approximation

    prefac = (2.0 / config.spindims) * config.sph_vol / (sqrt(2) * pi ** 2)
    contrib_unbound = prefac * fd_int_complete(mu, config.beta, 1.0)

    # return the function whose roots are to be found
    f_root = contrib_bound + contrib_unbound - nele

    return f_root
