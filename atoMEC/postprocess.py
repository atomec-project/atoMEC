"""
Module containing routines to post-process data from AA calculations.

Functions
---------
* :class:`IPR` : Holds the inverse participation ratio, derived quantities and \
                 the routines required to compute them.
"""

# standard libraries
import numpy as np

# internal libraries
from . import mathtools


def calc_IPR_mat(eigfuncs, xgrid):
    r"""
    Calculate the inverse participation ratio for all eigenfunctions (see notes).

    Parameters
    ----------
    eigfuncs : ndarray
        transformed radial KS orbitals :math:`P_{nl}(x)=\exp(x/2)X_{nl}(x)`
    xgrid : ndarray
        the logarithmic grid

    Returns
    -------
    IPR_mat : ndarray
        the matrix of all IPR values

    Notes
    -----
    The inverse participation ratio for a state `i` is usually defined as

    .. math:: \mathrm{IPR}_i = \int \mathrm{d}{\mathbf{r}} |\Psi_i(\mathbf{r})|^4

    It is typically used as a localization measure.

    WARNING: the current definition in this version is not mathematically correct.
    It does not include the proper contribution from the spherical harmonics
    :math:`|Y_l^m(\theta,\phi)|^4`. This is omitted because it makes little difference
    to the flavour of the results but complicates things. Currently, this function
    does not correctly produce the expected limits (even if the spherical harmonic
    contribution is correctly accounted for). Use at your own peril...
    """
    # get the dimensions for the IPR matrix
    spindims = np.shape(eigfuncs)[0]
    lmax = np.shape(eigfuncs)[1]
    nmax = np.shape(eigfuncs)[2]

    IPR_mat = np.zeros((spindims, lmax, nmax))

    # compute the IPR matrix
    # FIXME: add spherical harmonic term
    for i in range(spindims):
        for l in range(lmax):
            for n in range(nmax):
                # compute |X_nl(x)|^4 = |P_nl(x)|^4 * exp(-2x)
                Psi4 = eigfuncs[i, l, n, :] ** 4.0 * np.exp(-2 * xgrid)
                # integrate over sphere
                IPR_mat[i, l, n] = mathtools.int_sphere(Psi4, xgrid)

    return IPR_mat
