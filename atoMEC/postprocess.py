"""
Module containing routines to post-process data from AA calculations.

Classes
-------
* :class:`IPR` : Holds the inverse participation ratio, derived quantities and \
                 the routines required to compute them.
"""

from math import pi

import numpy as np


class IPR:
    r"""
    IPR is the inverse participation ratio, a measure of orbital localization.

    It can be used as an alternative method to compute the mean ionization state,
    as well as giving information about the localization characateristics of all
    orbitals.
    """

    def __init__(self, orbitals):

        self.orbitals = orbitals
        self._IPR_mat = np.zeros_like(orbitals.eigfuncs)
        self._MIS = None

    @property
    def IPR_mat(self):
        r"""ndarray: The inverse participation ratio matrix (of all orbitals)."""
        if np.all(self._IPR_mat == 0.0):
            self._IPR_mat = self.calc_IPR(self.orbitals.eigfuncs, self.orbitals._xgrid)
        return self._IPR_mat

    @property
    def MIS(self):
        r"""ndarray: The mean ionization state computed from the IPR."""
        if self._MIS is None:
            self._MIS = self.calc_MIS(self.orbitals, self.IPR_mat)
        return self._MIS

    @staticmethod
    def calc_IPR(eigfuncs, xgrid):
        r"""
        Calculate the inverse participation ratio for all eigenfunctions.

        Parameters
        ----------
        eigfuncs : ndarray
            transformed radial KS orbitals :math:`P(x)=\exp(x/2)R(x)`
        xgrid : ndarray
            the logarithmic grid
        """
        # get the dimensions for the IPR matrix
        spindims = np.shape(eigfuncs)[0]
        lmax = np.shape(eigfuncs)[1]
        nmax = np.shape(eigfuncs)[2]

        IPR_mat = np.zeros((spindims, lmax, nmax))

        # compute the fully delocalized limit (eigenfunctions constant)
        V = 4.0 / 3.0 * pi * np.exp(xgrid[-1]) ** 3.0
        psi_constant = V ** -2
        Psi4_c_int = np.trapz(psi_constant * np.exp(3 * xgrid), xgrid)

        # compute the IPR matrix
        for i in range(spindims):
            for l in range(lmax):
                for n in range(nmax):
                    Psi4 = eigfuncs[i, l, n, :] ** 4.0
                    Psi4_int = np.trapz(Psi4 * np.exp(3 * xgrid), xgrid)

                    # scale it by the fully delocalized limit
                    IPR_mat[i, l, n] = 1 - Psi4_c_int / Psi4_int

        return IPR_mat
