"""
The localization module handles routines designed to measure electron localization.

Classes
-------
* :class:`ELFTools` : Holds functions to compute the electron localization function\
                      (ELF) and related quantities e.g. number of electrons per shell.

Functions
---------
* :func:`calc_IPR_mat`: Computes the inverse participation ratio (IPR) for the orbitals.
"""

# standard packages
from math import pi

# external packages
from scipy.signal import argrelmin
import numpy as np

# internal modules
from atoMEC import staticKS, mathtools, check_inputs, arrays


class ELFTools:
    r"""
    ELF is the electron localization function, a measure of electron localization.

    It can be used as a tool for qualitative insight as well as an additional method to
    compute the mean ionization state. This class contains routines to compute the ELF
    and other related useful properties.

    Parameters
    ----------
    Atom : atoMEC.Atom
        the Atom object
    model : models.ISModel
        the model object
    orbitals : staticKS.Orbitals
        the orbitals object
    density : staticKS.Density
        the density object
    method : str, optional
        the method used for the ELF, "orbitals" or "density"
    """

    def __init__(self, Atom, model, orbitals, density, method="orbitals"):
        self._orbs = orbitals
        self._density = density
        self._eigfuncs = self._orbs.eigfuncs
        self._xgrid = self._orbs._xgrid
        self.method = method

        if orbitals.grid_type != "log":
            raise check_inputs.InputError.ELF_error(
                "Sqrt grid is not yet supported for ELF calculations."
                "Please switch to log grid."
            )

        # density and occupation numbers have to include bound and free contributions
        self._totdensity = self._density.total
        self._occnums_w = self._orbs.occnums_w

        # extrapolate the spin number and number of grid points
        spindims = np.shape(self._eigfuncs)[1]
        ngrid = np.shape(self._eigfuncs)[4]

        self._ELF = arrays.zeros((spindims, ngrid))
        self._epdc = arrays.zeros((spindims, ngrid))
        self._N_shell = None

    @property
    def ELF(self):
        r"""ndarray: the electron localization function for a given system."""
        if np.all(self._ELF == 0.0):
            self._ELF = self.calc_ELF(
                self._eigfuncs,
                self._occnums_w,
                self._xgrid,
                self._totdensity,
                self.method,
            )
        return self._ELF

    @property
    def epdc(self):
        r"""ndarray: the electron pair density curvature."""
        if self.method == "orbitals":
            self._epdc = self.calc_epdc(
                self._eigfuncs, self._occnums_w, self._xgrid, self._totdensity
            )
        elif self.method == "density":
            self._epdc = self.calc_epdc_dens(self._xgrid, self._totdensity)
        return self._epdc

    @property
    def N_shell(self):
        r"""ndarray: the number of electrons in each 'shell'."""
        if self._N_shell is None:
            xargs_min = self.get_minima(self.ELF, self._xgrid)
            self._N_shell = self.calc_N_shell(
                xargs_min, self._xgrid, self._totdensity, self.ELF
            )
        return self._N_shell

    @staticmethod
    def calc_ELF(eigfuncs, occnums, xgrid, density, method="orbitals"):
        r"""
        Compute the ELF (see notes).

        Parameters
        ----------
        eigfuncs : ndarray
            the (modified) KS eigenfunctions :math:`P(x)=\exp(x/2)X(x)`
        occnums : ndarray
            the orbital occupations
        xgrid : ndarray
            the logarithmic grid
        density : ndarray
            the electron density

        Returns
        -------
        ELF : ndarray
            the electron localization function

        Notes
        -----
        The ELF is defined as

        .. math:: \mathrm{ELF}^\sigma(r) = \frac{1}{1 + (\chi^\sigma(r))^2}\,, \\
            \chi^\sigma(r) = D^\sigma(r) / D^\sigma_0(r),

        where :math:`D^\sigma(r)` and :math:`D^\sigma_0(r)` are the electron
        pair density functions for the system of interest and the uniform
        electron gas (UEG) respectively.
        """
        # compute the UEG electron pair density curvature
        D_0 = (0.3) * (3 * pi**2) ** (2.0 / 3.0) * (density) ** (5.0 / 3.0)

        # compute the main electron pair density curvature
        if method == "orbitals":
            D = ELFTools.calc_epdc(eigfuncs, occnums, xgrid, density)
        elif method == "density":
            D = ELFTools.calc_epdc_dens(xgrid, density)

        # compute the ratio chi
        chi = D / D_0

        # compute the ELF
        ELF = 1.0 / (1.0 + chi**2)

        return ELF

    @staticmethod
    def calc_epdc(eigfuncs, occnums, xgrid, density):
        r"""
        Calculate the electron pair density curvature (see notes).

        Parameters
        ----------
        eigfuncs : ndarray
            the (modified) KS eigenfunctions :math:`P(x)=\exp(x/2)X(x)`
        occnums : ndarray
            the orbital occupations
        xgrid : ndarray
            the logarithmic grid
        density : ndarray
            the electron density

        Returns
        -------
        epdc : ndarray
            the electron pair density curvature

        Notes
        -----
        The epdc is defined as

        .. math::
            D^\sigma(r) = \tau^\sigma(r) - \frac{1}{8}\frac{|\nabla\rho(r)|^2}{\rho(r)},

        where :math:`\tau^\sigma(r)` is the local kinetic energy density.
        """
        # first compute the contribution from the KED
        tau = staticKS.Energy.calc_E_kin_dens(
            eigfuncs, occnums, xgrid, "log", method="B"
        )

        # compute the density gradient using chain rule
        grad_dens = np.exp(-xgrid) * np.gradient(density, xgrid, axis=-1)

        # compute epdc
        epdc = tau - 0.125 * (grad_dens) ** 2 / density

        return epdc

    @staticmethod
    def calc_epdc_dens(xgrid, density):
        r"""
        Calculate the electron pair density curvature (see notes).

        Parameters
        ----------
        xgrid : ndarray
            the logarithmic grid
        density : ndarray
            the electron density

        Returns
        -------
        epdc : ndarray
            the electron pair density curvature

        Notes
        -----
        The epdc is defined as

        .. math::
            D^\sigma(r) = D_0^\sigma(r) - \frac{1}{9}\frac{|\nabla\rho(r)|^2}{\rho(r)}\
            + \frac{1}{6} \nabla^2 \rho(r),

        where :math:`D_0^\sigma(r)=(3/10)(3\pi^2)^{2/3} \rho^{5/3}(r)`
        is the epdc for the UEG.
        """
        # compute the density gradient using chain rule
        grad_dens = np.exp(-xgrid) * np.gradient(density, xgrid, axis=-1)

        # compute the laplacian of the density
        lap_dens = np.exp(-3.0 * xgrid) * np.gradient(
            np.exp(2 * xgrid) * grad_dens, xgrid, axis=-1
        )

        # compute the UEG electron pair density curvature
        D_0 = (0.3) * (3 * pi**2) ** (2.0 / 3.0) * (density) ** (5.0 / 3.0)

        # compute epdc
        epdc = D_0 - (grad_dens) ** 2.0 / (9.0 * density) + lap_dens / 6.0

        return epdc

    @staticmethod
    def get_minima(ELF, xgrid):
        """
        Get the locations of the minima of the ELF.

        Parameters
        ----------
        ELF : ndarray
            the electron localization function
        xgrid : ndarray
            the logarithmic grid

        Returns
        -------
        xargs_min : list of ints
            list of locations of the minima and first and last points
        """
        # tolerance for determining whether something is a "true minimum"
        # FIXME: might be a better way of doing this
        tol = 1e-3
        spindims = np.shape(ELF)[0]

        # initialize list for the min args
        if spindims == 1:
            xargs_min = [[0]]
        elif spindims == 2:
            xargs_min = [[0], [0]]

        # search for the minimum arguments
        for i in range(spindims):
            xargs_0 = argrelmin(ELF[i])[0]
            # xargs_1 = argrelmax(ELF[i])[0]
            # xargs_0 = np.sort(np.concatenate((xargs_0, xargs_1)))
            for xarg in xargs_0:
                if ELF[i, xarg] < 1 - tol and ELF[i, xarg] > tol:
                    xargs_min[i].append(xarg)

        if spindims == 1:
            xargs_min[0].append(len(xgrid) - 1)
        elif spindims == 2:
            xargs_min[0].append(len(xgrid) - 1)
            xargs_min[1].append(len(xgrid) - 1)

        return xargs_min

    @staticmethod
    def calc_N_shell(xargs_min, xgrid, density, ELF):
        """
        Compute the number of electrons in each shell from the ELF.

        The number of electrons per shell is determined by integrating
        the electron density between successive minima of the ELF.

        Parameters
        ----------
        xargs_min : list
            list of minima, as well as the first and last points
        xgrid : ndarray
            the logarithmic grid
        density : ndarray
            the electron density
        ELF : ndarray
            the electron localization function

        Returns
        -------
        N_shell : list of floats
            list of the number of electrons per shell
        """
        spindims = len(xargs_min)
        N_shell = xargs_min  # initialize N_shell with the right shape

        for i in range(spindims):
            for j in range(len(xargs_min[i]) - 1):
                # determine the part of the xgrid lying between two minima
                xgrid_partition = xgrid[xargs_min[i][j] : xargs_min[i][j + 1] + 1]

                # determine the part of the density lying between two minima
                density_partition = density[i][
                    xargs_min[i][j] : xargs_min[i][j + 1] + 1
                ]

                # this will be going...
                # ELF_partition = ELF[i][xargs_min[i][j] : xargs_min[i][j + 1]]

                # integrate over the density between two minima
                N_shell[i][j] = mathtools.int_sphere(
                    density_partition, xgrid_partition, "log"
                )

        # delete the last element in N_shell (superfluous)
        for i in range(spindims):
            N_shell[i].pop()

        return N_shell


def MIS_count(model, orbitals, core_orbs):
    r"""
    Calculate the MIS using the "counting" method.

    Parameters
    ----------
    model : models.ISModel
        the ISModel object
    orbitals : staticKS.orbitals
        the KS orbitals object
    core_orbs : list of tuples
        the core orbitals

    Returns
    -------
    MIS : np.ndarray
        the mean ionization state
    """
    # start by retrieving the total electron number
    MIS = model.nele.astype(np.float64)

    # loop over the core orbitals and subtract their occupation numbers
    for core_orb in core_orbs:
        l, n = core_orb
        MIS -= np.sum(orbitals.occnums_w[:, :, l, n], axis=0)

    return MIS


def calc_IPR_mat(eigfuncs, xgrid, grid_type=None):
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

    .. warning::
       The current definition in this version is not mathematically correct.
       It does not include the proper contribution from the spherical harmonics
       :math:`|Y_l^m(\theta,\phi)|^4`. This is omitted as it makes little difference
       to the flavour of the results but complicates things. Currently, this function
       does not correctly produce the expected limits (even if the spherical harmonic
       contribution is correctly accounted for). Use at your own peril...

    """
    if grid_type is None:
        grid_type = "log"
        print(
            "No grid type provided, assuming logarathmic grid."
            "Please check if correct"
        )
    # get the dimensions for the IPR matrix
    nkpts = np.shape(eigfuncs)[0]
    spindims = np.shape(eigfuncs)[1]
    lmax = np.shape(eigfuncs)[2]
    nmax = np.shape(eigfuncs)[3]

    IPR_mat = arrays.zeros((nkpts, spindims, lmax, nmax))

    # compute the IPR matrix
    # FIXME: add spherical harmonic term
    for k in range(nkpts):
        for sp in range(spindims):
            for l in range(lmax):
                for n in range(nmax):
                    # compute |X_nl(x)|^4 = |P_nl(x)|^4 * exp(-2x)
                    if grid_type == "log":
                        Psi4 = eigfuncs[k, sp, l, n, :] ** 4.0 * np.exp(-2 * xgrid)
                    else:
                        Psi4 = eigfuncs[k, sp, l, n, :] ** 4.0
                    # integrate over sphere
                    IPR_mat[k, sp, l, n] = mathtools.int_sphere(Psi4, xgrid, grid_type)

    return IPR_mat
