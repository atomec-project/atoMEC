"""
The staticKS module handles time-indepenent KS quantities such as the KS orbitals.

This module is a 'workhouse' for much of the average-atom functionality.
Computation of static KS quantities is designed to be quite general.
The main assumption used everywhere is that we have a logarithmically spaced grid.

Classes
-------
* :class:`Orbitals` : Holds the KS orbitals (transformed to the log grid) and related \
                      quantities such as eigenvalues and occupation numbers
* :class:`Density` : Holds the KS density (bound/unbound components) and routines \
                     to compute it.
* :class:`Potential` : Holds the KS potential (split into individual components) and \
                       the routines to compute them.
* :class:`Energy` : Holds the free energy and all internal components (KS quantities \
                    and entropy) and the routines required to compute them.
* :class:`EnergyAlt` : Holds the free energy and all internal components (KS \
                       quantities and entropy) and the routines required to compute \
                       them. N.B. the :class:`EnergyAlt` class constructs the energy \
                       functional in an alternative manner to the main :class:`Energy` \
                       class.

Functions
---------
* :func:`log_grid` : Sets up the logarithmic (and associated real-space) grid on which \
                     all computations rely.
"""

# standard packages

# external packages
import numpy as np
from math import sqrt, pi, exp

# internal modules
from . import config
from . import numerov
from . import mathtools
from . import xc

# from . import writeoutput


# the logarithmic grid
def log_grid(x_r):
    """
    Set up the logarithmic (and real) grid, defined up to x_r.

    The leftmost grid point is given by the `config.grid_params["x0"]` parameter.

    Parameters
    ----------
    x_r : float
        The RHS grid point (in logarithmic space)

    Returns
    -------
    xgrid, rgrid : tuple of ndarrays
        The grids in logarithmic (x) and real (r) space
    """
    # grid in logarithmic co-ordinates
    xgrid = np.linspace(config.grid_params["x0"], x_r, config.grid_params["ngrid"])
    # grid in real space co-ordinates
    rgrid = np.exp(xgrid)

    config.xgrid = xgrid
    config.rgrid = rgrid

    return xgrid, rgrid


class Orbitals:
    """Class holding the KS orbitals, associated quantities and relevant routines."""

    def __init__(self, xgrid):

        self._xgrid = xgrid
        self._eigfuncs = np.zeros(
            (
                config.band_params["nkpts"],
                config.spindims,
                config.lmax,
                config.nmax,
                config.grid_params["ngrid"],
            )
        )
        self._eigvals = np.zeros(
            (config.band_params["nkpts"], config.spindims, config.lmax, config.nmax)
        )
        self._occnums = np.zeros_like(self._eigvals)
        self._occnums_w = np.zeros_like(self._eigvals)
        self._ldegen = np.zeros_like(self._eigvals)
        self._DOS = np.ones_like(self._eigvals)
        self._eigs_min_guess = np.zeros(
            (config.band_params["nkpts"], config.spindims, config.lmax)
        )
        self._eigvals_min = np.zeros(
            (config.band_params["nkpts"], config.spindims, config.lmax)
        )
        self._eigvals_max = np.zeros(
            (config.band_params["nkpts"], config.spindims, config.lmax)
        )
        self._kpt_int_weight = np.ones_like(self._eigvals)
        self._occ_weight = np.zeros_like(self._eigvals)

    @property
    def eigvals(self):
        r"""ndarray: The KS eigenvalues :math:`\epsilon_{nl}^\sigma`."""
        if np.all(self._eigvals == 0.0):
            raise Exception("Eigenvalues have not been initialized")
        return self._eigvals

    @property
    def eigfuncs(self):
        r"""
        ndarray: The radial KS eigenfunctions on the logarithmic grid.

        Related to real-grid KS radial orbitals by
        :math:`X_{nl}^{\sigma}(x)=e^{x/2} R_{nl}^{\sigma}(r).`
        """
        if np.all(self._eigfuncs == 0.0):
            raise Exception("Eigenfunctions have not been initialized")
        return self._eigfuncs

    @property
    def kpt_int_weight(self):
        r"""ndarray: The integration weight for the Fermi-Dirac / DOS integral."""
        if np.all(self._kpt_int_weight == 0.0):
            raise Exception("Band weightings have not been initialized")
        return self._kpt_int_weight

    @property
    def occnums_w(self):
        r"""ndarray: Weighted KS occupation numbers."""
        self._occnums_w = self.occnums * self.occ_weight
        return self._occnums_w

    @property
    def occnums(self):
        r"""ndarray: Bare KS occupation numbers (Fermi-Dirac)."""
        if np.all(self._occnums == 0.0):
            raise Exception("Occupation numbers have not been initialized")
        return self._occnums

    @property
    def occ_weight(self):
        r"""
        ndarray: KS occupation number weightings.

        The occupation weighting is the product of the DOS, degeneracy of the
        angular momentum, and the integration weight.
        """
        self._occ_weight = self.DOS * self.ldegen * self.kpt_int_weight
        return self._occ_weight

    @property
    def ldegen(self):
        r"""ndarray: Angular momentum degeneracy matrix."""
        self._ldegen = self.make_ldegen(self.eigvals)
        return self._ldegen

    @property
    def DOS(self):
        r"""ndarray: Density of states (DOS) matrix."""
        if config.bc == "bands":
            self._DOS = self.make_DOS_bands(
                self.eigvals_min, self.eigvals_max, self.eigvals
            )
        return self._DOS

    @property
    def eigvals_min(self):
        r"""ndarray: Lower bound (for bands bc) of KS eigenvalues."""
        if np.all(self._eigvals_min == 0.0):
            raise Exception("eigs_min has not been initialized")
        return self._eigvals_min

    @property
    def eigvals_max(self):
        r"""ndarray: Upper bound (for bands bc) of KS eigenvalues."""
        if np.all(self._eigvals_max == 0.0):
            raise Exception("eigs_min has not been initialized")
        return self._eigvals_max

    def compute(self, potential, bc, init=False, eig_guess=False):
        """
        Compute the orbitals and their eigenvalues with the given potential.

        Parameters
        ----------
        potential : ndarray
            the KS (or alternatively chosen) potential
        init : bool, optional
            whether orbitals are being computed for first time

        Returns
        -------
        None
        """
        # ensure the potential has the correct dimensions
        v = np.zeros((config.spindims, config.grid_params["ngrid"]))

        # set v to equal the input potential
        v[:] = potential

        if eig_guess:
            if bc != "bands":
                self._eigs_min_guess[0] = numerov.calc_eigs_min(v, self._xgrid, bc)
            else:
                self._eigs_min_guess[0] = numerov.calc_eigs_min(
                    v, self._xgrid, "neumann"
                )
                self._eigs_min_guess[1] = numerov.calc_eigs_min(
                    v, self._xgrid, "dirichlet"
                )

        # solve the KS equations
        if bc != "bands":
            self._eigfuncs[0], self._eigvals[0] = numerov.matrix_solve(
                v,
                self._xgrid,
                bc,
                eigs_min_guess=self._eigs_min_guess[0],
            )

            self._kpt_int_weight = np.ones_like(self._eigvals)
        else:
            eigfuncs_l, self._eigvals_min = numerov.matrix_solve(
                v,
                self._xgrid,
                "neumann",
                eigs_min_guess=self._eigs_min_guess[0],
            )

            eigfuncs_u, self._eigvals_max = numerov.matrix_solve(
                v,
                self._xgrid,
                "dirichlet",
                eigs_min_guess=self._eigs_min_guess[1],
            )

            self._eigvals, self._eigfuncs, self._kpt_int_weight = self.calc_bands(
                v, eigfuncs_l
            )

        # guess the chemical potential if initializing
        if init:
            config.mu = np.zeros((config.spindims))

        return

    # @writeoutput.timing
    def calc_bands(self, v, eigfuncs_l):
        """
        Compute the eigenfunctions which fill the energy bands.

        Parameters
        ----------
        v : ndarray
            the KS potential
        eigfuncs_l : ndarray
            the lower bound (neumann) eigenfunctions

        Returns
        -------
        eigvals : ndarray
            the orbital energies across all the bands
        eigfuncs : ndarray
            the KS eigenfunctions for all energies covered
        """
        # initialize some arrays
        eigfuncs = np.zeros_like(self._eigfuncs)
        eigvals = np.zeros_like(self._eigvals)
        kpt_int_weight = np.zeros_like(self._eigvals)

        # the energy band
        e_gap_arr = self.eigvals_max - self.eigvals_min

        # make the energy band array
        e_arr = np.linspace(
            self.eigvals_min, self.eigvals_max, config.band_params["nkpts"]
        )

        # propagate the numerov equation
        eigfuncs = numerov.calc_wfns_e_grid(self._xgrid, v, e_arr)

        # eigenvalues by default are equal to the energy band array
        eigvals = e_arr

        # make the k point integral weighting
        delta_E_plus = np.zeros_like(e_arr)
        delta_E_plus[1:] = e_arr[1:] - e_arr[:-1]
        delta_E_minus = np.zeros_like(e_arr)
        delta_E_minus[:-1] = e_arr[1:] - e_arr[:-1]
        kpt_int_weight = 0.5 * (delta_E_minus + delta_E_plus)

        # modify eigenvalues, kpt_int_weight and eigenfunctions when the gap is too
        # small to be considered a band
        eigvals = np.where(
            e_gap_arr[np.newaxis, :] >= config.band_params["de_min"],
            eigvals,
            self.eigvals_min[np.newaxis, :],
        )

        kpt_int_weight = np.where(
            e_gap_arr[np.newaxis, :] >= config.band_params["de_min"],
            kpt_int_weight,
            1.0 / config.band_params["nkpts"],
        )

        eigfuncs = np.where(
            e_gap_arr[np.newaxis, :, :, :, np.newaxis] >= config.band_params["de_min"],
            eigfuncs,
            eigfuncs_l[np.newaxis, :],
        )

        return eigvals, eigfuncs, kpt_int_weight

    def occupy(self):
        """
        Occupy the KS orbitals according to Fermi-Dirac statistics.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # compute the chemical potential using the eigenvalues
        config.mu = mathtools.chem_pot(self)

        # compute the occupation numbers using the chemical potential
        self._occnums = self.calc_occnums(self.eigvals, config.mu)

        return

    @staticmethod
    def calc_occnums(eigvals, mu):
        """
        Compute the Fermi-Dirac occupations for the eigenvalues.

        Parameters
        ----------
        eigvals : ndarray
            the KS eigenvalues
        lbound : ndarray
            the array of bound eigenvalues
        mu : aray_like
            the chemical potential

        Returns
        -------
        occnums : ndarray
            the orbital occupations multiplied with (2l+1) degeneracy
        """
        occnums = np.zeros_like(eigvals)

        for band in range(config.band_params["nkpts"]):
            for i in range(config.spindims):
                if config.nele[i] != 0:
                    occnums[band, i] = mathtools.fermi_dirac(
                        eigvals[band, i], mu[i], config.beta
                    )

        return occnums

    @staticmethod
    def make_ldegen(eigvals):
        r"""
        Construct the lbound matrix denoting the bound states and their degeneracies.

        For each spin channel,
        :math:`L^\mathrm{B}_{ln}=(2l+1)\times\Theta(\epsilon_{nl})`

        Parameters
        ----------
        eigvals : ndarray
            the KS orbital eigenvalues

        Returns
        -------
        lbound_mat : ndarray
            the lbound matrix
        """
        ldegen_mat = np.zeros_like(eigvals)

        if config.unbound == "quantum":
            for l in range(config.lmax):
                ldegen_mat[:, :, l] = (2.0 / config.spindims) * (2 * l + 1.0)
        elif config.unbound == "ideal":
            for l in range(config.lmax):
                ldegen_mat[:, :, l] = (2.0 / config.spindims) * np.where(
                    eigvals[:, :, l] < 0.0, 2 * l + 1.0, 0.0
                )

        # force bound levels if there are convergence issues

        if config.force_bound:
            for levels in config.force_bound:
                sp = levels[0]
                l = levels[1]
                n = levels[2]
                ldegen_mat[sp, l, n] = (2.0 / config.spindims) * (2 * l + 1.0)

        return ldegen_mat

    @staticmethod
    def make_DOS_bands(eigs_min, eigs_max, eigvals):
        r"""
        Compute the density-of-states using the method of Massacrier (see notes, [6]_).

        Parameters
        ----------
        eigs_min : ndarray
            the lower bound of the energy band
        eigs_max : ndarray
            the upper bound of the energy band
        eigvals : ndarray
            the KS energy eigenvalues

        Returns
        -------
        dos : ndarray
            the density-of-states

        Notes
        -----
        The density-of-states is defined in this model as:

        .. math::
            g(\epsilon) = \frac{2}{\pi  \delta^2} \sqrt{(\epsilon^+ - \epsilon)\
                        (\epsilon - \epsilon^-)},\
            \delta = \frac{1}{2} (\epsilon^+ - \epsilon^-),

        where :math:`\epsilon^\pm` are the upper and lower band limits respectively.

        References
        ----------
        .. [6] Massacrier, G. et al, Reconciling ionization energies and band gaps of
           warm dense matter derived with ab initio simulations and average atom models,
           Physical Review Research 3.2 (2021): 023026.
           `DOI:10.1103/PhysRevResearch.3.023026
           <https://doi.org/10.1103/PhysRevResearch.3.023026>`__.
        """
        # get the eigenvalue difference in a correctly shaped array
        eig_diff = np.einsum(
            "ijk,lijk->lijk", eigs_max - eigs_min, np.ones_like(eigvals)
        )

        # compute delta and (e_+ - e) * (e - e_-)
        delta = 0.5 * eig_diff
        hub_func = (eigs_max - eigvals) * (eigvals - eigs_min)

        # take the sqrt when the energy gap is big enough to justify a band
        f_sqrt = np.where(
            eig_diff > config.band_params["de_min"],
            np.sqrt(np.abs(hub_func)),
            1.0,
        )

        # compute the pre-factor when the energy gap is large enough for a band
        prefac = np.where(
            eig_diff > config.band_params["de_min"],
            2.0 / (pi * delta ** 2.0),
            1.0,
        )

        # compute the dos
        dos = prefac * f_sqrt

        return dos

    @staticmethod
    def calc_DOS_sum(eigs_min, eigs_max, ldegen):
        r"""
        Compute the summed density-of-states using the method of Massacrier et al.

        This function sums the DOS over `l` and `n` quantum numbers and is used for
        writing the output of the DOS only. Refer to `make_DOS_bands` for details on
        the functional form of the DOS.

        Parameters
        ----------
        eigs_min : ndarray
            the lower bound of the energy band
        eigs_max : ndarray
            the upper bound of the energy band
        ldegen : ndarray
            degeneracy matrix

        Returns
        -------
        e_arr : ndarray
            sorted array of energies
        fd_dist : ndarray
            Fermi-Dirac occupation numbers
        DOS_sum : ndarray
            Density-of-states multiplied and summed over (l,n) quanutum numbers.
        """
        # create the gapped array
        e_gap_arr = eigs_max - eigs_min

        nspin, lmax, nmax = np.shape(eigs_max)

        e_arr_dummy = Orbitals.make_e_arr(eigs_min, eigs_max, 0)
        e_arr = np.zeros((len(e_arr_dummy), nspin))
        dos_knl = np.zeros((len(e_arr_dummy), nspin, lmax, nmax))
        fd_dist = np.zeros((len(e_arr_dummy), nspin))

        for sp in range(nspin):

            # make the total energy array
            e_arr[:, sp] = Orbitals.make_e_arr(eigs_min, eigs_max, sp)

            for i, e in enumerate(e_arr[:, sp]):

                # compute delta and (e_+ - e) * (e - e_-)
                delta = 0.5 * e_gap_arr
                hub_func = (eigs_max - e) * (e - eigs_min)

                # take the sqrt when the energy gap is big enough to justify a band
                f_sqrt = np.where(hub_func > 0, np.sqrt(np.abs(hub_func)), 0.0)

                # compute the pre-factor when the energy gap is large enough for a band
                prefac = np.where(
                    e_gap_arr > config.band_params["de_min"],
                    2.0 / (pi * delta ** 2.0),
                    0.0,
                )

                # compute the dos
                dos_knl[i, sp] = prefac[sp] * f_sqrt[sp]

                # compute the Fermi-Dirac distribution
                fd_dist[i, sp] = mathtools.fermi_dirac(e, config.mu[sp], config.beta)

        ldegen0 = ldegen[0, 0, :, 0]

        # sum over the l and n axes
        DOS_sum = np.einsum("ijkl,k->ij", dos_knl, ldegen0)

        return e_arr, fd_dist, DOS_sum

    @staticmethod
    def make_e_arr(eigvals_min, eigvals_max, sp):
        """Make the energy array for the bands boundary condition.

        Energies are populated from the lowest band up to the maximum specified energy,
        with band gaps (forbidden energies) accounted for. The array is non-linear in
        order to optimize computation time.

        Parameters
        ----------
        eigvals_min : ndarray
            the lower bound of the energy bands
        eigvals_max : ndarray
            the upper bound of the energy bands

        Returns
        -------
        e_tot_arr : ndarray
            the banded energy array
        """
        # flatten and sort the minimum and maximum eigenvalues
        eigs_min = eigvals_min[sp].flatten()
        eigs_max = eigvals_max[sp].flatten()
        eigs_min = eigs_min[np.argsort(eigs_min)]
        eigs_max = eigs_max[np.argsort(eigs_max)]

        # make an array of the band energy differences
        e_gap_arr = eigvals_max - eigvals_min

        # start array from lowest eigenvalue to be treated as a band
        e_min = np.amin(
            eigvals_min[np.where(e_gap_arr >= config.band_params["de_min"])]
        )

        # make the energy array
        e_tot_arr = np.array([])
        for p in range(len(eigs_min) - 1):

            # ignore energies below the minimum for bands
            if eigs_min[p] < e_min:
                continue

            # populate linearly spaced energies in the band
            else:
                e_pt_arr = np.linspace(
                    eigs_min[p], eigs_max[p], config.band_params["nkpts"]
                )
                e_tot_arr = np.concatenate((e_tot_arr, e_pt_arr))

        # sort the array
        e_tot_arr = np.sort(e_tot_arr)
        return e_tot_arr


class Density:
    """
    Class holding the static KS density and routines required to compute its components.

    Parameters
    ----------
    orbs : :obj:`Orbitals`
        The KS orbitals object
    """

    def __init__(self, orbs):
        self._xgrid = orbs._xgrid
        self._total = np.zeros((config.spindims, config.grid_params["ngrid"]))
        self._bound = {
            "rho": np.zeros((config.spindims, config.grid_params["ngrid"])),
            "N": np.zeros((config.spindims)),
        }
        self._unbound = {
            "rho": np.zeros((config.spindims, config.grid_params["ngrid"])),
            "N": np.zeros((config.spindims)),
        }

        self._orbs = orbs

    @property
    def total(self):
        """ndarray: Total KS density :math:`n(r)` or :math:`n(x)`."""
        if np.all(self._total == 0.0):
            self._total = self.bound["rho"] + self.unbound["rho"]
        return self._total

    @property
    def bound(self):
        """
        :obj:`dict` of :obj:`ndarrays`: Bound part of KS density.

        Contains the keys `rho` and `N` denoting the bound density and number of bound
        electrons respectively.
        """
        if np.all(self._bound["rho"] == 0.0):
            self._bound = self.construct_rho_orbs(
                self._orbs.eigfuncs, self._orbs.occnums_w, self._xgrid
            )
        return self._bound

    @property
    def unbound(self):
        """
        :obj:`dict` of :obj:`ndarrays`: Unbound part of KS density.

        Contains the keys `rho` and `N` denoting the
        unbound density and number of unbound electrons respectively
        """
        if np.all(self._unbound["rho"]) == 0.0:
            if config.unbound == "ideal":
                self._unbound = self.construct_rho_unbound()
        return self._unbound

    @staticmethod
    def construct_rho_orbs(eigfuncs, occnums, xgrid):
        """
        Construct a density from a set of discrete KS orbitals.

        Parameters
        ----------
        eigfuncs : ndarray
            the radial eigenfunctions on the xgrid
        occnums : ndarray
            the orbital occupations
        xgrid : ndarray
            the logarithmic grid

        Returns
        -------
        dens : dict of ndarrays
            contains the keys `rho` and `N` denoting the density
            and number of electrons respectively
        """
        dens = {}  # initialize empty dict

        # first of all construct the density
        # rho_b(r) = \sum_{n,l} (2l+1) f_{nl} |R_{nl}(r)|^2
        # occnums in atoMEC are defined as (2l+1)*f_{nl}

        # R_{nl}(r) = exp(x/2) P_{nl}(x), P(x) are eigfuncs
        orbs_R = np.exp(-xgrid / 2.0) * eigfuncs
        orbs_R_sq = orbs_R ** 2.0

        # sum over the (l,n) dimensions of the orbitals to get the density
        dens["rho"] = np.einsum("ijkl,ijklm->jm", occnums, orbs_R_sq)

        # compute the number of unbound electrons
        dens["N"] = np.sum(occnums, axis=(0, 2, 3))

        return dens

    @staticmethod
    def construct_rho_unbound():
        """
        Construct the unbound part of the density.

        Parameters
        ----------
        orbs : ndarray
            the radial eigenfunctions on the xgrid
        xgrid : ndarray
            the logarithmic grid

        Returns
        -------
        rho_unbound : dict of ndarrays
            contains the keys `rho` and `N` denoting the unbound density
            and number of unbound electrons respectively
        """
        rho_unbound = np.zeros((config.spindims, config.grid_params["ngrid"]))
        N_unbound = np.zeros((config.spindims))

        # so far only the ideal approximation is implemented
        if config.unbound == "ideal":

            # unbound density is constant
            for i in range(config.spindims):
                prefac = (2.0 / config.spindims) * 1.0 / (sqrt(2) * pi ** 2)
                n_ub = prefac * mathtools.fd_int_complete(
                    config.mu[i], config.beta, 1.0
                )
                rho_unbound[i] = n_ub
                N_unbound[i] = n_ub * config.sph_vol

            unbound = {"rho": rho_unbound, "N": N_unbound}

        return unbound


class Potential:
    """Class holding the KS potential and the routines required to compute it."""

    def __init__(self, density):

        self._v_s = np.zeros_like(density.total)
        self._v_en = np.zeros((config.grid_params["ngrid"]))
        self._v_ha = np.zeros((config.grid_params["ngrid"]))
        self._v_xc = {
            "x": np.zeros_like(density.total),
            "c": np.zeros_like(density.total),
            "xc": np.zeros_like(density.total),
        }
        self._density = density.total
        self._xgrid = density._xgrid

    @property
    def v_s(self):
        r"""
        ndarray: The full KS potential.

        Given by :math:`v_\mathrm{s} = v_\mathrm{en} + v_\mathrm{ha} + v_\mathrm{xc}`.
        """
        if np.all(self._v_s == 0.0):
            self._v_s = self.v_en + self.v_ha + self.v_xc["xc"]
        return self._v_s

    @property
    def v_en(self):
        r"""ndarray: The electron-nuclear potential."""
        if np.all(self._v_en == 0.0):
            self._v_en = self.calc_v_en(self._xgrid)
        return self._v_en

    @property
    def v_ha(self):
        r"""ndarray: The Hartree potential."""
        if np.all(self._v_ha == 0.0):
            self._v_ha = self.calc_v_ha(self._density, self._xgrid)
        return self._v_ha

    @property
    def v_xc(self):
        r"""
        :obj:`dict` of :obj:`ndarrays`: The exchange-correlation potential.

        Contains the keys `x`, `c` and `xc` denoting exchange, correlation,
        and exchange + correlation respectively.
        """
        if np.all(self._v_xc["xc"] == 0.0):
            self._v_xc = xc.v_xc(self._density, self._xgrid, config.xfunc, config.cfunc)
        return self._v_xc

    @staticmethod
    def calc_v_en(xgrid):
        r"""
        Construct the electron-nuclear potential.

        The electron-nuclear potential is given by
        :math:`v_\mathrm{en} (x) = -Z * \exp(-x)`
        on the logarithmic grid
        """
        v_en = -config.Z * np.exp(-xgrid)

        return v_en

    @staticmethod
    def calc_v_ha(density, xgrid):
        r"""
        Construct the Hartree potential (see notes).

        Parameters
        ----------
        density : ndarray
            the total KS density
        xgrid : ndarray
            the logarithmic grid

        Notes
        -----
        :math:`v_\mathrm{ha}` is defined on the r-grid as:

        .. math::
            v_\mathrm{ha}(r) = 4\pi\int_0^r \mathrm{d}r' r'^2 \frac{n(r')}{\max(r,r')}

        On the x-grid:

        .. math::
            v_\mathrm{ha}(x) = 4\pi\Big\{\exp(-x)\int_{x0}^x \mathrm{d}x' n(x') \
            \exp(3x') -\int_x^{\log(r_s)} \mathrm{d}x' n(x') \exp(2x') \Big\}
        """
        # initialize v_ha
        v_ha = np.zeros_like(xgrid)

        # construct the total (sum over spins) density
        rho = np.sum(density, axis=0)

        # loop over the x-grid
        # this may be a bottleneck...
        for i, x0 in enumerate(xgrid):

            # set up 'upper' and 'lower' parts of the xgrid (x<=x0; x>x0)
            x_u = xgrid[np.where(x0 < xgrid)]
            x_l = xgrid[np.where(x0 >= xgrid)]

            # likewise for the density
            rho_u = rho[np.where(x0 < xgrid)]
            rho_l = rho[np.where(x0 >= xgrid)]

            # now compute the hartree potential
            int_l = exp(-x0) * np.trapz(rho_l * np.exp(3.0 * x_l), x_l)
            int_u = np.trapz(rho_u * np.exp(2 * x_u), x_u)

            # total hartree potential is sum over integrals
            v_ha[i] = 4.0 * pi * (int_l + int_u)

        return v_ha


class Energy:
    r"""Class holding information about the KS total energy and relevant routines."""

    def __init__(self, orbs, dens):

        # inputs
        self._orbs = orbs
        self._dens = dens.total
        self._xgrid = dens._xgrid

        # initialize attributes
        self._F_tot = 0.0
        self._E_tot = 0.0
        self._entropy = {"tot": 0.0, "bound": 0.0, "unbound": 0.0}
        self._E_kin = {"tot": 0.0, "bound": 0.0, "unbound": 0.0}
        self._E_en = 0.0
        self._E_ha = 0.0
        self._E_xc = {"xc": 0.0, "x": 0.0, "c": 0.0}

    @property
    def F_tot(self):
        r"""
        :obj:`dict` of :obj:`floats`: The free energy.

        Contains the keys `F`, `E` and `S` for free energy, internal energy and
        total entropy. :math:`F = E - TS`
        """
        if self._F_tot == 0.0:
            self._F_tot = self.E_tot - config.temp * self.entropy["tot"]
        return self._F_tot

    @property
    def E_tot(self):
        r"""
        float: The total KS internal energy.

        Given by :math:`E=T_\mathrm{s}+E_\mathrm{en}+E_\mathrm{ha}+F_\mathrm{xc}`.
        """
        if self._E_tot == 0.0:
            self._E_tot = self.E_kin["tot"] + self.E_en + self.E_ha + self.E_xc["xc"]
        return self._E_tot

    @property
    def entropy(self):
        """
        :obj:`dict` of :obj:`floats`: The total entropy.

        Contains `bound` and `unbound` keys.
        """
        if self._entropy["tot"] == 0.0:
            self._entropy = self.calc_entropy(self._orbs)
        return self._entropy

    @property
    def E_kin(self):
        """
        :obj:`dict` of :obj:`floats`: KS kinetic energy.

        Contains `bound` and `unbound` keys.
        """
        if self._E_kin["tot"] == 0.0:
            self._E_kin = self.calc_E_kin(self._orbs, self._xgrid)
        return self._E_kin

    @property
    def E_en(self):
        """float: The electron-nuclear energy."""
        if self._E_en == 0.0:
            self._E_en = self.calc_E_en(self._dens, self._xgrid)
        return self._E_en

    @property
    def E_ha(self):
        """float: The Hartree energy."""
        if self._E_ha == 0.0:
            self._E_ha = self.calc_E_ha(self._dens, self._xgrid)
        return self._E_ha

    @property
    def E_xc(self):
        """
        :obj:`dict` of :obj:`floats`: The exchange-correlation energy.

        Contains the keys `x`, `c` and `xc` for exchange,
        correlation and exchange + correlation respectively
        """
        if self._E_xc["xc"] == 0.0:
            self._E_xc = xc.E_xc(self._dens, self._xgrid, config.xfunc, config.cfunc)
        return self._E_xc

    def calc_E_kin(self, orbs, xgrid):
        """
        Compute the kinetic energy.

        Kinetic energy is in general different for bound and unbound components.
        This routine is a wrapper calling the respective functions.

        Parameters
        ----------
        orbs : :obj:`Orbitals`
            the KS orbitals object
        xgrid : ndarray
            the logarithmic grid

        Returns
        -------
        E_kin : dict of ndarrays
            Contains `tot`, `bound` and `unbound` keys
        """
        E_kin = {}

        # bound part
        E_kin["bound"] = self.calc_E_kin_orbs(orbs.eigfuncs, orbs.occnums_w, xgrid)

        # unbound part
        if config.unbound == "ideal":
            E_kin["unbound"] = self.calc_E_kin_unbound()
        else:
            E_kin["unbound"] = 0.0

        # total
        E_kin["tot"] = E_kin["bound"] + E_kin["unbound"]

        return E_kin

    @staticmethod
    def calc_E_kin_orbs(eigfuncs, occnums, xgrid):
        """
        Compute the kinetic energy contribution from discrete KS orbitals.

        Parameters
        ----------
        eigfuncs : ndarray
            the radial KS orbitals on the log grid
        occnums : ndarray
            the orbital occupations
        xgrid : ndarray
            the logarithmic grid

        Returns
        -------
        E_kin : float
            the kinetic energy
        """
        # compute the kinetic energy density (using default method A)
        e_kin_dens = Energy.calc_E_kin_dens(eigfuncs, occnums, xgrid)

        # FIXME: this is necessary because the Laplacian is not accurate at the boundary
        for i in range(config.spindims):
            e_kin_dens[i, -3:] = e_kin_dens[i, -4]

        # integrate over sphere
        E_kin = mathtools.int_sphere(np.sum(e_kin_dens, axis=0), xgrid)

        return E_kin

    @staticmethod
    def calc_E_kin_dens(eigfuncs, occnums, xgrid, method="A"):
        """
        Calculate the local kinetic energy density (KED).

        There are multiple definitions in the literature of the local KED, see notes.

        Parameters
        ----------
        eigfuncs : ndarray
            the radial KS orbitals on the log grid
        occnums : ndarray
            the orbital occupations
        xgrid : ndarray
            the logarithmic grid
        method : str, optional
            the definition used for KED, can be 'A' or 'B' (see notes).

        Returns
        -------
        e_kin_dens : ndarray
            the local kinetic energy density

        Notes
        -----
        The methods 'A' and 'B' in this function are given according to the definitions
        in [3]_ and [4]_.

        They of course (should) both integrate to the same kinetic energy. The
        definition 'B' is the one used in the usual definition of the electron
        localization function [5]_.

        References
        ----------
        .. [3] H. Jiang, The local kinetic energy density revisited,
           New J. Phys. 22 103050 (2020), `DOI:10.1088/1367-2630/abbf5d
           <https://doi.org/10.1088/1367-2630/abbf5d>`__.
        .. [4] L. Cohen, Local kinetic energy in quantum mechanics,
           J. Chem. Phys. 70, 788 (1979), `DOI:10.1063/1.437511
           <https://doi.org/10.1063/1.437511>`__.
        .. [5] A. Savin et al., Electron Localization in Solid-State Structures of the
           Elements: the Diamond Structure, Angew. Chem. Int. Ed. Engl. 31: 187-188
           (1992), `DOI:10.1002/anie.199201871
           <https://doi.org/10.1002/anie.199201871>`__.
        """
        if method == "A":
            # compute the grad^2 component
            grad2_orbs = mathtools.laplace(eigfuncs, xgrid)

            # compute the (l+1/2)^2 component
            l_arr = np.fromiter(
                ((l + 0.5) ** 2.0 for l in range(config.lmax)), float, config.lmax
            )
            lhalf_orbs = np.einsum("k,ijklm->ijklm", l_arr, eigfuncs)

            # add together and multiply by eigfuncs*exp(-3x)
            prefac = np.exp(-3.0 * xgrid) * eigfuncs
            kin_orbs = prefac * (grad2_orbs - lhalf_orbs)

            # multiply and sum over occupation numbers
            e_kin_dens = -0.5 * np.einsum("ijkl,ijklm->jm", occnums, kin_orbs)

        elif method == "B":

            # compute the gradient of the orbitals
            grad_eigfuncs = np.gradient(eigfuncs, xgrid, axis=-1, edge_order=2)

            # chain rule to convert from dP_dx to dX_dr
            grad_orbs = np.exp(-1.5 * xgrid) * (grad_eigfuncs - 0.5 * eigfuncs)

            # square it
            grad_orbs_sq = grad_orbs ** 2.0

            # multiply and sum over occupation numbers
            e_kin_dens = 0.5 * np.einsum("ijkl,ijklm->jm", occnums, grad_orbs_sq)

        return e_kin_dens

    @staticmethod
    def calc_E_kin_unbound():
        r"""
        Compute the contribution from unbound (continuum) electrons to kinetic energy.

        Parameters
        ----------
        orbs : :obj:`Orbitals`
            the KS orbitals object
        xgrid : ndarray
            the logarithmic grid

        Returns
        -------
        E_kin_unbound : float
            the unbound kinetic energy

        Notes
        -----
        Currently only "ideal" (uniform) approximation for unbound electrons supported.

        .. math::
            T_\mathrm{ub} = \sum_\sigma \frac{N^\sigma\times V}{\sqrt{2}\pi^2}\
            I_{3/2}(\mu,\beta),

        where :math:`I_{3/2}(\mu,\beta)` denotes the complete Fermi-Diract integral
        """
        # currently only ideal treatment supported
        if config.unbound == "ideal":
            E_kin_unbound = 0.0  # initialize
            for i in range(config.spindims):
                prefac = (2.0 / config.spindims) * config.sph_vol / (sqrt(2) * pi ** 2)
                E_kin_unbound += prefac * mathtools.fd_int_complete(
                    config.mu[i], config.beta, 3.0
                )

        return E_kin_unbound

    def calc_entropy(self, orbs):
        """
        Compute the KS entropy.

        Entropy is in general different for bound and unbound orbitals.
        This function is a wrapper which calls the respective functions.

        Parameters
        ----------
        orbs : :obj:`Orbitals`
            the KS orbitals object

        Returns
        -------
        S : dict of floats
           contains `tot`, `bound` and `unbound` keys
        """
        S = {}

        # bound part
        S["bound"] = self.calc_S_orbs(orbs.occnums, orbs.occ_weight)

        # unbound part
        if config.unbound == "ideal":
            S["unbound"] = self.calc_S_unbound()
        else:
            S["unbound"] = 0.0

        # total
        S["tot"] = S["bound"] + S["unbound"]

        return S

    @staticmethod
    def calc_S_orbs(occnums, degen):
        r"""
        Compute the KS (non-interacting) entropy for specified orbitals (see notes).

        Parameters
        ----------
        occnums : ndarray
            orbital occupation numbers
        degen : ndarray
            product of dos and degeneracy array (:math:`(2l+1)`)

        Returns
        -------
        S : float
            the bound contribution to the entropy

        Notes
        -----
        The entropy of non-interacting (KS) electrons is given by:

        .. math::
            S_\mathrm{b} = -\sum_{s,l,n} g_{ln} (2l+1) [ f_{nls} \log(f_{nls}) \
                           + (1-f_{nls}) (\log(1-f_{nls}) ]
        """
        # replace zeros in occupation numbers with finite numbers (for taking log)
        occnums_mod1 = np.where(occnums > 1e-5, occnums, 0.5)
        occnums_mod2 = np.where(occnums < 1.0 - 1e-5, occnums, 0.5)

        # now compute the terms in the square bracket
        term1 = occnums * np.log(occnums_mod1)
        term2 = (1.0 - occnums) * np.log(1.0 - occnums_mod2)

        # multiply by degeneracy (dos * (2l+1))
        g_nls = degen * (term1 + term2)

        # sum over all quantum numbers to get the total entropy
        S_orbs = -np.sum(g_nls)

        return S_orbs

    @staticmethod
    def calc_S_unbound():
        r"""
        Compute the unbound contribution to the entropy.

        Parameters
        ----------
        orbs : :obj:`Orbitals`
            the KS orbitals object

        Returns
        -------
        S_unbound : float
            the unbound entropy term

        Notes
        -----
        Currently only "ideal" (uniform) treatment of unbound electrons is supported.

        .. math::
            S_\mathrm{ub} = \sum_\sigma \frac{V}{\sqrt{2}\pi^2} I_{1/2}(\mu,\beta)

        where :math:`I_{1/2}(\mu,\beta)` is the complete Fermi-Dirac integral of order
        :math:`1/2`
        """
        # currently only ideal treatment supported
        if config.unbound == "ideal":
            S_unbound = 0.0  # initialize
            for i in range(config.spindims):
                if config.nele[i] > 1e-5:
                    prefac = (
                        (2.0 / config.spindims) * config.sph_vol / (sqrt(2) * pi ** 2)
                    )

                    S_unbound -= prefac * mathtools.ideal_entropy_int(
                        config.mu[i], config.beta, 1.0
                    )
                else:
                    S_unbound += 0.0

        return S_unbound

    @staticmethod
    def calc_E_en(density, xgrid):
        r"""
        Compute the electron-nuclear energy.

        Parameters
        ----------
        density : ndarray
            the (spin) KS density
        xgrid : ndarray
            the logarithmic grid

        Returns
        -------
        E_en : float
             the electron-nuclear energy

        Notes
        -----
        Electron-nuclear energy is given by

        .. math:: E_{en} = 4\pi\int_0^{r_s} \mathrm{d}{r} r^2 n(r) v_{en}(r),

        where :math:`r_s` denotes the radius of the sphere of interest
        """
        # sum the density over the spin axes to get the total density
        dens_tot = np.sum(density, axis=0)

        # compute the integral
        v_en = Potential.calc_v_en(xgrid)
        E_en = mathtools.int_sphere(dens_tot * v_en, xgrid)

        return E_en

    @staticmethod
    def calc_E_ha(density, xgrid):
        r"""
        Compute the Hartree energy.

        Parameters
        ----------
        density : ndarray
            the (spin) KS density
        xgrid : ndarray
            the logarithmic grid

        Returns
        -------
        E_ha : float
            the Hartree energy

        Notes
        -----
        The Hartree energy is given by

        .. math:: E_\mathrm{ha} = 2\pi\int_0^{r_s}\mathrm{d}r r^2 n(r) v_\mathrm{ha}(r),

        where :math:`v_\mathrm{ha}(r)` is the Hartree potential
        """
        # sum density over spins to get total density
        dens_tot = np.sum(density, axis=0)

        # compute the integral
        v_ha = Potential.calc_v_ha(density, xgrid)
        E_ha = 0.5 * mathtools.int_sphere(dens_tot * v_ha, xgrid)

        return E_ha


class EnergyAlt:
    r"""Class holding information about the KS energy and associated routines.

    N.B. this computes the total energy functional in an alternative way to the main
    :class:`Energy` object. In this class, the total energy is first calculated from the
    sum of the eigenvalues and then :math:`\int \mathrm{d}r v_\mathrm{Hxc}(r) n(r)`
    is subtracted to obtain the sum over kinetic and electron-nuclear energies.

    In general, this gives a different result compared to the :class:`Energy`, due
    to the way the unbound electrons are treated. Which result is 'correct' depends on
    how we interpret the unbound energy (i.e., is it purely a kinetic term or not).
    """

    def __init__(self, orbs, dens, pot):

        self._orbs = orbs
        self._dens = dens.total
        self._xgrid = dens._xgrid
        self._pot = pot

        # initialize attributes
        self._F_tot = 0.0
        self._E_tot = 0.0
        self._entropy = {"tot": 0.0, "bound": 0.0, "unbound": 0.0}
        self._E_eps = 0.0
        self._E_unbound = 0.0
        self._E_v_hxc = 0.0
        self._E_ha = 0.0
        self._E_xc = {"xc": 0.0, "x": 0.0, "c": 0.0}

    @property
    def F_tot(self):
        r"""
        :obj:`dict` of :obj:`floats`: The free energy.

        Contains the keys `F`, `E` and `S` for free energy, internal energy and
        total entropy. :math:`F = E - TS`
        """
        if self._F_tot == 0.0:
            self._F_tot = self.E_tot - config.temp * self.entropy["tot"]
        return self._F_tot

    @property
    def E_tot(self):
        r"""
        float: The total KS internal energy.

        Given by :math:`E=T_\mathrm{s}+E_\mathrm{en}+E_\mathrm{ha}+F_\mathrm{xc}`.
        """
        if self._E_tot == 0.0:
            self._E_tot = (
                self.E_eps + self.E_unbound - self.E_v_hxc + self.E_ha + self.E_xc["xc"]
            )
        return self._E_tot

    @property
    def E_eps(self):
        r"""
        float: The sum of the (weighted) eigenvalues.

        Given by :math:`E=\sum_{nl\sigma} (2l+1) f_{nl}^\sigma \epsilon_{nl}^\sigma`
        """
        if self._E_eps == 0.0:
            self._E_eps = np.sum(self._orbs.occnums_w * self._orbs.eigvals)
        return self._E_eps

    @property
    def E_unbound(self):
        r"""float: The energy of the unbound part of the electron density."""
        if self._E_unbound == 0.0:
            self._E_unbound = Energy.calc_E_kin_unbound(self._orbs, self._xgrid)
        return self._E_unbound

    @property
    def E_v_hxc(self):
        r"""float: The integral :math:`\int \mathrm{d}r v_\mathrm{Hxc}(r) n(r)`."""
        if self._E_v_hxc == 0.0:
            self._E_v_hxc = self.calc_E_v_hxc(self._dens, self._pot, self._xgrid)
        return self._E_v_hxc

    @property
    def entropy(self):
        """
        :obj:`dict` of :obj:`floats`: The total entropy.

        Contains `bound` and `unbound` keys.
        """
        if self._entropy["tot"] == 0.0:
            self._entropy = self.calc_entropy(self._orbs)
        return self._entropy

    @property
    def E_ha(self):
        """float: The Hartree energy."""
        if self._E_ha == 0.0:
            self._E_ha = Energy.calc_E_ha(self._dens, self._xgrid)
        return self._E_ha

    @property
    def E_xc(self):
        """
        :obj:`dict` of :obj:`floats`: The exchange-correlation energy.

        Contains the keys `x`, `c` and `xc` for exchange,
        correlation and exchange + correlation respectively
        """
        if self._E_xc["xc"] == 0.0:
            self._E_xc = xc.E_xc(self._dens, self._xgrid, config.xfunc, config.cfunc)
        return self._E_xc

    @staticmethod
    def calc_E_v_hxc(dens, pot, xgrid):
        r"""
        Compute the compensating integral term over the Hxc-potential (see notes).

        Parameters
        ----------
        dens : ndarray
            the (spin) KS density
        pot : :class:`Potential`
            the KS potential object
        xgrid : ndarray
            the logarithmic grid

        Returns
        -------
        E_v_hxc : float
            the compensating Hxc-potential integral

        Notes
        -----
        In this construction of the KS energy functional, the sum over the KS
        eigenvalues must be compensated by an integral of the Hxc-potential.
        This integral is given by:

        .. math::
            E = 4\pi\sum_\sigma\int \mathrm{d}r r^2 n^\sigma(r) v_\mathrm{Hxc}^\sigma(r)

        """
        # first compute the hartree contribution, which is twice the hartree energy
        E_v_ha = 2.0 * Energy.calc_E_ha(dens, xgrid)

        # now compute the xc contribution (over each spin channel)
        E_v_xc = 0.0
        for i in range(config.spindims):
            v_xc = pot.v_xc["xc"][i]
            E_v_xc = E_v_xc + mathtools.int_sphere(dens[i] * v_xc, xgrid)

        # compute the term due to the constant shift introduced in the potential
        if config.v_shift:
            v_shift = pot.v_s[:, -1]
            E_const = -np.sum(v_shift * config.nele)
        else:
            E_const = 0.0

        return E_v_ha + E_v_xc + E_const

    def calc_entropy(self, orbs):
        """
        Compute the KS entropy.

        Entropy is in general different for bound and unbound orbitals.
        This function is a wrapper which calls the respective functions.

        Parameters
        ----------
        orbs : :obj:`Orbitals`
            the KS orbitals object

        Returns
        -------
        S : dict of floats
           contains `tot`, `bound` and `unbound` keys
        """
        S = {}

        # bound part
        S["bound"] = Energy.calc_S_bound(orbs)

        # unbound part
        S["unbound"] = Energy.calc_S_unbound(orbs)

        # total
        S["tot"] = S["bound"] + S["unbound"]

        return S
