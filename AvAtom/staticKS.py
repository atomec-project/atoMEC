"""
Module which computes time-independent properties from the average-atom setup
"""

# standard packages

# external packages
import numpy as np
from math import sqrt, pi, exp

# internal modules
import config
import numerov
import mathtools
import xc

# the logarithmic grid
def log_grid(x_r):
    """
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
    """
    The orbitals object has the following attributes:
    - eigfuncs (numpy array)    :   the orbitals defined on the numerical grid
    - eigvals  (numpy array)    :   KS orbital eigenvalues
    - occnums  (numpy array)    :   KS orbital occupation numbers
    """

    def __init__(self, xgrid):
        """
        Initializes the orbital attributes to empty numpy arrays
        """

        self._xgrid = xgrid
        self._eigfuncs = np.zeros(
            (config.spindims, config.lmax, config.nmax, config.grid_params["ngrid"])
        )
        self._eigvals = np.zeros((config.spindims, config.lmax, config.lmax))
        self._occnums = np.zeros_like(self._eigvals)
        self._lbound = np.zeros_like(self._eigvals)

    @property
    def eigvals(self):
        if np.all(self._eigvals == 0.0):
            raise Exception("Eigenvalues have not been initialized")
        return self._eigvals

    @property
    def eigfuncs(self):
        if np.all(self._eigfuncs == 0.0):
            raise Exception("Eigenfunctions have not been initialized")
        return self._eigfuncs

    @property
    def occnums(self):
        if np.all(self._occnums == 0.0):
            raise Exception("Occnums have not been initialized")
        return self._occnums

    @property
    def lbound(self):
        if np.all(self._lbound == 0.0):
            raise Exception("lbound has not been initialized")
        return self._lbound

    def compute(self, potential, init=False):
        """
        Compute the orbitals and their eigenvalues with the given potential
        """

        # ensure the potential has the correct dimensions
        v = np.zeros((config.spindims, config.grid_params["ngrid"]))

        # set v to equal the input potential
        v[:] = potential

        # solve the KS equations
        self._eigfuncs, self._eigvals = numerov.matrix_solve(v, self._xgrid)

        # compute the lbound array
        self._lbound = self.make_lbound(self.eigvals)

        # guess the chemical potential if initializing
        if init:
            config.mu = np.zeros((config.spindims))

        return

    def occupy(self):
        """
        Occupies the orbitals according to Fermi Dirac statistics.
        The chemical potential is calculated to satisfy charge neutrality
        within the Voronoi sphere
        """

        # compute the chemical potential using the eigenvalues
        config.mu = mathtools.chem_pot(self)

        # compute the occupation numbers using the chemical potential
        self._occnums = self.calc_occnums(self.eigvals, self.lbound, config.mu)

    @staticmethod
    def calc_occnums(eigvals, lbound, mu):
        """
        Computes the Fermi-Dirac occupations for the eigenvalues
        """

        occnums = np.zeros_like(eigvals)

        for i in range(config.spindims):
            if config.nele[i] != 0:
                occnums[i] = lbound[i] * mathtools.fermi_dirac(
                    eigvals[i], mu[i], config.beta
                )

        return occnums

    @staticmethod
    def make_lbound(eigvals):
        """
        Constructs the 'lbound' attribute
        For each spin channel, lbound(l,n)=(2l+1)*Theta(eps_n)
        """

        lbound_mat = np.zeros_like(eigvals)

        for l in range(config.lmax):
            lbound_mat[:, l] = (2.0 / config.spindims) * np.where(
                eigvals[:, l] < 0, 2 * l + 1.0, 0.0
            )

        return lbound_mat


class Density:
    """
    The Density object has the following attributes:
    - rho_tot (np array)       : the total density n(r)
    - rho_bound (np array)     : the bound part of the density n(r)
    - rho_unbound (np array)   : the unbound part of the density n(r)
    - N_bound (np array)       : the number of bound electrons
    - N_unbound (np array)     : the number of unbound electrons
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
        if np.all(self._total == 0.0):
            self._total = self.bound["rho"] + self.unbound["rho"]
        return self._total

    @property
    def bound(self):
        if np.all(self._bound["rho"] == 0.0):
            self._bound = self.construct_rho_bound(self._orbs, self._xgrid)
        return self._bound

    @property
    def unbound(self):
        if np.all(self._unbound["rho"]) == 0.0:
            self._unbound = self.construct_rho_unbound(self._orbs)
        return self._unbound

    @staticmethod
    def construct_rho_bound(orbs, xgrid):
        """
        Constructs the bound part of the density

        Inputs:
        - orbs (object)    : the orbitals object
        """

        bound = {}  # initialize empty dict

        # first of all construct the density
        # rho_b(r) = \sum_{n,l} (2l+1) f_{nl} |R_{nl}(r)|^2
        # occnums in AvAtom are defined as (2l+1)*f_{nl}

        # R_{nl}(r) = exp(x/2) P_{nl}(x), P(x) are eigfuncs
        orbs_R = np.exp(-xgrid / 2.0) * orbs.eigfuncs
        orbs_R_sq = orbs_R ** 2.0

        # sum over the (l,n) dimensions of the orbitals to get the density
        bound["rho"] = np.einsum("ijk,ijkl->il", orbs.occnums, orbs_R_sq)

        # compute the number of unbound electrons
        bound["N"] = np.sum(orbs.occnums, axis=(1, 2))

        return bound

    @staticmethod
    def construct_rho_unbound(orbs):
        """
        Constructs the bound part of the density
        """

        rho_unbound = np.zeros((config.spindims, config.grid_params["ngrid"]))
        N_unbound = np.zeros((config.spindims))

        # so far only the ideal approximation is implemented
        if config.unbound == "ideal":

            # unbound density is constant
            for i in range(config.spindims):
                if config.nele[i] > 1e-5:
                    prefac = (2.0 / config.spindims) * 1.0 / (sqrt(2) * pi ** 2)
                    n_ub = prefac * mathtools.fd_int_complete(
                        config.mu[i], config.beta, 1.0
                    )
                    rho_unbound[i] = n_ub
                    N_unbound[i] = n_ub * config.sph_vol
                else:
                    N_unbound[i] = 0.0
                    rho_unbound[i] = 0.0

        unbound = {"rho": rho_unbound, "N": N_unbound}

        return unbound


class Potential:
    """
    The potential object has the following attributes:
    - v_s   (np array)    : KS potential
    - v_en  (np array)    : electron-nuclear potential
    - v_ha   (np array)    : Hartree potential
    - v_xc  (np array)    : xc-potential
    """

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
        if np.all(self._v_s == 0.0):
            self._v_s = self.v_en + self.v_ha + self.v_xc["xc"]
        return self._v_s

    @property
    def v_en(self):
        if np.all(self._v_en == 0.0):
            self._v_en = self.calc_v_en(self._xgrid)
        return self._v_en

    @property
    def v_ha(self):
        if np.all(self._v_ha == 0.0):
            self._v_ha = self.calc_v_ha(self._density, self._xgrid)
        return self._v_ha

    @property
    def v_xc(self):
        if np.all(self._v_xc["xc"] == 0.0):
            self._v_xc = xc.v_xc(self._density, self._xgrid, config.xfunc, config.cfunc)
        return self._v_xc

    @staticmethod
    def calc_v_en(xgrid):
        """
        Constructs the electron-nuclear potential
        v_en (x) = -Z * exp(-x)
        """

        v_en = -config.Z * np.exp(-xgrid)

        return v_en

    @staticmethod
    def calc_v_ha(density, xgrid):
        """
        Constructs the Hartree potential
        On the r-grid:
        v_ha(r) = 4*pi* \int_0^rn_s dr' n(r') r'^2 / max(r,r')
        On the x-grid:
        v_ha(x) = 4*pi* { exp(-x) \int_x0^x dx' n(x') exp(3x')
                         - \int_x^log(r_s) dx' n(x') exp(2x') }

        Inputs:
        - density (np array)  : density
        """

        # initialize v_ha
        v_ha = np.zeros_like(xgrid)

        # construct the total (sum over spins) density
        rho = np.sum(density, axis=0)

        # loop over the x-grid
        # this may be a bottleneck...
        for i, x0 in enumerate(xgrid):

            # set up 'upper' and 'lower' parts of the xgrid (x<=x0; x>x0)
            x_u = xgrid[np.where(x0 <= xgrid)]
            x_l = xgrid[np.where(x0 > xgrid)]

            # likewise for the density
            rho_u = rho[np.where(x0 <= xgrid)]
            rho_l = rho[np.where(x0 > xgrid)]

            # now compute the hartree potential
            int_l = exp(-x0) * np.trapz(rho_l * np.exp(3.0 * x_l), x_l)
            int_u = np.trapz(rho_u * np.exp(2 * x_u), x_u)

            # total hartree potential is sum over integrals
            v_ha[i] = 4.0 * pi * (int_l + int_u)

        return v_ha


class Energy:
    """
    Holds the energy which includes the following attributes
    Note:
    F = E - TS
    E = E_kin + E_en + E_ha + E_xc

    - F_tot (dict np arrays)    : the total free energy F
      contains: F, E, S
    - E_kin (dict np arrays)    : the kinetic energy E_kin
      contains: bound, unbound
    - E_en (np array)           : the electron-nuclear energy E_en
    - E_ha (np array)           : the Hartree energy E_ha
    - E_xc (dict np arrays)     : the xc energy
      contains : xc, x, c

    Inputs:
    - orbs (object)          : the orbitals object
    - dens  (object)         : the density object
    - pot (object)           : the potential object
    """

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
        if self._F_tot == 0.0:
            self._F_tot = self.E_tot - config.temp * self.entropy["tot"]
        return self._F_tot

    @property
    def E_tot(self):
        if self._E_tot == 0.0:
            self._E_tot = self.E_kin["tot"] + self.E_en + self.E_ha + self.E_xc["xc"]
        return self._E_tot

    @property
    def entropy(self):
        if self._entropy["tot"] == 0.0:
            self._entropy = self.calc_entropy(self._orbs)
        return self._entropy

    @property
    def E_kin(self):
        if self._E_kin["tot"] == 0.0:
            self._E_kin = self.calc_E_kin(self._orbs)
        return self._E_kin

    @property
    def E_en(self):
        if self._E_en == 0.0:
            self._E_en = self.calc_E_en(self._dens, self._xgrid)
        return self._E_en

    @property
    def E_ha(self):
        if self._E_ha == 0.0:
            self._E_ha = self.calc_E_ha(self._dens, self._xgrid)
        return self._E_ha

    @property
    def E_xc(self):
        if self._E_xc["xc"] == 0.0:
            # self._E_xc = xc.XCEnergy(self._dens, config.xfunc, config.cfunc).E_xc
            self._E_xc = xc.E_xc(self._dens, self._xgrid, config.xfunc, config.cfunc)
        return self._E_xc

    def calc_E_kin(self, orbs):
        """
        Kinetic energy computation is in general different
        for bound and unbound components.
        This wrapper calls the respective functions

        Inputs:
        - orbs (object)                : the orbitals object
        Returns:
        - E_kin (dict of np arrays)    : kinetic energy
          E_kin = {"bound" : E_b, "unbound" : E_ub}
        """

        E_kin = {}

        # bound part
        E_kin["bound"] = self.calc_E_kin_bound(orbs)

        # unbound part
        E_kin["unbound"] = self.calc_E_kin_unbound(orbs)

        # total
        E_kin["tot"] = E_kin["bound"] + E_kin["unbound"]

        return E_kin

    @staticmethod
    def calc_E_kin_bound(orbs):
        """
        Computes the kinetic energy contribution from the bound electrons

        Inputs:
        - orbs (object)        : the orbitals object
        Returns:
        - E_kin_bound (float)  : kinetic energy
        """

        # compute the grad^2 component
        grad2_orbs = mathtools.laplace(orbs.eigfuncs, orbs._xgrid)

        # compute the (l+1/2)^2 component
        l_arr = np.array([(l + 0.5) ** 2.0 for l in range(config.lmax)])
        lhalf_orbs = np.einsum("j,ijkl->ijkl", l_arr, orbs.eigfuncs)

        # add together and multiply by eigfuncs*exp(-3x)
        prefac = np.exp(-3.0 * orbs._xgrid) * orbs.eigfuncs
        kin_orbs = prefac * (grad2_orbs - lhalf_orbs)

        # multiply and sum over occupation numbers
        e_kin_dens = np.einsum("ijk,ijkl->l", orbs.occnums, kin_orbs)

        # integrate over sphere
        E_kin_bound = -0.5 * mathtools.int_sphere(e_kin_dens, orbs._xgrid)

        return E_kin_bound

    @staticmethod
    def calc_E_kin_unbound(orbs):
        """
        Computes the unbound contribution to the kinetic energy

        Inputs:
        - orbs (object)          : the orbitals object
        Returns:
        - E_kin_unbound (float)  : the unbound K.E.
        """

        # currently only ideal treatment supported
        if config.unbound == "ideal":
            E_kin_unbound = 0.0  # initialize
            for i in range(config.spindims):
                prefac = config.nele[i] * config.sph_vol / (sqrt(2) * pi ** 2)
                E_kin_unbound += prefac * mathtools.fd_int_complete(
                    config.mu[i], config.beta, 3.0
                )

        return E_kin_unbound

    def calc_entropy(self, orbs):
        """
        Entropy is in general computed differently for bound / unbound orbitals
        This wrapper calls the respective bound and unbound components

        Inputs
        - orbs (object)      : the orbitals object
        """

        S = {}

        # bound part
        S["bound"] = self.calc_S_bound(orbs)

        # unbound part
        S["unbound"] = self.calc_S_unbound(orbs)

        # total
        S["tot"] = S["bound"] + S["unbound"]

        return S

    @staticmethod
    def calc_S_bound(orbs):
        """
        Computes the contribution of the bound states to the entropy
        S_bo = -\sum_{s,l,n} (2l+1) [ f_{nls} log(f_{nls})
                                     + (1-f_{nls}) (log(1-f_{nls}) ]

        Inputs:
        - orbs (object)      : the orbitals object
        Returns
        - S_bound (float)    : the bound contribution to entropy
        """

        # the occupation numbers are stored as f'_{nl}=f_{nl}*(2l+1)
        # we first need to map them back to their 'pure' form f_{nl}
        lbound_inv = np.zeros_like(orbs.lbound)
        for l in range(config.lmax):
            lbound_inv[:, l] = (config.spindims / 2.0) * np.where(
                orbs.eigvals[:, l] < 0, 1.0 / (2 * l + 1.0), 0.0
            )

        # pure occupation numbers (with zeros replaced by finite values)
        occnums_pu = lbound_inv * orbs.occnums
        occnums_mod1 = np.where(occnums_pu > 1e-5, occnums_pu, 0.5)
        occnums_mod2 = np.where(occnums_pu < 1.0 - 1e-5, occnums_pu, 0.5)

        # now compute the terms in the square bracket
        term1 = occnums_pu * np.log(occnums_mod1)
        term2 = (1.0 - occnums_pu) * np.log(1.0 - occnums_mod2)

        # multiply by (2l+1) factor
        g_nls = orbs.lbound * (term1 + term2)

        # sum over all quantum numbers to get the total entropy
        S_bound = np.sum(g_nls)

        return S_bound

    @staticmethod
    def calc_S_unbound(orbs):
        """
        Computes the unbound contribution to the entropy

        Inputs:
        - orbs (object)          : the orbitals object
        Returns:
        - S_unbound (float)      : the unbound entropy
        """

        # currently only ideal treatment supported
        if config.unbound == "ideal":
            S_unbound = 0.0  # initialize
            for i in range(config.spindims):
                if config.nele[i] > 1e-5:
                    prefac = (
                        (2.0 / config.spindims) * config.sph_vol / (sqrt(2) * pi ** 2)
                    )

                    S_unbound -= prefac * mathtools.fd_int_complete(
                        config.mu[i], config.beta, 1.0
                    )
                else:
                    S_unbound += 0.0

        return S_unbound

    @staticmethod
    def calc_E_en(density, xgrid):
        """
        Computes the electron-nuclear energy
        E_en = \int dr v_en(r) n(r)

        Inputs:
        - density (np array) : density
        """

        # sum the density over the spin axes to get the total density
        dens_tot = np.sum(density, axis=0)

        # compute the integral
        v_en = Potential.calc_v_en(xgrid)
        E_en = mathtools.int_sphere(dens_tot * v_en, xgrid)

        return E_en

    @staticmethod
    def calc_E_ha(density, xgrid):
        """
        Computes the Hartree energy
        E_ha = 1/2 \int dr \int dr' n(r)n(r')/|r-r'|
        Uses the pre-computed hartree potential
        E_ha = 1/2 /int dr n(r) v_ha(r)

        Inputs:
        - density (np array)    : the density object
        - pot  (object)         : the potential object
        Returns:
        - E_ha (float)       : the hartree energy
        """

        # sum density over spins to get total density
        dens_tot = np.sum(density, axis=0)

        # compute the integral
        v_ha = Potential.calc_v_ha(density, xgrid)
        E_ha = 0.5 * mathtools.int_sphere(dens_tot * v_ha, xgrid)

        return E_ha
