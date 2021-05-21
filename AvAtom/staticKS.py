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


class Orbitals:
    """
    The orbitals object has the following attributes:
    - eigfuncs (numpy array)    :   the orbitals defined on the numerical grid
    - eigvals  (numpy array)    :   KS orbital eigenvalues
    - occnums  (numpy array)    :   KS orbital occupation numbers
    """

    def __init__(self):
        """
        Initializes the orbital attributes to empty numpy arrays
        """

        self._eigfuncs = None
        self._eigvals = None
        self._occnums = None
        self._lbound = None

    @property
    def eigvals(self):
        if self._eigvals is None:
            raise Exception("Eigenvalues have not been initialized")
        return self._eigvals

    @property
    def eigfuncs(self):
        if self._eigfuncs is None:
            raise Exception("Eigenfunctions have not been initialized")
        return self._eigfuncs

    @property
    def occnums(self):
        if self._occnums is None:
            raise Exception("Occnums have not been initialized")
        return self._occnums

    @property
    def lbound(self):
        if self._lbound is None:
            raise Exception("lbound has not been initialized")
        return self._lbound

    def SCF_init(self, atom):
        """
        Initializes the KS orbitals before an SCF cycle using the bare clmb potential
        """

        # compute the bare coulomb potential
        # v_en = -atom.at_chrg * np.exp(-config.xgrid)

        v_en = np.zeros((config.spindims, config.grid_params["ngrid"]))

        for i in range(config.spindims):
            v_en[i] = -config.Z * np.exp(-config.xgrid)

        # solve the KS equations with the bare coulomb potential
        self._eigfuncs, self._eigvals = numerov.matrix_solve(v_en, config.xgrid)

        # compute the lbound array
        self._lbound = self.make_lbound(self.eigvals)

        # initial guess for the chemical potential
        config.mu = np.zeros((config.spindims))

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
            lbound_mat[:, l] = np.where(eigvals[:, l] < 0, 2 * l + 1.0, 0.0)

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

    def __init__(self):

        self.rho_tot = np.zeros((config.spindims, config.grid_params["ngrid"]))
        self.rho_bound = np.zeros((config.spindims, config.grid_params["ngrid"]))
        self.rho_unbound = np.zeros_like(self.rho_bound)
        self.N_bound = np.zeros((config.spindims))
        self.N_unbound = np.zeros_like(self.N_bound)

    def construct(self, orbs):
        """
        Constructs the density

        Inputs:
        - orbs (object)    : the orbitals object
        """

        # construct the bound part of the density
        self.rho_bound, self.N_bound = self.construct_rho_bound(orbs)
        # construct the unbound part
        self.rho_unbound, self.N_unbound = self.construct_rho_unbound()

        # sum to get the total density
        self.rho_tot = self.rho_bound + self.rho_unbound

    @staticmethod
    def construct_rho_bound(orbs):
        """
        Constructs the bound part of the density

        Inputs:
        - orbs (object)    : the orbitals object
        """

        # first of all construct the density
        # rho_b(r) = \sum_{n,l} (2l+1) f_{nl} |R_{nl}(r)|^2
        # occnums in AvAtom are defined as (2l+1)*f_{nl}

        # R_{nl}(r) = exp(x/2) P_{nl}(x), P(x) are eigfuncs
        orbs_R = np.exp(-config.xgrid / 2.0) * orbs.eigfuncs
        orbs_R_sq = orbs_R ** 2.0

        # sum over the (l,n) dimensions of the orbitals to get the density
        rho_bound = np.einsum("ijk,ijkl->il", orbs.occnums, orbs_R_sq)

        # compute the number of unbound electrons
        N_bound = np.sum(orbs.occnums, axis=(1, 2))

        return rho_bound, N_bound

    @staticmethod
    def construct_rho_unbound():
        """
        Constructs the bound part of the density
        """

        rho_unbound = np.zeros((config.spindims, config.grid_params["ngrid"]))
        N_unbound = np.zeros((config.spindims))

        # so far only the ideal approximation is implemented
        if config.unbound == "ideal":

            # unbound density is constant
            for i in range(config.spindims):
                prefac = config.nele[i] / (sqrt(2) * pi ** 2)
                n_ub = prefac * mathtools.fd_int_complete(
                    config.mu[i], config.beta, 0.5
                )
                rho_unbound[i] = n_ub
                N_unbound[i] = n_ub * config.sph_vol

        return rho_unbound, N_unbound

    def write_to_file(self):
        # this routine should probably be moved to a more appropriate place
        """
        Writes the density (on the r-grid) to file
        """

        fname = "density.csv"

        if config.spinpol == True:
            headstr = (
                "r"
                + 7 * " "
                + "n^up_b"
                + 4 * " "
                + "n^up_ub"
                + 3 * " "
                + "n^dw_b"
                + 4 * " "
                + "n^dw_ub"
                + 3 * " "
            )
            data = np.column_stack(
                [
                    config.rgrid,
                    self.rho_bound[0],
                    self.rho_unbound[0],
                    self.rho_bound[1],
                    self.rho_unbound[1],
                ]
            )
        else:
            headstr = "r" + 8 * " " + "n_b" + 6 * " " + "n^_ub" + 3 * " "
            data = np.column_stack(
                [config.rgrid, self.rho_bound[0], self.rho_unbound[0]]
            )

        np.savetxt(fname, data, fmt="%8.3e", header=headstr)


class Potential:
    """
    The potential object has the following attributes:
    - v_s   (np array)    : KS potential
    - v_en  (np array)    : electron-nuclear potential
    - v_ha   (np array)    : Hartree potential
    - v_xc  (np array)    : xc-potential
    """

    def __init__(self):
        self.v_s = np.zeros((config.spindims, config.grid_params["ngrid"]))
        self.v_en = np.zeros((1, config.grid_params["ngrid"]))
        self.v_ha = np.zeros((1, config.grid_params["ngrid"]))
        self.v_xc = np.zeros((config.spindims, config.grid_params["ngrid"]))
        self.v_x = np.zeros_like(self.v_xc)
        self.v_c = np.zeros_like(self.v_x)

    def construct(self, density):
        """
        Constructs the components of the KS potential

        Inputs:
        - density (object)   : the density object
        """

        # construct the electron-nuclear potential
        self.v_en[0] = -config.Z * np.exp(-config.xgrid)

        # construct the Hartree potential
        self.v_ha[0] = self.calc_v_ha(density)

        # extract the xc components from the potential object
        pot_xc = xc.XCPotential(density, config.xfunc, config.cfunc)
        self.v_x = pot_xc.vx
        self.v_c = pot_xc.vc
        self.v_xc = pot_xc.vxc

        # sum the potentials to get the total KS potential
        self.v_s = self.v_en + self.v_ha + self.v_xc

    @staticmethod
    def calc_v_ha(density):
        """
        Constructs the Hartree potential
        On the r-grid:
        v_ha(r) = 4*pi* \int_0^r_s dr' n(r') r'^2 / max(r,r')
        On the x-grid:
        v_ha(x) = 4*pi* { exp(-x) \int_x0^x dx' n(x') exp(3x')
                         - \int_x^log(r_s) dx' n(x') exp(2x') }

        Inputs:
        - density (object)  : density object
        """

        # rename xgrid for ease
        xgrid = config.xgrid

        # construct the total (sum over spins) density
        rho = np.sum(density.rho_tot, axis=0)

        # initialize v_hartree potential
        v_ha = np.zeros((config.grid_params["ngrid"]))

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

    def write_to_file(self):
        # this routine should probably be moved to a more appropriate place
        """
        Writes the potential (on the r-grid) to file
        """

        fname = "potential.csv"

        if config.spinpol == True:
            headstr = (
                "r"
                + 7 * " "
                + "v_en"
                + 4 * " "
                + "v_ha"
                + 3 * " "
                + "v^up_xc"
                + 4 * " "
                + "v^dw_xc"
                + 3 * " "
            )
            data = np.column_stack(
                [
                    config.rgrid,
                    self.v_en[0],
                    self.v_ha[0],
                    self.v_xc[0],
                    self.v_xc[1],
                ]
            )
        else:
            headstr = "r" + 8 * " " + "v_en" + 6 * " " + "v_ha" + 3 * " "
            data = np.column_stack(
                [config.rgrid, self.v_en[0], self.v_ha[0], self.v_xc[0]]
            )

        np.savetxt(fname, data, fmt="%8.3e", header=headstr)
