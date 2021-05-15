"""
Module which computes time-independent properties from the average-atom setup
"""

# standard packages

# external packages
import numpy as np

# internal modules
import config
import numerov
import mathtools


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
        self.eigfuncs = np.zeros(
            (2, config.lmax, config.nmax, config.grid_params["ngrid"])
        )
        self.eigvals = np.zeros((2, config.lmax, config.nmax))
        self.occnums = np.zeros((2, config.lmax, config.nmax))
        self.lbound = np.zeros((2, config.lmax, config.nmax))

    def SCF_init(self, atom):
        """
        Initializes the KS orbitals before an SCF cycle using the bare clmb potential
        """

        # compute the bare coulomb potential
        v_en_up = -atom.at_chrg * np.exp(-config.xgrid)
        v_en = [v_en_up, v_en_up]

        # solve the KS equations with the bare coulomb potential
        self.eigfuncs, self.eigvals = numerov.matrix_solve(self, v_en, config.xgrid)

        # compute the lbound array
        self.make_lbound()

        # # keep only the real parts
        # self.eigfuncs = eigfuncs
        # self.eigvals = eigvals

        # initial guess for the chemical potential
        config.mu = [0.0, 0.0]

    def occupy(self):
        """
        Occupies the orbitals according to Fermi Dirac statistics.
        The chemical potential is calculated to satisfy charge neutrality
        within the Voronoi sphere
        """

        # compute the chemical potential using the eigenvalues
        config.mu = mathtools.chem_pot(self)

        # compute the occupation numbers using the chemical potential
        self.occnums = self.calc_occnums(config.mu)

        return True

    def calc_occnums(self, mu):
        """
        Computes the Fermi-Dirac occupations for the eigenvalues
        """

        # initialize the occnums to have the same format
        occnums = self.eigvals

        if config.spinpol == True:
            for i in range(2):
                occnums[i] = self.lbound[i] * mathtools.fermi_dirac(
                    self.eigvals[i], mu[i], config.beta
                )
        else:
            occnums[0] = self.lbound[0] * mathtools.fermi_dirac(
                self.eigvals[0], mu[0], config.beta
            )
            occnums[1] = occnums[0]

        return occnums

    def make_lbound(self):
        """
        Constructs the 'lbound' attribute
        For each spin channel, lbound(l,n)=(2l+1)*Theta(eps_n)
        """

        for l in range(config.lmax):
            self.lbound[:, l] = np.where(self.eigvals[:, l] < 0, 2 * l + 1.0, 0.0)
