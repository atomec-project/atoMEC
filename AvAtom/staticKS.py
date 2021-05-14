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
        self.eigfuncs = np.zeros((1))
        self.eigvals = np.zeros((1))
        self.occnums = np.zeros((1))

    def SCF_init(self, atom):
        """
        Initializes the KS orbitals before an SCF cycle using the bare clmb potential
        """

        # compute the bare coulomb potential
        v_en_up = -atom.at_chrg * np.exp(-config.xgrid)
        v_en = [v_en_up, v_en_up]

        # solve the KS equations with the bare coulomb potential
        eigfuncs, eigvals = numerov.matrix_solve(v_en, config.xgrid)

        # redefine lmax
        config.lmax = len(eigvals[0])

        # keep only the real parts
        self.eigfuncs = eigfuncs
        self.eigvals = eigvals

        # initial guess for the chemical potential
        config.mu = [0.0, 0.0]

    def occupy(self):
        """
        Occupies the orbitals according to Fermi Dirac statistics.
        The chemical potential is calculated to satisfy charge neutrality
        within the Voronoi sphere
        """

        # compute the chemical potential using the eigenvalues
        config.mu = mathtools.chem_pot(self.eigvals)
        print(config.mu)

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
                for l in range(config.lmax):
                    occnums[i][l] = mathtools.fermi_dirac(
                        self.eigvals[i][l], mu[i], config.beta
                    )

        else:
            for l in range(config.lmax):
                occnums[0][l] = mathtools.fermi_dirac(
                    self.eigvals[0][l], mu[0], config.beta
                )
            occnums[1] = occnums[0]

        return occnums
