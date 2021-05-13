"""
Module which computes time-independent properties from the average-atom setup
"""

# standard packages

# external packages
import numpy as np

# internal modules
import config
import numerov


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

        self.eigfuncs = eigfuncs
        self.eigvals = eigvals
