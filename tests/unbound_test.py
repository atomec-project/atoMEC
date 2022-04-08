#!/usr/bin/env python3
"""
Boundary conditions test

Runs an SCF calculation with the three possible boundary conditions,
and checks the total free energy.
"""

from atoMEC import Atom, models, config
import pytest
import numpy as np


# expected values and tolerance
unbound_expected = -10.091898
accuracy = 1e-3


class Test_unbound:
    """
    Test class for different boundary conditions.

    Checks the free energy for an SCF calculation is given by the expected value.
    """

    def test_bcs(self):

        # parallel
        config.numcores = 1

        assert np.isclose(self._run(), unbound_expected, atol=accuracy)

    @staticmethod
    def _run():
        """
        Run an SCF calculation for an He Atom with unbound = "Ideal"

        Returns
        -------
        F_tot : float
            the total free energy
        """

        # set up the atom and model
        Be_at = Atom(
            "He",
            25,
            radius=3.17506,
            units_temp="eV",
            units_radius="angstrom",
            write_info=False,
        )
        model = models.ISModel(Be_at, unbound="ideal", write_info=False)

        # run the SCF calculation
        output = model.CalcEnergy(
            3, 3, scf_params={"maxscf": 3}, write_info=False, force_bound=[[0, 0, 0]]
        )

        # extract the total free energy
        F_tot = output["energy"].F_tot
        return F_tot


if __name__ == "__main__":
    print(Test_unbound._run())
