#!/usr/bin/env python3
"""
Boundary conditions test

Runs an SCF calculation with the three possible boundary conditions,
and checks the total free energy.
"""

from atoMEC import Atom, models, config
import pytest
import numpy as np

# parallel
config.numcores = -1

# expected values and tolerance
pressure_expected = 0.013395564
accuracy = 1e-7


class Test_pressure:
    """
    Test class for different boundary conditions.

    Checks the free energy for an SCF calculation is given by the expected value.
    """

    def test_bcs(self):

        assert np.isclose(self._run(), pressure_expected, atol=accuracy)

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
        Li_at = Atom("Li", 10, radius=2.5, units_temp="eV", write_info=False)
        model = models.ISModel(Li_at, unbound="quantum", write_info=False)

        # run the SCF calculation
        output = model.CalcEnergy(
            3,
            3,
            scf_params={"maxscf": 3},
            write_info=False,
        )

        pressure = model.CalcPressure(Li_at, output)

        return pressure


if __name__ == "__main__":
    print(Test_pressure._run())
