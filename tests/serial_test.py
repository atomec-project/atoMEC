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
dense_expected = -0.562437
coarse_expected = -0.567817
accuracy = 1e-3


class Test_serial:
    """
    Test class for different boundary conditions.

    Checks the free energy for an SCF calculation is given by the expected value.
    """

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (400, coarse_expected),
            (5001, dense_expected),
        ],
    )
    def test_serial(self, test_input, expected):

        # serial
        config.numcores = 0

        assert np.isclose(self._run(test_input), expected, atol=accuracy)

    @staticmethod
    def _run(ngrid):
        """
        Run an SCF calculation for an He Atom with unbound = "Ideal"

        Returns
        -------
        F_tot : float
            the total free energy
        """

        # set up the atom and model
        Al_at = Atom("H", 0.075, radius=4.0, density=0.042)
        model = models.ISModel(Al_at, unbound="quantum", write_info=False, bc="bands")

        # run the SCF calculation
        output = model.CalcEnergy(
            2,
            2,
            scf_params={"maxscf": 1, "mixfrac": 0.7},
            band_params={"nkpts": 30},
            grid_params={"ngrid": ngrid},
            write_info=True,
        )

        # extract the total free energy
        F_tot = output["energy"].F_tot
        print(F_tot)
        return F_tot


if __name__ == "__main__":
    print(Test_serial._run())
