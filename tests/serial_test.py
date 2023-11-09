#!/usr/bin/env python3
"""Test serial execution of code."""

from atoMEC import Atom, models, config
import pytest
import numpy as np


# expected values and tolerance
dense_expected = -0.5620111902408967
coarse_expected = -0.56341460946266
accuracy = 1e-3


class TestSerial:
    """
    Test class for running calculations in serial.

    Checks the free energy for an SCF calculation is given by the expected value,
    and also uses a very coarse and very dense grid.
    """

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (250, coarse_expected),
            (10001, dense_expected),
        ],
    )
    def test_serial(self, test_input, expected):
        """Check free energy is given by expected value."""
        # serial
        config.numcores = 0

        assert np.isclose(self._run(test_input), expected, atol=accuracy)

    @staticmethod
    def _run(ngrid):
        """
        Run an SCF calculation in serial for a Hydrogen atom.

        Parameters
        ----------
        ngrid : int
            the number of grid points

        Returns
        -------
        F_tot : float
            the total free energy
        """
        # set up the atom and model
        Al_at = Atom("H", 0.075, radius=4.0, density=0.042)
        model = models.ISModel(Al_at, unbound="quantum", bc="bands")

        # run the SCF calculation
        output = model.CalcEnergy(
            2,
            2,
            scf_params={"maxscf": 1, "mixfrac": 0.7},
            band_params={"nkpts": 30},
            grid_params={"ngrid": ngrid, "ngrid_coarse": 300},
            grid_type="log",
        )

        # extract the total free energy
        F_tot = output["energy"].F_tot
        return F_tot


if __name__ == "__main__":
    config.numcores = 0
    dense = TestSerial._run(10001)
    coarse = TestSerial._run(250)
    print("dense_expected =", dense)
    print("coarse_expected =", coarse)
