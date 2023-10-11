#!/usr/bin/env python3
"""Test the total free energy for different spinmag values."""

from atoMEC import Atom, models, config
import pytest
import numpy as np


# expected values and tolerance
singlet_expected = -37.89758629854574
triplet_expected = -37.88133924584101
accuracy = 1e-3


class TestSpin:
    """
    Test class for different boundary conditions.

    Checks the free energy for an SCF calculation is given by the expected value.
    """

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (0, singlet_expected),
            (2, triplet_expected),
        ],
    )
    def test_spinmag(self, test_input, expected):
        """Check free energy for different spinmag values."""
        # parallel
        config.numcores = -1

        assert np.isclose(self._run(test_input), expected, atol=accuracy)

    @staticmethod
    def _run(spinmag):
        """
        Run an SCF calculation for a C Atom with given spinmag.

        Parameters
        ----------
        spinmag : int
            the spin magnetization

        Returns
        -------
        F_tot : float
            the total free energy
        """
        # set up the atom and model
        C_at = Atom("C", 5000, density=1.0, units_temp="K")
        model = models.ISModel(
            C_at,
            spinpol=True,
            spinmag=spinmag,
            unbound="quantum",
            bc="bands",
            xfunc_id="gga_x_pbe",
            cfunc_id="gga_c_pbe",
        )

        # run the SCF calculation
        output = model.CalcEnergy(
            4,
            5,
            scf_params={"maxscf": 4, "mixfrac": 0.3},
            band_params={"nkpts": 50},
            grid_params={"ngrid": 1000, "ngrid_coarse": 300},
        )

        # extract the total free energy
        F_tot = output["energy"].F_tot
        return F_tot


if __name__ == "__main__":
    config.numcores = -1
    singlet = TestSpin._run(0)
    triplet = TestSpin._run(2)
    print("singlet_expected =", singlet)
    print("triplet_expected =", triplet)
