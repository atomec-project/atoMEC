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
singlet_expected = -37.6906246
triplet_expected = -37.6732473
accuracy = 10 * config.conv_params["econv"]


class Test_spinmag:
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
    def test_bcs(self, test_input, expected):

        assert np.isclose(self._run(test_input), expected, atol=accuracy)

    @staticmethod
    def _run(spinmag):
        """
        Run an SCF calculation for a Be Atom with given boundary condition.

        Parameters
        ----------
        bc : str
            the boundary condition

        Returns
        -------
        F_tot : float
            the total free energy
        """

        # set up the atom and model
        C_at = Atom("C", 5000, density=1.0, units_temp="K", write_info=True)
        model = models.ISModel(
            C_at,
            spinpol=True,
            spinmag=spinmag,
            unbound="quantum",
            write_info=True,
            bc="bands",
        )

        # run the SCF calculation
        output = model.CalcEnergy(
            4,
            5,
            scf_params={"maxscf": 4},
            band_params={"nkpts": 50},
            write_info=True,
        )

        # extract the total free energy
        F_tot = output["energy"].F_tot
        return F_tot


if __name__ == "__main__":
    print("singlet energy = ", Test_spinmag._run(0))
    print("triplet energy = ", Test_spinmag._run(2))
