#!/usr/bin/env python3
"""
Boundary conditions test.

Runs an SCF calculation with the three possible boundary conditions,
and checks the total free energy.
"""

from atoMEC import Atom, models, config
import pytest
import numpy as np


# expected values and tolerance
dirichlet_expected = -14.02657778662206
neumann_expected = -15.112195227107968
bands_expected = -14.68311960864459

accuracy = 1e-3


class TestBcs:
    """
    Test class for different boundary conditions.

    Checks the free energy for an SCF calculation is given by the expected value.
    """

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            ("dirichlet", dirichlet_expected),
            ("neumann", neumann_expected),
            ("bands", bands_expected),
        ],
    )
    def test_bcs(self, test_input, expected):
        """Check free energy for the three boundary conditions."""
        # parallel
        config.numcores = -1

        assert np.isclose(self._run(test_input), expected, atol=accuracy)

    @staticmethod
    def _run(bc):
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
        Be_at = Atom("Be", 0.1, radius=3.0)
        model = models.ISModel(Be_at, bc=bc, unbound="quantum")

        if bc == "bands":
            nkpts = 50
        else:
            nkpts = 1

        # run the SCF calculation
        output = model.CalcEnergy(
            4,
            4,
            scf_params={"maxscf": 5, "mixfrac": 0.3},
            grid_params={"ngrid": 1000},
            band_params={"nkpts": nkpts},
        )

        # extract the total free energy
        F_tot = output["energy"].F_tot
        return F_tot


if __name__ == "__main__":
    config.numcores = -1
    dirichlet = TestBcs._run("dirichlet")
    neumann = TestBcs._run("neumann")
    bands = TestBcs._run("bands")
    print("dirichlet_expected =", dirichlet)
    print("neumann_expected =", neumann)
    print("bands_expected =", bands)
