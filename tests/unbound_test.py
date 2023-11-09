#!/usr/bin/env python3
"""Test free energy for ideal treatment of unbound electrons."""
from atoMEC import Atom, models, config
import numpy as np


# expected values and tolerance
unbound_expected = -10.085061150480994
accuracy = 1e-3


class TestUnbound:
    """
    Test class for ideal treatment of unbound electrons.

    Checks the free energy for an SCF calculation is given by the expected value.
    Also uses the force_bound parameter.
    """

    def test_unbound(self):
        """Check free energy for SCF calc with unbound=ideal."""
        # test "parallel" with 1 core
        config.numcores = 1

        assert np.isclose(self._run(), unbound_expected, atol=accuracy)

    @staticmethod
    def _run():
        """
        Run an SCF calculation for a Be atom with unbound="ideal".

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
        )
        model = models.ISModel(Be_at, unbound="ideal", bc="dirichlet")

        # run the SCF calculation
        output = model.CalcEnergy(
            3,
            3,
            scf_params={"maxscf": 3, "mixfrac": 0.3},
            grid_params={"ngrid": 1000, "ngrid_coarse": 300},
            force_bound=[[0, 0, 0]],
        )

        # extract the total free energy
        F_tot = output["energy"].F_tot
        return F_tot


if __name__ == "__main__":
    config.numcores = -1
    unbound = TestUnbound._run()
    print("unbound_expected =", unbound)
