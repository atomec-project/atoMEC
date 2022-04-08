#!/usr/bin/env python3
"""
Boundary conditions test

Runs an SCF calculation with the three possible boundary conditions,
and checks the total free energy.
"""

from atoMEC import Atom, models, config, staticKS
import pytest
import numpy as np


# expected values and tolerance
ideal_expected = -213.124985
quantum_expected = -208.999350
accuracy = 1e-3


class TestEnergyAlt:
    """
    Test class for different boundary conditions.

    Checks the free energy for an SCF calculation is given by the expected value.
    """

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            ("ideal", ideal_expected),
            ("quantum", quantum_expected),
        ],
    )
    def test_energy_alt(self, test_input, expected):

        # parallel
        config.numcores = -1

        assert np.isclose(self._run(test_input), expected, atol=accuracy)

    @staticmethod
    def _run(unbound):
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
        Mg_at = Atom("Mg", 0.3, radius=1, write_info=False, units_radius="Bohr")
        model = models.ISModel(
            Mg_at,
            unbound=unbound,
            write_info=False,
            bc="neumann",
        )

        # run the SCF calculation
        output = model.CalcEnergy(
            10,
            5,
            scf_params={"maxscf": 6, "mixfrac": 0.3},
            write_info=True,
        )

        energy_alt = staticKS.EnergyAlt(
            output["orbitals"], output["density"], output["potential"]
        )
        F_tot = energy_alt.F_tot
        return F_tot


if __name__ == "__main__":
    print("ideal energy = ", TestEnergyAlt._run("ideal"))
    print("quantum energy = ", TestEnergyAlt._run("quantum"))
