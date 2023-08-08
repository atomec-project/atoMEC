#!/usr/bin/env python3
"""
Test for the EnergyAlt class.

Runs a calculation with ideal and quantum unbound treatments, then constructs the
EnergyAlt object (alternative energy constructor) and checks the total free energy.
"""

from atoMEC import Atom, models, config, staticKS
import pytest
import numpy as np


# expected values and tolerance
ideal_expected = -214.33339222837958
quantum_expected = -207.98637746900067
accuracy = 1e-3


class TestEnergyAlt:
    """
    Test class for EnergyAlt object.

    Checks the EnergyAlt free energy is given by the expected value.
    """

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            ("ideal", ideal_expected),
            ("quantum", quantum_expected),
        ],
    )
    def test_energy_alt(self, test_input, expected):
        """Check the EnergyAlt total free energy."""
        # parallel
        config.numcores = -1

        assert np.isclose(self._run(test_input), expected, atol=accuracy)

    @staticmethod
    def _run(unbound):
        """
        Run an SCF calculation for a Be Atom with given boundary condition.

        Parameters
        ----------
        unbound : str
            unbound electron treatment

        Returns
        -------
        F_tot : float
            the total free energy
        """
        # set up the atom and model
        Mg_at = Atom("Mg", 0.3, radius=1, units_radius="Bohr")
        model = models.ISModel(
            Mg_at,
            unbound=unbound,
            bc="neumann",
        )

        # run the SCF calculation
        output = model.CalcEnergy(
            10, 5, scf_params={"maxscf": 6, "mixfrac": 0.3}, grid_params={"ngrid": 1000}
        )

        # construct the EnergyAlt object
        energy_alt = staticKS.EnergyAlt(
            output["orbitals"], output["density"], output["potential"]
        )

        F_tot = (
            energy_alt.E_kin["tot"]
            + energy_alt.E_ha
            + energy_alt.E_en
            + energy_alt.E_xc["xc"]
            - Mg_at.temp * energy_alt.entropy["tot"]
        )

        F_tot = 0.5 * (F_tot + energy_alt.F_tot)

        return F_tot


if __name__ == "__main__":
    config.numcores = -1
    ideal = TestEnergyAlt._run("ideal")
    quantum = TestEnergyAlt._run("quantum")
    print("ideal_expected =", ideal)
    print("quantum_expected =", quantum)
