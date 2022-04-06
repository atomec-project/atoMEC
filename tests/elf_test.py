#!/usr/bin/env python3
"""
Boundary conditions test

Runs an SCF calculation with the three possible boundary conditions,
and checks the total free energy.
"""

from atoMEC import Atom, models, config
from atoMEC.postprocess import localization
import pytest
import numpy as np

# parallel
config.numcores = -1

# expected values and tolerance
orbitals_expected = 2.2047
density_expected = 2.1266
accuracy = 0.001


class Test_ELF:
    """
    Test class for different boundary conditions.

    Checks the free energy for an SCF calculation is given by the expected value.
    """

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            ("orbitals", orbitals_expected),
            ("density", density_expected),
        ],
    )
    def test_ELF(self, test_input, expected):

        SCF_output = self._run_SCF()

        assert np.isclose(
            self._run_ELF(SCF_output, test_input), expected, atol=accuracy
        )

    @staticmethod
    def _run_SCF():
        """
        Run an SCF calculation for an He Atom with unbound = "Ideal"

        Returns
        -------
        F_tot : float
            the total free energy
        """

        # set up the atom and model
        Al_at = Atom("Al", 0.01, radius=10.0, units_temp="eV", write_info=False)
        model = models.ISModel(Al_at, unbound="ideal", write_info=False)

        # run the SCF calculation
        output = model.CalcEnergy(
            3,
            3,
            scf_params={"maxscf": 10, "mixfrac": 0.5},
            write_info=False,
        )

        return output

    @staticmethod
    def _run_ELF(input_SCF, method):

        ELF = localization.ELFTools(
            input_SCF["orbitals"], input_SCF["density"], method=method
        )

        N_0 = ELF.N_shell[0][0]

        return N_0


if __name__ == "__main__":
    SCF_out = Test_ELF._run_SCF()
    print(Test_ELF._run_ELF(SCF_out, "orbitals"))
    print(Test_ELF._run_ELF(SCF_out, "density"))
