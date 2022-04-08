#!/usr/bin/env python3
"""
Boundary conditions test

Runs an SCF calculation with the three possible boundary conditions,
and checks the total free energy.
"""

from atoMEC import Atom, models, config
from atoMEC.postprocess import localization
import pytest
from pytest_lazyfixture import lazy_fixture
import functools
import numpy as np


# expected values and tolerance
orbitals_expected = 1.1035
density_expected = 2.1266
IPR_expected = 78.2132
epdc_orbs_expected = 0.7096
epdc_dens_expected = 0.7096
accuracy = 0.001


class Test_ELF:
    """
    Test class for different boundary conditions.

    Checks the free energy for an SCF calculation is given by the expected value.
    """

    #    @pytest.fixture(autouse=True)
    #    def _setup(self):
    # parallel
    #        config.numcores = -1

    #        self.SCF_output = self._run_SCF()

    @pytest.fixture(autouse=True, scope="class")
    def SCF_nospin_output(self):
        config.numcores = -1
        return self._run_SCF(False)

    @pytest.fixture(autouse=True, scope="class")
    def SCF_spin_output(self):
        config.numcores = -1
        return self._run_SCF(True)

    @pytest.mark.parametrize(
        "SCF_input,method,expected",
        [
            (lazy_fixture("SCF_spin_output"), "orbitals", orbitals_expected),
            (lazy_fixture("SCF_nospin_output"), "density", density_expected),
        ],
    )
    def test_ELF(self, SCF_input, method, expected):

        assert np.isclose(
            self._run_ELF(SCF_input, method),
            expected,
            atol=accuracy,
        )

    @pytest.mark.parametrize(
        "SCF_input,method,expected",
        [
            (lazy_fixture("SCF_spin_output"), "orbitals", epdc_orbs_expected),
            (lazy_fixture("SCF_spin_output"), "density", epdc_dens_expected),
        ],
    )
    def test_epdc(self, SCF_input, method, expected):

        assert np.isclose(
            self._run_epdc(SCF_input, method),
            expected,
            atol=accuracy,
        )

    def test_IPR(self, SCF_nospin_output):

        assert np.isclose(self._run_IPR(SCF_nospin_output), IPR_expected, atol=accuracy)

    @staticmethod
    def _run_SCF(spinpol):
        """
        Run an SCF calculation for an He Atom with unbound = "Ideal"

        Returns
        -------
        F_tot : float
            the total free energy
        """

        # set up the atom and model
        Al_at = Atom("Al", 0.01, radius=10.0, units_temp="eV", write_info=False)
        model = models.ISModel(
            Al_at, unbound="ideal", spinpol=spinpol, write_info=False
        )

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

    @staticmethod
    def _run_epdc(input_SCF, method):

        ELF = localization.ELFTools(
            input_SCF["orbitals"],
            input_SCF["density"],
            method=method,
        )

        epdc = ELF.epdc

        D_0 = (
            (0.3)
            * (3 * np.pi ** 2) ** (2.0 / 3.0)
            * (input_SCF["density"].total) ** (5.0 / 3.0)
        )

        n = 150
        ELF_0 = 1 / (1 + (epdc / D_0) ** 2)[0][n]

        print(ELF_0)
        return ELF_0

    @staticmethod
    def _run_IPR(input_SCF):

        orbitals = input_SCF["orbitals"].eigfuncs
        xgrid = input_SCF["orbitals"]._xgrid

        IPR_mat = localization.calc_IPR_mat(orbitals, xgrid)

        return IPR_mat[0, 0, 0, 0]


if __name__ == "__main__":
    SCF_out = Test_ELF._run_SCF()
    print(Test_ELF._run_ELF(SCF_out, "orbitals"))
    print(Test_ELF._run_ELF(SCF_out, "density"))
    print(Test_ELF._run_epdc(SCF_out))
    print(Test_ELF._run_IPR(SCF_out))
