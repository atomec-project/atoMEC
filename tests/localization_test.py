#!/usr/bin/env python3
"""
Test localization module.

Checks output from the ELF class and IPR funcs is as expected under various inputs.
"""

from atoMEC import Atom, models, config
from atoMEC.postprocess import localization
import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np


# expected values and tolerance
orbitals_expected = 1.1015
density_expected = 2.1252
IPR_expected = 78.2107
epdc_orbs_expected = 8.7733
epdc_dens_expected = 3.5307
accuracy = 0.001


class TestLocalization:
    """
    Test class for localization module.

    First runs an SCF calculation to generate KS orbitals and density with and
    without spin polarization. Then uses this input to check the N_shell and epdc
    properties in the ELF class, and the IPR matrix.
    """

    @pytest.fixture(autouse=True, scope="class")
    def SCF_nospin_output(self):
        """Run a spin-unpolarized SCF calc and save the output."""
        config.numcores = -1
        return self._run_SCF(False)

    @pytest.fixture(autouse=True, scope="class")
    def SCF_spin_output(self):
        """Run a spin-polarized SCF calc and save the output."""
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
        """Test the ELF through the N_shell property."""
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
        """Test the epdc function."""
        assert np.isclose(
            self._run_epdc(SCF_input, method),
            expected,
            atol=accuracy,
        )

    def test_IPR(self, SCF_nospin_output):
        """Test the IPR function."""
        assert np.isclose(self._run_IPR(SCF_nospin_output), IPR_expected, atol=accuracy)

    @staticmethod
    def _run_SCF(spinpol):
        """
        Run an SCF calculation for an He Atom with unbound = "Ideal".

        Returns
        -------
        output : dict of objects
            the output dictionary containing density, orbitals etc
        """
        # parallel
        config.numcores = -1

        # set up the atom and model
        Al_at = Atom("Al", 0.01, radius=5.0, units_temp="eV")
        model = models.ISModel(Al_at, unbound="quantum", spinpol=spinpol)

        # run the SCF calculation
        output = model.CalcEnergy(
            3,
            2,
            scf_params={"mixfrac": 0.3, "maxscf": 50},
            grid_params={"ngrid": 1000},
        )

        return output

    @staticmethod
    def _run_ELF(input_SCF, method):
        """
        Compute the electron number in the n=1 shell.

        Parameters
        ----------
        input_SCF : dict of objects
            the SCF input
        method : the method used to compute the ELF

        Returns
        -------
        N_0 : float
            number of electrons in the n=1 shell
        """
        ELF = localization.ELFTools(
            input_SCF["orbitals"], input_SCF["density"], method=method
        )

        N_0 = ELF.N_shell[0][0]

        return N_0

    @staticmethod
    def _run_epdc(input_SCF, method):
        """
        Compute the ELF via the epdc function.

        Parameters
        ----------
        input_SCF : dict of objects
            the SCF input
        method : the method used to compute the ELF

        Returns
        -------
        ELF_int : float
            integral of the ELF function (spin-up channel)
        """
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

        ELF_func = 1 / (1 + (epdc / D_0) ** 2)

        ELF_int = np.trapz(ELF_func, ELF._xgrid)[0]
        return ELF_int

    @staticmethod
    def _run_IPR(input_SCF):
        """
        Compute the IPR matrix.

        Parameters
        ----------
        input_SCF : dict of objects
            the SCF input

        Returns
        -------
        IPR_0 : float
            the [0,0,0,0] element of the IPR matrix
        """
        orbitals = input_SCF["orbitals"].eigfuncs
        xgrid = input_SCF["orbitals"]._xgrid

        IPR_mat = localization.calc_IPR_mat(orbitals, xgrid)
        IPR_0 = IPR_mat[0, 0, 0, 0]

        return IPR_0


if __name__ == "__main__":
    SCF_out = TestLocalization._run_SCF(False)
    print(TestLocalization._run_ELF(SCF_out, "orbitals"))
    print(TestLocalization._run_ELF(SCF_out, "density"))
    print(TestLocalization._run_epdc(SCF_out))
    print(TestLocalization._run_IPR(SCF_out))
