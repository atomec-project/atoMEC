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
epdc_orbs_expected = 9.22831052983569
epdc_dens_expected = 3.527486593490338
density_expected = 2.1469328620811075
orbitals_expected = 1.100028932668539
IPR_expected = 78.19952984379802
accuracy = 0.01


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
        "SCF_input,method,spinpol,expected",
        [
            (lazy_fixture("SCF_spin_output"), "orbitals", True, orbitals_expected),
            (lazy_fixture("SCF_nospin_output"), "density", False, density_expected),
        ],
    )
    def test_ELF(self, SCF_input, method, spinpol, expected):
        """Test the ELF through the N_shell property."""
        assert np.isclose(
            self._run_ELF(SCF_input, method, spinpol),
            expected,
            atol=accuracy,
        )

    @pytest.mark.parametrize(
        "SCF_input,method,spinpol,expected",
        [
            (lazy_fixture("SCF_spin_output"), "orbitals", True, epdc_orbs_expected),
            (lazy_fixture("SCF_spin_output"), "density", True, epdc_dens_expected),
        ],
    )
    def test_epdc(self, SCF_input, method, spinpol, expected):
        """Test the epdc function."""
        assert np.isclose(
            self._run_epdc(SCF_input, method, spinpol),
            expected,
            atol=accuracy,
        )

    def test_IPR(self, SCF_nospin_output):
        """Test the IPR function."""
        assert np.isclose(self._run_IPR(SCF_nospin_output), IPR_expected, atol=accuracy)

    @staticmethod
    def _run_SCF(spinpol):
        """
        Run an SCF calculation for an He Atom with unbound = "quantum".

        Returns
        -------
        output_dict : dictionary
            the output dictionary containing Atom, model and SCF output
        """
        # parallel
        config.numcores = -1

        # set up the atom and model
        Al_at = Atom("Al", 0.01, radius=5.0, units_temp="eV")
        model = models.ISModel(
            Al_at, unbound="quantum", bc="dirichlet", spinpol=spinpol
        )

        # run the SCF calculation
        output = model.CalcEnergy(
            3,
            2,
            scf_params={"mixfrac": 0.3, "maxscf": 50},
            grid_params={"ngrid": 1000},
        )

        output_dict = {"Atom": Al_at, "model": model, "SCF_out": output}

        return output_dict

    @staticmethod
    def _run_ELF(input_SCF, method, spinpol):
        """
        Compute the electron number in the n=1 shell.

        Parameters
        ----------
        input_SCF : dict
            dictionary with Atom, model and SCF output
        method : the method used to compute the ELF

        Returns
        -------
        N_0 : float
            number of electrons in the n=1 shell
        """
        orbs = input_SCF["SCF_out"]["orbitals"]
        density = input_SCF["SCF_out"]["density"]

        atom = Atom(
            input_SCF["Atom"].species.name,
            input_SCF["Atom"].temp,
            density=input_SCF["Atom"].density,
            write_info=False,
        )

        model = models.ISModel(
            atom,
            unbound=input_SCF["model"].unbound,
            spinpol=input_SCF["model"].spinpol,
            write_info=False,
            bc="dirichlet",
        )

        ELF = localization.ELFTools(
            input_SCF["Atom"], input_SCF["model"], orbs, density, method=method
        )

        N_0 = ELF.N_shell[0][0]

        return N_0

    @staticmethod
    def _run_epdc(input_SCF, method, spinpol):
        """
        Compute the ELF via the epdc function.

        Parameters
        ----------
        input_SCF : dict
            dictionary with Atom, model and SCF output
        method : the method used to compute the ELF

        Returns
        -------
        ELF_int : float
            integral of the ELF function (spin-up channel)
        """
        orbs = input_SCF["SCF_out"]["orbitals"]
        density = input_SCF["SCF_out"]["density"]

        atom = Atom(
            input_SCF["Atom"].species.name,
            input_SCF["Atom"].temp,
            density=input_SCF["Atom"].density,
            write_info=False,
        )

        model = models.ISModel(
            atom,
            unbound=input_SCF["model"].unbound,
            spinpol=input_SCF["model"].spinpol,
            write_info=False,
            bc="dirichlet",
        )

        ELF = localization.ELFTools(
            input_SCF["Atom"], input_SCF["model"], orbs, density, method=method
        )

        epdc = ELF.epdc

        D_0 = (0.3) * (3 * np.pi**2) ** (2.0 / 3.0) * (density.total) ** (5.0 / 3.0)

        ELF_func = 1 / (1 + (epdc / D_0) ** 2)

        ELF_int = np.trapz(ELF_func, ELF._xgrid)[0]

        return ELF_int

    @staticmethod
    def _run_IPR(input_SCF):
        """
        Compute the IPR matrix.

        Parameters
        ----------
        input_SCF : dict
            dictionary with Atom, model and SCF output

        Returns
        -------
        IPR_0 : float
            the [0,0,0,0] element of the IPR matrix
        """
        eigfuncs = input_SCF["SCF_out"]["orbitals"].eigfuncs
        xgrid = input_SCF["SCF_out"]["orbitals"]._xgrid

        IPR_mat = localization.calc_IPR_mat(eigfuncs, xgrid)
        IPR_0 = IPR_mat[0, 0, 0, 0]

        return IPR_0


if __name__ == "__main__":
    config.numcores = -1
    SCF_spin_out = TestLocalization._run_SCF(True)
    SCF_no_spin_out = TestLocalization._run_SCF(False)
    print(
        "epdc_orbs_expected =",
        TestLocalization._run_epdc(SCF_spin_out, "orbitals", True),
    )
    print(
        "epdc_dens_expected =",
        TestLocalization._run_epdc(SCF_spin_out, "density", True),
    )
    print(
        "density_expected =",
        TestLocalization._run_ELF(SCF_no_spin_out, "density", False),
    )
    print(
        "orbitals_expected =", TestLocalization._run_ELF(SCF_spin_out, "orbitals", True)
    )
    print("IPR_expected =", TestLocalization._run_IPR(SCF_no_spin_out))
