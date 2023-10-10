#!/usr/bin/env python3
"""
Pressure test.

Compute pressure with the various avaiable methods in `postprocess.pressure`
and check they return the expected result.
"""

from atoMEC import Atom, models, config, mathtools
from atoMEC.unitconv import ha_to_gpa as h2g
from atoMEC.postprocess import pressure
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture


# expected values and tolerance
finite_diff_expected_A = 172.7901141261863
finite_diff_expected_B = 244.0758379313636
stress_tensor_expected_rr = 149.73033807657444
stress_tensor_expected_tr = 111.06081510721712
virial_expected_corr = 143.24751643303316
virial_expected_nocorr = 178.4773754883355
ideal_expected = 103.18020549074609
ion_expected = 165.19118614722603


accuracy = 0.1


class TestPressure:
    """Test class for the CalcPressure function."""

    @pytest.fixture(autouse=True, scope="class")
    def SCF_output(self):
        """Run a spin-unpolarized SCF calc and save the output."""
        config.numcores = -1
        return self._run_SCF()

    @pytest.mark.parametrize(
        "SCF_input,method,expected",
        [
            (lazy_fixture("SCF_output"), "A", finite_diff_expected_A),
            (lazy_fixture("SCF_output"), "B", finite_diff_expected_B),
        ],
    )
    def test_finite_diff(self, SCF_input, method, expected):
        """Run the finite difference pressure method."""
        config.numcores = -1
        assert np.isclose(
            self._run_finite_diff(SCF_input, method),
            expected,
            atol=accuracy,
        )

    @pytest.mark.parametrize(
        "SCF_input,only_rr,expected",
        [
            (lazy_fixture("SCF_output"), True, stress_tensor_expected_rr),
            (lazy_fixture("SCF_output"), False, stress_tensor_expected_tr),
        ],
    )
    def test_stress_tensor(self, SCF_input, only_rr, expected):
        """Run the stress tensor pressure method."""
        assert np.isclose(
            self._run_stress_tensor(SCF_input, only_rr),
            expected,
            atol=accuracy,
        )

    @pytest.mark.parametrize(
        "SCF_input,use_correction,expected",
        [
            (lazy_fixture("SCF_output"), True, virial_expected_corr),
            (lazy_fixture("SCF_output"), False, virial_expected_nocorr),
        ],
    )
    def test_virial(self, SCF_input, use_correction, expected):
        """Run the virial pressure method."""
        assert np.isclose(
            self._run_virial(SCF_input, use_correction),
            expected,
            atol=accuracy,
        )

    def test_ideal(self, SCF_output):
        """Run the ideal ionic pressure method."""
        assert np.isclose(self._run_ideal(SCF_output), ideal_expected, atol=accuracy)

    def test_ion(self, SCF_output):
        """Run the ideal ionic pressure method."""
        assert np.isclose(
            self._run_ion(SCF_output["Atom"]), ion_expected, atol=accuracy
        )

    @staticmethod
    def _run_SCF():
        """
        Run an SCF calculation for an Li atom.

        Returns
        -------
        output_dict : dictionary
            the Atom, model and SCF output in a dictionary
        """
        # set up the atom and model
        Li_at = Atom("Li", 10, radius=2.5, units_temp="eV")
        model = models.ISModel(Li_at, unbound="quantum", v_shift=False, bc="bands")
        # run the SCF calculation
        output = model.CalcEnergy(
            3,
            3,
            scf_params={"maxscf": 5},
            grid_params={"ngrid": 1000},
            band_params={"nkpts": 50},
            verbosity=1,
            grid_type="log",
        )

        output_dict = {"Atom": Li_at, "model": model, "SCF_out": output}

        return output_dict

    @staticmethod
    def _run_finite_diff(SCF_input, method):
        """
        Compute pressure via the finite difference method.

        Inputs
        ------
        SCF_input : dictionary
            Atom, model and SCF output information

        Returns
        -------
        P_e : float
            electronic pressure
        """
        P_e = h2g * SCF_input["model"].CalcPressure(
            SCF_input["Atom"], SCF_input["SCF_out"], method=method
        )

        return P_e

    @staticmethod
    def _run_stress_tensor(SCF_input, only_rr):
        """
        Compute pressure via the stress tensor method.

        Inputs
        ------
        SCF_input : dictionary
            Atom, model and SCF output information

        Returns
        -------
        P_e : float
            electronic pressure
        """
        orbs = SCF_input["SCF_out"]["orbitals"]
        pot = SCF_input["SCF_out"]["potential"]

        P_e = h2g * pressure.stress_tensor(
            SCF_input["Atom"], SCF_input["model"], orbs, pot, only_rr
        )

        return P_e

    @staticmethod
    def _run_virial(SCF_input, use_correction):
        """
        Compute pressure via the virial method.

        Inputs
        ------
        SCF_input : dictionary
            Atom, model and SCF output information

        Returns
        -------
        P_e : float
            electronic pressure
        """
        energy = SCF_input["SCF_out"]["energy"]
        rho = SCF_input["SCF_out"]["density"]
        orbs = SCF_input["SCF_out"]["orbitals"]
        pot = SCF_input["SCF_out"]["potential"]

        P_e = h2g * pressure.virial(
            SCF_input["Atom"],
            SCF_input["model"],
            energy,
            rho,
            orbs,
            pot,
            use_correction=use_correction,
        )

        return P_e

    @staticmethod
    def _run_ideal(scf_input):
        """
        Compute the ideal electron pressure.

        Inputs
        ------
        SCF_input : dictionary
            Atom, model and SCF output information

        Returns
        -------
        P_e : float
            ideal electron pressure
        """
        orbs = scf_input["SCF_out"]["orbitals"]
        atom = scf_input["Atom"]
        model = scf_input["model"]
        # reverse engineer chem pot
        # this is HORRIBLE and HAS TO CHANGE
        config.spindims = np.shape(orbs.eigvals)[1]
        config.unbound = model.unbound
        chem_pot = mathtools.chem_pot(orbs)
        P_e = h2g * pressure.ideal_electron(atom, chem_pot)

        return P_e

    @staticmethod
    def _run_ion(Atom):
        """
        Compute the ideal gas pressure.

        Inputs
        ------
        SCF_input : dictionary
            Atom, model and SCF output information

        Returns
        -------
        P_ion : float
            ionic pressure
        """
        P_ion = h2g * pressure.ions_ideal(Atom)

        return P_ion


if __name__ == "__main__":
    config.numcores = -1
    SCF_out = TestPressure._run_SCF()
    fd_a = TestPressure._run_finite_diff(SCF_out, "A")
    fd_b = TestPressure._run_finite_diff(SCF_out, "B")
    print("finite_diff_expected_A =", fd_a)
    print("finite_diff_expected_B =", fd_b)
    print(
        "stress_tensor_expected_rr =",
        TestPressure._run_stress_tensor(SCF_out, True),
    )
    print(
        "stress_tensor_expected_tr =",
        TestPressure._run_stress_tensor(SCF_out, False),
    )
    print("virial_expected_corr =", TestPressure._run_virial(SCF_out, True))
    print("virial_expected_nocorr =", TestPressure._run_virial(SCF_out, False))
    print("ideal_expected =", TestPressure._run_ideal(SCF_out))
    print("ion_expected =", TestPressure._run_ion(SCF_out["Atom"]))
