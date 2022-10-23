#!/usr/bin/env python3
"""
Pressure test.

Compute pressure with the various avaiable methods in `postprocess.pressure`
and check they return the expected result.
"""

from atoMEC import Atom, models, config
from atoMEC.postprocess import pressure
import numpy as np
import pytest


# expected values and tolerance
finite_diff_expected = 0.0133944
stress_tensor_expected = 0.0044425
virial_expected = 0.013604
ion_expected = 0.0056149
accuracy_1 = 5e-4
accuracy_2 = 1e-4


class TestPressure:
    """Test class for the CalcPressure function."""

    @pytest.fixture(autouse=True, scope="class")
    def SCF_output(self):
        """Run a spin-unpolarized SCF calc and save the output."""
        config.numcores = -1
        return self._run_SCF()

    def test_finite_diff(self, SCF_output):
        """Run the finite difference pressure method."""
        config.numcores = -1
        assert np.isclose(
            self._run_finite_diff(SCF_output), finite_diff_expected, atol=accuracy_1
        )

    def test_stress_tensor(self, SCF_output):
        """Run the stress tensor pressure method."""
        assert np.isclose(
            self._run_stress_tensor(SCF_output), stress_tensor_expected, atol=accuracy_2
        )

    def test_virial(self, SCF_output):
        """Run the virial pressure method."""
        assert np.isclose(
            self._run_virial(SCF_output), virial_expected, atol=accuracy_1
        )

    def test_ion(self, SCF_output):
        """Run the ideal ionic pressure method."""
        assert np.isclose(
            self._run_ion(SCF_output["Atom"]), ion_expected, atol=accuracy_2
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
        model = models.ISModel(Li_at, unbound="quantum", v_shift=False)

        # run the SCF calculation
        output = model.CalcEnergy(
            3,
            3,
            scf_params={"maxscf": 5},
            grid_params={"ngrid": 1000},
            verbosity=1,
        )

        output_dict = {"Atom": Li_at, "model": model, "SCF_out": output}

        return output_dict

    @staticmethod
    def _run_finite_diff(SCF_input):
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
        P_e = SCF_input["model"].CalcPressure(SCF_input["Atom"], SCF_input["SCF_out"])

        return P_e

    @staticmethod
    def _run_stress_tensor(SCF_input):
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

        P_e = pressure.stress_tensor(SCF_input["Atom"], SCF_input["model"], orbs, pot)

        return P_e

    @staticmethod
    def _run_virial(SCF_input):
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

        P_e = pressure.virial(SCF_input["Atom"], SCF_input["model"], energy, rho)

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
        P_ion = pressure.ions_ideal(Atom)

        return P_ion


if __name__ == "__main__":
    SCF_out = TestPressure._run_SCF()
    print("Finite diff pressure: ", TestPressure._run_finite_diff(SCF_out))
    print("Stress tensor pressure: ", TestPressure._run_stress_tensor(SCF_out))
    print("Virial pressure: ", TestPressure._run_virial(SCF_out))
    print("Ion pressure: ", TestPressure._run_ion(SCF_out["Atom"]))
