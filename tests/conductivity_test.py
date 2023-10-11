#!/usr/bin/env python3
"""Test conductivity module."""

from atoMEC import Atom, models, config
from atoMEC.postprocess import conductivity
import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np


# expected values and tolerance
N_cc_expected_1_0 = 2.454442757747575
N_cc_expected_3_1 = 6.727831527604185
N_tt_expected_1_0 = 6.833387102730367
N_tt_expected_3_1 = 6.833387102730367
N_vv_expected_1_0 = 0.0
N_vv_expected_3_1 = 0.0
N_cv_expected_1_0 = 4.1257136059769985
N_cv_expected_3_1 = 0.00749276366862972
expected_integral_4 = 1.3416407864998738
expected_integral_2 = 0.447213595499958
expected_sum_rule = 0.29294247872666707
expected_prop_sig_vv = 0.0
expected_prop_sig_cv = 0.0014275605940770114
expected_prop_N_tot = 6.08906292713134
expected_prop_N_free = 4.5527425486281805

accuracy = 0.001


class TestConductivity:
    """
    Test class for conduction module.

    Calculates number of electrons vie the cond_tot function,
    and checks the sum rule for an unfinished SCF calculation.
    """

    @pytest.fixture(autouse=True, scope="class")
    def SCF_output(self):
        """Run a spin-unpolarized SCF calc and save the output."""
        config.numcores = -1
        return self._run_SCF()

    @pytest.mark.parametrize(
        "SCF_input,method,val_orb,expected",
        [
            (lazy_fixture("SCF_output"), "cc", (1, 0), N_cc_expected_1_0),
            (lazy_fixture("SCF_output"), "tt", (1, 0), N_tt_expected_1_0),
            (lazy_fixture("SCF_output"), "vv", (1, 0), N_vv_expected_1_0),
            (lazy_fixture("SCF_output"), "cv", (1, 0), N_cv_expected_1_0),
            (lazy_fixture("SCF_output"), "cc", (3, 1), N_cc_expected_3_1),
            (lazy_fixture("SCF_output"), "tt", (3, 1), N_tt_expected_3_1),
            (lazy_fixture("SCF_output"), "vv", (3, 1), N_vv_expected_3_1),
            (lazy_fixture("SCF_output"), "cv", (3, 1), N_cv_expected_3_1),
        ],
    )
    def test_cond_tot(self, SCF_input, method, val_orb, expected):
        """Run the cond_tot function (under Kubo-Greenwood class)."""
        assert np.isclose(
            self._run_cond_tot(SCF_input, method, val_orb),
            expected,
            atol=accuracy,
        )

    @pytest.mark.parametrize(
        "SCF_input,method,expected",
        [
            (lazy_fixture("SCF_output"), "sig_vv", expected_prop_sig_vv),
            (lazy_fixture("SCF_output"), "sig_cv", expected_prop_sig_cv),
            (lazy_fixture("SCF_output"), "N_tot", expected_prop_N_tot),
            (lazy_fixture("SCF_output"), "N_free", expected_prop_N_free),
        ],
    )
    def test_prop(self, SCF_input, method, expected):
        """Run _run_prop function to check properties of the Kubo-Greenwood class."""
        assert np.isclose(
            self._run_prop(SCF_input, method),
            expected,
            atol=accuracy,
        )

    @pytest.mark.parametrize(
        "order,expected",
        [
            (2, expected_integral_2),
            (4, expected_integral_4),
        ],
    )
    def test_integrals(self, order, expected):
        """Test the spherical harmonic integrals."""
        assert np.isclose(self._run_int_calc(order), expected, atol=accuracy)

    @pytest.mark.parametrize(
        "SCF_input,expected",
        [
            (lazy_fixture("SCF_output"), expected_sum_rule),
        ],
    )
    def test_sum_rule(self, SCF_input, expected):
        """Run the sum_rule func from the conductivity module."""
        assert np.isclose(self._run_sum_rule(SCF_input), expected, atol=accuracy)

    @staticmethod
    def _run_SCF():
        r"""
        Run an SCF calculation for a Fluorine atom.

        Parameters
        ----------
        spinpol: bool
            If the SCF calculation is spin polarized.

        Returns
        -------
        output_dict : dictionary
            the Atom, model and SCF output in a dictionary

        """
        # parallel
        config.numcores = -1

        # set up the atom and model
        F_at = Atom("F", 0.4, radius=5.0)
        model = models.ISModel(F_at, unbound="quantum", bc="dirichlet")

        # run the SCF calculation
        output = model.CalcEnergy(
            4,
            4,
            scf_params={"mixfrac": 0.3, "maxscf": 6},
            grid_params={"ngrid": 1200, "ngrid_coarse": 300},
        )

        output_dict = {"Atom": F_at, "model": model, "SCF_out": output}

        return output_dict

    @staticmethod
    def _run_cond_tot(input_SCF, which, val_orb):
        """
        Compute the number of conducting electrons via the cond_tot function.

        Parameters
        ----------
        input_SCF : dictionary
            dictionary of Atom, model and SCF output
        which : the componenet for which the number of electrons is computed.
        val_orb : tuple of ints
            defines the valance orbital

        Returns
        -------
        N : float
            number of conductiong electrons in the component.

        """
        orbs = input_SCF["SCF_out"]["orbitals"]

        cond = conductivity.KuboGreenwood(
            input_SCF["Atom"], input_SCF["model"], orbs, valence_orbs=[val_orb]
        )

        N = cond.cond_tot(component=which)[1]
        return N

    @staticmethod
    def _run_prop(input_SCF, which):
        """
        Return properties from the Kubo-Greenwood class.

        Parameters
        ----------
        input_SCF : dictionary
            dictionary of Atom, model and SCF output
        which : the property to be calculated

        Returns
        -------
        prop: float
            The property's value
        """
        orbs = input_SCF["SCF_out"]["orbitals"]

        cond = conductivity.KuboGreenwood(
            input_SCF["Atom"], input_SCF["model"], orbs, valence_orbs=[(2, 0)]
        )

        if which == "sig_vv":
            prop = cond.sig_vv
        elif which == "sig_cv":
            prop = cond.sig_cv
        elif which == "N_tot":
            prop = cond.N_tot
        else:
            prop = cond.N_free
        return prop

    @staticmethod
    def _run_int_calc(order):
        """
        Calculate a spherical harmonic integral.

        Parameters
        ----------
        order : int
            the order of the P integral to be calculated.

        Returns
        -------
        integ: float
            the resulting integral.

        """
        spherical_har = conductivity.SphHamInts()
        integ = spherical_har.P_int(order, 1, 2, 1)
        return integ

    @staticmethod
    def _run_sum_rule(input_SCF):
        """
        Check the sum rule of an unconveged set of orbitals.

        Parameters
        ----------
        input_SCF : dictionary
            dictionary of Atom, model and SCF output

        Returns
        -------
        sum_rule : float
            the result of trying to apply the sum rule to an
            arbitrary inconverged orbital.

        """
        orbs = input_SCF["SCF_out"]["orbitals"]
        cond = conductivity.KuboGreenwood(input_SCF["Atom"], input_SCF["model"], orbs)
        sum_rule = cond.check_sum_rule(2, 2, 0)[0]
        return sum_rule


if __name__ == "__main__":
    config.numcores = -1
    SCF_out = TestConductivity._run_SCF()
    print("N_cc_expected_1_0 =", TestConductivity._run_cond_tot(SCF_out, "cc", (1, 0)))
    print("N_cc_expected_3_1 =", TestConductivity._run_cond_tot(SCF_out, "cc", (3, 1)))
    print("N_tt_expected_1_0 =", TestConductivity._run_cond_tot(SCF_out, "tt", (1, 0)))
    print("N_tt_expected_3_1 =", TestConductivity._run_cond_tot(SCF_out, "tt", (3, 1)))
    print("N_vv_expected_1_0 =", TestConductivity._run_cond_tot(SCF_out, "vv", (1, 0)))
    print("N_vv_expected_3_1 =", TestConductivity._run_cond_tot(SCF_out, "vv", (3, 1)))
    print("N_cv_expected_1_0 =", TestConductivity._run_cond_tot(SCF_out, "cv", (1, 0)))
    print("N_cv_expected_3_1 =", TestConductivity._run_cond_tot(SCF_out, "cv", (3, 1)))
    print("expected_integral_4 =", TestConductivity._run_int_calc(4))
    print("expected_integral_2 =", TestConductivity._run_int_calc(2))
    print("expected_sum_rule =", TestConductivity._run_sum_rule(SCF_out))
    print("expected_prop_sig_vv =", TestConductivity._run_prop(SCF_out, "sig_vv"))
    print("expected_prop_sig_cv =", TestConductivity._run_prop(SCF_out, "sig_cv"))
    print("expected_prop_N_tot =", TestConductivity._run_prop(SCF_out, "N_tot"))
    print("expected_prop_N_free =", TestConductivity._run_prop(SCF_out, "N_free"))
