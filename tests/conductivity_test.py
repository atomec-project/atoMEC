#!/usr/bin/env python3
"""
Test conductivity module.

"""

from atoMEC import Atom, models, config
from atoMEC.postprocess import conductivity
import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np


# expected values and tolerance
#orbitals_expected = 1.1015
#density_expected = 2.1252
#IPR_expected = 78.2107
#epdc_orbs_expected = 8.7733
#epdc_dens_expected = 3.5307
N_cc_expected=4.59790383269
N_tt_expected=4.59790383269
N_vv_expected=0.0
N_cv_expected=0.0
expected_integral_4=1.341640786499
expected_integral_2=0.447213595499
expected_sum_rule=0.2940819621
accuracy = 0.001


class TestConductivity:
    """
    Test class for conduction module.

    First runs an SCF calculation to generate KS orbitals and density with and
    without spin polarization. Then uses this input to check the N_shell and epdc
    properties in the ELF class, and the IPR matrix.
    """

    @pytest.fixture(autouse=True, scope="class")
    def SCF_output(self):
        """Run a spin-unpolarized SCF calc and save the output."""
        config.numcores = -1
        return self._run_SCF(False)

    @pytest.mark.parametrize(
        "SCF_input,method,expected",
        [
            (lazy_fixture("SCF_output"), "cc", N_cc_expected),
            (lazy_fixture("SCF_output"), "tt", N_tt_expected),
            (lazy_fixture("SCF_output"), "vv", N_vv_expected),
            (lazy_fixture("SCF_output"), "cv", N_cv_expected),
        ],
    )
    def test_cond_tot(self, SCF_input, method, expected):
        """Run the cond_tot function (under Kubo-Greenwood class)."""
        assert np.isclose(
            self._run_cond_tot(SCF_input, method),
            expected,
            atol=accuracy,
        )
    

    @pytest.mark.parametrize(
        "order,expected",
        [
            (2,expected_integral_2),
            (4,expected_integral_4),
        ],
    )
    def test_integrals(self,order,expected):
        """ Test the spherical harmonic integrals. """
        assert np.isclose(self._run_int_calc(order),expected,atol=accuracy)

    @pytest.mark.parametrize(
        "SCF_input,expected",
        [
            (lazy_fixture("SCF_output"),expected_sum_rule),
        ],
    )
    
    def test_sum_rule(self,SCF_input,expected):
        assert np.isclose(self._run_sum_rule(SCF_input),expected,atol=accuracy)
    
    @staticmethod
    def _run_SCF(spinpol):
        """
        Run an SCF calculation for a Fluorine atom

        Returns
        -------
        output : dict of objects
            the output dictionary containing density, orbitals etc
        """
        # parallel
        config.numcores = -1

        # set up the atom and model
        F_at = Atom("F", 0.4, radius=5.0)
        model = models.ISModel(F_at, unbound="quantum",bc="dirichlet")

        # run the SCF calculation
        output = model.CalcEnergy(4,4,scf_params={"mixfrac": 0.3, "maxscf": 6},grid_params={"ngrid": 1200})

        return output
    
    @staticmethod

    def _run_cond_tot(input_SCF, which):
        """
        Compute the number of conducting electrons via the cond_tot function

        Parameters
        ----------
        input_SCF : dict of objects
            the SCF input
        which : the componenet for which the number of electrons is computed.

        Returns
        -------
        N : float
            number of conductiong electrons in the component
        """
        cond=conductivity.KuboGreenwood(input_SCF["orbitals"])

        N=cond.cond_tot(component=which)[1]
        return N

    @staticmethod
    def _run_int_calc(order):
        """
        """
        spherical_har=conductivity.SphHamInts()
        integ=spherical_har.P_int(order,1,2,1)
        return integ

    @staticmethod
    def _run_sum_rule(input_SCF): 
        """
        """
        cond=conductivity.KuboGreenwood(input_SCF["orbitals"])
        sum_rule=cond.check_sum_rule(2,2,0)[0]
        return sum_rule

if __name__ == "__main__":
    SCF_out = TestConduction._run_SCF(False)
    print(TestConduction._run_cond_tot(SCF_out, "cc"))
    print(TestConduction._run_cond_tot(SCF_out, "tt"))
    print(TestConduction._run_cond_tot(SCF_out, "cv"))
    print(TestConduction._run_cond_tot(SCF_out, "vv"))
