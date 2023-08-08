#!/usr/bin/env python3
"""
Exchange-correlation functionals test.

Runs an SCF calculation with the possible classes of functional (LDA, T-LDA,
special functionals) and checks free energy is as expected.
"""

from atoMEC import Atom, models, config
import pytest
import numpy as np


# expected values and tolerance
lda_expected = -158.66453567079228
gdsmfb_expected = -158.6079090856573
no_xc_expected = -144.49795725080637
no_hxc_expected = -249.50772961499766
gga_expected = -159.37528906276566
accuracy = 1e-3


class TestFuncs:
    """
    Test class for different functionals.

    Checks the free energy for an SCF calculation is given by the expected value.
    """

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            ("lda", lda_expected),
            ("gdsmfb", gdsmfb_expected),
            ("no_xc", no_xc_expected),
            ("no_hxc", no_hxc_expected),
            ("gga", gga_expected),
        ],
    )
    def test_funcs(self, test_input, expected):
        """Check the free energy for different functionals."""
        # parallel
        config.numcores = -1

        assert np.isclose(self._run(test_input), expected, atol=accuracy)

    @staticmethod
    def _run(func):
        """
        Run an SCF calculation for an Na atom with given functionals.

        Parameters
        ----------
        func : str
            the xc functional

        Returns
        -------
        F_tot : float
            the total free energy
        """
        # set up the atom
        Na_at = Atom("Na", 0.25, radius=2.0)

        if func == "lda":
            xfunc_id = "lda_x"
            cfunc_id = "lda_c_pw"
        elif func == "gdsmfb":
            xfunc_id = "lda_xc_gdsmfb"
            cfunc_id = "None"
        elif func == "no_xc":
            xfunc_id = "None"
            cfunc_id = "None"
        elif func == "no_hxc":
            xfunc_id = "hartree"
            cfunc_id = "None"
        elif func == "gga":
            xfunc_id = "gga_x_pbe"
            cfunc_id = "gga_c_pbe"

        model = models.ISModel(
            Na_at,
            unbound="quantum",
            xfunc_id=xfunc_id,
            cfunc_id=cfunc_id,
        )

        # run the SCF calculation
        output = model.CalcEnergy(
            6,
            6,
            scf_params={"maxscf": 5, "mixfrac": 0.3},
            grid_params={"ngrid": 1000},
        )

        # extract the total free energy
        F_tot = output["energy"].F_tot
        return F_tot


if __name__ == "__main__":
    config.numcores = -1
    lda = TestFuncs()._run("lda")
    gdsmfb = TestFuncs()._run("gdsmfb")
    no_xc = TestFuncs()._run("no_xc")
    no_hxc = TestFuncs()._run("no_hxc")
    pbe = TestFuncs()._run("gga")
    print("lda_expected =", lda)
    print("gdsmfb_expected =", gdsmfb)
    print("no_xc_expected =", no_xc)
    print("no_hxc_expected =", no_hxc)
    print("gga_expected =", pbe)
