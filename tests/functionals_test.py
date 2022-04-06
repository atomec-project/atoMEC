#!/usr/bin/env python3
"""
Boundary conditions test

Runs an SCF calculation with the possible classes of functional (LDA, T-LDA,
special functionals) and checks free energy is as expected.
"""

from atoMEC import Atom, models, config
import pytest
import numpy as np

# parallel
config.numcores = -1

# expected values and tolerance
lda_expected = -159.210841
gdsmfb_expected = -159.154166
no_xc_expected = -145.019411
no_hxc_expected = -249.507728
accuracy = 10 * config.conv_params["econv"]


class Test_funcs:
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
        ],
    )
    def test_bcs(self, test_input, expected):

        assert np.isclose(self.__run(test_input), expected, atol=accuracy)

    @staticmethod
    def __run(func):
        """
        Run an SCF calculation for an Na atom with given functionals.

        Parameters
        ----------
        bc : str
            the boundary condition

        Returns
        -------
        F_tot : float
            the total free energy
        """

        # set up the atom
        Na_at = Atom("Na", 0.25, radius=2.0, write_info=False)

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

        model = models.ISModel(
            Na_at,
            unbound="quantum",
            write_info=False,
            xfunc_id=xfunc_id,
            cfunc_id=cfunc_id,
        )

        # run the SCF calculation
        output = model.CalcEnergy(
            6,
            6,
            scf_params={"maxscf": 5},
            write_info=False,
        )

        # extract the total free energy
        F_tot = output["energy"].F_tot
        return F_tot


# if __name__ == "__main__":
#     print("lda=", Test_funcs().run("lda"))
#     print("gdsmfb=", Test_funcs().run("gdsmfb"))
#     print("no_xc=", Test_funcs().run("no_xc"))
#     print("no_hxc=", Test_funcs().run("no_hxc"))
