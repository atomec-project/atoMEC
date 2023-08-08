#!/usr/bin/env python3
"""
Gram Schmidt Test.

Run an SCF calculation and uses the resulting orbs to calculate overlap
integrals and perform Gram-Schmidt orthonormalization.
"""

from atoMEC import Atom, models, staticKS
import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np

# expected values and tolerance
self_overlap_expected = 1.00000
overlap_expected = 0.041051392274703176
accuracy = 0.0001


class TestGS:
    """Test class for Gram Schmidt."""

    @pytest.fixture(autouse=True, scope="class")
    def SCF_output(self):
        """Run a SCF calculation."""
        return self._run_SCF()

    @pytest.mark.parametrize(
        "input_SCF,case,expected",
        [
            (lazy_fixture("SCF_output"), "self", self_overlap_expected),
            (lazy_fixture("SCF_output"), "other", overlap_expected),
        ],
    )
    def test_overlap(self, input_SCF, case, expected):
        """Run overlap integral after orthonormalizatation."""
        assert np.isclose(
            self._run_overlap(input_SCF, case),
            expected,
            atol=accuracy,
        )

    @staticmethod
    def _run_SCF():
        """
        Run and SCF calculation for a Sodium atom.

        Parameters
        ----------
            None

        Returns
        -------
        output : dict of objects
            the output dictonary containing density, orbitals etc.

        """
        # set up Atom and Model
        Na_at = Atom("Na", 0.7, radius=5.0)
        model = models.ISModel(Na_at, unbound="ideal", bc="dirichlet")

        # run SCF calculation
        output = model.CalcEnergy(
            4, 4, scf_params={"mixfrac": 0.3, "maxscf": 5}, grid_params={"ngrid": 1000}
        )

        return output

    @staticmethod
    def _run_overlap(input_SCF, case):
        """
        Orthonormalizes the orbs and calculates ovelap integrals.

        Parameters
        ----------
        input_SCF: dict of objects
            output of SCF calculation.
        case: str
            is the calculation done for self or non-self ovelap

        Returns
        -------
        norm: float
            the value of the overlap integral

        """
        xgrid = input_SCF["orbitals"]._xgrid
        GS = staticKS.GramSchmidt(input_SCF["orbitals"].eigfuncs, xgrid)
        ortho = GS.make_ortho()
        if case == "self":
            norm = GS.prod_eigfuncs(ortho[0, 0, 0, 1], ortho[0, 0, 0, 1], xgrid)
        else:
            norm = np.abs(GS.prod_eigfuncs(ortho[0, 0, 0, 1], ortho[0, 0, 1, 2], xgrid))

        return norm


if __name__ == "__main__":
    SCF_out = TestGS._run_SCF()
    print("self_overlap_expected =", TestGS._run_overlap(SCF_out, "self"))
    print("overlap_expected =", TestGS._run_overlap(SCF_out, "other"))
