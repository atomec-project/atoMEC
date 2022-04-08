#!/usr/bin/env python3
"""
Boundary conditions test

Runs an SCF calculation with the three possible boundary conditions,
and checks the total free energy.
"""

from atoMEC import Atom, models, config
from atoMEC.check_inputs import InputError
import pytest
import numpy as np


# expected values and tolerance
orbitals_expected = 2.2047
density_expected = 2.1266
IPR_expected = 78.2132
accuracy = 0.001


class TestAtom:
    """
    Test class for different boundary conditions.

    Checks the free energy for an SCF calculation is given by the expected value.
    """

    @pytest.mark.parametrize(
        "ele_input",
        [
            (14),
            ("Bf"),
        ],
    )
    def test_element(self, ele_input):

        with pytest.raises(SystemExit):
            atom = Atom(ele_input, 0.05, radius=1)

    def test_temp_units(self):

        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05, radius=1, units_temp="jk")

    @pytest.mark.parametrize("temp_input", [("a"), (-0.2)])
    def test_temp(self, temp_input):

        with pytest.raises(SystemExit):
            atom = Atom("H", temp_input, radius=1)

    def test_charge(self):

        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05, radius=1.0, charge="jk")

    def test_radius_units(self):

        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05, radius=1.0, units_radius="cm")

    def test_density_units(self):

        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05, density=0.1, units_density="ggcm3")

    @pytest.mark.parametrize("rad_input", [("a"), (-0.2)])
    def test_radius(self, rad_input):

        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05, radius=rad_input)

    @pytest.mark.parametrize("dens_input", [("a"), (-0.2)])
    def test_density(self, dens_input):

        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05, density=dens_input)

    def test_rad_dens_1(self):

        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05, radius=2.0, density=10.0)

    def test_rad_dens_2(self):

        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05)


class TestModel:

    """
    Test class for different boundary conditions.

    Checks the free energy for an SCF calculation is given by the expected value.
    """

    @pytest.mark.parametrize(
        "xc_input",
        [
            (5.0),
            ("lca"),
            ("gga_x_pbe"),
        ],
    )
    def test_xc(self, xc_input):
        atom = Atom("Al", 0.05, radius=1)

        with pytest.raises((SystemExit, TypeError)):

            model = models.ISModel(atom, xfunc_id=xc_input)

    @pytest.mark.parametrize(
        "unbound_input",
        [
            (5.0),
            ("thomas_fermi"),
            ("ideal"),
        ],
    )
    def test_unbound(self, unbound_input):

        atom = Atom("Al", 0.05, radius=1)

        with pytest.raises(SystemExit):

            model = models.ISModel(atom, unbound=unbound_input, bc="bands")

    @pytest.mark.parametrize(
        "bcs_input",
        [
            (5.0),
            ("timsbc"),
        ],
    )
    def test_bcs(self, bcs_input):

        atom = Atom("Al", 0.05, radius=1)

        with pytest.raises(SystemExit):

            model = models.ISModel(atom, bc=bcs_input)

    def test_spinpol(self):

        atom = Atom("Al", 0.05, radius=1)

        with pytest.raises(SystemExit):

            model = models.ISModel(atom, spinpol="a")

    @pytest.mark.parametrize(
        "spinmag_input",
        [([5.0, "Al"]), ([2, "Al"]), ([1, "Be"])],
    )
    def test_spinmag(self, spinmag_input):

        atom = Atom(spinmag_input[1], 0.05, radius=1)

        with pytest.raises(SystemExit):

            model = models.ISModel(atom, spinmag=spinmag_input[0])

    def test_v_shift(self):

        atom = Atom("Al", 0.05, radius=1)

        with pytest.raises(SystemExit):

            model = models.ISModel(atom, v_shift="a")


class TestSCF:

    """
    Test class for different boundary conditions.

    Checks the free energy for an SCF calculation is given by the expected value.
    """

    @pytest.mark.parametrize(
        "grid_input",
        [
            ({"ngrid": "a"}),
            ({"ngrid": -100}),
            ({"ngrid_coarse": "a"}),
            ({"ngrid_coarse": -100}),
            ({"x0": -2}),
        ],
    )
    def test_ngrid_params(self, grid_input):

        atom = Atom("Al", 0.05, radius=1)
        model = models.ISModel(atom)

        with pytest.raises(SystemExit):

            model.CalcEnergy(3, 3, grid_params=grid_input)

    @pytest.mark.parametrize(
        "conv_input",
        [
            ({"nconv": "a"}),
            ({"vconv": -1e-3}),
        ],
    )
    def test_conv_params(self, conv_input):

        atom = Atom("Al", 0.05, radius=1)
        model = models.ISModel(atom)

        with pytest.raises(SystemExit):

            model.CalcEnergy(3, 3, conv_params=conv_input)

    @pytest.mark.parametrize(
        "scf_input",
        [
            ({"maxscf": "a"}),
            ({"maxscf": 0}),
            ({"mixfrac": "a"}),
            ({"mixfrac": 0}),
        ],
    )
    def test_scf_params(self, scf_input):

        atom = Atom("Al", 0.05, radius=1)
        model = models.ISModel(atom)

        with pytest.raises(SystemExit):

            model.CalcEnergy(3, 3, scf_params=scf_input)

    @pytest.mark.parametrize(
        "bands_input",
        [
            ({"nkpts": "a"}),
            ({"nkpts": 5}),
            ({"de_min": "a"}),
            ({"de_min": -0.1}),
        ],
    )
    def test_band_params(self, bands_input):

        atom = Atom("Al", 0.05, radius=1)
        model = models.ISModel(atom, bc="bands", unbound="quantum")

        with pytest.raises(SystemExit):

            model.CalcEnergy(3, 3, band_params=bands_input)
