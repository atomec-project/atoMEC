#!/usr/bin/env python3
"""Test that exceptions are raised for incorrect specification of inputs."""

from atoMEC import Atom, models
import pytest


class TestAtom:
    """Test class for raising exceptions in the Atom object."""

    @pytest.mark.parametrize(
        "ele_input",
        [
            (14),
            ("Bf"),
        ],
    )
    def test_element(self, ele_input):
        """Check chemical species input."""
        with pytest.raises(SystemExit):
            atom = Atom(ele_input, 0.05, radius=1)
            return atom

    def test_temp_units(self):
        """Check temperature units input."""
        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05, radius=1, units_temp="jk")
            return atom

    @pytest.mark.parametrize("temp_input", [("a"), (-0.2)])
    def test_temp(self, temp_input):
        """Check temperature input."""
        with pytest.raises(SystemExit):
            atom = Atom("H", temp_input, radius=1)
            return atom

    def test_charge(self):
        """Check charge input."""
        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05, radius=1.0, charge="jk")
            return atom

    def test_radius_units(self):
        """Check radius units input."""
        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05, radius=1.0, units_radius="cm")
            return atom

    def test_density_units(self):
        """Check density units input."""
        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05, density=0.1, units_density="ggcm3")
            return atom

    @pytest.mark.parametrize("rad_input", [("a"), (-0.2)])
    def test_radius(self, rad_input):
        """Check radius input."""
        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05, radius=rad_input)
            return atom

    @pytest.mark.parametrize("dens_input", [("a"), (-0.2)])
    def test_density(self, dens_input):
        """Check density input."""
        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05, density=dens_input)
            return atom

    def test_rad_dens_1(self):
        """Check radius and density compatibility."""
        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05, radius=2.0, density=10.0)
            return atom

    def test_rad_dens_2(self):
        """Check one of radius or density specified."""
        with pytest.raises(SystemExit):
            atom = Atom("H", 0.05)
            return atom


class TestModel:
    """Test class for raising exceptions in the ISModel object."""

    @pytest.mark.parametrize(
        "xc_input",
        [(5.0), ("lca"), ("gga_z_pbc")],
    )
    def test_xc(self, xc_input):
        """Test the exchange-correlation (xc) input."""
        atom = Atom("Al", 0.05, radius=1)

        with pytest.raises((SystemExit, TypeError)):
            model = models.ISModel(atom, xfunc_id=xc_input)
            return model

    @pytest.mark.parametrize(
        "unbound_input",
        [
            (5.0),
            ("thomas_fermi"),
            ("ideal"),
        ],
    )
    def test_unbound(self, unbound_input):
        """Test unbound input."""
        atom = Atom("Al", 0.05, radius=1)

        with pytest.raises(SystemExit):
            model = models.ISModel(atom, unbound=unbound_input, bc="bands")
            return model

    @pytest.mark.parametrize(
        "bcs_input",
        [
            (5.0),
            ("timsbc"),
        ],
    )
    def test_bcs(self, bcs_input):
        """Test boundary conditions input."""
        atom = Atom("Al", 0.05, radius=1)

        with pytest.raises(SystemExit):
            model = models.ISModel(atom, bc=bcs_input)
            return model

    def test_spinpol(self):
        """Test spin polarization input."""
        atom = Atom("Al", 0.05, radius=1)

        with pytest.raises(SystemExit):
            model = models.ISModel(atom, spinpol="a")
            return model

    @pytest.mark.parametrize(
        "spinmag_input",
        [([5.0, "Al"]), ([2, "Al"]), ([1, "Be"])],
    )
    def test_spinmag(self, spinmag_input):
        """Test spin magnetization input."""
        atom = Atom(spinmag_input[1], 0.05, radius=1)

        with pytest.raises(SystemExit):
            model = models.ISModel(atom, spinmag=spinmag_input[0])
            return model

    def test_v_shift(self):
        """Test v shift input."""
        atom = Atom("Al", 0.05, radius=1)

        with pytest.raises(SystemExit):
            model = models.ISModel(atom, v_shift="a")
            return model


class TestCalcEnergy:
    """Test class for inputs to the ISModel.CalcEnergy function."""

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
    def test_grid_params(self, grid_input):
        """Test the grid_params input."""
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
        """Test the conv_params input."""
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
        """Test the scf_params input."""
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
        """Test the band params input."""
        atom = Atom("Al", 0.05, radius=1)
        model = models.ISModel(atom, bc="bands", unbound="quantum")

        with pytest.raises(SystemExit):
            model.CalcEnergy(3, 3, band_params=bands_input)
