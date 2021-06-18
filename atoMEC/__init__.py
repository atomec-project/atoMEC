# standard libraries
from math import pi

# external libraries
import mendeleev

# global imports
from . import config
from . import check_inputs
from .models import *
from . import writeoutput


class Atom:
    """
    The Atom class defines the main atom object containing the key information about the
    atomic species, temperature, density etc

    Parameters
    ----------
    species : str
        The chemical symbol for the atomic species, eg "He"
    temp : float
        The electronic temperature in hartree, eV or Kelvin
    radius : float, optional
        The radius of the Wigner-Seitz sphere, defined as 0.5*a_i,
        where a_i is the average inter-atomic distance
    density : float, optional
        The mass density of the material in g cm^-3
    charge : int, optional
        The overall net charge
    units_temp : str, optional
        The units of temperature, must be one of "ha", "ev" or "k"
    units_radius : str, optional
        The units of radius, must be one of "ang" or "bohr"
    units_density : str, optional
        The units of density, currently only "g/cm3" is supported
    write_output : bool, optional
        Whether to print atomic information, defaults True

    Attributes
    ----------
    species : str
        The chemical symbol for the atomic species
    temp : float
        The electronic temperature in hartree
    radius : float
        The radius of the Wigner-Seitz sphere, defined as 0.5*a_i,
        where a_i is the average inter-atomic distance
    density : float
        The mass density of the material in g cm^-3
    charge : int, optional
        The overall net charge
    at_chrg : int
        The atomic number Z
    at_mass : float
        The atomic mass
    nele : int
        The total electron number
    info : str
        Information about the atom
    """

    def __init__(
        self,
        species,
        temp,
        radius=-1,
        density=-1,
        charge=0,
        units_temp="ha",
        units_radius="bohr",
        units_density="g/cm3",
        write_info=True,
    ):

        # print the initial spiel
        print("\n" + "Welcome to atoMEC! \n")

        # input variables are checked later with getter / setter functions
        self.species = species
        self.units_temp = units_temp
        self.temp = temp
        self.charge = charge
        self.units_radius = units_radius
        self.units_density = units_density

        # radius and density need a special check to ensure compatibility
        radius_check, density_check = check_inputs.Atom().check_rad_dens_init(
            self,
            radius,
            density,
            self.units_radius,
            self.units_density,
        )

        self.radius = radius_check
        self.density = density_check

        if write_info:
            print(self.info)

    # below are the getter and setter attributes for all the class attributes

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, species):
        self._species = check_inputs.Atom().check_species(species)

    @property
    def at_chrg(self):
        chrg = self.species.atomic_number
        config.Z = chrg
        return chrg

    @property
    def at_mass(self):
        return self.species.atomic_weight

    @property
    def units_temp(self):
        return self._units_temp

    @units_temp.setter
    def units_temp(self, units_temp):
        self._units_temp = check_inputs.Atom().check_units_temp(units_temp)

    @property
    def temp(self):
        return self._temp

    @temp.setter
    def temp(self, temp):
        self._temp = check_inputs.Atom().check_temp(temp, self.units_temp)
        config.temp = self._temp
        config.beta = 1.0 / self._temp

    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self, charge):
        self._charge = check_inputs.Atom().check_charge(charge)

    @property
    def nele(self):
        return self.at_chrg + self._charge

    @property
    def units_radius(self):
        return self._units_radius

    @units_radius.setter
    def units_radius(self, units_radius):
        self._units_radius = check_inputs.Atom().check_units_radius(units_radius)

    @property
    def units_density(self):
        return self._units_density

    @units_density.setter
    def units_density(self, units_density):
        self._units_density = check_inputs.Atom().check_units_density(units_density)

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius):
        self._radius = check_inputs.Atom().check_radius(radius, self.units_radius)
        self._density = check_inputs.Atom().radius_to_dens(self, self._radius)
        config.r_s = self._radius
        config.sph_vol = (4.0 * pi * self._radius ** 3.0) / 3.0

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, density):
        self._density = check_inputs.Atom().check_density(density, self.units_density)
        self._radius = check_inputs.Atom().dens_to_radius(self, self._density)
        config.r_s = self._radius
        config.sph_vol = (4.0 * pi * self._radius ** 3.0) / 3.0

    @property
    def info(self):
        # write output info
        return writeoutput.write_atomic_data(self)
