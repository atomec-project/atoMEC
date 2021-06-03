# standard libraries
from math import pi

# external libraries
import mendeleev

# global imports
import config
import check_inputs
from models import *
import writeoutput


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
    ):

        # Input variables
        self.species = check_inputs.Atom().check_species(species)
        self.temp = check_inputs.Atom().check_temp(temp, units_temp)
        self.charge = check_inputs.Atom().check_charge(charge)

        # Fundamental atomic properties
        self.at_chrg = self.species.atomic_number  # atomic number
        config.Z = self.at_chrg
        self.at_mass = self.species.atomic_weight  # atomic mass
        self.nele = self.at_chrg + self.charge  # total electron number

        # Check the radius and density
        self.radius, self.density = check_inputs.Atom().check_density(
            self,
            radius,
            density,
            units_radius,
            units_density,
        )

        config.r_s = self.radius
        self.volume = (4.0 * pi * self.radius ** 3.0) / 3.0
        config.sph_vol = self.volume

        # set temperature and inverse temperature
        config.temp = self.temp
        config.beta = 1.0 / self.temp

        # write output info
        output_str = writeoutput.write_atomic_data(self)
        print(output_str)
