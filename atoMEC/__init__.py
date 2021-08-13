"""
atoMEC: Average-atom code for matter under extreme conditions.

Copyright (c) 2021 (in alphabetical order), Tim Callow, Attila Cangi, Eli Kraisler.
All rights reserved.

atoMEC is a python-based average-atom code for simulations of high energy density \
phenomena such as in warm dense matter.
Please see the README or the project wiki (https://atomec-project.github.io/atoMEC/) \
for more information.

Classes
-------
* :class:`Atom` : the main object for atoMEC calculations, containing information \
about physical material properties
"""

__version__ = "1.0.0"

# standard libraries
from math import pi

# global imports
from . import config
from . import check_inputs
from . import writeoutput


class Atom:
    r"""
    The principal object in atoMEC calculations which defines the material properties.

    The `Atom` contains key information about the physical properties of the material
    such as temperature, density, and charge. It does not contain any information \
    about approximations or choices of model.

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
        The mass density of the material in :math:`\mathrm{g\ cm}^{-3}`
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
        if write_info:
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
        """str: the chemical symbol for the atomic species."""
        return self._species

    @species.setter
    def species(self, species):
        self._species = check_inputs.Atom().check_species(species)

    @property
    def at_chrg(self):
        """int: the atomic charge Z."""
        chrg = self.species.atomic_number
        config.Z = chrg
        return chrg

    @property
    def at_mass(self):
        """float: the atomic mass (in a.u.)."""
        return self.species.atomic_weight

    @property
    def units_temp(self):
        """str: the units of temperature."""
        return self._units_temp

    @units_temp.setter
    def units_temp(self, units_temp):
        self._units_temp = check_inputs.Atom().check_units_temp(units_temp)

    @property
    def temp(self):
        """float: the electronic temperature in Hartree."""
        return self._temp

    @temp.setter
    def temp(self, temp):
        self._temp = check_inputs.Atom().check_temp(temp, self.units_temp)
        config.temp = self._temp
        config.beta = 1.0 / self._temp

    @property
    def charge(self):
        """int: the net charge of the atom."""
        return self._charge

    @charge.setter
    def charge(self, charge):
        self._charge = check_inputs.Atom().check_charge(charge)

    @property
    def nele(self):
        """int: the number of electrons in the atom.

        The total electron number is given by the sum of :obj:`at_chrg`
        and :obj:`charge`.
        """
        return self.at_chrg + self._charge

    @property
    def units_radius(self):
        """str: the units of the atomic radius."""
        return self._units_radius

    @units_radius.setter
    def units_radius(self, units_radius):
        self._units_radius = check_inputs.Atom().check_units_radius(units_radius)

    @property
    def units_density(self):
        """str: the units of the atomic density."""
        return self._units_density

    @units_density.setter
    def units_density(self, units_density):
        self._units_density = check_inputs.Atom().check_units_density(units_density)

    @property
    def radius(self):
        r"""float: radius of the Wigner-Seitz sphere.

        The radius is defined as :math:`a_i /2`,
        where :math:`a_i` is the average inter-atomic distance.
        """
        return self._radius

    @radius.setter
    def radius(self, radius):
        self._radius = check_inputs.Atom().check_radius(radius, self.units_radius)
        self._density = check_inputs.Atom().radius_to_dens(self, self._radius)
        config.r_s = self._radius
        config.sph_vol = (4.0 * pi * self._radius ** 3.0) / 3.0

    @property
    def density(self):
        r"""float: the mass density of the material in :math:`\mathrm{g\ cm}^{-3}`."""
        return self._density

    @density.setter
    def density(self, density):
        self._density = check_inputs.Atom().check_density(density)
        self._radius = check_inputs.Atom().dens_to_radius(self, self._density)
        config.r_s = self._radius
        config.sph_vol = (4.0 * pi * self._radius ** 3.0) / 3.0

    @property
    def info(self):
        """str: formatted information about the :obj:`Atom`'s attributes."""
        # write output info
        return writeoutput.write_atomic_data(self)
