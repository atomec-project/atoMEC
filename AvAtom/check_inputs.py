"""
This module checks inputs for errors
"""

# standard python packages
import sys

# external packages
from mendeleev import element
import sqlalchemy.orm.exc as ele_chk
import numpy as np
from math import pi

# internal packages
import constants
import unitconv
import xc
import config


class Atom:
    """
    Checks the inputs from the BuildAtom class
    """

    def check_species(self, species):
        """
        Checks the species is a string and corresponds to an actual element

        Inputs:
        - species (str)    : chemical symbol for atomic species
        """
        if isinstance(species, str) == False:
            raise InputError.species_error("element is not a string")
        else:
            try:
                return element(species)
            except ele_chk.NoResultFound:
                raise InputError.species_error("invalid element")

    def check_temp(self, temp, units_temp):
        """
        Checks the temperature is a float within a sensible range
        """
        units_accepted = ["ha", "ev", "k"]
        if units_temp.lower() not in units_accepted:
            raise InputError.temp_error("units of temperature are not recognised")

        elif isinstance(temp, (float, int)) == False:
            raise InputError.temp_error("temperature is not a number")
        else:
            # convert internal temperature to hartree
            if units_temp.lower() == "ev":
                temp = unitconv.ev_to_ha * temp
            elif units_temp.lower() == "k":
                temp = unitconv.K_to_ha * temp
            # check if temperature is within some reasonable limits
            if temp < 0:
                raise InputError.temp_error("temperature is negative")
            if temp < 0.01:
                InputWarning.temp_warning("low")
                return temp
            elif temp > 100.0:
                InputWarning.temp_warning("high")
                return temp
            else:
                return temp

    def check_charge(self, charge):
        """
        Checks the charge is an integer
        """
        if isinstance(charge, int) == False:
            raise InputError.charge_error()
        else:
            return charge

    def check_density(self, atom, radius, density, units_radius, units_density):
        """
        Checks that the density or radius is specified

        Inputs:
        - atom (object)     : atom object
        - density (float)   : material density
        - radius (float)    : voronoi sphere radius
        """

        radius_units_accepted = ["bohr", "angstrom", "ang"]
        density_units_accepted = ["g/cm3", "gcm3"]

        if isinstance(density, (float, int)) == False:
            raise InputError.density_error("Density is not a number")
        elif isinstance(radius, (float, int)) == False:
            raise InputError.density_error("Radius is not a number")
        elif units_radius.lower() not in radius_units_accepted:
            raise InputError.density_error("Radius units not recognised")
        elif units_density.lower() not in density_units_accepted:
            raise InputError.density_error("Density units not recognised")
        else:
            if units_radius == "angstrom" or units_radius == "ang":
                radius = unitconv.angstrom_to_bohr * radius
            if density == -1 and radius != -1:
                if radius < 0.1:
                    raise InputError.density_error(
                        "Radius must be a positive number greater than 0.1"
                    )
                else:
                    density = self.radius_to_dens(atom, radius)
            elif radius == -1 and density != -1:
                if density > 100 or density < 0:
                    raise InputError.density_error(
                        "Density must be a positive number less than 100"
                    )
                else:
                    radius = self.dens_to_radius(atom, density)
            elif radius != -1 and density != -1:
                density_test = self.radius_to_dens(atom, radius)
                if abs((density_test - density) / density) > 5e-2:
                    raise InputError.density_error(
                        "Both radius and density are specified but they are not compatible"
                    )
                else:
                    density = density_test
            elif radius == -1 and density == -1:
                raise InputError.density_error(
                    "One of radius or density must be specified"
                )

        return radius, density

    def radius_to_dens(self, atom, radius):
        """
        Convert the Voronoi sphere radius to a mass density
        """

        # radius in cm
        rad_cm = radius / unitconv.cm_to_bohr
        # volume in cm
        vol_cm = (4.0 * pi * rad_cm ** 3) / 3.0
        # atomic mass in g
        mass_g = constants.mp_g * atom.at_mass
        # density in g cm^-3
        density = mass_g / vol_cm

        return density

    def dens_to_radius(self, atom, density):
        """
        Convert the material density to Voronoi sphere radius
        """

        # compute atomic mass in g
        mass_g = constants.mp_g * atom.at_mass
        # compute volume and radius in cm^3/cm
        vol_cm = mass_g / density
        rad_cm = (3.0 * vol_cm / (4.0 * pi)) ** (1.0 / 3.0)
        # convert to a.u.
        radius = rad_cm * unitconv.cm_to_bohr

        return radius

    def check_spinmag(self, spinmag, nele):
        """
        Checks the spin magnetization is compatible with the total electron number
        """
        if isinstance(spinmag, int) == False:
            raise InputError.spinmag_error(
                "Spin magnetization is not a positive integer"
            )

        # computes the default value of spin magnetization
        if spinmag == -1:
            if nele % 2 == 0:
                spinmag = 0
            else:
                spinmag = 1
        elif spinmag > -1:
            if nele % 2 == 0 and spinmag % 2 != 0:
                raise InputError.spinmag_error(
                    "Spin magnetization is not compatible with total electron number"
                )
            elif nele % 2 != 0 and spinmag % 2 == 0:
                raise InputError.spinmag_error(
                    "Spin magnetization is not compatible with total electron number"
                )
        else:
            raise InputError.spinmag_error(
                "Spin magnetization is not a positive integer"
            )

        return spinmag

    def calc_nele(self, spinmag, nele):
        """
        Calculates the electron number in each spin channel from spinmag
        and total electron number
        """

        if config.spindims == 1:
            nele = np.array([nele], dtype=int)
        elif config.spindims == 2:
            nele_up = (nele + spinmag) / 2
            nele_dw = (nele - spinmag) / 2
            nele = np.array([nele_up, nele_dw], dtype=int)

        return nele


class ISModel:
    """
    Checks the inputs for the IS model class
    """

    def check_xc(self, x_func, c_func):
        """
        checks the exchange and correlation functionals are defined by libxc
        """

        # supported families of libxc functional by name
        names_supp = ["lda"]
        # supported families of libxc functional by id
        id_supp = [1]

        # check both the exchange and correlation functionals are valid
        x_func, err_x = xc.check_xc_func(x_func, id_supp)
        c_func, err_c = xc.check_xc_func(c_func, id_supp)

        if err_x == 1:
            raise InputError.xc_error("x functional is not an id (int) or name (str)")
        elif err_c == 1:
            raise InputError.xc_error("c functional is not an id (int) or name (str)")
        elif err_x == 2:
            raise InputError.xc_error(
                "x functional is not a valid name or id.\n \
Please choose from the valid inputs listed here: \n\
https://www.tddft.org/programs/libxc/functionals/"
            )
        elif err_c == 2:
            raise InputError.xc_error(
                "c functional is not a valid name or id. \n\
Please choose from the valid inputs listed here: \n\
https://www.tddft.org/programs/libxc/functionals/"
            )
        elif err_x == 3:
            raise InputError.xc_error(
                "This family of functionals is not yet supported by AvAtom. \n\
Supported families so far are: "
                + " ".join(names_supp)
            )
        elif err_c == 3:
            raise InputError.xc_error(
                "This family of functionals is not yet supported by AvAtom. \n\
Supported families so far are: "
                + " ".join(names_supp)
            )

        return x_func, c_func


class InputError(Exception):
    """
    Handles errors in inputs
    """

    def species_error(err_msg):
        """
        Raises an exception if there is an invalid species

        Inputs:
        - err_msg (str)     : error message printed
        """

        print("Error in atomic species input: " + err_msg)
        print("Species must be a chemical symbol, e.g. 'He'")
        sys.exit("Exiting AvAtom")

    def temp_error(err_msg):
        """
        Raises an exception if temperature is not a float

        Inputs:
        - err_msg (str)     : error message printed
        """
        print("Error in temperature input: " + err_msg)
        print("Temperature should be >0 and given in units of eV")
        sys.exit("Exiting AvAtom")

    def charge_error():
        """
        Raises an exception if charge is not an integer
        """
        print("Error in charge input: charge is not an integer")
        sys.exit("Exiting AvAtom")

    def density_error(err_msg):
        """
        Raises an exception if density is not a float or negative

        Inputs:
        - err_msg (str)     : error message printed
        """
        print("Error in density input: " + err_msg)
        sys.exit("Exiting AvAtom")

    def spinmag_error(err_msg):
        """
        Raises an exception if density is not a float or negative
        """

        print("Error in spinmag input: " + err_msg)
        sys.exit("Exiting AvAtom")

    def xc_error(err_msg):

        """
        Raises an exception if density is not a float or negative
        """

        print("Error in xc input: " + err_msg)
        sys.exit("Exiting AvAtom")


class InputWarning:
    """
    Warns user if inputs are considered outside of typical ranges, but proceeds anyway
    """

    def temp_warning(err):
        """
        Warning if temperature outside of sensible range

        Inputs:
        - temp (float)    : temperature in units of eV
        - err (str)       : "high" or "low"
        """
        print(
            "Warning: this input temperature is very "
            + err
            + ". Proceeding anyway, but results may not be accurate."
        )
        print("Normal temperature range for AvAtom is 0.01 -- 100 eV ")
