"""
This module checks inputs for errors
"""

# standard python packages
import sys

# external packages
from mendeleev import element
import sqlalchemy.orm.exc as ele_chk
import numpy as np

# internal packages


class Atom:
    """
    Checks the inputs from the BuildAtom class
    """

    def check_species(species):
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

    def check_temp(temp):
        """
        Checks the temperature is a float within a sensible range
        """
        if isinstance(temp, (float, int)) == False:
            raise InputError.temp_error("temperature is not a number")
        else:
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

    def check_charge(charge):
        """
        Checks the charge is an integer
        """
        if isinstance(charge, int) == False:
            raise InputError.charge_error()
        else:
            return charge

    def check_density(density):
        """
        Checks the charge is an integer
        """
        if isinstance(density, (float, int)) == False:
            raise InputError.density_error("Density is not a number")
        else:
            if density <= 0.0:
                raise InputError.density_error("Density is negative!")
            else:
                return density

    def check_spinmag(spinmag, nele):
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

    def calc_nele(spinmag, nele):
        """
        Calculates the electron number in each spin channel from spinmag
        and total electron number
        """

        nele_up = (nele + spinmag) / 2
        nele_dw = (nele - spinmag) / 2

        return np.array([nele_up, nele_dw])


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
