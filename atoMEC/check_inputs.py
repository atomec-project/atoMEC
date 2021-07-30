"""
The check_inputs module checks the validity of all user-defined inputs.

If inputs are invalid, InputError exceptions are raised. It also assigns
appropriate default inputs where none are supplied.

Classes
-------
* :class:`Atom` : Check the inputs from the :class:`atoMEC.Atom` object.
* :class:`ISModel` : Check the inputs from the :obj:`atoMEC.models.ISModel` class.
* :class:`EnergyCalcs` : Check the inputs from the\
 :func:`atoMEC.models.ISModel.CalcEnergy` function.
* :class:`InputError` : Exit atoMEC and print relevant input error message.
* :class:`InputWarning` : Warn if inputs are considered outside of typical ranges.
"""

# standard python packages
import sys

# external packages
from mendeleev import element
import sqlalchemy.orm.exc as ele_chk
import numpy as np
from math import pi

# internal packages
from . import unitconv
from . import xc
from . import config


# define some custom types

intc = (int, np.integer)  # unfifying type for integers


class Atom:
    """Check the inputs from the Atom class."""

    def check_species(self, species):
        """
        Check the species is a string and corresponds to an actual element.

        Parameters
        ----------
        species : str
            chemical symbol for atomic species

        Returns
        -------
        None

        Raises
        ------
        InputError.species_error
            Chemical symbol is not valid
        """
        if not isinstance(species, str):
            raise InputError.species_error("element is not a string")
        else:
            try:
                return element(species)
            except ele_chk.NoResultFound:
                raise InputError.species_error("invalid element")

    def check_units_temp(self, units_temp):
        """
        Check the units of temperature are accepted.

        Parameters
        ----------
        units_temp : str
            units of temperature

        Returns
        -------
        units_temp : str
            units of temperature (if valid input) converted to lowercase

        Raises
        ------
        InputError.temp_error
            unit of temperature is not accepted, i.e. not one of "ha", "ev" or "k"
        """
        units_accepted = ["ha", "ev", "k"]
        if units_temp.lower() not in units_accepted:
            raise InputError.temp_error("units of temperature are not recognised")
        return units_temp.lower()

    def check_temp(self, temp, units_temp):
        """
        Check the temperature is a float within a sensible range.

        Parameters
        ----------
        temp : float
             temperature (in any accepted units)
        units_temp : str
            units of temperature

        Returns
        -------
        temp : float
            temperature in units of Hartree

        Raises
        ------
        InputError.temp_error
            input temperature is not a positive number
        InputWarning.temp_warning
            input temperature is not inside a well-tested range
        """
        if not isinstance(temp, (float, intc)):
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
                print(InputWarning.temp_warning("low"))
                return temp
            elif temp > 3.5:
                print(InputWarning.temp_warning("high"))
                return temp
            else:
                return temp

    def check_charge(self, charge):
        """
        Check the net charge is an integer.

        Parameters
        ----------
        charge : int
            the net charge

        Returns
        -------
        charge : int
            the net charge (if input valid)

        Raises
        ------
        InputError.charge_error
            if charge is not an integer
        """
        if not isinstance(charge, intc):
            raise InputError.charge_error()
        else:
            return charge

    def check_units_radius(self, units_radius):
        """
        Check the units of radius are accepted.

        Parameters
        ----------
        units_radius : str
            units of radius

        Returns
        -------
        units_radius : str
            units of radius (if accepted) converted to lowercase

        Raises
        ------
        InputError.density_error
            if units of radius are not one of "bohr", "angstrom" or "ang"
        """
        radius_units_accepted = ["bohr", "angstrom", "ang"]
        if units_radius.lower() not in radius_units_accepted:
            raise InputError.density_error("Radius units not recognised")

        units_radius = units_radius.lower()
        return units_radius

    def check_units_density(self, units_density):
        """
        Check the units of density are accepted.

        Parameters
        ----------
        units_density : str
            units of density

        Returns
        -------
        units_density : str
            units of density (if accepted) converted to lowercase

        Raises
        ------
        InputError.density_error
            if units of density are not one of "g/cm3" or "gcm3"
        """
        density_units_accepted = ["g/cm3", "gcm3"]

        if units_density.lower() not in density_units_accepted:
            raise InputError.density_error("Density units not recognised")

        return units_density.lower()

    def check_radius(self, radius, units_radius):
        """
        Check the Wigner-Seitz radius is valid and reasonable.

        Parameters
        ----------
        radius : float or int
            Wigner-Seitz radius (in input units)
        units_radius : str
            input units of radius

        Returns
        -------
        radius : float
             Wigner-Seitz radius in Hartree units (Bohr)

        Raises
        ------
        InputError.density_error
            if the radius is not a positive number > 0.1
        """
        if not isinstance(radius, (float, intc)):
            raise InputError.density_error("Radius is not a number")

        else:
            if units_radius == "angstrom" or units_radius == "ang":
                radius = unitconv.angstrom_to_bohr * radius
            if radius < 0.1:
                raise InputError.density_error(
                    "Radius must be a positive number greater than 0.1"
                )
        return radius

    def check_density(self, density):
        r"""
        Check the mass density is valid and reasonable.

        Parameters
        ----------
        density : float or int
            mass density (in :math:`\mathrm{g\ cm}^{-3}`)

        Returns
        -------
        density : float
            mass density (in :math:`\mathrm{g\ cm}^{-3}`) if input accepted

        Raises
        ------
        InputError.density_error
            if the density is not a positive number <= 100
        """
        if not isinstance(density, (float, intc)):
            raise InputError.density_error("Density is not a number")
        else:
            if density > 100 or density < 0:
                raise InputError.density_error(
                    "Density must be a positive number less than 100"
                )

        return density

    def check_rad_dens_init(self, atom, radius, density, units_radius, units_density):
        """
        Check that at least one of radius or density is specified and reasonable.

        In case both are specified, check they are compatible.

        Parameters
        ----------
        Atom : atoMEC.Atom
            the main Atom object
        radius : float or int
            Wigner-Seitz radius
        density : float or int
            mass density
        units_radius : str
            units of radius
        units_density : str
            units of density

        Returns
        -------
        radius, density : tuple of floats
            the Wigner-Seitz radius and mass density if inputs are valid

        Raises
        ------
        InputError.density_error
            if neither density nor radius is not given, or if one is invalid,
            or if both are given and they are incompatible
        """
        if not isinstance(density, (float, intc)):
            raise InputError.density_error("Density is not a number")
        if not isinstance(radius, (float, intc)):
            raise InputError.density_error("Radius is not a number")
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
                        "Both radius and density are specified but they are not"
                        " compatible"
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
        Convert the Voronoi sphere radius to a mass density.

        Parameters
        ----------
        atom : atoMEC.Atom
            the main Atom object
        radius : float
            the Wigner-Seitz radius

        Returns
        -------
        density : float
            the mass density
        """
        # radius in cm
        rad_cm = radius / unitconv.cm_to_bohr
        # volume in cm
        vol_cm = (4.0 * pi * rad_cm ** 3) / 3.0
        # atomic mass in g
        mass_g = config.mp_g * atom.at_mass
        # density in g cm^-3
        density = mass_g / vol_cm

        return density

    def dens_to_radius(self, atom, density):
        """
        Convert the mass density to a Wigner-Seitz radius.

        Parameters
        ----------
        atom : atoMEC.Atom
            the main Atom object
        density : float
            the mass density

        Returns
        -------
        radius : float
            the Wigner-Seitz radius
        """
        # compute atomic mass in g
        mass_g = config.mp_g * atom.at_mass
        # compute volume and radius in cm^3/cm
        vol_cm = mass_g / density
        rad_cm = (3.0 * vol_cm / (4.0 * pi)) ** (1.0 / 3.0)
        # convert to a.u.
        radius = rad_cm * unitconv.cm_to_bohr

        return radius


class ISModel:
    """Check the inputs for the IS model class."""

    def check_xc(xc_func, xc_type):
        """
        Check the exchange and correlation functionals are accepted.

        Parameters
        ----------
        xc_func : str or int
            the libxc name or id of the x/c functional
        xc_type : str
            type i.e. "exchange" or "correlation"

        Returns
        -------
        xc_func : str
            the libxc name of the x/c functional (if valid input)

        Raises
        ------
        InputError.xc_error
            if xc functional is not a valid libxc input or is not supported
            by the current version of atoMEC
        """
        # supported families of libxc functional by name
        names_supp = ["lda"]
        # supported families of libxc functional by id
        id_supp = [1]

        # check both the exchange and correlation functionals are valid
        xc_func, err_xc = xc.check_xc_func(xc_func, id_supp)

        if err_xc == 1:
            raise InputError.xc_error(
                xc_type + " functional is not an id (int) or name (str)"
            )
        elif err_xc == 2:
            raise InputError.xc_error(
                xc_type
                + " functional is not a valid name or id.\n                 Please"
                " choose from the valid inputs listed here: \n               "
                " https://www.tddft.org/programs/libxc/functionals/"
            )
        elif err_xc == 3:
            raise InputError.xc_error(
                "This family of "
                + xc_type
                + " functionals is not yet supported by atoMEC. \n               "
                " Supported families so far are: " + " ".join(names_supp)
            )

        return xc_func

    def check_unbound(unbound):
        """
        Check the unbound electron input is accepted.

        Parameters
        ----------
        unbound : str
            defines the treatment of the unbound electrons

        Returns
        -------
        unbound : str
            treatment of unbound electrons (if input valid)

        Raises
        ------
        InputError.unbound_error
            if the treatment of unbound electrons is not a valid input
        """
        # list all possible treatments for unbound electrons
        unbound_permitted = ["ideal"]

        # convert unbound to all lowercase
        unbound.lower()

        if not isinstance(unbound, str):
            raise InputError.unbound_error(
                "Unbound electron description is not a string"
            )
        else:
            if unbound not in unbound_permitted:
                err_msg = (
                    "Treatment of unbound electrons not recognised. \n                "
                    " Allowed treatments are: " + [ub for ub in unbound_permitted]
                )
                raise InputError.unbound_error(err_msg)

        return unbound

    def check_bc(bc):
        """
        Check the boundary condition is accepted.

        Parameters
        ----------
        bc : str
            the boundary condition used to solve KS eqns
            (can be either "dirichlet" or "neumann")

        Returns
        -------
        bc : str
            the boundary condition used to solve KS eqns (lowercase)

        Raises
        ------
        InputError.bc_error
            if the boundary condition is not recognised
        """
        # list permitted boundary conditions
        bcs_permitted = ["dirichlet", "neumann"]

        # convert to lowercase
        bc.lower()

        if not isinstance("bc", str):
            raise InputError.bc_error("Boundary condition is not a string")
        else:
            if bc not in bcs_permitted:
                err_msg = (
                    "Boundary condition is not recognised. \n                 Allowed"
                    " boundary conditions are: " + [b for b in bcs_permitted]
                )
                raise InputError.bc_error(err_msg)

        return bc

    def check_spinpol(spinpol):
        """
        Check the spin polarization is a boolean.

        Parameters
        ----------
        spinpol : bool
           whether spin polarized calculation is done

        Returns
        -------
        spinpol : bool
            same as input unless error raised

        Raises
        ------
        InputError.spinpol_error
            if the spin polarization is not a bool
        """
        if not isinstance(spinpol, bool):
            raise InputError.spinpol_error("Spin polarization is not of type bool")

        return spinpol

    def check_spinmag(spinmag, nele):
        """
        Check the spin magnetization is compatible with the total electron number.

        Also compute a default value if none is specified.

        Parameters
        ----------
        spinmag : int
            the spin magnetization (e.g. 1 for a doublet state)
        nele : int
            the total number of electrons

        Returns
        -------
        spinmag : int
            the spin magnetization if input valid

        Raises
        ------
        InputError.spinmag_error
            if spinmag input is not an integer or incompatible with electron number
        """
        if not isinstance(spinmag, intc):
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

    def check_v_shift(v_shift):
        """
        Check the potential shift is a boolean.

        Parameters
        ----------
        v_shift : bool
           whether potential is shifted or not

        Returns
        -------
        v_shift : bool
        same as input unless error raised

        Raises
        ------
        InputError.v_shift_error
            if the potential shift is not a bool
        """
        if not isinstance(v_shift, bool):
            raise InputError.v_shift_error("Potential shift is not of type bool")

        return v_shift

    def calc_nele(spinmag, nele, spinpol):
        """
        Calculate the electron number in each spin channel (if spin polarized).

        Parameters
        ----------
        spinmag : int
            the spin magnetization
        nele : int
            total electron number
        spinpol : bool
            spin polarization

        Returns
        -------
        nele : array of ints
            number of electrons in each spin channel if spin-polarized, else
            just total electron number
        """
        if not spinpol:
            nele = np.array([nele], dtype=int)
        else:
            nele_up = (nele + spinmag) / 2
            nele_dw = (nele - spinmag) / 2
            nele = np.array([nele_up, nele_dw], dtype=int)

        return nele


class EnergyCalcs:
    """Check inputs for CalcEnergy calculations."""

    @staticmethod
    def check_grid_params(grid_params):
        r"""
        Check grid parameters are reasonable, or assigns if empty.

        Parameters
        ----------
        grid_params : dict
            Can contain the keys `ngrid` (``int``, number of grid points)
            and `x0` (`float`, LHS grid point for log grid)

        Returns
        -------
        grid_params : dict
            dictionary of grid parameters as follows:
            {
            `ngrid` (``int``)   : number of grid points,
            `x0`    (``float``) : LHS grid point takes form
            :math:`r_0=\exp(x_0)`; :math:`x_0` can be specified
            }

        Raises
        ------
        InputError.grid_error
            if grid inputs are invalid or outside a reasonable range
        InputError.ngrid_warning
            if `ngrid` is outside a reasonable convergence range
        """
        # First assign the keys ngrid and x0 if they are not given
        try:
            ngrid = grid_params["ngrid"]
        except KeyError:
            ngrid = config.grid_params["ngrid"]

        try:
            x0 = grid_params["x0"]
        except KeyError:
            x0 = config.grid_params["x0"]

        # check that ngrid is an integer
        if not isinstance(ngrid, intc):
            raise InputError.grid_error("Number of grid points not an integer!")
        # check that ngrid is a positive number
        if ngrid < 0:
            raise InputError.grid_error("Number of grid points must be positive")
        elif ngrid < 500:
            print(InputWarning.ngrid_warning("low", "inaccurate"))
        elif ngrid > 5000:
            print(InputWarning.ngrid_warning("high", "expensive"))

        # check that x0 is reasonable
        if x0 > -3:
            raise InputError.grid_error(
                "x0 is too high, calculation will likely not converge"
            )

        grid_params = {"ngrid": ngrid, "x0": x0}

        return grid_params

    @staticmethod
    def check_conv_params(input_params):
        """
        Check convergence parameters are reasonable, or assigns if empty.

        Parameters
        ----------
        input_params : dict of floats
            Can contain the keys `econv`, `nconv` and `vconv`, for energy,
            density and potential convergence parameters

        Returns
        -------
        conv_params : dict of floats
            dictionary of convergence parameters as follows:
            {
            `econv` (``float``) : convergence for total energy,
            `nconv` (``float``) : convergence for density,
            `vconv` (``float``) : convergence for electron number
            }

        Raises
        ------
        InputError.conv_error
            if a convergence parameter is invalid (not float or negative).
        """
        conv_params = {}
        # loop through the convergence parameters
        for conv in ["econv", "nconv", "vconv"]:
            # assign value if not given
            try:
                x_conv = input_params[conv]
            except KeyError:
                x_conv = config.conv_params[conv]

            # check float
            if not isinstance(x_conv, float):
                raise InputError.conv_error(conv + " is not a float!")
            # check > 0
            elif x_conv < 0:
                raise InputError.conv_error(conv + " cannot be negative")

            conv_params[conv] = x_conv

        return conv_params

    @staticmethod
    def check_scf_params(input_params):
        """
        Check scf parameters are reasonable, or assigns if empty.

        Parameters
        ----------
        input_params : dict
            can contain the keys `maxscf` and `mixfrac` for max scf cycle
            and potential mixing fraction

        Returns
        -------
        scf_params : dict
            dictionary with the following keys:
            {
            `maxscf`   (``int``)    : max number scf cycles,
            `mixfrac`  (``float``)    : mixing fraction
            }

        Raises
        ------
        InputError.SCF_error
            if the SCF parameters are not of correct type or in valid range
        """
        scf_params = {}

        # assign value to scf param if it is not specified
        for p in ["maxscf", "mixfrac"]:
            try:
                scf_params[p] = input_params[p]
            except KeyError:
                scf_params[p] = config.scf_params[p]

        # check maxscf is an integer
        maxscf = scf_params["maxscf"]
        if not isinstance(maxscf, intc):
            raise InputError.SCF_error("maxscf is not an integer!")
        # check it is at least 1
        elif maxscf < 1:
            raise InputError.SCF_error("maxscf must be at least 1")

        # check mixfrac is a float
        mixfrac = scf_params["mixfrac"]
        if not isinstance(mixfrac, float):
            raise InputError.SCF_error("mixfrac is not a float!")
        # check it lies between 0,1
        elif mixfrac < 0 or mixfrac > 1:
            raise InputError.SCF_error("mixfrac must be in range [0,1]")

        return scf_params


class InputError(Exception):
    """Exit atoMEC and print relevant input error message."""

    def species_error(err_msg):
        """
        Raise an exception if there is an invalid species.

        Parameters
        ----------
        err_msg : str
            error message printed

        Returns
        -------
        None
        """
        print("Error in atomic species input: " + err_msg)
        print("Species must be a chemical symbol, e.g. 'He'")
        sys.exit("Exiting atoMEC")

    def temp_error(err_msg):
        """
        Raise an exception if temperature is not a float.

        Parameters
        ----------
        err_msg : str
            error message printed

        Returns
        -------
        None
        """
        print("Error in temperature input: " + err_msg)
        print("Temperature should be >0 and given in units of eV")
        sys.exit("Exiting atoMEC")

    def charge_error():
        """
        Raise an exception if charge is not an integer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print("Error in charge input: charge is not an integer")
        sys.exit("Exiting atoMEC")

    def density_error(err_msg):
        """
        Raise an exception if density is not a float or negative.

        Parameters
        ----------
        err_msg : str
            error message printed

        Returns
        -------
        None
        """
        print("Error in density input: " + err_msg)
        sys.exit("Exiting atoMEC")

    def spinmag_error(err_msg):
        """
        Raise an exception if density is not a float or negative.

        Parameters
        ----------
        err_msg : str
            error message printed

        Returns
        -------
        None
        """
        print("Error in spinmag input: " + err_msg)
        sys.exit("Exiting atoMEC")

    def xc_error(err_msg):
        """
        Raise an exception if density is not a float or negative.

        Parameters
        ----------
        err_msg : str
            error message printed

        Returns
        -------
        None
        """
        print("Error in xc input: " + err_msg)
        sys.exit("Exiting atoMEC")

    def unbound_error(err_msg):
        """
        Raise exception if unbound not str or in permitted values.

        Parameters
        ----------
        err_msg : str
            the error message printed

        Returns
        -------
        None
        """
        print("Error in unbound electron input: " + err_msg)
        sys.exit("Exiting atoMEC")

    def bc_error(err_msg):
        """
        Raise exception if unbound not str or in permitted values.

        Parameters
        ----------
        err_msg : str
            the error message printed

        Returns
        -------
        None
        """
        print("Error in boundary condition input: " + err_msg)
        sys.exit("Exiting atoMEC")

    def spinpol_error(err_msg):
        """
        Raise exception if spinpol not a boolean.

        Parameters
        ----------
        err_msg : str
            the error message printed

        Returns
        -------
        None
        """
        print("Error in spin polarization input: " + err_msg)
        sys.exit("Exiting atoMEC")

    def grid_error(err_msg):
        """
        Raise exception if error in grid inputs.

        Parameters
        ----------
        err_msg : str
            the error message printed

        Returns
        -------
        None
        """
        print("Error in grid inputs: " + err_msg)
        sys.exit("Exiting atoMEC")

    def conv_error(err_msg):
        """
        Raise exception if error in convergence inputs.

        Parameters
        ----------
        err_msg : str
            the error message printed

        Returns
        -------
        None
        """
        print("Error in convergence inputs: " + err_msg)
        sys.exit("Exiting atoMEC")

    def SCF_error(err_msg):
        """
        Raise exception if error in SCF inputs.

        Parameters
        ----------
        err_msg : str
            the error message printed

        Returns
        -------
        None
        """
        print("Error in scf_params input: " + err_msg)
        sys.exit("Exiting atoMEC")


class InputWarning:
    """Warn if inputs are considered outside of typical ranges, but proceed anyway."""

    def temp_warning(err):
        """
        Warn if temperature outside of sensible range.

        Parameters
        ----------
        err : str
            custom part of error message

        Returns
        -------
        warning : str
            full errror message
        """
        warning = (
            "Warning: this input temperature is very "
            + err
            + ". Proceeding anyway, but results may not be accurate. \n"
            + "Normal temperature range for atoMEC is 0.01 -- 100 eV \n"
        )
        return warning

    def ngrid_warning(err1, err2):
        """
        Warn if grid params outside of sensible range.

        Parameters
        ----------
        err1 : str
            first custom part of error message
        err2 : str
            second custom part of error message

        Returns
        -------
        warning : str
            full error message
        """
        warning = (
            "Warning: number of grid points is very "
            + err1
            + ". Proceeding anyway, but results may be "
            + err2
            + "\n"
            + "Suggested grid range is between 1000-5000 but should be tested wrt"
            " convergence \n"
        )
        return warning
