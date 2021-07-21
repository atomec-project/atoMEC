"""
The xc module calculates exchange-correlation energies and potentials.

Mostly it sets up inputs and makes appropriate calls to the libxc package.
It also includes a class for customized xc-functionals and checks the
validity of inputs.

Classes
-------
* :class:`XCFunc` : handles atoMEC-defined functionals (not part of libxc package)

Functions
---------
* :func:`check_xc_func` : check the xc func input is recognised and valid for atoMEC
* :func:`set_xc_func`   : initialize the xc functional object
* :func:`v_xc`          : return the xc potential on the grid
* :func:`E_xc`          : return the xc energy
* :func:`calc_xc`       : compute the xc energy or potential depending on arguments
"""

# standard libs

# external libs
import pylibxc
import numpy as np

# internal libs
from . import config
from . import mathtools

# list of special codes for functionals not defined by libxc
xc_special_codes = ["hartree", "None"]


class XCFunc:
    """
    Defines the XCFunc object for 'special' (non-libxc) funcs.

    This is desgined to align with some proprerties of libxc func objects
    in order to make certain general calls more straightforward

    Parameters
    ----------
    xc_code: str
         the string identifier for the x/c func

    Notes
    -----
    The XCFunc object contains no public attributes, but it does contain
    the following private attributes (named so to match libxc):

    _xc_code : str
        the string identifier for the x/c functional (not a libxc attribute)
    _xc_func_name : str
         name of the functional
    _number : int
        functional id
    _family : str
        the family to which the xc functional belongs
    """

    def __init__(self, xc_code):
        self._xc_code = xc_code
        self._family = "special"

        self.__xc_func_name = None
        self.__number = None

    # defines the xc functional name
    @property
    def _xc_func_name(self):
        if self.__xc_func_name is None:
            if self._xc_code == "hartree":
                self.__xc_name = "- hartree"
            elif self._xc_code == "None":
                self.__xc_name = "none"
        return self.__xc_name

    # defines the number id for the xc functional
    @property
    def _number(self):
        if self.__number is None:
            if self._xc_code == "hartree":
                self.__xc_number = -1
            elif self._xc_code == "None":
                self.__xc_number = 0
        return self.__xc_number


def check_xc_func(xc_code, id_supp):
    """
    Check the xc input string or id is valid.

    Parameters
    ----------
    xc_code: str or int
        the name or libxc id of the xc functional
    id_supp: list of ints
        supported families of xc functional

    Returns
    -------
    xc_func_name: str or None
        the xc functional name
    """
    # check the xc code is either a string descriptor or integer id
    if not isinstance(xc_code, (str, int)):
        err = 1

    # when xc code is one of the special atoMEC defined functionals
    if xc_code in xc_special_codes:
        xc_func_name = xc_code
        err = 0

    # make the libxc object functional
    else:
        # checks if the libxc code is recognised
        try:

            xc_func = pylibxc.LibXCFunctional(xc_code, "unpolarized")

            # check the xc family is supported
            if xc_func._family in id_supp:
                err = 0
                xc_func_name = xc_func._xc_func_name
            else:
                err = 3
                xc_func_name = None

        except KeyError:
            err = 2
            xc_func_name = None

    return xc_func_name, err


def set_xc_func(xc_code):
    """
    Initialize the xc functional object.

    Parameters
    ----------
    xc_code: str or int
        the name or id of the libxc functional

    Returns
    -------
    xc_func: :obj:`XCFunc` or :obj:`pylibxc.LibXCFunctional`
    """
    # when xc code is one of the special atoMEC defined functionals
    if xc_code in xc_special_codes:
        xc_func = XCFunc(xc_code)

    else:
        # whether the xc functional is spin polarized
        if config.spindims == 2:
            xc_func = pylibxc.LibXCFunctional(xc_code, "polarized")
        else:
            xc_func = pylibxc.LibXCFunctional(xc_code, "unpolarized")

    return xc_func


def v_xc(density, xgrid, xfunc, cfunc):
    """
    Retrive the xc potential.

    Parameters
    ----------
    density: ndarray
        the KS density on the log grid
    xgrid: ndarray
        the log grid
    xfunc: :obj:`XCFunc` or :obj:`pylibxc.LibXCFunctional`
        the exchange functional object
    cfunc: :obj:`XCFunc` or :obj:`pylibxc.LibXCFunctional`
        the correlation functional object

    Returns
    -------
    _v_xc: dict of ndarrays
        dictionary containing terms `x`, `c` and `xc` for exchange, correlation
        and exchange + correlation respectively
    """
    # initialize the _v_xc dict (leading underscore to distinguish from function name)
    _v_xc = {}

    # compute the exchange potential
    _v_xc["x"] = calc_xc(density, xgrid, xfunc, "v_xc")

    # compute the correlation potential
    _v_xc["c"] = calc_xc(density, xgrid, cfunc, "v_xc")

    # sum to get the total xc potential
    _v_xc["xc"] = _v_xc["x"] + _v_xc["c"]

    return _v_xc


def E_xc(density, xgrid, xfunc, cfunc):
    """
    Retrieve the xc energy.

    Parameters
    ----------
    density: ndarray
        the KS density on the log grid
    xgrid: ndarray
        the log grid
    xfunc: :obj:`XCFunc` or :obj:`pylibxc.LibXCFunctional`
        the exchange functional object
    cfunc: :obj:`XCFunc` or :obj:`pylibxc.LibXCFunctional`
        the correlation functional object

    Returns
    -------
    _E_xc: dict of floats
        dictionary containing terms `x`, `c` and `xc` for exchange, correlation
        and exchange + correlation respectively
    """
    # initialize the _E_xc dict (leading underscore to distinguish from function name)
    _E_xc = {}

    # get the total density
    dens_tot = np.sum(density, axis=0)

    # compute the exchange energy
    ex_libxc = calc_xc(density, xgrid, xfunc, "e_xc")
    _E_xc["x"] = mathtools.int_sphere(ex_libxc * dens_tot, xgrid)

    # compute the correlation energy
    ec_libxc = calc_xc(density, xgrid, cfunc, "e_xc")
    _E_xc["c"] = mathtools.int_sphere(ec_libxc * dens_tot, xgrid)

    # sum to get the total xc potential
    _E_xc["xc"] = _E_xc["x"] + _E_xc["c"]

    return _E_xc


def calc_xc(density, xgrid, xcfunc, xctype):
    """
    Compute the x/c energy density or potential depending on `xctype` input.

    Parameters
    ----------
    density: ndarray
        the KS density on the log grid
    xgrid: ndarray
        the log grid
    xcfunc: :obj:`XCFunc` or :obj:`pylibxc.LibXCFunctional`
        the exchange or correlation functional object
    xctype: str
        the quantity to return, `e_xc` for energy density or `v_xc` for potential

    Returns
    -------
    xc_arr: ndarray
        xc energy density or potential depending on `xctype`
    """
    # initialize the temperature if required
    # this would be better placed in the set_xc_func routine;
    # but for now that does not respond to a change in the atomic temperature
    xc_temp_funcs = ["lda_xc_gdsmfb", "lda_xc_ksdt"]
    if xcfunc._xc_func_name in xc_temp_funcs:
        xcfunc.set_ext_params([config.temp])

    # determine the dimensions of the xc_arr based on xctype
    if xctype == "e_xc":
        xc_arr = np.zeros((config.grid_params["ngrid"]))
    elif xctype == "v_xc":
        xc_arr = np.zeros_like(density)

    # case where there is no xc func
    if xcfunc._number == 0:
        xc_arr[:] = 0.0

    # special case in which xc = -hartree
    elif xcfunc._number == -1:

        # import the staticKS module
        from . import staticKS

        if xctype == "v_xc":
            xc_arr[:] = -staticKS.Potential.calc_v_ha(density, xgrid)
        elif xctype == "e_xc":
            xc_arr = -0.5 * staticKS.Potential.calc_v_ha(density, xgrid)

    else:
        # lda
        if xcfunc._family == 1:
            # lda just needs density as input
            # messy transformation for libxc - why isn't tranpose working??
            rho_libxc = np.zeros((config.grid_params["ngrid"], config.spindims))
            for i in range(config.spindims):
                rho_libxc[:, i] = density[i, :]
            inp = {"rho": rho_libxc}
            # compute the xc potential and energy density
            out = xcfunc.compute(inp)
            # extract the energy density
            if xctype == "e_xc":
                xc_arr = out["zk"].transpose()[0]
            elif xctype == "v_xc":
                xc_arr = out["vrho"].transpose()

    return xc_arr
