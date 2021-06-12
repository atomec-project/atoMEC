"""
Handles everything connected to the exchange-correlation term
"""

# standard libs

# external libs
import pylibxc
import numpy as np

# internal libs
import config
import mathtools


class XCFunc:
    """
    Class which defines the XCFunc object for 'special' (non-libxc) funcs
    Desgined to align with some proprerties of libxc func objects

    Parameters
    ----------
    xc_code: str
         The string identifier for the x/c func

    Attributes
    ----------
    _xc_func_name : str
         String which describes the functional
    _number : int
        Functional id
    _family : str
        The family to which the xc functional belongs
    """

    def __init__(self, xc_code):
        self._xc_func_name = self.get_name(xc_code)
        self._number = self.get_id(xc_code)
        self._family = "special"

    # defines the xc functional name
    @staticmethod
    def get_name(xc_code):
        """
        Parameters
        ----------
        xc_code : str
            String defining the xc functional

        Returns
        -------
        str:
            A name identifying the functional
        """
        if xc_code == "hartree":
            xc_name = "- hartree"
        elif xc_code == "None":
            xc_name = "none"
        return xc_name

    @staticmethod
    # defines an id for the xc functional
    def get_id(xc_code):
        """
        Parameters
        ----------
        xc_code : str
            String defining the xc functional

        Returns
        -------
        int:
            A number identifying the functional
        """

        if xc_code == "hartree":
            xc_number = -1
        elif xc_code == "None":
            xc_number = 0
        return xc_number


def check_xc_func(xc_code, id_supp):
    """
    Checks there is a valid libxc code (or "None" for no x/c term)
    Then initializes the libxc object

    Inputs:
    - xc_code (str / int)     : the name or id of the libxc functional
    - id_supp (int)           : list of supported xc family ids
    Returns:
    - xc_func (object / int)  : the xc functional object from libxc (or 0)
    - err (int)               : the error code
    """

    # check the xc code is either a string descriptor or integer id
    if isinstance(xc_code, (str, int)) == False:
        err = 1
        xc_func_id = 0

    # case when xc code is not a libxc functional
    xc_special_codes = ["hartree", "None"]
    if xc_code in xc_special_codes:
        xc_func_id = xc_code
        err = 0

    # make the libxc object functional
    else:
        # checks if the libxc code is recognised
        try:

            # don't need to worry about polarization yet
            xc_func = pylibxc.LibXCFunctional(xc_code, "unpolarized")

            # check the xc family is supported
            if xc_func._family in id_supp:
                err = 0
            else:
                err = 3
                xc_func = 0
        except KeyError:
            err = 2
            xc_func = 0

    return xc_func._xc_func_name, err


def set_xc_func(xc_code):

    if config.spindims == 2:
        xc_func = pylibxc.LibXCFunctional(xc_code, "polarized")
    else:
        xc_func = pylibxc.LibXCFunctional(xc_code, "unpolarized")

    # initialize the temperature if required
    xc_temp_funcs = ["lda_xc_gdsmfb", "lda_xc_ksdt", 577, 259]

    if xc_code in xc_temp_funcs:
        xc_func.set_ext_params([config.temp])

    return xc_func


def v_xc(density, xgrid, xfunc, cfunc):
    """
    Wrapper function which computes the exchange and correlation potentials
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
    Wrapper function which computes the exchange and correlation energies
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
    Computes the xc energy density and potential
    Returns either the energy or potential depending what is requested
    by xc type
    """

    # determine the dimensions of the xc_arr based on xctype
    if xctype == "e_xc":
        xc_arr = np.zeros((config.grid_params["ngrid"]))
    elif xctype == "v_xc":
        xc_arr = np.zeros_like(density)

    # case where there is no xc func
    if xcfunc._number == 0:
        xc_arr = 0.0

    # special case in which xc = -hartree
    elif xcfunc._number == -1:

        # import the staticKS module
        import staticKS

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
