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
        xc_func = 0

    # if "None" is chosen (ie no exchange correlation)
    elif xc_code == "None":
        err = 0
        xc_func = 0

    # special code to choose xc = - hartree
    elif xc_code == "hartree":
        err = 0
        xc_func = -1

    # make the libxc object functional
    else:
        # checks if the libxc code is recognised
        try:
            if config.spinpol == True:
                xc_func = pylibxc.LibXCFunctional(xc_code, "polarized")
            else:
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

    # initialize the temperature if required
    xc_temp_funcs = ["lda_xc_gdsmfb", "lda_xc_ksdt", 577, 259]

    if xc_code in xc_temp_funcs:
        xc_func.set_ext_params([config.temp])

    return xc_func, err


def v_xc(density, xfunc, cfunc):
    """
    Wrapper function which computes the exchange and correlation potentials
    """

    # initialize the _v_xc dict (leading underscore to distinguish from function name)
    _v_xc = {}

    # compute the exchange potential
    _v_xc["x"] = calc_xc(density, xfunc, "v_xc")

    # compute the correlation potential
    _v_xc["c"] = calc_xc(density, cfunc, "v_xc")

    # sum to get the total xc potential
    _v_xc["xc"] = _v_xc["x"] + _v_xc["c"]

    return _v_xc


def E_xc(density, xfunc, cfunc):
    """
    Wrapper function which computes the exchange and correlation energies
    """

    # initialize the _E_xc dict (leading underscore to distinguish from function name)
    _E_xc = {}

    # get the total density
    dens_tot = np.sum(density.rho_tot, axis=0)

    # compute the exchange energy
    ex_libxc = calc_xc(density, xfunc, "e_xc")
    _E_xc["x"] = mathtools.int_sphere(ex_libxc * dens_tot)

    # compute the correlation energy
    ec_libxc = calc_xc(density, cfunc, "e_xc")
    _E_xc["c"] = mathtools.int_sphere(ec_libxc * dens_tot)

    # sum to get the total xc potential
    _E_xc["xc"] = _E_xc["x"] + _E_xc["c"]

    return _E_xc


def calc_xc(density, xcfunc, xctype):
    """
    Computes the xc energy density and potential
    Returns either the energy or potential depending what is requested
    by xc type
    """

    # case where there is no xc func
    if xcfunc == 0:
        xc_arr = np.zeros_like(density.rho_tot)

    # special case in which xc = -hartree
    elif xcfunc == -1:

        # import the staticKS module
        import staticKS

        if xctype == "v_xc":
            xc_arr = np.zeros_like(density.rho_tot)
            xc_arr[:] = -staticKS.Potential.calc_v_ha(density)
        elif xctype == "e_xc":
            xc_arr = -0.5 * staticKS.Potential.calc_v_ha(density)

    else:
        # lda
        if xcfunc._family == 1:
            # lda just needs density as input
            # messy transformation for libxc - why isn't tranpose working??
            rho_libxc = np.zeros((config.grid_params["ngrid"], config.spindims))
            for i in range(config.spindims):
                rho_libxc[:, i] = density.rho_tot[i, :]
            inp = {"rho": rho_libxc}
            # compute the xc potential and energy density
            out = xcfunc.compute(inp)
            # extract the energy density
            if xctype == "e_xc":
                xc_arr = out["zk"].transpose()[0]
            elif xctype == "v_xc":
                xc_arr = out["vrho"].transpose()

    return xc_arr
