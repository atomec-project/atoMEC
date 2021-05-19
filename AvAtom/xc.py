"""
Handles everything connected to the exchange-correlation term
"""

# standard libs

# external libs
import pylibxc

# internal libs
import config


def check_xc_func(xc_code):
    """
    Checks there is a valid libxc code (or "None" for no x/c term)
    Then initializes the libxc object

    Inputs:
    - xc_code (str / int)     : the name or id of the libxc functional
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

    # make the libxc object functional
    else:
        try:
            err = 0
            if config.spinpol == "True":
                xc_func = pylibxc.LibXCFunctional(xc_code, "polarized")
            else:
                xc_func = pylibxc.LibXCFunctional(xc_code, "unpolarized")

        except KeyError:
            err = 2
            xc_func = 0

    # initialize the temperature if required
    xc_temp_funcs = ["lda_xc_gdsmfb", "lda_xc_ksdt", 577, 259]

    if xc_code in xc_temp_funcs:
        func.set_ext_params([config.temp])

    return xc_func, err
