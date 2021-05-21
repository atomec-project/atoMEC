"""
Handles everything connected to the exchange-correlation term
"""

# standard libs

# external libs
import pylibxc
import numpy as np

# internal libs
import config
from staticKS import Orbitals


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

    # make the libxc object functional
    else:
        # checks if the libxc code is recognised
        try:
            if config.spinpol == "True":
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


class XCPotential:

    """
    Holds the XC potential object and the routines to compute it
    Inputs:
    - density (object)     : the density object
    - xfunc   (object)     : the exchange functional
    - cfunc   (object)     : the correlation functional
    """

    def __init__(self, density, xfunc, cfunc):

        # initialize the properties
        self.density = density
        self.xfunc = xfunc
        self.cfunc = cfunc
        self._vx = None
        self._vc = None
        self._vxc = None

        # compute or retrieve the x and c potentials

    @property
    def vx(self):
        if self._vx is None:
            self._vx = self.calc_vxc(self.density, self.xfunc)
        return self._vx

    @property
    def vc(self):
        if self._vc is None:
            self._vc = self.calc_vxc(self.density, self.cfunc)
        return self._vc

    @property
    def vxc(self):
        if self._vxc is None:
            if self._vx is None or self._vc is None:
                raise Exception(
                    "Cannot compute v_xc as either vx or vc is uninitialized"
                )
            else:
                self._vxc = self._vx + self._vc
        return self._vxc

    @staticmethod
    def calc_vxc(density, xcfunc):
        """
        Calls libxc to compute the chosen exchange or correlation potential
        """

        # case where there is no xc func
        if xcfunc == 0:
            vxc = np.zeros_like(density.rho_tot)

        else:
            vxc = np.zeros_like(density.rho_tot)

            # lda
            if xcfunc._family == 1:
                # lda just needs density as input
                inp = {"rho": density.rho_tot}
                # compute the xc potential and energy density
                out = xcfunc.compute(inp)
                # extract the potential
                vxc_libxc = out["vrho"][:, 0]

            # tranpose back to AvAtom array style
            vxc = np.reshape(vxc_libxc, (config.spindims, config.grid_params["ngrid"]))

        return vxc
