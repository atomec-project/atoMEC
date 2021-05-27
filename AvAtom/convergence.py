"""
Module which handles convergence routines
"""

# standard libraries

# external libraries
import numpy as np

# internal modules
import mathtools
import config


class SCF:
    """
    Handles any routines related to the SCF cycle
    """

    def __init__(self):

        self.conv_vals = {
            "dE": 0.0,
            "drho": np.zeros((2)),
            "dpot": np.zeros((2)),
            "complete": False,
        }

        self._energy = np.zeros((2))
        self._potential = np.zeros((2, config.spindims, config.grid_params["ngrid"]))
        self._density = np.zeros((2, config.spindims, config.grid_params["ngrid"]))

    def check_conv(self, E_free, pot, dens, iscf):

        conv_vals = {}

        # first update the energy, potential and density attributes
        self._energy[0] = E_free
        self._potential[0] = pot
        self._density[0] = dens

        # compute the change in energy
        conv_vals["dE"] = abs(self._energy[0] - self._energy[1])

        # compute the change in potential
        dv = np.abs(self._potential[0] - self._potential[1])
        # integrate over sphere to return a number
        conv_vals["dpot"] = mathtools.int_sphere(dv)

        # compute the change in density
        dn = np.abs(self._density[0] - self._density[1])
        # integrate over sphere to return a number
        conv_vals["drho"] = mathtools.int_sphere(dn)

        # reset the energy, potential and density attributes
        self._energy[1] = E_free
        self._potential[1] = pot
        self._density[1] = dens

        # see if the convergence criteria are satisfied
        conv_energy = False
        conv_rho = False
        conv_pot = False
        conv_vals["complete"] = False

        if iscf > 3:
            if conv_vals["dE"] < config.conv_params["econv"]:
                conv_energy = True
            if np.all(conv_vals["drho"] < config.conv_params["nconv"]):
                conv_rho = True
            if np.all(conv_vals["dpot"] < config.conv_params["vconv"]):
                conv_pot = True
            if conv_energy and conv_rho and conv_pot:
                conv_vals["complete"] = True

        return conv_vals
