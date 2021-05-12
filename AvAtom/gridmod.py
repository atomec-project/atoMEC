# import standard packages

# import external packages
import numpy as np
import math

# import internal packages
import config


class GridSetup:
    """
    Holds the numerical grid (both the logarithmic and real space grid)

    Inputs
    - atom (object)      : the main atom object
    """

    def __init__(self, atom):

        xl = config.grid_params["x0"]  # left hand most grid point in x co-ordinates
        xr = math.log(atom.radius)  # right hand most grid point in x co-ords

        # define the x-grid
        self.xgrid = np.linspace(xl, xr, config.grid_params["ngrid"])
        # define the r-grid
        self.rgrid = np.exp(self.xgrid)
