# import standard packages

# import external packages
import numpy as np
from mendeleev import element
from math import pi

# import internal packages
import unit_conv
import constants
import check_inputs
import config
import staticKS
import gridmod


class Atom:
    """
    The Atom class defines the main object which is to be used for calculations

    Mandatory inputs:
    - species (str)    : atomic species
    - temp (float)     : system temperature in eV
    - density (float)  : material density (in g cm^-3)
    Optional inputs:
    - charge (int)     : net charge
    """

    def __init__(self, species, density, temp, charge=0):

        print("Initializing AvAtom calculation")

        # Input variables
        self.species = check_inputs.Atom.check_species(species)
        self.density = check_inputs.Atom.check_density(density)
        self.temp = check_inputs.Atom.check_temp(temp)
        self.charge = check_inputs.Atom.check_charge(charge)

        # Fundamental atomic properties
        self.at_chrg = self.species.atomic_number  # atomic number
        self.at_mass = self.species.atomic_weight  # atomic mass
        self.nele = self.at_chrg + self.charge

        # Compute the radius and volume of average atom model

        # compute atomic mass in g
        mass_g = constants.mp_g * self.at_mass
        # compute volume and radius in cm^3/cm
        vol_cm = mass_g / self.density
        rad_cm = (3.0 * vol_cm / (4.0 * pi)) ** (1.0 / 3.0)
        # Convert to a.u.
        self.radius = unit_conv.cm_to_bohr(rad_cm)
        self.volume = (4.0 * pi * self.radius ** 3.0) / 3.0

    class ISModel:
        def __init__(
            self,
            xfunc=config.xfunc,
            cfunc=config.cfunc,
            bc=config.bc,
            spinpol=config.spinpol,
            unbound=config.unbound,
        ):
            """
            Defines the parameters used for an energy calculation.
            These are choices for the theoretical model, not numerical parameters for implementation

            Inputs (all optional):
            - xfunc    (str)   : code for libxc exchange functional     (use "None" for no exchange func)
            - cfunc    (str)   : code for libxc correlation functional  (use "None" for no correlation func)
            - bc       (int)   : choice of boundary condition (1 or 2)
            - spinpol  (bool)  : spin-polarized calculation
            - unbound  (str)   : treatment of unbound electrons
            """

            # Input variables
            self.xfunc = xfunc
            config.xfunc = self.xfunc
            self.cfunc = cfunc
            config.cfunc = self.cfunc
            self.bc = bc
            config.bc = self.bc
            self.spinpol = spinpol
            config.spinpol = self.spinpol
            self.unbound = unbound
            config.unbound = self.unbound

    def CalcEnergy(
        self,
        grid_params=config.grid_params,
        conv_params=config.conv_params,
        scf_params=config.scf_params,
    ):

        """
        Runs a self-consistent calculation to minimize the Kohn--Sham free energy functional

        Inputs (optional):
        - grid_params (dict)   : dictionary of grid parameters as follows
          {'ngrid'    (int)    : number of grid points
           'x0'       (float)  : LHS grid point takes form r0=exp(x0); x0 can be specified }
        - conv_params (dict)   : dictionary of convergence parameters as follows
          {'econv'    (float)  : convergence for total energy
           'nconv'    (float)  : convergence for density
           'numconv'  (float)  : convergence for electron number}
        - scf_params  (dict)   : dictionary for scf cycle parameters as follows
          {'maxscf'   (int)    : maximum number of scf cycles
           'mixfrac'  (float)  : density mixing fraction}
        """

        # reset global parameters if they are changed
        config.grid_params = grid_params
        config.conv_params = conv_params
        config.scf_params = scf_params

        # set up the grids
        grid = gridmod.GridSetup(self)

        # initialize orbitals
        orbs = staticKS.Orbitals()
        orbs.SCF_init(self, grid)
        # print(orbs.eigfuncs)
        # occupy orbitals
        # orbs.occupy()
        # construct density
        # rho=KSvars.Density.construct(orbs,grid.xgrid)
        # construct potential
        # pot=KSvars.Potential.construct(rho,grid.xgrid)
