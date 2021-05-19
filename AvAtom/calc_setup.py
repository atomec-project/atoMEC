# import standard packages

# import external packages
import numpy as np
from mendeleev import element
from math import pi

# import internal packages
import constants
import check_inputs
import config
import staticKS
import gridmod
import xc


class Atom:
    """
    The Atom class defines the main object which is to be used for calculations

    Mandatory inputs:
    - species (str)     : atomic species
    - temp (float)      : system temperature in eV
    One of the following must be specified:
    - radius (float)    : radius of Voronoi sphere
    - density (float)   : material density (in g cm^-3)
    Optional inputs:
    - charge (int)      : net charge
    - spinmag (int>0)   : spin magnetization (default -1 assigns spin automatically)
    """

    def __init__(self, species, temp, radius=-1, density=-1, charge=0, spinmag=-1):

        print("Initializing AvAtom calculation")

        # Input variables
        self.species = check_inputs.Atom().check_species(species)
        # self.density = check_inputs.Atom.check_density(density)
        self.temp = check_inputs.Atom().check_temp(temp)
        self.charge = check_inputs.Atom().check_charge(charge)

        # Fundamental atomic properties
        self.at_chrg = self.species.atomic_number  # atomic number
        config.Z = self.at_chrg
        self.at_mass = self.species.atomic_weight  # atomic mass
        nele_tot = self.at_chrg + self.charge  # total electron number

        # Check the radius and density
        self.radius, self.density = check_inputs.Atom().check_density(
            self, radius, density
        )

        # spin magnetization has to be compatible with total electron number
        self.spinmag = check_inputs.Atom().check_spinmag(spinmag, nele_tot)

        # calculate electron number in each spin channel
        self.nele = check_inputs.Atom().calc_nele(self.spinmag, nele_tot)
        config.nele = self.nele

        config.r_s = self.radius
        self.volume = (4.0 * pi * self.radius ** 3.0) / 3.0
        config.sph_vol = self.volume

        # compute inverse temperature
        config.temp = self.temp
        config.beta = 1.0 / self.temp

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

            # check the xc functionals are ok
            self.xfunc = xfunc
            self.cfunc = cfunc
            config.xfunc, config.cfunc = check_inputs.ISModel().check_xc(xfunc, cfunc)

            self.bc = bc
            config.bc = self.bc
            self.spinpol = spinpol
            config.spinpol = self.spinpol
            if config.spinpol == True:
                config.spindims = 2
            else:
                config.spindims = 1
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
        gridmod.grid_setup()

        # initialize orbitals
        orbs = staticKS.Orbitals()
        orbs.SCF_init(self)
        # occupy orbitals
        orbs.occupy()
        print("Eigenvalues")
        print(orbs.eigvals)
        # print(orbs.occnums)
        print("Occupations")
        print(orbs.occnums)
        # construct density
        rho = staticKS.Density()
        rho.construct(orbs)
        print("Unbound electrons")
        print(rho.N_unbound)
        rho.write_to_file()
        print("Computing KS potential")
        # construct potential
        pot = staticKS.Potential()
        pot.construct(rho)
