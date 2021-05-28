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
import convergence


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
    """

    def __init__(
        self,
        species,
        temp,
        radius=-1,
        density=-1,
        charge=0,
        units_temp="ha",
        units_radius="bohr",
        units_density="g/cm3",
    ):

        # Input variables
        self.species = check_inputs.Atom().check_species(species)
        # self.density = check_inputs.Atom.check_density(density)
        self.temp = check_inputs.Atom().check_temp(temp, units_temp)
        self.charge = check_inputs.Atom().check_charge(charge)

        # Fundamental atomic properties
        self.at_chrg = self.species.atomic_number  # atomic number
        config.Z = self.at_chrg
        self.at_mass = self.species.atomic_weight  # atomic mass
        self.nele = self.at_chrg + self.charge  # total electron number

        # Check the radius and density
        self.radius, self.density = check_inputs.Atom().check_density(
            self,
            radius,
            density,
            units_radius,
            units_density,
        )

        config.r_s = self.radius
        self.volume = (4.0 * pi * self.radius ** 3.0) / 3.0
        config.sph_vol = self.volume

        # set temperature and inverse temperature
        config.temp = self.temp
        config.beta = 1.0 / self.temp


class ISModel:
    def __init__(
        self,
        atom,
        xfunc=config.xfunc,
        cfunc=config.cfunc,
        bc=config.bc,
        spinpol=config.spinpol,
        spinmag=-1,
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
        - spinmag (int)    : spin-magentization
        - unbound  (str)   : treatment of unbound electrons
        """

        # Input variables

        # check the spin polarization
        self.spinpol = spinpol
        config.spinpol = self.spinpol

        # set the spinpol param (leading dimension for density, orbitals etc)
        if config.spinpol == True:
            config.spindims = 2
        else:
            config.spindims = 1

        # spin magnetization has to be compatible with total electron number
        self.spinmag = check_inputs.Atom().check_spinmag(spinmag, atom.nele)

        # calculate electron number in (each) spin channel
        self.nele = check_inputs.Atom().calc_nele(self.spinmag, atom.nele)
        config.nele = self.nele

        print("nele = ", config.nele)

        # check the xc functionals
        self.xfunc = xfunc
        self.cfunc = cfunc
        config.xfunc, config.cfunc = check_inputs.ISModel().check_xc(xfunc, cfunc)

        self.bc = bc
        config.bc = self.bc

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
        # use coulomb potential as initial guess
        v_init = staticKS.Potential.calc_v_en()
        orbs.compute(v_init, init=True)

        # occupy orbitals
        orbs.occupy()

        # write the initial spiel
        # scf_string = self.print_scf_init()

        # initialize the convergence object
        conv = convergence.SCF()

        for iscf in range(config.scf_params["maxscf"]):

            # construct density
            rho = staticKS.Density(orbs)

            # construct potential
            pot = staticKS.Potential(rho)

            # compute energies
            energy = staticKS.Energy(orbs, rho)
            E_free = energy.F_tot

            # mix potential
            if iscf > 1:
                alpha = config.scf_params["mixfrac"]
                v_s = alpha * pot.v_s + (1 - alpha) * v_s_old
            else:
                v_s = pot.v_s

            # update the orbitals with the KS potential
            orbs.compute(v_s)
            orbs.occupy()

            # update old potential
            v_s_old = v_s

            # test convergence
            conv_vals = conv.check_conv(E_free, v_s, rho.total, iscf)

            #     scf_string = self.print_scf_cycle(conv_vals)
            #     print(scf_string)

            if conv_vals["complete"]:
                break

            print(
                iscf,
                E_free,
                conv_vals["dE"],
                np.max(conv_vals["dpot"]),
                np.max(conv_vals["drho"]),
            )

        print(orbs.eigvals)
        print(orbs.occnums)

        print(rho.unbound["N"])

        print("kinetic energy = ", energy.E_kin)
        print("electron-nuclear energy = ", energy.E_en)
        print("Hartree energy = ", energy.E_ha)
        print("xc energy = ", energy.E_xc)
        print("entropy = ", energy.entropy)

        rho.write_to_file()


#     # construct density
#     rho.construct(orbs)

#     # construct potential
#     pot.construct(rho)

#     # mix potential
#     pot.mix()

#     # compute the energy
#     energy.compute(orbs, rho, pot)

#     # test convergence
#     conv_vals = convergence.SCF_check(energy, rho, pot)

#     scf_string = self.print_scf_cycle(conv_vals)
#     print(scf_string)

#     if conv_vals["complete"]:
#         break

# scf_string = self.print_scf_complete(conv_vals)
