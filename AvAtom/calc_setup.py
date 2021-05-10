# import standard packages

# import external packages
import numpy as np
from mendeleev import element
from math import pi

# import internal packages
import unit_conv
import constants
import check_inputs

class BuildAtom:
    '''
    Initializes the atom to be used in all calculations
    
    Mandatory inputs:
    - species (str)    : atomic species 
    - temp (float)     : system temperature in eV
    - density (float)  : material density (in g cm^-3)
    Optional inputs:
    - charge (int)     : net charge
    '''

    def __init__(self,species,density,temp,charge=0):

        print("Initializing AvAtom calculation")

        # Input variables
        self.species  = check_inputs.Atom.check_species(species)
        self.temp     = check_inputs.Atom.check_temp(temp)
        self.charge   = check_inputs.Atom.check_charge(charge)
        self.density  = check_inputs.Atom.check_density(density)

        # Fundamental atomic properties
        self.at_chrg = self.species.atomic_number # atomic number
        self.at_mass = self.species.atomic_weight # atomic mass
        self.nele = self.at_chrg + self.charge
        
        # Compute the radius and volume of average atom model

        # compute atomic mass in g
        mass_g = constants.mp_g*self.at_mass
        # compute volume and radius in cm^3/cm
        vol_cm = mass_g/self.density
        rad_cm = (3.*vol_cm/4.*pi)**(1./3.)
        # Convert to a.u.
        self.radius = unit_conv.cm_to_bohr(rad_cm)
        self.volume = (4.*pi*self.radius**3.)/3.


class Energy:


    def __init__(self,xfunc='LDA',cfunc='LDAPW',bc=1,spinpol=False,unbound='Ideal'):
        '''
        Defines the parameters used for an energy calculation.
        These are choices for the theoretical model, not numerical parameters for implementation
        
        Inputs (all optional):
        - xfunc    (str)   : code for libxc exchange functional     (use "None" for no exchange func)
        - cfunc    (str)   : code for libxc correlation functional  (use "None" for no correlation func)
        - bc       (int)   : choice of boundary condition (1 or 2)
        - spinpol  (bool)  : spin-polarized calculation
        - unbound  (str)   : treatment of unbound electrons
        '''

        #Input variables
        self.xfunc=xfunc
        self.cfunc=func
        self.bc=bc
        self.spinpol=spinpol
        self.unbound=unbound
