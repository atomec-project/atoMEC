#!/usr/bin/env python3
"""
Simple example using default settings wherever possible
"""

from atoMEC import Atom, models, config

# use all cores
config.numcores = -1

atom_species = "Be"  # helium
r_s = 2.35  # Wigner-Seitz radius of room-temp Be
temperature = 25  # temperature in eV

# initialize the atom object
Be = Atom(atom_species, radius=r_s, temp=temperature, units_temp="eV")

# initialize the model
model = models.ISModel(Be, bc="dirichlet")

# compute the total energy
# define the number of levels to scan for
# note that nmax should be much greater than the actual levels interested in
# and should be tested for convergence
nmax = 40
lmax = 3
scf_output = model.CalcEnergy(nmax, lmax, grid_params={"ngrid": 2000})

# with the energy output compute the pressure
# we print the scf info to see what's happening
pressure = model.CalcPressure(Be, scf_output, write_info=True)
