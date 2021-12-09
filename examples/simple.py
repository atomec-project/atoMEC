#!/usr/bin/env python3
"""
Simple example using default settings wherever possible
"""

from atoMEC import Atom, models, config

config.numcores = 0

atom_species = "He"  # helium
r_s = 3.0  # Wigner-Seitz radius
temperature = 0.01  # temp in hartree

# initialize the atom object
He = Atom(atom_species, radius=r_s, temp=temperature)

# initialize the model
model = models.ISModel(He, bc="neumann")

# compute the total energy
# define the number of levels to scan for, should be tested for convergence
nmax = 2
lmax = 2
output = model.CalcEnergy(nmax, lmax, grid_params={"ngrid": 1000})

print("Total free energy = ", output["energy"].F_tot)
