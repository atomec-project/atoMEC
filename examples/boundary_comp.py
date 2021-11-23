#!/usr/bin/env python3
"""
Compares dirichlet and neumann boundary conditions for Aluminium
"""

from atoMEC import models, Atom, config

# use parallelization to make things slightly quicker
config.numcores = 5

# initialize the atom
Al = Atom("Al", density=2.7, temp=5, units_temp="eV")

# set up the model with lda exchange and correlation
model = models.ISModel(Al, bc="dirichlet")

# compute the total energy
dirichlet_out = model.CalcEnergy(
    4, 4, grid_params={"ngrid": 1000}, scf_params={"mixfrac": 0.7}
)
energy_dirichlet = dirichlet_out["energy"]

# now change to neumann bc
model.bc = "neumann"

# print new model info
print(model.info)

# compute the total energy again
neumann_out = model.CalcEnergy(25, 4)
energy_neumann = neumann_out["energy"]

print("Total free energy with dirichlet :" + str(energy_dirichlet.F_tot))
print("Total free energy with neumann :" + str(energy_neumann.F_tot))
