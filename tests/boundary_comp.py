"""
Compares dirichlet and neumann boundary conditions for Aluminium
"""

import AvAtom

# initialize the atom
atom = AvAtom.Atom("Al", density=2.7, temp=5, units_temp="eV")

# set up the model with lda exchange and correlation
model = AvAtom.ISModel(atom, bc="dirichlet")

# compute the total energy
energy_dirichlet = model.CalcEnergy(
    4, 5, grid_params={"ngrid": 2000}, scf_params={"mixfrac": 0.7}
)

# now change to neumann bc
model.bc = "neumann"

# compute the total energy again
energy_neumann = model.CalcEnergy(
    4, 4, grid_params={"ngrid": 2000}, scf_params={"mixfrac": 0.7}
)

print("Total free energy with dirichlet :" + str(energy_dirichlet.F_tot))
print("Total free energy with neumann :" + str(energy_neumann.F_tot))
