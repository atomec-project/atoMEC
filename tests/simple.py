"""
Simple example using default settings wherever possible
"""

from atoMEC import Atom, models

atom_species = "He"  # helium
r_s = 3.0  # Wigner-Seitz radius
temperature = 0.1  # temp in hartree

# initialize the atom object
He = Atom(atom_species, radius=r_s, temp=temperature)

# initialize the model
model = models.ISModel(He)

# compute the total energy
# define the number of levels to scan for
# note that nmax should be much greater than the actual levels interested in
# and should be tested for convergence
nmax = 20
lmax = 2
energy = model.CalcEnergy(nmax, lmax)

print("Total free energy = ", energy.F_tot)
