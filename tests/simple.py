"""
Simple example using default settings wherever possible
"""

import AvAtom

atom_species = "He"  # helium
r_s = 3.0  # Wigner-Seitz radius
temperature = 0.1  # temp in hartree

# initialize the atom object
He_atom = AvAtom.Atom(atom_species, radius=r_s, temp=temperature)

# initialize the model
model = AvAtom.ISModel(He_atom)

# compute the total energy
# define the number of levels to scan for
nmax = 2
lmax = 2
energy = model.CalcEnergy(lmax, nmax)

print("Total free energy = ", energy.F_tot)
