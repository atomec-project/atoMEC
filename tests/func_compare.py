"""
Compares lda and 'exact' xc functional for Hydrogen
"""

import AvAtom

# initialize the atom
atom = AvAtom.Atom("H", radius=4.0, temp=10, units_temp="eV")

# set up the model with lda exchange and correlation
# also use spinpol
model = AvAtom.ISModel(atom, xfunc_id="lda_x", cfunc_id="lda_c_pw", spinpol=True)

# compute the total energy
energy_lda = model.CalcEnergy(3, 3, scf_params={"mixfrac": 0.6})

# now change the exchange and correlation functionals
model.xfunc_id = "hartree"
model.cfunc_id = "None"

# compute the total energy again
energy_clmb = model.CalcEnergy(3, 3, scf_params={"mixfrac": 0.6})

print("Total free energy with LDA :" + str(energy_lda.F_tot))
print("Total free energy with bare clmb :" + str(energy_clmb.F_tot))
