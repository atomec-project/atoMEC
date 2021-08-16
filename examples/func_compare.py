#!/usr/bin/env python3
"""
Compares lda and 'exact' xc functional for Hydrogen
"""

from atoMEC import Atom, models, config

config.numcores = 0

# initialize the atom
atom = Atom("H", radius=4.0, temp=0.01)

# set up the model with lda exchange and correlation
# also use spin polarization
model = models.ISModel(atom, xfunc_id="lda_x", cfunc_id="lda_c_pw", spinpol=True)

# compute the total energy
lda_out = model.CalcEnergy(20, 3, scf_params={"mixfrac": 0.6})
energy_lda = lda_out["energy"]

# now change the exchange and correlation functionals
model.xfunc_id = "hartree"
model.cfunc_id = "None"

# print the model info again
print(model.info)

# compute the total energy again
clmb_out = model.CalcEnergy(20, 3)
energy_clmb = clmb_out["energy"]

print("Total free energy with LDA :" + str(energy_lda.F_tot))
print("Total free energy with bare clmb :" + str(energy_clmb.F_tot))
