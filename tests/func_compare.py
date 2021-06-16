"""
Compares lda and 'exact' xc functional for Hydrogen
"""

from atoMEC import Atom, models

# initialize the atom
atom = Atom("H", radius=4.0, temp=10, units_temp="eV")

# set up the model with lda exchange and correlation
# also use spin polarization
model = models.ISModel(atom, xfunc_id="lda_x", cfunc_id="lda_c_pw", spinpol=True)

# compute the total energy
energy_lda = model.CalcEnergy(20, 3, scf_params={"mixfrac": 0.6})

# now change the exchange and correlation functionals
model.xfunc_id = "hartree"
model.cfunc_id = "None"

# print the model info again
print(model.info)

# compute the total energy again
energy_clmb = model.CalcEnergy(20, 3)

print("Total free energy with LDA :" + str(energy_lda.F_tot))
print("Total free energy with bare clmb :" + str(energy_clmb.F_tot))
