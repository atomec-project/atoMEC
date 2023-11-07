"""Test template file for atoMEC."""

from atoMEC import Atom, models, config, mathtools, unitconv
from atoMEC.postprocess import pressure
import json
import time

st_time = time.time()

config.numcores = -1

# load input params and set up the atom and model objects
with open("input.json", "r") as f:
    input = json.load(f)

atom = Atom(input["species"], input["temp"], density=input["rho"], units_temp="eV")
model = models.ISModel(atom, bc="bands", unbound="quantum")

# the scf calculation
out = model.CalcEnergy(
    input["nmax"],
    input["lmax"],
    grid_params={"ngrid": input["ngrid"], "s0": 1e-4},
    conv_params={"nconv": input["nconv"], "vconv": 1e-1, "econv": 1e-2},
    band_params={"nkpts": input["nkpts"]},
    scf_params={"maxscf": 30},
    write_info=True,
    grid_type="sqrt",
)

# stress tensor pressure
P_st_rr = pressure.stress_tensor(
    atom, model, out["orbitals"], out["potential"], only_rr=True
)

# virial pressure
P_vir_nocorr = pressure.virial(
    atom,
    model,
    out["energy"],
    out["density"],
    out["orbitals"],
    out["potential"],
    use_correction=False,
    method="B",
)

# compute ideal pressure
chem_pot = mathtools.chem_pot(out["orbitals"])
P_elec_id = pressure.ideal_electron(atom, chem_pot)

end_time = time.time()
tdiff = end_time - st_time

output_dict = {
    "species": input["species"],
    "temp": input["temp"],
    "rho": input["rho"],
    "P_st_rr": unitconv.ha_to_gpa * P_st_rr,
    "P_vir_nocorr": unitconv.ha_to_gpa * P_vir_nocorr,
    "P_id": unitconv.ha_to_gpa * P_elec_id,
    "time": tdiff,
}

with open("output.json", "w") as f:
    json.dump(output_dict, f)
