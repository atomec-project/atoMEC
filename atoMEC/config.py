"""Configuration file to store global parameters."""

# physical constants
mp_g = 1.6726219e-24  # mass of proton in grams

# model parameters
spinpol = False  # spin-polarized functional
xfunc_id = "lda_x"  # exchange functional (libxc ref)
cfunc_id = "lda_c_pw"  # correlation functional (libxc ref)
bc = "bands"  # boundary condition
unbound = "quantum"  # treatment for unbound electrons
v_shift = True  # whether to shift the KS potential vertically

# numerical grid for static calculations
grid_params = {"ngrid": 500, "x0": -12, "ngrid_coarse": 300, "s0": 1e-3}
# convergence parameters for static calculations
conv_params = {
    "econv": 1.0e-6,
    "nconv": 1.0e-5,
    "vconv": 1.0e-4,
    "eigtol": 1.0e-4,
    "bandtol": 1.0e-3,
}
# scf parameters
scf_params = {"maxscf": 50, "mixfrac": 0.3}
# band parameters for massacrier band model
band_params = {"nkpts": 30, "de_min": 1e-3}

# forced bound energy levels (default none)
force_bound = []

grid_type = "sqrt"

# parallelization
numcores = 0  # defaults to serial
