"""
The pressure module contains functions to compute the pressure with various approaches.

Functions
---------
* :func:`finite_diff` : Calculate electronic pressure with finite-difference method.
* :func:`stress_tensor` : Calculate electronic pressure with stress-tensor method.
* :func:`virial` : Calculate electronic pressure with virial method.
* :func:`calc_Wd_xc` : Calculate the derivative contribution to virial pressure.
* :func:`ions_ideal` : Calculate ionic pressure with ideal gas law.
"""

# external libs
import numpy as np

# internal libs
from atoMEC import mathtools, xc, config
from atoMEC.check_inputs import InputError


def finite_diff(
    atom,
    model,
    orbs,
    pot,
    conv_params={},
    scf_params={},
    force_bound=[],
    write_info=False,
    verbosity=0,
    dR=0.01,
):
    r"""
    Calculate the electronic pressure using the finite differences method.

    Parameters
    ----------
    atom : atoMEC.Atom
        The main atom object
    model : models.ISModel
        The ISModel object
    orbs : staticKS.Orbitals
        the orbitals object
    pot : staticKS.Potential
        the potential object
    conv_params : dict, optional
        dictionary of convergence parameters as follows:
        {
        `econv` (``float``)  : convergence for total energy,
        `nconv` (``float``)  : convergence for density,
        `vconv` (``float``)  : convergence for electron number,
        `eigtol` (``float``) : tolerance for eigenvalues
        }
    scf_params : dict, optional
       dictionary for scf cycle parameters as follows:
       {
       `maxscf`  (``int``)   : maximum number of scf cycles,
       `mixfrac` (``float``) : density mixing fraction
       }
    force_bound : list of list of ints, optional
        force certain levels to be bound, for example:
        `force_bound = [0, 1, 0]`
        forces the orbital quith quantum numbers :math:`\sigma=0,\ l=1,\ n=0` to be
        always bound even if it has positive energy. This prevents convergence
        issues.
    verbosity : int, optional
        how much information is printed at each SCF cycle.
        `verbosity=0` prints the total energy and convergence values (default)
        `verbosity=1` prints the above and the KS eigenvalues and occupations.
    write_info : bool, optional
        prints the scf cycle and final parameters
        defaults to False
    dR : float, optional
        radius difference for finite difference calculation
        defaults to 0.01

    Returns
    -------
    P_e : float
        electronic pressure in Ha
    """
    # set up grid and band dictionaries
    grid_params = {}
    band_params = {}

    # if inheriting params from CalcEnergy original function
    eigfuncs = orbs.eigfuncs
    (
        band_params["nkpts"],
        config.spindims,
        lmax,
        nmax,
        grid_params["ngrid"],
    ) = np.shape(eigfuncs)
    grid_params["x0"] = orbs._xgrid[0]

    # initialize the main radius we are interested in
    main_rad = atom.radius

    # change main radius by +dR
    atom.radius = main_rad + dR

    # calculate free energy for new radius and store it
    output1 = model.CalcEnergy(
        nmax,
        lmax,
        grid_params=grid_params,
        band_params=band_params,
        scf_params=scf_params,
        conv_params=conv_params,
        force_bound=force_bound,
        verbosity=verbosity,
        write_info=write_info,
        guess=True,
        guess_pot=pot.v_s,
        write_density=False,
        write_potential=False,
    )
    F1 = output1["energy"].F_tot

    # change main radius by -dR
    atom.radius = main_rad - dR

    # calculate free energy for new radius and store it
    output2 = model.CalcEnergy(
        nmax,
        lmax,
        grid_params=grid_params,
        band_params=band_params,
        scf_params=scf_params,
        conv_params=conv_params,
        force_bound=force_bound,
        verbosity=verbosity,
        write_info=write_info,
        guess=True,
        guess_pot=pot.v_s,
        write_density=False,
        write_potential=False,
    )
    F2 = output2["energy"].F_tot

    dFdR = (F1 - F2) / (2 * dR)  # finite differences
    dRdV = 1 / (4 * np.pi * main_rad**2)  # V = sphere of radius R (main_rad) volume

    # calculate pressure by thermodynamic definition p = -dFdV and chain rule
    P_e = -dFdR * dRdV

    # convert atom.radius back to its correct value
    atom.radius = main_rad

    return P_e


def stress_tensor(Atom, model, orbs, pot):
    r"""Calculate the pressure with the stress tensor approach [9]_.

    Parameters
    ----------
    orbs : staticKS.Orbitals
        the orbitals object
    pot : staticKS.Potential
        the potential object

    Returns
    -------
    P_e : float
        the electronic pressure

    References
    ----------
    .. [9] G. Faussurier and C. Blancard, Pressure in warm and hot dense matter
       using the average-atom model, Phys. Rev. E 99, 053201 (2019)
       `DOI:10.1103/PhysRevE.99.053201 <https://doi.org/10.1103/PhysRevE.99.053201>`__.
    """
    # retrive the dimensions of the eigenvalues
    config.band_params["nkpts"], config.spindims, config.lmax, config.nmax, config.grid_params["ngrid"] = np.shape(orbs.eigfuncs)
    lmax, nmax = config.lmax, config.nmax
    
    # set the xgrid
    xgrid = orbs._xgrid

    # first compute the gradient of the orbitals
    deriv_orbs = np.gradient(orbs.eigfuncs, xgrid, axis=-1, edge_order=2)

    # chain rule to convert from dP_dx to dX_dr
    grad_orbs = np.exp(-1.5 * xgrid) * (deriv_orbs - 0.5 * orbs.eigfuncs)

    # compute the "gradient" term
    grad_sq = grad_orbs ** 2

    # compute the l*(l+1) array
    l_arr = np.fromiter((l * (l + 1.0) for l in range(lmax)), float, lmax)

    # get the X(R)^2 term
    orb_sq = orbs.eigfuncs ** 2 * np.exp(-xgrid)

    # compute the l(l+1)/r^2 X(R)^2 term
    lsq_term = np.exp(-2 * xgrid) * np.einsum("k,ijklm->ijklm", l_arr, orb_sq)

    # compute the eps * X(R)^2 term
    v_E_arr = (
        -pot.v_s[np.newaxis, :, np.newaxis, np.newaxis, :]
        + orbs.eigvals[:, :, :, :, np.newaxis]
    )
    eps_term = 2 * v_E_arr * orb_sq

    # sum the orbital based terms
    sum_terms = grad_sq + lsq_term + eps_term

    # put everything together
    P_arr = np.einsum("ijkl,ijklm->m", orbs.occnums_w, sum_terms) / 6

    # return the value of P at the sphere edge
    P_e = P_arr[-1]

    return P_e


def virial(atom, model, energy, density):
    r"""Compute the pressure using the virial theorem (see notes).

    Parameters
    ----------
    atom : atoMEC.Atom
        the Atom object
    model : models.ISModel
        the ISModel object
    energy : staticKS.Energy
        the Energy object
    density : staticKS.Density
        the density object

    Returns
    -------
    P_e : float
        the electronic pressure

    Notes
    -----
    The virial pressure is given by the formula [10]_

    .. math::

        P &= \frac{2T + E_\mathrm{en} + E_\mathrm{Ha} + W_\mathrm{xc}}{3V}\ , \\
        W_\mathrm{xc} &= 3 (W^\mathrm{d}_\mathrm{xc} - E_\mathrm{xc})

    References
    ----------
    .. [10] P. Legrand and F. Perrot, Virial theorem and pressure calculations in the
        GGA, J. Phys.: Condens. Matter 13 (2001) 287–301
        `DOI:10.1088/0953-8984/13/2/306
        <https://doi.org/10.1088/0953-8984/13/2/306>`__.
    """
    # compute the sphere volume
    sph_vol = (4.0 * np.pi / 3.0) * atom.radius**3

    # compute the derivative term of the W_xc component
    Wd_x = calc_Wd_xc(model.xfunc_id, density)
    Wd_c = calc_Wd_xc(model.cfunc_id, density)

    # compute total W_xc component
    W_xc = -3 * energy.E_xc["xc"] + 3 * (Wd_x + Wd_c)

    # compute E_V = 2*T + U + W_xc
    E_V = 2 * energy.E_kin["tot"] + energy.E_en + energy.E_ha + W_xc

    # compute the virial pressure
    P_e = E_V / (3 * sph_vol)

    return P_e


def calc_Wd_xc(xc_func_id, density):
    r"""Compute the 'derivative' component of the xc term in virial formula (see notes).

    Parameters
    ----------
    xc_func_id : str
        the exchange or correlation functional id
    density : staticKS.Density
        the density object

    Returns
    -------
    Wd_xc : float
        the derivative component of the xc term in the virial formula.

    Notes
    -----
    In the LDA, Wd_xc is given by [10]_

    .. math::

        W^\mathrm{d}_\mathrm{xc} =  \int \mathrm{d}r v_\mathrm{xc}(r) n(r)

    GGA implementation is still to come.
    """
    # retrieve spin and grid dims
    spindims, ngrid = np.shape(density.total)

    # set up xc_func object
    xc_func = xc.set_xc_func(xc_func_id)

    if xc_func_id == "None":
        return 0.0
    # not set up for GGA funcs yet
    elif xc_func._family == 2:
        raise InputError.xc_error("Virial method not yet set up for GGAs")
    else:
        v_xc = xc.calc_xc(density.total, density._xgrid, xc_func, "v_xc")

    # multiply by density, integrate over sphere and sum over spins
    Wd_xc = 0.0
    for sp in range(spindims):
        Wd_xc += mathtools.int_sphere(density.total[sp] * v_xc[sp], density._xgrid)

    return Wd_xc


def ions_ideal(atom):
    r"""Compute the the ideal gas pressure for the ions.

    In atomic units, the ideal gas pressure is simply P = T/V.

    Parameters
    ----------
    atom : atoMEC.Atom
        the Atom object

    Returns
    -------
    P_ion : float
        the ionic pressure
    """
    # compute the volume
    V = (4.0 * np.pi / 3.0) * atom.radius**3

    # compute pressure with ideal gas law
    P_ion = atom.temp / V

    return P_ion
