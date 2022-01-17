"""
Contains the routines which generate printed output (to screen and separate files).

Classes
-------
* :class:`SCF` : Write information about the self-consistent field (SCF) cycle.

Functions
---------
* :func:`write_atomic_data` : Write information about the main Atom object.
* :func:`write_ISModel_data` : Write information about the IS model.
* :func:`density_to_csv` : Write the KS density to file.
* :func:`potential_to_csv` : Write the KS potential to file.
* :func:`timing`: Generate timing information for given input function.
"""

# standard libs
from functools import wraps
import time

# external libs
import numpy as np
import tabulate

# internal libs
from . import unitconv
from . import config

# define some line spacings
spc = "\n"
dblspc = "\n \n"


def write_atomic_data(atom):
    """
    Write information about the main Atom object.

    Parameters
    ----------
    atom : atoMEC.Atom
        the main Atom object

    Returns
    -------
    output_str : str
        formatted description of the Atom attributes
    """
    # the initial spiel
    init_str = "Atomic information:" + dblspc

    # information about the atomic species
    species_str = "{preamble:30s}: {species:2s} ".format(
        preamble="Atomic species", species=atom.species.symbol
    )
    at_chrg_str = "{preamble:30s}: {chrg:<3d} / {weight:<.3f}".format(
        preamble="Atomic charge / weight", chrg=atom.at_chrg, weight=atom.at_mass
    )
    nvalence_str = "{preamble:30s}: {nval:<3d}".format(
        preamble="Valence electrons", nval=atom.nvalence
    )
    spec_info = species_str + spc + at_chrg_str + spc + nvalence_str + spc

    # information about the atomic / mass density
    rho_str = "{preamble:30s}: {rho:<.3g} g cm^-3".format(
        preamble="Mass density", rho=atom.density
    )
    rad_ang = atom.radius / unitconv.angstrom_to_bohr
    rad_str = "{preamble:30s}: {rad_b:<.4g} Bohr / {rad_a:<.4g} Angstrom".format(
        preamble="Voronoi sphere radius", rad_b=atom.radius, rad_a=rad_ang
    )
    rho_info = rho_str + spc + rad_str + spc

    # information about the temperature
    temp_ev = atom.temp / unitconv.ev_to_ha
    temp_K = atom.temp / unitconv.K_to_ha
    temp_str = "{preamble:30s}: {t_ha:<.4g} Ha /  {t_ev:<.4g} eV / {t_k:<.4g} K".format(
        preamble="Electronic temperature", t_ha=atom.temp, t_ev=temp_ev, t_k=temp_K
    )
    temp_info = temp_str + spc

    # information about the dimensionless parameters
    WS_radius_str = "{preamble:30s}: {rad:<.4g} (Bohr)".format(
        preamble="Wigner-Seitz radius", rad=atom.WS_radius
    )
    gamma_i_str = "{preamble:30s}: {gamma:<.4g}".format(
        preamble="Ionic coupling parameter", gamma=atom.gamma_ion
    )
    theta_e_str = "{preamble:30s}: {theta:<.4g}".format(
        preamble="Electron degeneracy parameter", theta=atom.theta_e
    )

    dim_params_str = WS_radius_str + spc + gamma_i_str + spc + theta_e_str + spc

    # put all into a single string
    output_str = init_str + spec_info + rho_info + temp_info + dim_params_str + spc

    return output_str


def write_ISModel_data(ISModel):
    """
    Write information about the approximations used for the IS model.

    Parameters
    ----------
    ISModel : models.ISModel
        The ISModel object

    Returns
    -------
    output_str : str
        formatted description of the ISModel attributes
    """
    # the initial spiel
    init_str = "Using Ion-Sphere model" + spc + "Ion-sphere model parameters: " + dblspc

    # spin-pol information
    spinpol_str = "{preamble:30s}: {spin}".format(
        preamble="Spin-polarized", spin=ISModel.spinpol
    )
    spinpol_info = spinpol_str + spc

    # number of electrons in each spin channel info
    nele = ISModel.nele
    if ISModel.spinpol:
        pre = "Number of up / down electrons"
        nele_str = "{preamble:30s}: {Nu} / {Nd}".format(
            preamble=pre, Nu=nele[0], Nd=nele[1]
        )
    else:
        pre = "Number of electrons"
        nele_str = "{preamble:30s}: {Ne}".format(preamble=pre, Ne=nele[0])
    nele_info = nele_str + spc

    # exchange functional
    xfunc_str = "{preamble:30s}: {xfunc}".format(
        preamble="Exchange functional", xfunc=ISModel.xfunc_id
    )
    cfunc_str = "{preamble:30s}: {cfunc}".format(
        preamble="Correlation functional", cfunc=ISModel.cfunc_id
    )
    xc_info = xfunc_str + spc + cfunc_str + spc

    # boundary condition
    bc_str = "{preamble:30s}: {bc}".format(preamble="Boundary condition", bc=ISModel.bc)
    bc_info = bc_str + spc

    # unbound electrons
    ub_str = "{preamble:30s}: {ub}".format(
        preamble="Unbound electron treatment", ub=ISModel.unbound
    )
    ub_info = ub_str + spc

    # potential shift
    vshift_str = "{preamble:30s}: {spin}".format(
        preamble="Shift KS potential", spin=ISModel.v_shift
    )
    vshift_info = vshift_str + spc

    output_str = (
        init_str
        + spinpol_info
        + nele_info
        + xc_info
        + bc_info
        + ub_info
        + vshift_info
        + spc
    )

    return output_str


class SCF:
    """Write information about the self-consistent field (SCF) cycle."""

    @staticmethod
    def write_init():
        """
        Write the initial spiel for an SCF calculation.

        Parameters
        ----------
        None

        Returns
        -------
        output_str : str
            header string for SCF cycle
        """
        # the initial message
        init_str = "Starting SCF energy calculation" + dblspc

        # spacing between terms
        termspc = 3 * " "

        # main output string
        E_str = "{E:12s}".format(E="E_free (Ha)")
        dE_str = "{dE:2s} ({dE_x:<4.1e})".format(
            dE="dE", dE_x=config.conv_params["econv"]
        )
        dn_str = "{dn:2s} ({dn_x:<4.1e})".format(
            dn="dn", dn_x=config.conv_params["nconv"]
        )
        dv_str = "{dv:2s} ({dv_x:<4.1e})".format(
            dv="dv", dv_x=config.conv_params["vconv"]
        )

        main_str = termspc.join(["iscf", E_str, dE_str, dn_str, dv_str]) + spc
        buffer_str = 65 * "-"

        output_str = init_str + main_str + buffer_str
        return output_str

    @staticmethod
    def write_cycle(iscf, E_free, conv_vals):
        """
        Write output string for each SCF iteration.

        Parameters
        ----------
        iscf : int
            the iteration number
        E_free : float
            the total free energy
        conv_vals: dict
            dictionary of convergence values

        Returns
        -------
        output_str : str
            free energy and convergence values for the i-th iteration
        """
        termspc = 3 * " "

        iscf_str = "{i:4d}".format(i=iscf)
        E_str = "{E:12.7f}".format(E=E_free)
        dE_str = "{dE:12.3e}".format(dE=conv_vals["dE"])
        dn_str = "{dn:12.3e}".format(dn=np.max(conv_vals["drho"]))
        dv_str = "{dv:12.3e}".format(dv=np.max(conv_vals["dpot"]))

        output = termspc.join([iscf_str, E_str, dE_str, dn_str, dv_str])

        return output

    def write_final(self, energy, orbitals, density, conv_vals):
        """
        Write final post-SCF information about energy, orbitals and convergence.

        Parameters
        ----------
        energy : staticKS.Energy
            the energy object
        orbitals : staticKS.Orbitals
            the orbitals object
        density: staticKS.Density
            the density object
        conv_vals : dict
            dictionary of convergence values

        Returns
        -------
        output_str : str
            information about whether the calculation converged, the final energies
            (broken down into components), and the orbital energies and occupations
        """
        output_str = 65 * "-" + spc

        # write whether convergence cycle succesfully completed
        if conv_vals["complete"]:
            output_str += "SCF cycle converged" + dblspc
        else:
            output_str += (
                output_str
                + "SCF cycle did not converge in "
                + str(config.scf_params["maxscf"])
                + " iterations"
                + dblspc
            )

        # write the total energies
        output_str += self.write_final_energies(energy) + spc

        # write the chemical potential and mean ionization state
        occs_pos = np.where(orbitals.eigvals > 0, orbitals.occnums_w, 0)
        N_ub = np.sum(occs_pos, axis=(0, 2, 3)) + density.unbound["N"]

        if config.spindims == 2:
            mu_str = "Chemical potential (u/d)"
            chem_pot_str = "{mu:30s} : {mu1:7.3f} / {mu2:<7.3f}".format(
                mu=mu_str, mu1=config.mu[0], mu2=config.mu[1]
            )
            N_ub_str = "Mean ionization state (u/d)"
            MIS_str = "{Nub:30s} : {Nub1:7.3f} / {Nub2:<7.3f}".format(
                Nub=N_ub_str, Nub1=N_ub[0], Nub2=N_ub[1]
            )
        elif config.spindims == 1:
            mu_str = "Chemical potential"
            chem_pot_str = "{mu:30s} : {mu1:7.3f}".format(mu=mu_str, mu1=config.mu[0])
            N_ub_str = "Mean ionization state"
            MIS_str = "{Nub:30s} : {Nub1:7.3f}".format(Nub=N_ub_str, Nub1=N_ub[0])

        output_str += spc.join([chem_pot_str, MIS_str])

        eigvals, occnums = self.write_orb_info(orbitals)
        if config.bc == "bands":
            output_str += (
                dblspc + "Orbital eigenvalues (band limits) (Ha) :" + dblspc + eigvals
            )
            output_str += (
                spc + "Weighted band occupations (sum over k pts) :" + dblspc + occnums
            )
        else:
            output_str += dblspc + "Orbital eigenvalues (Ha) :" + dblspc + eigvals
            output_str += (
                spc + "Orbital occupations (2l+1) * f_{nl} :" + dblspc + occnums
            )

        return output_str

    @staticmethod
    def write_final_energies(energy):
        """
        Write formatted KS energy information (by component).

        Parameters
        ----------
        energy : staticKS.Energy
            the KS energy object

        Returns
        -------
        output_str : str
            formatted output string of energies by component
        """
        output_str = "Final energies (Ha)" + dblspc
        box_str = 45 * "-" + spc
        output_str += box_str

        # write the kinetic energy information
        E_kin = energy.E_kin
        KE_str = (
            "{KE:30s} : {KE_x:10.4f}".format(KE="Kinetic energy", KE_x=E_kin["tot"])
            + spc
        )
        KE_str += (
            4 * " "
            + "{KE:26s} : {KE_x:10.4f}".format(KE="orbitals", KE_x=E_kin["bound"])
            + spc
        )
        KE_str += (
            4 * " "
            + "{KE:26s} : {KE_x:10.4f}".format(
                KE="unbound ideal approx.", KE_x=E_kin["unbound"]
            )
            + spc
        )

        output_str += KE_str

        # electron-nuclear contribution
        en_str = (
            "{en:30s} : {E_en:10.4f}".format(
                en="Electron-nuclear energy", E_en=energy.E_en
            )
            + spc
        )
        output_str += en_str

        # hartree contribution
        ha_str = (
            "{Ha:30s} : {E_ha:10.4f}".format(Ha="Hartree energy", E_ha=energy.E_ha)
            + spc
        )
        output_str += ha_str

        # exchange-correlation (broken down into components)
        E_xc = energy.E_xc
        xc_str = (
            "{xc:30s} : {xc_x:10.4f}".format(
                xc="Exchange-correlation energy", xc_x=E_xc["xc"]
            )
            + spc
        )
        xc_str += (
            4 * " "
            + "{xc:26s} : {xc_x:10.4f}".format(xc="exchange", xc_x=E_xc["x"])
            + spc
        )
        xc_str += (
            4 * " "
            + "{xc:26s} : {xc_x:10.4f}".format(xc="correlation", xc_x=E_xc["c"])
            + spc
        )

        output_str += xc_str

        # total energy
        tot_E_str = (
            box_str
            + "{tot:30s} : {E_tot:10.4f}".format(tot="Total energy", E_tot=energy.E_tot)
            + spc
            + box_str
        )
        output_str += tot_E_str

        # entropy (split into bound / unbound)
        ent = energy.entropy
        ent_str = "{S:30s} : {S_x:10.4f}".format(S="Entropy", S_x=ent["tot"]) + spc
        ent_str += (
            4 * " "
            + "{S:26s} : {S_x:10.4f}".format(S="orbitals", S_x=ent["bound"])
            + spc
        )
        ent_str += (
            4 * " "
            + "{S:26s} : {S_x:10.4f}".format(
                S="unbound ideal approx.", S_x=ent["unbound"]
            )
            + spc
        )

        output_str += ent_str

        # total free energy F = E - T * S
        tot_F_str = (
            box_str
            + "{F:30s} : {F_x:10.4f}".format(F="Total free energy", F_x=energy.F_tot)
            + spc
            + box_str
        )
        output_str += tot_F_str

        return output_str

    @staticmethod
    def write_orb_info(orbitals):
        """
        Write formatted KS orbital information.

        Information up to to the highest occupied level +1 (for both n and l)
        is included; remaining levels are truncated.

        Parameters
        ----------
        orbitals : staticKS.Orbitals
            the KS orbitals object

        Returns
        -------
        eigs_occs : tuple[str,str]
            a tuple containing formatted eigenvalue table eigval_tbl,
            and formatted occupation numbers table occnum_tbl
        """
        # loop over the spin dimensions
        eigval_tbl = ""
        occnum_tbl = ""

        for i in range(config.spindims):

            # truncate the table otherwise it becomes too large

            lmax_new = min(config.lmax, 8)
            nmax_new = min(config.nmax, 5)

            # define row and column headers
            headers = [n + 1 for n in range(nmax_new)]
            headers[0] = "n=l+1"
            RowIDs = [*range(lmax_new)]
            RowIDs[0] = "l=0"

            if config.bc == "bands":
                eigvals_list = [orbitals.eigvals_min, orbitals.eigvals_max]
            else:
                eigvals_list = [orbitals.eigvals[0]]

            for eigvals in eigvals_list:
                eigvals_new = eigvals[i, :lmax_new, :nmax_new]

                # the eigenvalue table
                eigval_tbl += (
                    tabulate.tabulate(
                        eigvals_new,
                        headers,
                        tablefmt="presto",
                        showindex=RowIDs,
                        floatfmt="7.3f",
                        stralign="right",
                    )
                    + dblspc
                )

            # sum up the occupation numbers over each band
            occnums_band = np.sum(orbitals.occnums_w, axis=0)
            occnums_new = occnums_band[i, :lmax_new, :nmax_new]

            # the occnums table
            occnum_tbl += (
                tabulate.tabulate(
                    occnums_new,
                    headers,
                    tablefmt="presto",
                    showindex=RowIDs,
                    floatfmt="7.3f",
                    stralign="right",
                )
                + dblspc
            )

        return eigval_tbl, occnum_tbl


def density_to_csv(rgrid, density, filename):
    """
    Write the density (on the r-grid) to file.

    Parameters
    ----------
    rgrid : ndarray
        real-space grid
    density : staticKS.Density
        the KS density object
    filename : str
        name of the file to write to
    """
    if config.spindims == 2:
        headstr = (
            "r (a_0)"
            + 4 * " "
            + "n^u (orbs)"
            + 3 * " "
            + "n^u (ideal)"
            + 2 * " "
            + "n^d (orbs)"
            + 3 * " "
            + "n^d (ideal)"
            + 1 * " "
        )
        data = np.column_stack(
            [
                rgrid,
                density.bound["rho"][0],
                density.unbound["rho"][0],
                density.bound["rho"][1],
                density.unbound["rho"][1],
            ]
        )
    else:
        headstr = "r (a_0)" + 4 * " " + "n (orbs)" + 5 * " " + "n (ideal)" + 3 * " "
        data = np.column_stack(
            [rgrid, density.bound["rho"][0], density.unbound["rho"][0]]
        )

    np.savetxt(filename, data, fmt="%11.6e", header=headstr)

    return


def potential_to_csv(rgrid, potential, filename):
    """
    Write the potential (on the r-grid) to file.

    Parameters
    ----------
    rgrid : ndarray
        real-space grid
    density : staticKS.Potential
        the KS potential object
    filename : str
        name of the file to write to
    """
    if config.spindims == 2:
        headstr = (
            "r"
            + 7 * " "
            + "v_en"
            + 4 * " "
            + "v_ha"
            + 3 * " "
            + "v^up_xc"
            + 4 * " "
            + "v^dw_xc"
            + 3 * " "
        )
        data = np.column_stack(
            [
                rgrid,
                potential.v_en,
                potential.v_ha,
                potential.v_xc["xc"][0],
                potential.v_xc["xc"][0],
            ]
        )
    else:
        headstr = "r" + 8 * " " + "v_en" + 6 * " " + "v_ha" + 3 * " "
        data = np.column_stack(
            [rgrid, potential.v_en, potential.v_ha, potential.v_xc["xc"][0]]
        )

    np.savetxt(filename, data, fmt="%8.3e", header=headstr)

    return


def eigs_occs_to_csv(orbitals, filename):
    """
    Write all the orbital energies and their occupations to file.

    Parameters
    ----------
    orbitals: staticKS.Orbitals
        the orbitals object
    filename : str
        name of the file to write to

    Returns
    -------
    None
    """
    data_tot = np.array([])
    for sp in range(config.spindims):
        # flatten out the relevant matrices
        # and sort by energy eigenvalue
        eigs_sp = orbitals.eigvals[:, sp].flatten()
        idr = np.argsort(eigs_sp)
        eigs_sp = eigs_sp[idr]
        occs_sp = orbitals.occnums[:, sp].flatten()[idr]
        dos_sp = orbitals.DOS[:, sp].flatten()[idr]
        ldegen_sp = orbitals.ldegen[:, sp].flatten()[idr]
        band_weight_sp = orbitals.kpt_int_weight[:, sp].flatten()[idr]

        data = np.column_stack([eigs_sp, occs_sp, dos_sp, ldegen_sp, band_weight_sp])
        try:
            data_tot = np.concatenate((data_tot, data), axis=-1)
        except ValueError:
            data_tot = data

    headstr = config.spindims * (
        "eigs"
        + 5 * " "
        + "occs"
        + 6 * " "
        + "dos"
        + 7 * " "
        + "l_degen"
        + 3 * " "
        + "k_int_wt"
        + 4 * " "
    )

    np.savetxt(filename, data_tot, fmt="%8.3e", header=headstr)

    return


def dos_to_csv(orbitals, filename):
    """
    Write the energy eigenvalues, Fermi-Dirac occupations and DOS to file.

    Parameters
    ----------
    orbitals: staticKS.Orbitals
        the orbitals object
    filename : str
        name of the file to write to

    Returns
    -------
    None
    """
    data_tot = np.array([])
    for sp in range(config.spindims):

        # compute the dos (*(2l+1)) and FD dist in amenable format
        e_arr, fd_arr, DOS_arr = orbitals.calc_DOS_sum(
            orbitals.eigvals_min, orbitals.eigvals_max, orbitals.ldegen
        )

        data = np.column_stack([e_arr[:, sp], fd_arr[:, sp], DOS_arr[:, sp]])

        try:
            data_tot = np.concatenate((data_tot, data), axis=-1)
        except ValueError:
            data_tot = data

    headstr = config.spindims * (
        "energy" + 5 * " " + "fd occ" + 6 * " " + "dos" + 10 * " "
    )

    np.savetxt(
        filename,
        data_tot,
        fmt="%10.5e",
        header=headstr,
    )

    return


# timing wrapper
def timing(f):
    """
    Generate timing information for given input function.

    Parameters
    ----------
    f : func
        the function to be timed
    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap
