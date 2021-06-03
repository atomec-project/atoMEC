"""
Handles all output, writing to files etc
"""

# standard libs

# external libs

# internal libs
import unitconv


def write_atomic_data(atom):
    """
    Writes information about the atomic object

    Parameters
    ----------
    atom : obj
        The atomic object

    Returns
    -------
    str:
        The formatted text output string
    """

    # define some line spacings
    spc = "\n"
    dblspc = "\n \n"

    # the initial spiel
    init_str = "Welcome to AvAtom!" + dblspc + "Atomic information:" + dblspc

    # information about the atomic species
    species_str = "{preamble:25s}: {species:2s} ".format(
        preamble="Atomic species", species=atom.species.symbol
    )
    at_chrg_str = "{preamble:25s}: {chrg:<3d} / {weight:<.3f}".format(
        preamble="Atomic charge / weight", chrg=atom.at_chrg, weight=atom.at_mass
    )
    spec_info = species_str + spc + at_chrg_str + spc

    # information about the net charge / electron number
    net_chrg_str = "{preamble:25s}: {chrg:<3d}".format(
        preamble="Net charge", chrg=atom.charge
    )
    nele_str = "{preamble:25s}: {nele:<3d}".format(
        preamble="Number of electrons", nele=atom.nele
    )
    nele_info = net_chrg_str + spc + nele_str + spc

    # information about the atomic / mass density
    rho_str = "{preamble:25s}: {rho:<.3g} g cm^-3".format(
        preamble="Mass density", rho=atom.density
    )
    rad_ang = atom.radius / unitconv.angstrom_to_bohr
    rad_str = "{preamble:25s}: {rad_b:<.4g} Bohr / {rad_a:<.4g} Angstrom".format(
        preamble="Wigner-Seitz radius", rad_b=atom.radius, rad_a=rad_ang
    )
    rho_info = rho_str + spc + rad_str + spc

    # information about the temperature
    temp_ev = atom.temp / unitconv.ev_to_ha
    temp_K = atom.temp / unitconv.K_to_ha
    temp_str = "{preamble:25s}: {t_ha:<.4g} Ha /  {t_ev:<.4g} eV / {t_k:<.4g} K".format(
        preamble="Electronic temperature", t_ha=atom.temp, t_ev=temp_ev, t_k=temp_K
    )
    temp_info = temp_str + spc

    # put all into a single string
    output = init_str + spec_info + nele_info + rho_info + temp_info

    return output
