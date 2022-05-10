import numpy as np
from atoMEC import unitconv


def stress_tensor(orbs, pot):

    r"""Calculate the pressure with the stress tensor approach."""

    nkpts, spindims, lmax, nmax = np.shape(orbs.eigvals)

    # set the xgrid and value of r at sphere edge
    xgrid = orbs._xgrid
    rmax = np.exp(xgrid[-1])

    # first compute the gradient of the orbitals

    # take the derivative of orb2
    # compute the gradient of the orbitals
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
        pot.v_s[np.newaxis, :, np.newaxis, np.newaxis, :]
        - orbs.eigvals[:, :, :, :, np.newaxis]
    )
    eps_term = -2 * v_E_arr * orb_sq

    # sum the orbital based terms
    sum_terms = grad_sq + lsq_term + eps_term

    # put everything together
    P = np.einsum("ijkl,ijklm->m", orbs.occnums_w, sum_terms) / 6

    return P


def ions_ideal(atom):

    # compute the volume
    V = (4.0 * np.pi / 3.0) * atom.radius ** 3

    # compute pressure with ideal gas law
    P_ion = atom.temp / V

    return P_ion
