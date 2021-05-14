"""
Routines for solving the KS equations via Numerov's method
"""

# standard libs

# external libs
import numpy as np
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import eigh, eig

# from staticKS import Orbitals

# internal libs
import config
import mathtools


def matrix_solve(v, xgrid):
    """
    Solves the KS equations via diagonalization of a matrix via the method described in the following paper:
    Mohandas Pillai, Joshua Goglio, and Thad G. Walker , "Matrix Numerov method for solving Schrödinger’s equation",
    American Journal of Physics 80, 1017-1019 (2012) https://doi.org/10.1119/1.4748813

    Inputs:
    - v (numpy array)          : KS potential
    - grid (numpy array)       : logarithmic grid

    Outputs:
    - eigfuncs (numpy array)   : KS orbitals on the logarithmic grid
    - eigvals  (numpy array)   : KS orbital eigenvalues
    """

    # initialize the eigenfunctions and their eigenvalues
    eigfuncs = [[], []]
    eigvals = [[], []]

    # define the spacing of the xgrid
    dx = xgrid[1] - xgrid[0]
    # number of grid points
    N = config.grid_params["ngrid"]

    # Set-up the following matrix diagonalization problem
    # H*|u>=E*B*|u>; H=T+B*V; T=-p*A
    # |u> is related to the radial eigenfunctions R(r) via R(x)=exp(x/2)u(x)

    # off-diagonal matrices
    I_minus = np.eye(N, k=-1)
    I_zero = np.eye(N)
    I_plus = np.eye(N, k=1)

    V_up = np.zeros((N, N))
    V_dw = V_up  # potential matrix on log grid
    p = np.zeros((N, N))  # transformation for kinetic term on log grid
    np.fill_diagonal(p, np.exp(-2 * xgrid))

    # see referenced paper for definitions of A and B matrices
    A = np.matrix((I_minus - 2 * I_zero + I_plus) / dx ** 2)
    B = np.matrix((I_minus + 10 * I_zero + I_plus) / 12)

    # von neumann boundary conditions
    if config.bc == "neumann":
        A[N - 2, N - 1] = 2 * dx ** (-2)
        B[N - 2, N - 1] = 2 * B[N - 1, N - 1]
        A[N - 1, N - 1] = A[N - 1, N - 1] + 1.0 / dx
        B[N - 1, N - 1] = B[N - 1, N - 1] - dx / 12.0

    # construct kinetic energy matrix
    T = -0.5 * p * A

    # A new Hamiltonian has to be re-constructed for every value of l and each spin channel if spin-polarized
    for l in range(config.lmax):

        # diagonalize Hamiltonian using scipy
        if config.spinpol == True:

            # fill potential matrices
            np.fill_diagonal(V_up, v[0] + 0.5 * (l + 0.5) ** 2 * np.exp(-2 * xgrid))
            np.fill_diagonal(V_dw, v[1] + 0.5 * (l + 0.5) ** 2 * np.exp(-2 * xgrid))

            # construct Hamiltonians
            H_up = T + B * V_up
            H_dw = T + B * V_dw

            # we seek the lowest nmax eigenvalues from sparse matrix diagonalization
            # use `shift-invert mode' (sigma=0) and pick lowest magnitude ("LM") eigs
            # sigma=0 seems to cause numerical issues so use a small offset
            eigs_up, vecs_up = eigs(H_up, k=config.nmax, M=B, which="LM", sigma=0.0001)
            eigs_dw, vecs_dw = eigs(H_dw, k=config.nmax, M=B, which="LM", sigma=0.0001)

            # update up spin orbitals
            eigfuncs[0], eigvals[0] = update_orbitals(
                eigfuncs[0], eigvals[0], vecs_up, eigs_up
            )
            # update down spin orbitals
            eigfuncs[1], eigvals[1] = update_orbitals(
                eigfuncs[1], eigvals[1], vecs_up, eigs_up
            )

        else:
            # fill potential matrices
            np.fill_diagonal(V_up, v[0] + 0.5 * (l + 0.5) ** 2 * np.exp(-2 * xgrid))

            # construct Hamiltonians
            H_up = T + B * V_up

            # we seek the lowest nmax eigenvalues from sparse matrix diagonalization
            # use `shift-invert mode' (sigma=0) and pick lowest magnitude ("LM") eigs
            # sigma=0 seems to cause numerical issues so use a small offset
            eigs_up, vecs_up = eigs(H_up, k=config.nmax, M=B, which="LM", sigma=0)

            # update spin-up orbitals
            eigfuncs[0], eigvals[0] = update_orbitals(
                eigfuncs[0], eigvals[0], vecs_up, eigs_up
            )

            # spin-down orbitals are identical to spin-up orbitals
            eigfuncs[1] = eigfuncs[0]
            eigvals[1] = eigvals[0]

    return eigfuncs, eigvals


def update_orbitals(eigfuncs, eigvals, l_eigfuncs, l_eigvals):
    """
    Updates the set of KS orbitals stored. If ideal or TF treatment of unbound electrons is chosen, only the bound orbitals are kept
    and the rest are thrown away. Quantum treatment of unbound electrons is not yet implemented but will require a different approach.

    Inputs:
    - eigfuncs (np array)     : the set of orbitals to be updated
    - eigvals  (np array)     : the set of eigenvalues to be updated
    - l_eigfuncs (np array)   : the eigenfunctions resulting from the chosen value of l
    - l_eigvals  (np array)   : the eigenvalues resulting from the chosen value of l
    Returns:
    - eigfuncs (np array)     : updated orbitals
    - eigvals  (np array)     : updated orbital energies
    """

    # Sort eigenvalues in ascending order
    idr = np.argsort(l_eigvals)
    l_eigvals = np.array(l_eigvals[idr])
    l_eigfuncs = np.array(np.transpose(l_eigfuncs)[idr])

    if config.unbound == "ideal":

        # location of bound eigenvalues (E<0)
        boundvals = np.where(l_eigvals < 0)

        # keep only bound states
        l_eigfuncs = l_eigfuncs[boundvals]
        l_eigvals = l_eigvals[boundvals]

        # update orbitals with new bound states
        if l_eigvals.size != 0:
            # normalize the eigenfunctions within the cell
            l_eigfuncs = mathtools.normalize_orbs(l_eigfuncs)
            eigfuncs.append(l_eigfuncs.real)
            eigvals.append(l_eigvals.real)

    return eigfuncs, eigvals
