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

    N = config.grid_params["ngrid"]

    # initialize the eigenfunctions and their eigenvalues
    eigfuncs = np.zeros((config.spindims, config.lmax, config.nmax, N))
    eigvals = np.zeros((config.spindims, config.lmax, config.nmax))

    # define the spacing of the xgrid
    dx = xgrid[1] - xgrid[0]
    # number of grid points

    # Set-up the following matrix diagonalization problem
    # H*|u>=E*B*|u>; H=T+B*V; T=-p*A
    # |u> is related to the radial eigenfunctions R(r) via R(x)=exp(x/2)u(x)

    # off-diagonal matrices
    I_minus = np.eye(N, k=-1)
    I_zero = np.eye(N)
    I_plus = np.eye(N, k=1)

    V_mat = np.zeros((N, N))
    p = np.zeros((N, N))  # transformation for kinetic term on log grid
    np.fill_diagonal(p, np.exp(-2 * xgrid))

    # see referenced paper for definitions of A and B matrices
    A = np.matrix((I_minus - 2 * I_zero + I_plus) / dx ** 2)
    B = np.matrix((I_minus + 10 * I_zero + I_plus) / 12)

    # von neumann boundary conditions
    if config.bc == "neumann":
        A[N - 2, N - 1] = 2 * dx ** (-2)
        B[N - 2, N - 1] = 2 * B[N - 2, N - 1]
        A[N - 1, N - 1] = A[N - 1, N - 1] + 1.0 / dx
        B[N - 1, N - 1] = B[N - 1, N - 1] - dx / 12.0

    # construct kinetic energy matrix
    T = -0.5 * p * A

    # A new Hamiltonian has to be re-constructed for every value of l and each spin channel if spin-polarized
    for l in range(config.lmax):

        # diagonalize Hamiltonian using scipy
        for i in range(np.shape(v)[0]):

            # fill potential matrices
            np.fill_diagonal(V_mat, v[i] + 0.5 * (l + 0.5) ** 2 * np.exp(-2 * xgrid))

            # construct Hamiltonians
            H = T + B * V_mat

            # we seek the lowest nmax eigenvalues from sparse matrix diagonalization
            # use `shift-invert mode' (sigma=0) and pick lowest magnitude ("LM") eigs
            # sigma=0 seems to cause numerical issues so use a small offset
            eigs_up, vecs_up = eigs(H, k=config.nmax, M=B, which="LM", sigma=0.0001)

            eigfuncs[i, l], eigvals[i, l] = update_orbs(vecs_up, eigs_up, xgrid)

    return eigfuncs, eigvals


def update_orbs(l_eigfuncs, l_eigvals, xgrid):
    """
    Sorts the eigenvalues and functions by ascending order in energy
    and normalizes the eigenfunctions within the Voronoi sphere

    Inputs:
    - l_eigfuncs (np array)   : the eigenfunctions resulting from the chosen value of l
    - l_eigvals  (np array)   : the eigenvalues resulting from the chosen value of l
    Returns:
    - eigfuncs (np array)     : updated orbitals
    - eigvals  (np array)     : updated orbital energies
    """

    # Sort eigenvalues in ascending order
    idr = np.argsort(l_eigvals)
    eigvals = np.array(l_eigvals[idr].real)
    # under neumann bc the RHS pt is junk, convert to correct value
    if config.bc == "neumann":
        l_eigfuncs[-1] = l_eigfuncs[-2]
    eigfuncs = np.array(np.transpose(l_eigfuncs.real)[idr])
    eigfuncs = mathtools.normalize_orbs(eigfuncs, xgrid)  # normalize

    return eigfuncs, eigvals
