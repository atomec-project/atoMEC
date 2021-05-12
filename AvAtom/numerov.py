"""
Routines for solving the KS equations via Numerov's method
"""

# standard libs

# external libs
import numpy as np
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import eigh, eig

# internal libs
import config


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

    # define the spacing of the xgrid
    dx = xgrid[1] - xgrid[0]
    # number of grid points
    N = config.grid_params(["ngrid"])

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
    A = np.matrix((I_minus - 2 * I_zero + I_plus) / d ** 2)
    B = np.matrix((I_minus + 10 * I_zero + I_plus) / 12)

    # von neumann boundary conditions
    if config.bc == "Neumann":
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
            np.fill_diagonal(V_up, v[0, :] + 0.5 * (l + 0.5) ** 2 * np.exp(-2 * xgrid))
            np.fill_diagonal(V_dw, v[1, :] + 0.5 * (l + 0.5) ** 2 * np.exp(-2 * xgrid))

            # construct Hamiltonians
            H_up = T + B * V_up
            H_dw = T + B * V_dw

            # we seek the lowest nmax eigenvalues from sparse matrix diagonalization
            # use `shift-invert mode' (sigma=0) and pick lowest magnitude ("LM") eigs
            eigs_up, vecs_up = eigs(H_up, k=config.nmax, M=B, which="LM", sigma=0)
            eigs_dw, vecs_dw = eigs(H_dw, k=config.nmax, M=B, which="LM", sigma=0)
