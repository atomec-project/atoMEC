"""
The numerov module handles the routines required to solve the KS equations.

So far, there is only a single implementation which is based on a matrix
diagonalization. There is an option for parallelizing over the angular
quantum number `l`.

Functions
---------
* :func:`matrix_solve` : Solve the radial KS equation via matrix diagonalization of \
                         Numerov's method.
* :func:`KS_matsolve_parallel` : Solve the KS matrix diagonalization by parallelizing \
                                 over config.ncores.
* :func:`KS_matsolve_serial` : Solve the KS matrix diagonalization in serial.
* :func:`diag_H` : Diagonalize the Hamiltonian for given input potential.
* :func:`update_orbs` : Sort the eigenvalues and functions by ascending energies and \
                        normalize orbs.
"""


# standard libs
import os
import shutil

# external libs
import numpy as np
from scipy.sparse.linalg import eigs
from scipy import linalg
from scipy.interpolate import interp1d
from joblib import Parallel, delayed, dump, load

# from staticKS import Orbitals

# internal libs
from . import config
from . import mathtools


def calc_eigs_min(v, xgrid):
    """
    Compute an estimate for the minimum values of the KS eigenvalues.

    This estimate uses full diagonalization of the Hamiltonian on a coarse grid.
    The eigenvalue estimates are used in Scipy's sparse eigenvalue solver
    (for the full grid) which optimizes the performance of the solver.

    Parameters
    ----------
    v : ndarray
        the KS potential
    xgrid : ndarray
        the logarithmic grid (full)

    Returns
    -------
    eigs_min : ndarray
        array containing estimations of the lowest eigenvalue for each value of
        `l` angular quantum number and spin quantum number
    """
    # first of all create the coarse xgrid
    xgrid_coarse = np.linspace(xgrid[0], xgrid[-1], config.grid_params["ngrid_coarse"])

    # interpolate the potential onto the coarse grid
    func_interp = interp1d(xgrid, v, kind="cubic")
    v_coarse = func_interp(xgrid_coarse)

    # full diagonalization to estimate the lowest eigenvalues
    eigs_min = matrix_solve(v_coarse, xgrid_coarse, solve_type="guess")[1]

    return eigs_min


# @writeoutput.timing
def matrix_solve(v, xgrid, solve_type="full", eigs_min_guess=None):
    r"""
    Solve the radial KS equation via matrix diagonalization of Numerov's method.

    See notes for details of the implementation.

    Parameters
    ----------
    v : ndarray
        the KS potential on the log grid
    xgrid : ndarray
        the logarithmic grid
    solve_type : str, optional
        whether to do a "full" or "guess" calculation: "guess" estimates the lower
        bounds of the eigenvalues
    eigs_min_guess : ndarray, optional
        input guess for the lowest eigenvalues for given `l` and spin,
        should be dimension `config.spindims * config.lmax`

    Returns
    -------
    eigfuncs : ndarray
        the radial KS eigenfunctions on the log grid
    eigvals : ndarray
        the KS eigenvalues

    Notes
    -----
    The implementation is based on [2]_.

    The matrix diagonalization is of the form:

    .. math::

        \hat{H} \lvert X \rangle &= \lambda \hat{B} \lvert X \rangle\ , \\
        \hat{H}                  &= \hat{T} + \hat{B}\times\hat{V}\ ,   \\
        \hat{T}                  &= -0.5\times\hat{p}\times\hat{A}\ .

    where :math:`\hat{p}=\exp(-2x)`.
    See [2]_ for the definitions of the matrices :math:`\hat{A}` and :math:`\hat{B}`.

    References
    ----------
    .. [2] M. Pillai, J. Goglio, and T. G. Walker , Matrix Numerov method for solving
       Schrödinger’s equation, American Journal of Physics 80,
       1017-1019 (2012) `DOI:10.1119/1.4748813 <https://doi.org/10.1119/1.4748813>`__.
    """
    if eigs_min_guess is None:
        eigs_min_guess = np.zeros((config.spindims, config.lmax))

    # N = config.grid_params["ngrid"]
    N = np.size(xgrid)

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

    # solve in serial or parallel - serial mostly useful for debugging
    if config.numcores > 0:
        eigfuncs, eigvals = KS_matsolve_parallel(
            T, B, v, xgrid, solve_type, eigs_min_guess
        )
    else:
        eigfuncs, eigvals = KS_matsolve_serial(
            T, B, v, xgrid, solve_type, eigs_min_guess
        )

    return eigfuncs, eigvals


def KS_matsolve_parallel(T, B, v, xgrid, solve_type, eigs_min_guess):
    """
    Solve the KS matrix diagonalization by parallelizing over config.ncores.

    Parameters
    ----------
    T : ndarray
        kinetic energy array
    B : ndarray
        off-diagonal array (for RHS of eigenvalue problem)
    v : ndarray
        KS potential array
    xgrid : ndarray
        the logarithmic grid
    solve_type : str
        whether to do a "full" or "guess" calculation: "guess" estimates the lower
        bounds of the eigenvalues
    eigs_min_guess : ndarray
        input guess for the lowest eigenvalues for given `l` and spin,
        should be dimension `config.spindims * config.lmax`

    Returns
    -------
    eigfuncs : ndarray
        radial KS wfns
    eigvals : ndarray
        KS eigenvalues
    """
    # compute the number of grid points
    N = np.size(xgrid)

    # Compute the number pmax of distinct diagonizations to be solved
    pmax = config.spindims * config.lmax

    # now flatten the potential matrix over spins
    v_flat = np.zeros((pmax, N))
    eigs_guess_flat = np.zeros((pmax))
    for i in range(np.shape(v)[0]):
        for l in range(config.lmax):
            v_flat[l + (i * config.lmax)] = v[i] + 0.5 * (l + 0.5) ** 2 * np.exp(
                -2 * xgrid
            )
            eigs_guess_flat[l + (i * config.lmax)] = eigs_min_guess[i, l]

    # make temporary folder to store arrays
    joblib_folder = "./joblib_memmap"
    try:
        os.mkdir(joblib_folder)
    except FileExistsError:
        pass

    # dump and load the large numpy arrays from file
    data_filename_memmap = os.path.join(joblib_folder, "data_memmap")
    dump((T, B, v_flat), data_filename_memmap)
    T, B, v_flat = load(data_filename_memmap, mmap_mode="r")

    # set up the parallel job
    with Parallel(n_jobs=config.numcores) as parallel:
        X = parallel(
            delayed(diag_H)(
                q,
                T,
                B,
                v_flat,
                xgrid,
                config.nmax,
                config.bc,
                eigs_guess_flat,
                solve_type,
            )
            for q in range(pmax)
        )

    # remove the joblib arrays
    try:
        shutil.rmtree(joblib_folder)
    except:  # noqa
        print("Could not clean-up automatically.")

    if solve_type == "full":
        # retrieve the eigfuncs and eigvals from the joblib output
        eigfuncs_flat = np.zeros((pmax, config.nmax, N))
        eigvals_flat = np.zeros((pmax, config.nmax))
        for q in range(pmax):
            eigfuncs_flat[q] = X[q][0]
            eigvals_flat[q] = X[q][1]

        # unflatten eigfuncs / eigvals so they return to original shape
        eigfuncs = eigfuncs_flat.reshape(config.spindims, config.lmax, config.nmax, N)
        eigvals = eigvals_flat.reshape(config.spindims, config.lmax, config.nmax)

        return eigfuncs, eigvals

    elif solve_type == "guess":

        for q in range(pmax):
            eigs_guess_flat[q] = X[q][1]
        eigfuncs_null = X[:][0]

        eigs_guess = eigs_guess_flat.reshape(config.spindims, config.lmax)

        return eigfuncs_null, eigs_guess


def KS_matsolve_serial(T, B, v, xgrid, solve_type, eigs_min_guess):
    """
    Solve the KS equations via matrix diagonalization in serial.

    Parameters
    ----------
    T : ndarray
        kinetic energy array
    B : ndarray
        off-diagonal array (for RHS of eigenvalue problem)
    v : ndarray
        KS potential array
    xgrid : ndarray
        the logarithmic grid
    solve_type : str
        whether to do a "full" or "guess" calculation: "guess" estimates the lower
        bounds of the eigenvalues
    eigs_min_guess : ndarray
        input guess for the lowest eigenvalues for given `l` and spin,
        should be dimension `config.spindims * config.lmax`

    Returns
    -------
    eigfuncs : ndarray
        radial KS wfns
    eigvals : ndarray
        KS eigenvalues
    """
    # compute the number of grid points
    N = np.size(xgrid)
    # initialize empty potential matrix
    V_mat = np.zeros((N, N))

    # initialize the eigenfunctions and their eigenvalues
    eigfuncs = np.zeros((config.spindims, config.lmax, config.nmax, N))
    eigvals = np.zeros((config.spindims, config.lmax, config.nmax))
    eigs_guess = np.zeros((config.spindims, config.lmax))

    # A new Hamiltonian has to be re-constructed for every value of l and each spin
    # channel if spin-polarized
    for l in range(config.lmax):

        # diagonalize Hamiltonian using scipy
        for i in range(np.shape(v)[0]):

            # fill potential matrices
            np.fill_diagonal(V_mat, v[i] + 0.5 * (l + 0.5) ** 2 * np.exp(-2 * xgrid))

            # construct Hamiltonians
            H = T + B * V_mat

            # we seek the lowest nmax eigenvalues from sparse matrix diagonalization
            # use 'shift-invert mode' to find the eigenvalues nearest in magnitude to
            # the estimated lowest eigenvalue from full diagonalization on coarse grid
            if solve_type == "full":
                eigs_up, vecs_up = eigs(
                    H,
                    k=config.nmax,
                    M=B,
                    which="LM",
                    sigma=eigs_min_guess[i, l],
                    tol=config.conv_params["eigtol"],
                )

                eigfuncs[i, l], eigvals[i, l] = update_orbs(
                    vecs_up, eigs_up, xgrid, config.bc
                )

            elif solve_type == "guess":

                # estimate the lowest eigenvalues for a given value of l
                eigs_up = linalg.eigvals(H, b=B, check_finite=False)

                # sort the eigenvalues to find the lowest
                idr = np.argsort(eigs_up)
                eigs_guess[i, l] = np.array(eigs_up[idr].real)[0]

                # dummy variable for the null eigenfucntions
                eigfuncs_null = eigfuncs

    if solve_type == "full":
        return eigfuncs, eigvals
    else:
        return eigfuncs_null, eigs_guess


def diag_H(p, T, B, v, xgrid, nmax, bc, eigs_guess, solve_type):
    """
    Diagonilize the Hamiltonian for the input potential v[p].

    Uses Scipy's sparse matrix solver scipy.sparse.linalg.eigs. This
    searches for the lowest magnitude `nmax` eigenvalues, so care
    must be taken to converge calculations wrt `nmax`.

    Parameters
    ----------
    p : int
       the desired index of the input array v to solve for
    T : ndarray
        the kinetic energy matrix
    B : ndarray
        the off diagonal matrix multiplying V and RHS
    v : ndarray
        KS potential array
    xgrid : ndarray
        the logarithmic grid
    nmax : int
        number of eigenvalues returned by the sparse matrix diagonalization
    bc : str
        the boundary condition

    Returns
    -------
    evecs : ndarray
        the KS radial eigenfunctions
    evals : ndarray
        the KS eigenvalues
    """
    # compute the number of grid points
    N = np.size(xgrid)
    # initialize empty potential matrix
    V_mat = np.zeros((N, N))

    # fill potential matrices
    # np.fill_diagonal(V_mat, v + 0.5 * (l + 0.5) ** 2 * np.exp(-2 * xgrid))
    np.fill_diagonal(V_mat, v[p])

    # construct Hamiltonians
    H = T + B * V_mat

    # we seek the lowest nmax eigenvalues from sparse matrix diagonalization
    # use 'shift-invert mode' to find the eigenvalues nearest in magnitude to
    # the estimated lowest eigenvalue from full diagonalization on coarse grid
    if solve_type == "full":
        evals, evecs = eigs(H, k=nmax, M=B, which="LM", sigma=eigs_guess[p])

        # sort and normalize
        evecs, evals = update_orbs(evecs, evals, xgrid, bc)

        return evecs, evals

    # estimate the lowest eigenvalues for a given value of l
    elif solve_type == "guess":
        evals = linalg.eigvals(H, b=B, check_finite=False)

        # sort the eigenvalues to find the lowest
        idr = np.argsort(evals)
        evals = np.array(evals[idr].real)[0]

        # dummy eigenvector for return statement
        evecs_null = np.zeros((N))

        return evecs_null, evals


def update_orbs(l_eigfuncs, l_eigvals, xgrid, bc):
    """
    Sort the eigenvalues and functions by ascending energies and normalize orbs.

    Parameters
    ----------
    l_eigfuncs : ndarray
        input (unsorted and unnormalized) eigenfunctions (for given l and spin)
    l_eigvals : ndarray
        input (unsorted) eigenvalues (for given l and spin)
    xgrid : ndarray
        the logarithmic grid
    bc : str
        the boundary condition

    Returns
    -------
    eigfuncs : ndarray
        sorted and normalized eigenfunctions
    eigvals : ndarray
        sorted eigenvalues in ascending energy
    """
    # Sort eigenvalues in ascending order
    idr = np.argsort(l_eigvals)
    eigvals = np.array(l_eigvals[idr].real)
    # under neumann bc the RHS pt is junk, convert to correct value
    if bc == "neumann":
        l_eigfuncs[-1] = l_eigfuncs[-2]
    eigfuncs = np.array(np.transpose(l_eigfuncs.real)[idr])
    eigfuncs = mathtools.normalize_orbs(eigfuncs, xgrid)  # normalize

    return eigfuncs, eigvals
