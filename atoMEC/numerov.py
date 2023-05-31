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
import sys
import shutil
import string
import random

# external libs
import numpy as np
from scipy.sparse.linalg import eigs
from scipy import linalg
from scipy.interpolate import interp1d
from joblib import Parallel, delayed, dump, load
from scipy import optimize

# from staticKS import Orbitals

# internal libs
from . import config
from . import mathtools

# from . import writeoutput


def solve(v, xgrid, bc, solve_method="matrix", eigs_min_guess=None, solve_type="full"):
    """Wrapper to solve either with matrix or linear method."""
    if solve_method == "matrix":
        return matrix_solve(
            v, xgrid, bc, solve_type=solve_type, eigs_min_guess=eigs_min_guess
        )
    elif solve_method == "linear":
        return linear_solve(v, xgrid, bc, eigs_min_guess)
    else:
        print("solver not recognized")


def calc_eigs_min(v, xgrid, bc, solve_type="guess_full"):
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
    eigs_min = matrix_solve(v_coarse, xgrid_coarse, bc, solve_type=solve_type)[1]

    return eigs_min


# @writeoutput.timing
def matrix_solve(v, xgrid, bc, solve_type="full", eigs_min_guess=None):
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
    else:
        eigs_min_guess = eigs_min_guess[:, :, 0]

    # define the spacing of the xgrid
    dx = xgrid[1] - xgrid[0]

    # number of grid pts
    N = np.size(xgrid)

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
    A = np.array((I_minus - 2 * I_zero + I_plus) / dx**2)
    B = np.array((I_minus + 10 * I_zero + I_plus) / 12)

    # von neumann boundary conditions
    if bc == "neumann":
        A[N - 2, N - 1] = 2 * dx ** (-2)
        B[N - 2, N - 1] = 2 * B[N - 2, N - 1]
        A[N - 1, N - 1] = A[N - 1, N - 1] + 1.0 / dx
        B[N - 1, N - 1] = B[N - 1, N - 1] - dx / 12.0

    # construct kinetic energy matrix
    T = -0.5 * p @ A

    # solve in serial or parallel - serial mostly useful for debugging
    if config.numcores == 0:
        eigfuncs, eigvals = KS_matsolve_serial(
            T, B, v, xgrid, bc, solve_type, eigs_min_guess
        )

    else:
        eigfuncs, eigvals = KS_matsolve_parallel(
            T, B, v, xgrid, bc, solve_type, eigs_min_guess
        )

    return eigfuncs, eigvals


def KS_matsolve_parallel(T, B, v, xgrid, bc, solve_type, eigs_min_guess):
    """
    Solve the KS matrix diagonalization by parallelizing over config.numcores.

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

    Notes
    -----
    The parallelization is done via the `joblib.Parallel` class of the `joblib` library,
    see here_ for more information.

    .. _here: https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html

    For "best" performance (i.e. exactly one core for each call of the diagonalization
    routine plus one extra for the "master" node), the number of cores should be chosen
    as `config.numcores = 1 + config.spindims * config.lmax`. However, if this number is
    larger than the total number of cores available, performance is hindered.

    Therefore for "good" performance, we can suggest:
    `config.numcores = max(1 + config.spindimgs * config.lmax, n_avail)`, where
    `n_avail` is the number of cores available.

    The above is just a guide for how to choose `config.numcores`, there may well
    be better choices. One example where it might not work is for particularly large
    numbers of grid points, when the memory required might be too large for a single
    core.

    N.B. if `config.numcores=-N` then `joblib` detects the number of available cores
    `n_avail` and parallelizes into `n_avail + 1 - N` separate jobs.
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

    # make temporary folder with random name to store arrays
    while True:
        try:
            joblib_folder = "atoMEC_tmpdata_" + "".join(
                random.choices(string.ascii_uppercase + string.digits, k=20)
            )
            os.mkdir(joblib_folder)
            break
        except FileExistsError as e:
            print(e)

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
                bc,
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

    elif solve_type == "guess_full":
        eigvals_flat = np.zeros((pmax, config.nmax))
        for q in range(pmax):
            eigvals_flat[q] = X[q][1][: config.nmax]
        eigfuncs_null = X[:][0]

        # unflatten eigfuncs / eigvals so they return to original shape
        eigvals = eigvals_flat.reshape(config.spindims, config.lmax, config.nmax)

        return eigfuncs_null, eigvals


def KS_matsolve_serial(T, B, v, xgrid, bc, solve_type, eigs_min_guess):
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
    eigs_guess = np.zeros((config.spindims, config.lmax, config.nmax))

    # A new Hamiltonian has to be re-constructed for every value of l and each spin
    # channel if spin-polarized
    for l in range(config.lmax):
        # diagonalize Hamiltonian using scipy
        for i in range(np.shape(v)[0]):
            # fill potential matrices
            np.fill_diagonal(V_mat, v[i] + 0.5 * (l + 0.5) ** 2 * np.exp(-2 * xgrid))

            # construct Hamiltonians
            H = T + B @ V_mat

            # if dirichlet solve on (N-1) x (N-1) grid
            if bc == "dirichlet":
                H_s = H[: N - 1, : N - 1]
                B_s = B[: N - 1, : N - 1]
            # if neumann don't change anything
            elif bc == "neumann":
                H_s = H
                B_s = B

            # we seek the lowest nmax eigenvalues from sparse matrix diagonalization
            # use 'shift-invert mode' to find the eigenvalues nearest in magnitude to
            # the estimated lowest eigenvalue from full diagonalization on coarse grid
            if solve_type == "full":
                eigs_up, vecs_up = eigs(
                    H_s,
                    k=config.nmax,
                    M=B_s,
                    which="LM",
                    sigma=eigs_min_guess[i, l],
                    tol=config.conv_params["eigtol"],
                )

                K = np.zeros((N, config.nmax))
                for n in range(config.nmax):
                    K[:, n] = (
                        -2 * np.exp(2 * xgrid) * (V_mat.diagonal() - eigs_up.real[n])
                    )
                eigfuncs[i, l], eigvals[i, l] = update_orbs(
                    vecs_up, eigs_up, xgrid, bc, K
                )

            elif solve_type == "guess":
                # estimate the lowest eigenvalues for a given value of l
                eigs_up = linalg.eigvals(H, b=B, check_finite=False)

                # sort the eigenvalues to find the lowest
                idr = np.argsort(eigs_up)
                eigs_guess[i, l] = np.array(eigs_up[idr].real)[0]

                # dummy variable for the null eigenfucntions
                eigfuncs_null = eigfuncs

            elif solve_type == "guess_full":
                # estimate the lowest eigenvalues for a given value of l
                eigs_up = linalg.eigvals(H, b=B, check_finite=False)

                # sort the eigenvalues to find the lowest
                idr = np.argsort(eigs_up)
                eigs_guess[i, l] = np.array(eigs_up[idr].real)[: config.nmax]

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
    H = T + B @ V_mat

    # if dirichlet solve on (N-1) x (N-1) grid
    if bc == "dirichlet":
        H_s = H[: N - 1, : N - 1]
        B_s = B[: N - 1, : N - 1]
    # if neumann don't change anything
    elif bc == "neumann":
        H_s = H
        B_s = B

    # we seek the lowest nmax eigenvalues from sparse matrix diagonalization
    # use 'shift-invert mode' to find the eigenvalues nearest in magnitude to
    # the estimated lowest eigenvalue from full diagonalization on coarse grid
    if solve_type == "full":
        evals, evecs = eigs(
            H_s,
            k=nmax,
            M=B_s,
            which="LM",
            tol=config.conv_params["eigtol"],
            sigma=eigs_guess[p],
        )

        # sort and normalize
        K = np.zeros((N, nmax))
        for n in range(nmax):
            K[:, n] = -2 * np.exp(2 * xgrid) * (V_mat.diagonal() - evals.real[n])
        evecs, evals = update_orbs(evecs, evals, xgrid, bc, K)

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

    elif solve_type == "guess_full":
        evals = linalg.eigvals(H, b=B, check_finite=False)

        # sort the eigenvalues to find the lowest
        idr = np.argsort(evals)
        evals = np.array(evals[idr].real)

        # dummy eigenvector for return statement
        evecs_null = np.zeros((N))

        return evecs_null, evals


def update_orbs(l_eigfuncs, l_eigvals, xgrid, bc, K):
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

    # resize l_eigfuncs from N-1 to N for dirichlet condition
    if bc == "dirichlet":
        N = np.size(xgrid)
        nmax = np.shape(l_eigfuncs)[1]
        l_eigfuncs_dir = np.zeros((N, nmax))
        l_eigfuncs_dir[:-1] = l_eigfuncs.real
        l_eigfuncs = l_eigfuncs_dir

    # manually propagate to final point for both boundary conditions
    dx = xgrid[1] - xgrid[0]
    h = (dx**2) / 12.0
    l_eigfuncs[-1] = (
        (2 - 10 * h * K[-2]) * l_eigfuncs[-2] - (1 + h * K[-3]) * l_eigfuncs[-3]
    ) / (1 + h * K[-1])

    # convert to correct dimensions
    eigfuncs = np.array(np.transpose(l_eigfuncs.real)[idr])
    eigfuncs = mathtools.normalize_orbs(eigfuncs, xgrid)  # normalize

    return eigfuncs, eigvals


# @writeoutput.timing
def calc_wfns_e_grid(xgrid, v, e_arr):
    """
    Compute all KS orbitals defined on the energy grid.

    This routine is used to propagate a set of orbitals defined with a fixed
    set of energies. It is used for the `bands` boundary condition in the
    `models.ISModel` class.

    Parameters
    ----------
    xgrid : ndarray
        the spatial (logarithmic) grid
    v : ndarray
        the KS potential
    e_arr : ndarray
        the energy grid
    Returns
    -------
    eigfuncs_e : ndarray
        the KS orbitals with defined energies
    """
    # size of spaital grid
    N = np.size(xgrid)

    dx = xgrid[1] - xgrid[0]
    x0 = xgrid[0]

    # dimensions of e_arr
    nkpts, spindims, lmax, nmax = np.shape(e_arr)

    # flatten energy array
    e_arr_flat = e_arr.flatten()

    # initialize the W (potential) and eigenfunction arrays
    W_arr = np.zeros((N, nkpts, spindims, lmax, nmax))
    eigfuncs_init = np.zeros_like(W_arr)

    # set up the flattened potential matrix
    # W = -2*exp(x)*(v - E) - (l + 1/2)^2
    # first set up the v - E array (matching dimensions)
    v_E_arr = (
        np.transpose(v)[:, np.newaxis, :, np.newaxis, np.newaxis] - e_arr[np.newaxis, :]
    )

    # muptiply by -2*exp(x) term
    v_E_arr = np.einsum("i,ijklm->ijklm", -2.0 * np.exp(2.0 * xgrid), v_E_arr)

    # add (l+1/2)^2 term and initial condition
    for l in range(lmax):
        W_arr[:, :, :, l] = v_E_arr[:, :, :, l] - (l + 0.5) ** 2
        eigfuncs_init[1, :, :, l] = np.exp((l + 0.5) * (x0 + dx))

    # flatten arrays for input to numerov propagation
    W_flat = W_arr.reshape((N, len(e_arr_flat)))
    eigfuncs_init_flat = eigfuncs_init.reshape((N, len(e_arr_flat)))

    # solve numerov eqn for the wfns
    eigfuncs_flat = num_propagate(xgrid, W_flat, e_arr_flat, eigfuncs_init_flat)

    # reshape the eigenfucntions
    eigfuncs_e = eigfuncs_flat.reshape((nkpts, spindims, lmax, nmax, N))

    return eigfuncs_e


def calc_wfns_no_kpts(xgrid, v, e_arr, bc, wfn=False):
    """
    Compute all KS orbitals defined on the energy grid.

    This routine is used to propagate a set of orbitals defined with a fixed
    set of energies. It is used for the `bands` boundary condition in the
    `models.ISModel` class.

    Parameters
    ----------
    xgrid : ndarray
        the spatial (logarithmic) grid
    v : ndarray
        the KS potential
    e_arr : ndarray
        the energy grid
    Returns
    -------
    eigfuncs_e : ndarray
        the KS orbitals with defined energies
    """
    # size of spaital grid
    N = np.size(xgrid)

    dx = xgrid[1] - xgrid[0]
    x0 = xgrid[0]

    # dimensions of e_arr
    spindims, lmax, nmax = np.shape(e_arr)

    # flatten energy array
    e_arr_flat = e_arr.flatten()

    # initialize the W (potential) and eigenfunction arrays
    W_arr = np.zeros((N, spindims, lmax, nmax))
    eigfuncs_init = np.zeros_like(W_arr)

    # set up the flattened potential matrix
    # W = -2*exp(x)*(v - E) - (l + 1/2)^2
    # first set up the v - E array (matching dimensions)
    v_E_arr = np.transpose(v)[:, :, np.newaxis, np.newaxis] - e_arr[np.newaxis, :]

    # muptiply by -2*exp(x) term
    v_E_arr = np.einsum("i,iklm->iklm", -2.0 * np.exp(2.0 * xgrid), v_E_arr)

    # add (l+1/2)^2 term and initial condition
    for l in range(lmax):
        W_arr[:, :, l] = v_E_arr[:, :, l] - (l + 0.5) ** 2
        eigfuncs_init[1, :, l] = np.exp((l + 0.5) * (x0 + dx))

    # flatten arrays for input to numerov propagation
    W_flat = W_arr.reshape((N, len(e_arr_flat)))
    eigfuncs_init_flat = eigfuncs_init.reshape((N, len(e_arr_flat)))

    # solve numerov eqn for the wfns
    eigfuncs_flat = num_propagate(xgrid, W_flat, e_arr_flat, eigfuncs_init_flat)

    if wfn:
        eigfuncs = eigfuncs_flat.reshape((spindims, lmax, nmax, N))
        return eigfuncs

    else:
        if bc == "dirichlet":
            deriv_diff_flat = eigfuncs_flat[:, -1]
        else:
            deriv_X_R = (eigfuncs_flat[:, -1] - eigfuncs_flat[:, -2]) / dx
            deriv_diff_flat = np.exp(-1.5 * xgrid[-1]) * (
                -0.5 * eigfuncs_flat[:, -1] + deriv_X_R
            )

        # reshape the eigenfucntions
        deriv_diff = deriv_diff_flat.reshape((spindims, lmax, nmax))

        return deriv_diff


# @writeoutput.timing
def num_propagate(xgrid, W, e_arr, eigfuncs_init, wfn=True):
    """
    Propagate the wfn manually for fixed energy with numerov scheme.

    Parameters
    ----------
    xgrid : ndarray
        the logarithmic grid
    W : ndarray
        flattened potential array (with angular momentum term)
    e_arr : ndarray
        flattened energy array
    eigfuncs_init : ndarray
        initial values for eigenfunctions
    Returns
    -------
    Psi_norm : ndarray
        normalized wavefunction
    """
    # define some initial grid parameters
    dx = xgrid[1] - xgrid[0]
    h = (dx**2) / 12.0  # a parameter for the numerov integration
    N = np.size(xgrid)  # size of grid

    # set the eigenfucntions to their initial values
    Psi = eigfuncs_init

    # Integration loop
    for i in range(2, N):
        Psi[i] = (
            2.0 * (1.0 - 5.0 * h * W[i - 1]) * Psi[i - 1]
            - (1.0 + h * W[i - 2]) * Psi[i - 2]
        ) / (1.0 + h * W[i])

    # normalize the wavefunction
    Psi = Psi.transpose()
    psi_sq = np.exp(-xgrid) * Psi**2  # convert from P_nl to X_nl and square
    integrand = 4.0 * np.pi * np.exp(3.0 * xgrid) * psi_sq
    norm = (np.trapz(integrand, x=xgrid)) ** (-0.5)
    Psi_norm = np.einsum("i,ij->ij", norm, Psi)

    return Psi_norm


def numerov_linear_solve(e_arr, xgrid, W, bc, l, wfn=False):
    dx = xgrid[1] - xgrid[0]
    h = (dx**2) / 12.0  # a parameter for the numerov integration
    N = np.size(xgrid)  # size of grid

    # set the eigenfucntions to their initial values
    Psi = np.zeros_like(xgrid)
    Psi[1] = np.exp((l + 0.5) * xgrid[1])

    W = W + 2.0 * np.exp(2 * xgrid) * e_arr

    # Integrate from the left
    for i in range(2, N):
        Psi[i] = (
            2.0 * (1.0 - 5.0 * h * W[i - 1]) * Psi[i - 1]
            - (1.0 + h * W[i - 2]) * Psi[i - 2]
        ) / (1.0 + h * W[i])

        Psi /= max(np.max(np.abs(Psi)), 1)

    # normalize the wavefunction - CHANGE HERE
    psi_sq = np.exp(-xgrid) * Psi**2  # convert from P_nl to X_nl and square
    integrand = 4.0 * np.pi * np.exp(3.0 * xgrid) * psi_sq
    norm = (np.trapz(integrand, x=xgrid)) ** (-0.5)
    Psi_norm = Psi * norm

    if bc == "dirichlet":
        deriv_diff = Psi_norm[-1]
    else:
        deriv_X_R = (Psi_norm[-1] - Psi_norm[-2]) / dx
        deriv_diff = np.exp(-1.5 * xgrid[-1]) * (-0.5 * Psi_norm[-1] + deriv_X_R)

    if wfn:
        return Psi_norm
    else:
        return deriv_diff


def find_root_manual(E_left, E_right, W, x, boundary, l, tol=1e-3, max_iter=100):
    dd_right = numerov_linear_solve(E_right, x, W, boundary, l)
    dd_left = numerov_linear_solve(E_left, x, W, boundary, l)

    if dd_right * dd_left >= 0:
        E_mid = (E_left + E_right) / 2
        return E_mid
    for i in range(max_iter):
        E_mid = (E_left + E_right) / 2
        dd_mid = numerov_linear_solve(E_mid, x, W, boundary, l)
        if np.abs(dd_mid) < tol:  # If the boundary condition is satisfied
            return E_mid
        else:
            if dd_mid * dd_left < 0:  # If the root is in the left half
                E_right = E_mid
                dd_right = numerov_linear_solve(E_right, x, W, boundary, l)
            else:  # If the root is in the right half
                E_left = E_mid
                dd_left = numerov_linear_solve(E_left, x, W, boundary, l)

    print("Warning: Maximum number of iterations reached")
    return E_mid


def find_root(E_left, E_right, W, x, boundary, l, tol=1e-3, max_iter=100):
    E_mid = (E_left + E_right) / 2
    bracket = np.array([E_left, E_right])
    args = (x, W, boundary, l)
    try:
        soln = optimize.root_scalar(
            numerov_linear_solve,
            x0=E_mid,
            args=args,
            method="brentq",
            bracket=bracket,
            options={"maxiter": max_iter, "xtol": tol},
        )
        if not soln.converged:
            print("Not converged")
        return soln.root

    except ValueError:
        return E_mid

    return soln.root


def linear_solve(v, xgrid, bc, eigs_guess, max_iter=100, tol=0.01):
    eigvals_converged = np.zeros_like(eigs_guess)
    d1, d2, d3 = np.shape(eigs_guess)
    eigfuncs = np.zeros((d1, d2, d3, len(xgrid)))

    E_bracket = 0.1 * np.abs(eigs_guess)
    E_upper = eigs_guess + E_bracket
    E_lower = eigs_guess - E_bracket

    for i in range(max_iter):
        E_mid = (E_upper + E_lower) / 2
        deriv_diff_l = calc_wfns_no_kpts(xgrid, v, E_lower, bc, wfn=False)
        deriv_diff_mid = calc_wfns_no_kpts(xgrid, v, E_mid, bc, wfn=False)

        if np.amax(np.abs(deriv_diff_mid)) < tol:
            eigfuncs = calc_wfns_no_kpts(xgrid, v, E_mid, bc, wfn=True)
            return eigfuncs, E_mid

        E_upper = np.where(deriv_diff_mid * deriv_diff_l < 0, E_mid, E_upper)
        E_lower = np.where(deriv_diff_mid * deriv_diff_l > 0, E_mid, E_lower)

    return eigfuncs, eigvals_converged
