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
import string
import random
import warnings

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

# from . import writeoutput


class LogSolver:
    def calc_eigs_min(self, v, xgrid, bc):
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
        xgrid_coarse = np.linspace(
            xgrid[0], xgrid[-1], config.grid_params["ngrid_coarse"]
        )

        # interpolate the potential onto the coarse grid
        func_interp = interp1d(xgrid, v, kind="cubic")
        v_coarse = func_interp(xgrid_coarse)

        # full diagonalization to estimate the lowest eigenvalues
        eigs_min = self.matrix_solve(v_coarse, xgrid_coarse, bc, solve_type="guess")[1]

        return eigs_min

    # @writeoutput.timing
    def matrix_solve(self, v, xgrid, bc, solve_type="full", eigs_min_guess=None):
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
            eigfuncs, eigvals = self.KS_matsolve_serial(
                T, B, v, xgrid, bc, solve_type, eigs_min_guess
            )

        else:
            eigfuncs, eigvals = self.KS_matsolve_parallel(
                T, B, v, xgrid, bc, solve_type, eigs_min_guess
            )

        return eigfuncs, eigvals

    def KS_matsolve_parallel(self, T, B, v, xgrid, bc, solve_type, eigs_min_guess):
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
                delayed(self.diag_H)(
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
            eigfuncs = eigfuncs_flat.reshape(
                config.spindims, config.lmax, config.nmax, N
            )
            eigvals = eigvals_flat.reshape(config.spindims, config.lmax, config.nmax)

            return eigfuncs, eigvals

        elif solve_type == "guess":
            for q in range(pmax):
                eigs_guess_flat[q] = X[q][1]
            eigfuncs_null = X[:][0]

            eigs_guess = eigs_guess_flat.reshape(config.spindims, config.lmax)

            return eigfuncs_null, eigs_guess

    def KS_matsolve_serial(self, T, B, v, xgrid, bc, solve_type, eigs_min_guess):
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
                np.fill_diagonal(
                    V_mat, v[i] + 0.5 * (l + 0.5) ** 2 * np.exp(-2 * xgrid)
                )

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
                            -2
                            * np.exp(2 * xgrid)
                            * (V_mat.diagonal() - eigs_up.real[n])
                        )
                    eigfuncs[i, l], eigvals[i, l] = self.update_orbs(
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

        if solve_type == "full":
            return eigfuncs, eigvals
        else:
            return eigfuncs_null, eigs_guess

    def diag_H(self, p, T, B, v, xgrid, nmax, bc, eigs_guess, solve_type):
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
            evecs, evals = self.update_orbs(evecs, evals, xgrid, bc, K)

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

    @staticmethod
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
    def calc_wfns_e_grid(self, xgrid, v, e_arr, eigfuncs_l, eigfuncs_u):
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
            np.transpose(v)[:, np.newaxis, :, np.newaxis, np.newaxis]
            - e_arr[np.newaxis, :]
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
        eigfuncs_flat = self.num_propagate(
            xgrid, W_flat, e_arr_flat, eigfuncs_init_flat
        )

        # reshape the eigenfucntions
        eigfuncs_e = eigfuncs_flat.reshape((nkpts, spindims, lmax, nmax, N))

        return eigfuncs_e

    # @writeoutput.timing
    @staticmethod
    def num_propagate(xgrid, W, e_arr, eigfuncs_init):
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


class SqrtSolver:
    def calc_eigs_min(self, v, xgrid, bc):
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
        xgrid_coarse = np.linspace(
            xgrid[0], xgrid[-1], config.grid_params["ngrid_coarse"]
        )

        # interpolate the potential onto the coarse grid
        func_interp = interp1d(xgrid, v, kind="cubic")
        v_coarse = func_interp(xgrid_coarse)

        # full diagonalization to estimate the lowest eigenvalues
        eigs_min = self.matrix_solve(v_coarse, xgrid_coarse, bc, solve_type="guess")[1]

        return eigs_min

    def matrix_solve(self, v, sgrid, bc, solve_type="full", eigs_min_guess=None):
        if eigs_min_guess is None:
            eigs_min_guess = np.zeros((config.spindims, config.lmax))

        # define the spacing of the xgrid
        ds = sgrid[1] - sgrid[0]

        # number of grid pts
        N = np.size(sgrid)

        # Set-up the following matrix diagonalization problem
        # H*|u>=E*B*|u>; H=T+B*V; T=-p*A
        # |u> is related to the radial eigenfunctions R(r) via R(x)=exp(x/2)u(x)

        # off-diagonal matrices
        I_minus = np.eye(N, k=-1)
        I_zero = np.eye(N)
        I_plus = np.eye(N, k=1)

        p = np.zeros((N, N))  # transformation for kinetic term on log grid

        np.fill_diagonal(p, 0.25 * sgrid**-2)

        # see referenced paper for definitions of A and B matrices
        A = np.array((I_minus - 2 * I_zero + I_plus) / ds**2)
        B = np.array((I_minus + 10 * I_zero + I_plus) / 12)

        # von neumann boundary conditions
        if bc == "neumann":
            s_R = sgrid[-1]
            A[N - 2, N - 1] = 2 * A[N - 2, N - 1]
            B[N - 2, N - 1] = 2 * B[N - 2, N - 1]
            A[N - 1, N - 1] = A[N - 1, N - 1] + 3 / (s_R * ds)
            B[N - 1, N - 1] = B[N - 1, N - 1] - ds / (4 * s_R)

        # construct kinetic energy matrix
        T = -0.5 * p @ A

        if config.numcores == 0:
            eigfuncs, eigvals = self.KS_matsolve_serial(
                T, B, v, sgrid, bc, solve_type, eigs_min_guess
            )
        else:
            eigfuncs, eigvals = self.KS_matsolve_parallel(
                T, B, v, sgrid, bc, solve_type, eigs_min_guess
            )

        return eigfuncs, eigvals

    def KS_matsolve_parallel(self, T, B, v, sgrid, bc, solve_type, eigs_min_guess):
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
        sgrid : ndarray
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
        N = np.size(sgrid)

        # Compute the number pmax of distinct diagonizations to be solved
        pmax = config.spindims * config.lmax

        # now flatten the potential matrix over spins
        v_flat = np.zeros((pmax, N))
        eigs_guess_flat = np.zeros((pmax))
        for i in range(np.shape(v)[0]):
            for l in range(config.lmax):
                v_flat[l + (i * config.lmax)] = (
                    v[i] + 3 / (32 * sgrid**4) + l * (l + 1) / (2 * sgrid**4)
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
                delayed(self.diag_H)(
                    q,
                    T,
                    B,
                    v_flat,
                    sgrid,
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
            eigfuncs = eigfuncs_flat.reshape(
                config.spindims, config.lmax, config.nmax, N
            )
            eigvals = eigvals_flat.reshape(config.spindims, config.lmax, config.nmax)

            return eigfuncs, eigvals

        elif solve_type == "guess":
            for q in range(pmax):
                eigs_guess_flat[q] = X[q][1]
            eigfuncs_null = X[:][0]

            eigs_guess = eigs_guess_flat.reshape(config.spindims, config.lmax)

            return eigfuncs_null, eigs_guess

    def KS_matsolve_serial(self, T, B, v, sgrid, bc, solve_type, eigs_min_guess):
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
        sgrid : ndarray
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
        N = np.size(sgrid)
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
                np.fill_diagonal(
                    V_mat, v[i] + 3 / (32 * sgrid**4) + l * (l + 1) / (2 * sgrid**4)
                )

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
                        K[:, n] = 8 * sgrid**2 * (V_mat.diagonal() - eigs_up.real[n])

                    # sort and normalize
                    eigfuncs[i, l], eigvals[i, l] = self.update_orbs(
                        vecs_up, eigs_up, sgrid, bc, K
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

    def diag_H(self, p, T, B, v, sgrid, nmax, bc, eigs_guess, solve_type):
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
        sgrid : ndarray
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
        N = np.size(sgrid)
        # initialize empty potential matrix
        V_mat = np.zeros((N, N))

        # fill potential matrices
        # np.fill_diagonal(V_mat, v + 0.5 * (l + 0.5) ** 2 * np.exp(-2 * sgrid))
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
                K[:, n] = 8 * sgrid**2 * (V_mat.diagonal() - evals.real[n])
            evecs, evals = self.update_orbs(evecs, evals, sgrid, bc, K)

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

    def calc_wfns_e_grid(self, sgrid, v, e_arr, eigfuncs_l, eigfuncs_u):
        """
        Compute all KS orbitals defined on the energy grid.

        This routine is used to propagate a set of orbitals defined with a fixed
        set of energies. It is used for the `bands` boundary condition in the
        `models.ISModel` class.

        Parameters
        ----------
        sgrid : ndarray
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
        N = np.size(sgrid)

        ds = sgrid[1] - sgrid[0]
        s0 = sgrid[0]

        # dimensions of e_arr
        nkpts, spindims, lmax, nmax = np.shape(e_arr)

        # flatten energy array
        e_arr_flat = e_arr.flatten()

        # initialize the W (potential) and eigenfunction arrays
        W_arr = np.zeros((N, nkpts, spindims, lmax, nmax))
        eigfuncs_init = np.zeros_like(W_arr)
        eigfuncs_backup = np.zeros((nkpts, spindims, lmax, nmax, N))
        for k in range(nkpts):
            eigfuncs_backup[k] = (
                k * eigfuncs_u + (nkpts - k - 1) * eigfuncs_l
            ) / (nkpts - 1)
            
            norm = mathtools.int_sphere(eigfuncs_backup[k]**2, sgrid, "sqrt") ** -0.5
            eigfuncs_backup[k] = np.einsum("ijkl,ijk->ijkl", eigfuncs_backup[k], norm)

        # set up the flattened potential matrix
        # W = -2*exp(x)*(v - E) - (l + 1/2)^2
        # first set up the v - E array (matching dimensions)

        v_E_arr = (
            np.transpose(v)[:, np.newaxis, :, np.newaxis, np.newaxis]
            - e_arr[np.newaxis, :]
        )

        # muptiply by -2*exp(x) term
        v_E_arr = np.einsum("i,ijklm->ijklm", -8 * sgrid**2, v_E_arr)

        # add (l+1/2)^2 term and initial condition
        for l in range(lmax):
            l_term = -4 * l * (l + 1) / (sgrid**2) - 3 / (4 * sgrid**2)
            W_arr[:, :, :, l] = (
                v_E_arr[:, :, :, l] + l_term[:, np.newaxis, np.newaxis, np.newaxis]
            )
            eigfuncs_init[1, :, :, l] = max((s0 + ds) ** (2 * l + 1), 1e-300)

        # flatten arrays for input to numerov propagation
        W_flat = W_arr.reshape((N, len(e_arr_flat)))
        eigfuncs_init_flat = eigfuncs_init.reshape((N, len(e_arr_flat)))
        # eigfuncs_backup_flat = eigfuncs_backup.reshape((N, len(e_arr_flat)))

        # solve numerov eqn for the wfns
        eigfuncs_flat = self.num_propagate(
            sgrid, W_flat, e_arr_flat, eigfuncs_init_flat
        )

        # reshape the eigenfucntions
        eigfuncs_e = eigfuncs_flat.reshape((nkpts, spindims, lmax, nmax, N))
        eigfuncs_0 = np.where(np.all(eigfuncs_e == 0, axis=-1))
        eigfuncs_e[eigfuncs_0] = eigfuncs_backup[eigfuncs_0]
        print(np.size(eigfuncs_e[eigfuncs_0]) / np.size(eigfuncs_e))

        return eigfuncs_e

    @staticmethod
    def update_orbs(l_eigfuncs, l_eigvals, sgrid, bc, K):
        """
        Sort the eigenvalues and functions by ascending energies and normalize orbs.

        Parameters
        ----------
        l_eigfuncs : ndarray
            input (unsorted and unnormalized) eigenfunctions (for given l and spin)
        l_eigvals : ndarray
            input (unsorted) eigenvalues (for given l and spin)
        sgrid : ndarray
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
            N = np.size(sgrid)
            nmax = np.shape(l_eigfuncs)[1]
            l_eigfuncs_dir = np.zeros((N, nmax))
            l_eigfuncs_dir[:-1] = l_eigfuncs.real
            l_eigfuncs = l_eigfuncs_dir

        # manually propagate to final point for both boundary conditions
        ds = sgrid[1] - sgrid[0]
        h = (ds**2) / 12.0
        l_eigfuncs[-1] = (
            (2 - 10 * h * K[-2]) * l_eigfuncs[-2] - (1 + h * K[-3]) * l_eigfuncs[-3]
        ) / (1 + h * K[-1])

        # convert to correct dimensions
        eigfuncs = np.array(np.transpose(l_eigfuncs.real)[idr])
        eigfuncs *= sgrid**-1.5
        eigfuncs = mathtools.normalize_orbs_sgrid(eigfuncs, sgrid)  # normalize

        return eigfuncs, eigvals

    @staticmethod
    def num_propagate(xgrid, W, e_arr, eigfuncs_init):
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
        with np.errstate(over="ignore"):
            Psi *= xgrid**-1.5
            psi_sq = Psi**2  # convert from P_nl to X_nl and square
            integrand = 8.0 * np.pi * xgrid**5 * psi_sq
            norm = (np.trapz(integrand, x=xgrid)) ** (-0.5)
            Psi_norm = np.einsum("i,ij->ij", norm, Psi)

        return Psi_norm
