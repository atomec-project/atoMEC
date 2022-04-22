"""
The conductivity module handles routines used to model the electrical conducivity.

So far just the Kubo-Greenwood method is implemented.

Classes
-------
* :class:`KuboGreenwood` : Holds various routines needed to compute the Kubo-Greenwood \
                           conducivity, including its various components. Also contains\
                           various properties related to the KG conducivity.
* :class:`SphHamInts`: Holds the routines to construct angular integrals from the \
                       spherical harmonic functions
* :class:`RadialInts`: Holds the routines to construct radial integrals from the radial\
                       KS orbitals
"""

# standard packages
import sys
from math import factorial
import functools

# external packages
import numpy as np
from scipy.special import lpmv
from scipy.integrate import quad

# internal modules
from atoMEC import mathtools


class KuboGreenwood:
    """Class for Kubo-Greenwood conductivity and MIS via TRK sum rule."""

    def __init__(self, orbitals, valence_orbs=[], nmax=0, lmax=0):

        self._orbitals = orbitals
        self._xgrid = orbitals._xgrid
        self._eigfuncs = orbitals._eigfuncs
        self._eigvals = orbitals.eigvals
        self._occnums = orbitals.occnums
        self._DOS_w = orbitals.DOS * orbitals.kpt_int_weight
        nbands, self._spindims, lmax_default, nmax_default = np.shape(self._eigvals)
        if self._spindims == 2:
            sys.exit(
                "Kubo-Greenwood is not yet set-up for spin-polarized calculations. \
Please run again with spin-unpolarized input."
            )
        if nmax == 0:
            self._nmax = nmax_default
        else:
            self._nmax = nmax
        if lmax == 0:
            self._lmax = lmax_default
        else:
            self._lmax = lmax
        self.valence_orbs = valence_orbs

    @property
    def all_orbs(self):
        r"""list of tuples: all the possible orbital pairings."""
        all_orbs_tmp = []
        for l in range(self._lmax):
            for n in range(self._nmax):
                all_orbs_tmp.append((l, n))
        self._all_orbs = all_orbs_tmp
        return self._all_orbs

    @property
    def cond_orbs(self):
        r"""list of tuples: all the conduction band orbital pairings."""
        cond_orbs_tmp = self.all_orbs
        for val_orbs in self.valence_orbs:
            cond_orbs_tmp.remove(val_orbs)
        self._cond_orbs = cond_orbs_tmp
        return self._cond_orbs

    @property
    def sig_tot(self):
        r"""ndarray: the integrated total conductivity."""
        self._sig_tot = self.calc_sig(
            self.R1_int_tt, self.R2_int_tt, self.all_orbs, self.all_orbs
        )
        return self._sig_tot

    @property
    def sig_cc(self):
        r"""ndarray: the integrated cc conductivity component."""
        self._sig_cc = self.calc_sig(
            self.R1_int_cc, self.R2_int_cc, self.cond_orbs, self.cond_orbs
        )
        return self._sig_cc

    @property
    def sig_vv(self):
        r"""ndarray: the integrated vv conductivity component."""
        self._sig_vv = self.calc_sig(
            self.R1_int_vv, self.R2_int_vv, self.valence_orbs, self.valence_orbs
        )
        return self._sig_vv

    @property
    def sig_cv(self):
        r"""ndarray: the integrated cv conductivity component."""
        self._sig_cv = self.calc_sig(
            self.R1_int_cv, self.R2_int_cv, self.cond_orbs, self.valence_orbs
        )
        return self._sig_cv

    @property
    def N_tot(self):
        r"""float: the total electron number from TRK sum-rule."""
        self._N_tot = self.sig_tot * (2 * self.sph_vol / np.pi)
        return self._N_tot

    @property
    def N_free(self):
        r"""float: the free electron number from TRK sum-rule."""
        self._N_free = self.sig_cc * (2 * self.sph_vol / np.pi)
        return self._N_free

    @property
    def sph_vol(self):
        r"""float: the volume of the sphere."""
        rmax = np.exp(self._xgrid)[-1]
        V = (4.0 / 3.0) * np.pi * rmax ** 3.0
        return V

    def cond_tot(self, component="tt", gamma=0.01, maxfreq=50, nfreq=500):
        """
        Calculate the chosen component of dynamical electrical conductivity sig(w).

        Parameters
        ----------
        component : str, optional
            the desired component of the conducivity e.g. "cc", "tt" etc
        gamma : float, optional
            smoothing factor
        maxfreq : float, optional
            maximum frequency to scan up to
        nfreq : int, optional
            number of points in the frequency grid

        Returns
        -------
        cond_tot_ : ndarray
            dynamical electrical conductivity
        """

        if component == "tt":
            R1_int = self.R1_int_tt
            R2_int = self.R2_int_tt
            orb_subset_1 = self.all_orbs
            orb_subset_2 = self.all_orbs
        elif component == "cc":
            R1_int = self.R1_int_cc
            R2_int = self.R2_int_cc
            orb_subset_1 = self.cond_orbs
            orb_subset_2 = self.cond_orbs
        elif component == "cv":
            R1_int = self.R1_int_cv
            R2_int = self.R2_int_cv
            orb_subset_1 = self.cond_orbs
            orb_subset_2 = self.valence_orbs
        elif component == "vv":
            R1_int = self.R1_int_vv
            R2_int = self.R2_int_vv
            orb_subset_1 = self.valence_orbs
            orb_subset_2 = self.valence_orbs
        else:
            sys.exit("Component of conducivity not recognised")

        cond_tot_ = self.calc_sig_func(
            R1_int, R2_int, orb_subset_1, orb_subset_2, maxfreq, nfreq, gamma
        )
        return cond_tot_

    @property
    @functools.lru_cache
    def R1_int_tt(self):
        """Total-total component of the R1 radial integral."""
        R1_int_tt_ = RadialInts.calc_R1_int_mat(
            self._eigfuncs,
            self._occnums,
            self._xgrid,
            self.all_orbs,
            self.all_orbs,
        )
        return R1_int_tt_

    @property
    @functools.lru_cache
    def R1_int_cc(self):
        """Conducting-conducting component of the R1 radial integral."""
        R1_int_cc_ = RadialInts.calc_R1_int_mat(
            self._eigfuncs,
            self._occnums,
            self._xgrid,
            self.cond_orbs,
            self.cond_orbs,
        )
        return R1_int_cc_

    @property
    @functools.lru_cache
    def R1_int_cv(self):
        """Conducting-valence component of the R1 radial integral."""
        R1_int_cv_ = RadialInts.calc_R1_int_mat(
            self._eigfuncs,
            self._occnums,
            self._xgrid,
            self.cond_orbs,
            self.valence_orbs,
        )
        return R1_int_cv_

    @property
    @functools.lru_cache
    def R1_int_vv(self):
        """Valence-valence component of the R1 radial integral."""
        R1_int_vv_ = RadialInts.calc_R1_int_mat(
            self._eigfuncs,
            self._occnums,
            self._xgrid,
            self.valence_orbs,
            self.valence_orbs,
        )
        return R1_int_vv_

    @property
    @functools.lru_cache
    def R2_int_tt(self):
        """Total-total component of the R2 radial integral."""
        R2_int_tt_ = RadialInts.calc_R2_int_mat(
            self._eigfuncs,
            self._occnums,
            self._xgrid,
            self.all_orbs,
            self.all_orbs,
        )
        return R2_int_tt_

    @property
    @functools.lru_cache
    def R2_int_cc(self):
        """Conducting-conducting component of the R2 radial integral."""
        R2_int_cc_ = RadialInts.calc_R2_int_mat(
            self._eigfuncs,
            self._occnums,
            self._xgrid,
            self.cond_orbs,
            self.cond_orbs,
        )
        return R2_int_cc_

    @property
    @functools.lru_cache
    def R2_int_cv(self):
        """Conducting-valence component of the R2 radial integral."""
        R2_int_cv_ = RadialInts.calc_R2_int_mat(
            self._eigfuncs,
            self._occnums,
            self._xgrid,
            self.cond_orbs,
            self.valence_orbs,
        )
        return R2_int_cv_

    @property
    @functools.lru_cache
    def R2_int_vv(self):
        """Valence-valence component of the R2 radial integral."""
        R2_int_vv_ = RadialInts.calc_R2_int_mat(
            self._eigfuncs,
            self._occnums,
            self._xgrid,
            self.valence_orbs,
            self.valence_orbs,
        )
        return R2_int_vv_

    def check_sum_rule(self, l, n, m):
        r"""
        Check the sum rule (see notes) for an orbital :math:`\phi_{nlm}` is satisfied.

        Parameters
        ----------
        l : int
            angular quantum number
        n : int
            principal quantum number
        m : int
            magnetic quantum number

        Returns
        -------
        sum_mom : ndarray
            the momentum sum rule (see notes)

        Notes
        -----
        The expression for the momentum sum rule is given by

        .. math::
            S_{p} = \sum_{(n_1,l_1,m_1)\neq (n,l,m)}\
            \frac{|\langle\phi_{nlm}|\nabla|\phi_{n_1 l_1 m_1}\rangle|^2} {\
            \epsilon_{n_1,l_1,m_1}-\epsilon_{n,l,m}}

        If the sum rule is satisfied, the summation above should equal 1/2.
        See Eq. (38) of Ref. [7]_ for an explanation of this sum rule.

        References
        ----------
        .. [7] Calderin, L. et al, Kubo--Greenwood electrical conductivity formulation
           and implementation for projector augmented wave datasets", Comp. Phys Comms.
           221 (2017): 118-142.
           `DOI:doi.org/10.1016/j.cpc.2017.08.008
           <https://doi.org/10.1016/j.cpc.2017.08.008>`__.
        """

        # set up the orbitals to sum over
        new_orbs = self.all_orbs
        new_orbs.remove((l, n))

        # initialize sum_mom and various indices
        nbands, nspin, lmax, nmax = np.shape(self._eigvals)
        sum_mom = np.zeros((nbands))

        # compute the sum rule
        for k in range(nbands):
            for l1, n1 in new_orbs:
                # the eigenvalue difference
                eig_diff = self._eigvals[k, 0, l1, n1] - self._eigvals[k, 0, l, n]
                # only states with |l-l_1|=1 contribute
                if abs(l1 - l) != 1:
                    continue
                else:
                    # scale eigenfunctions by sqrt(4 pi) due to different normalization
                    orb_l1n1 = np.sqrt(4 * np.pi) * self._eigfuncs[k, 0, l1, n1]
                    orb_ln = np.sqrt(4 * np.pi) * self._eigfuncs[k, 0, l, n]

                    # compute the matrix element <\phi|\grad|\phi> and its complex conjugate
                    if abs(m) > l1:
                        mel_sq = 0
                    else:
                        mel = self.calc_mel_grad_int(
                            orb_ln, orb_l1n1, l, n, l1, n1, m, self._xgrid
                        )
                        mel_cc = self.calc_mel_grad_int(
                            orb_l1n1, orb_ln, l1, n1, l, n, m, self._xgrid
                        )
                        mel_sq = np.abs(mel_cc * mel)
                    sum_mom[k] += mel_sq / eig_diff

        return sum_mom

    def calc_sig(self, R1_int, R2_int, orb_subset_1, orb_subset_2):
        r"""
        Compute the *integrated* dynamical conducivity for given subsets (see notes).

        Parameters
        ----------
        R1_int : ndarray
            the 'R1' radial component of the integrand (see notes)
        R2_int : ndarray
            the 'R2' radial component of the integrand (see notes)
        orb_subset_1 : list of tuples
            the first subset of orbitals to sum over
        orb_subset_2 : list of tuples
            the second subset of orbitals to sum over

        Returns
        -------
        sig : float
            the integrated dynamical conductivity

        Notes
        -----
        This function returns the integrated dynamical conductivity,
        :math:`\bar{\sigma}=\int_0^\infty d\omega \sigma(\omega)`. The conductivity
        :math:`\sigma(\omega)` is defined as

        .. math::
            \sigma_{S_1,S2}(\omega) = \frac{2\pi}{3V\omega}
            \sum_{i\in S_1}\sum_{j\in S_2} (f_i - f_j)\
            |\langle\phi_{i}|\nabla|\phi_{j}\rangle|^2\delta(\epsilon_j-\epsilon_i-\omega),

        where :math:`S_1,S_2` denote the subsets of orbitals specified in the function's
        paramaters, e.g. the conduction-conduction orbitals.

        In practise, the integral in the above equation is given by a discrete sum due
        to the presenence of the dirac-delta function.

        The paramaters `R1_int` and `R2_int` refer to radial integral components in the
        calculation of the matrix elements. See the supplementary information of
        Ref. [8]_ for more information on thse components, and the functions
        :func:`calc_R1_int_mat` and :func:`calc_R2_int_mat` for their definitions.

        References
        ----------
        .. [8] Callow, T.J. et al.,  "Accurate and efficient computation of mean
           ionization states with an average-atom Kubo-Greenwood approach."
           arXiv preprint arXiv:2203.05863 (2022).
           `<https://arxiv.org/abs/2203.05863>`__.
        """

        # get matrix dimensions
        nbands, nspin, lmax, nmax = np.shape(self._occnums)

        # compute the angular integrals (see functions for defns)
        P2_int = SphHamInts.P_mat_int(2, lmax)
        P4_int = SphHamInts.P_mat_int(4, lmax)

        # compute the products of the radial and angular integrals
        tmp_mat_1 = np.einsum("kabcd,ace->kabcde", R1_int, P2_int)
        tmp_mat_2 = np.einsum("kabcd,ace->kabcde", R2_int, P4_int)
        tmp_mat_3 = np.einsum("kcdab,cae->kabcde", R1_int, P2_int)
        tmp_mat_4 = np.einsum("kcdab,cae->kabcde", R2_int, P4_int)

        # compute the sum over the matrix element |< phi_nlm | nabla | phi_pqm >|^2
        mel_sq_mat = np.sum(
            np.abs((tmp_mat_1 + tmp_mat_2) * (tmp_mat_3 + tmp_mat_4)),
            axis=-1,
        )

        # compute the f_nl - f_pq matrix
        occ_diff_mat = self.calc_occ_diff_mat(self._occnums, orb_subset_1, orb_subset_2)
        # compute the (e_nl - e_pq)^-1 matrix
        eig_diff_mat = self.calc_eig_diff_mat(self._eigvals, orb_subset_1, orb_subset_2)

        # put it all together for the integrated conducivity
        sig_bare = np.einsum(
            "kln,klnpq->", self._DOS_w[:, 0], mel_sq_mat * occ_diff_mat / eig_diff_mat
        )

        # multiply by prefactor 2*pi/V
        sig = 2 * np.pi * sig_bare / self.sph_vol

        return sig

    def calc_sig_func(
        self, R1_int, R2_int, orb_subset_1, orb_subset_2, omega_max, n_freq, gamma
    ):

        r"""
        Compute the dynamical conducivity for given subsets (see notes).

        Parameters
        ----------
        R1_int : ndarray
            the 'R1' radial component of the integrand (see notes)
        R2_int : ndarray
            the 'R2' radial component of the integrand (see notes)
        orb_subset_1 : list of tuples
            the first subset of orbitals to sum over
        orb_subset_2 : list of tuples
            the second subset of orbitals to sum over
        omega_max : float
            maximum value of the frequency grid
        n_freq : int
            number of points in the frequency grid
        gamma : float
            smoothing factor for the Lorentzian

        Returns
        -------
        sig_omega, nele: tuple (ndarray, float)
            * sig_omega: 2d array containing frequency grid and conductivity
              :math:`\sigma(\omega)`
            * n_ele: the number of electrons from integration of :math:`\sigma(\omega)`;
              equivalent to N_ij (for orb subsets ij) in the limit :math:`\gamma\to 0`

        Notes
        -----
        This function returns the dynamical conductivity, :math:`\sigma(\omega)`,
        defined as

        .. math::
            \sigma_{S_1,S2}(\omega) &= \frac{2\pi}{3V\omega}
            \sum_{i\in S_1}\sum_{j\in S_2} (f_i - f_j)\
            |\langle\phi_{i}|\nabla|\phi_{j}\rangle|^2\
            \mathcal{L}(\epsilon_i, \epsilon_j, \gamma, \omega) \\
            \mathcal{L}(\epsilon_i, \epsilon_j, \gamma, \omega)&=\
            \frac{\gamma}{\pi}\frac{1}{\gamma^2+(\omega+[\epsilon_i-\epsilon_j)])^2}

        where :math:`S_1,S_2` denote the subsets of orbitals specified in the function's
        paramaters, e.g. the conduction-conduction orbitals.

        As can be seen in the above equation, the dirac-delta function in the definition
        of the KG conductivity (see `calc_sig` function) is represented by a Lorentzian
        distribution :math:`\mathcal{L}` to obtain a smooth conductivity function. In
        the limit :math:`\gamma\to 0`, the Lorentzian becomes a delta function.
        
        The paramaters `R1_int` and `R2_int` refer to radial integral components in the
        calculation of the matrix elements. See the supplementary information of
        Ref. [8]_ for more information on these components, and the functions
        :func:`calc_R1_int_mat` and :func:`calc_R2_int_mat` for their definitions.
        """

        # get the dimensions of the array
        nbands, nspin, lmax, nmax = np.shape(self._occnums)

        # compute the angular momenta integrals
        P2_int = SphHamInts.P_mat_int(2, lmax)
        P4_int = SphHamInts.P_mat_int(4, lmax)

        # put the angular and radial integrals together
        tmp_mat_1 = np.einsum("kabcd,ace->kabcde", R1_int, P2_int)
        tmp_mat_2 = np.einsum("kabcd,ace->kabcde", R2_int, P4_int)
        tmp_mat_3 = np.einsum("kcdab,cae->kabcde", R1_int, P2_int)
        tmp_mat_4 = np.einsum("kcdab,cae->kabcde", R2_int, P4_int)

        mel_sq_mat = np.sum(
            np.abs((tmp_mat_1 + tmp_mat_2) * (tmp_mat_3 + tmp_mat_4)),
            axis=-1,
        )

        # compute the occupation number and eigenvalue differences
        occ_diff_mat = self.calc_occ_diff_mat(self._occnums, orb_subset_1, orb_subset_2)
        eig_diff_mat = self.calc_eig_diff_mat(self._eigvals, orb_subset_1, orb_subset_2)

        # set up the frequency array - must start a bit above zero
        # sqrt spacing from origin gives faster convergence wrt nfreq
        omega_arr = np.linspace(1e-5, np.sqrt(omega_max), n_freq) ** 2

        # set up lorentzian: requires dummy array to get right shape
        sig_omega = np.zeros((np.size(omega_arr), 2))
        omega_dummy_mat = np.ones((nbands, lmax, nmax, lmax, nmax, n_freq))
        eig_diff_omega_mat = np.einsum(
            "nijkl,nijklm->nijklm", eig_diff_mat, omega_dummy_mat
        )
        eig_diff_lorentz_mat = mathtools.lorentzian(
            omega_arr, eig_diff_omega_mat, gamma
        )

        # put everythin together to get conductivity
        mat1 = np.einsum(
            "kln,klnpq->klnpq", self._DOS_w[:, 0], mel_sq_mat * occ_diff_mat
        )
        mat2 = eig_diff_lorentz_mat / eig_diff_omega_mat

        # assign sig(w) and w to sig_omega array dimensions
        sig_omega[:, 1] = (
            np.einsum("nijkl,nijklm->m", mat1, mat2) * 2 * np.pi / self.sph_vol
        )
        sig_omega[:, 0] = omega_arr

        # integrate and convert to get electron number
        N_ele = self.sig_to_N(np.trapz(sig_omega[:, 1], x=omega_arr), self.sph_vol)

        return sig_omega, N_ele

    @staticmethod
    def calc_occ_diff_mat(occnums, orb_subset_1, orb_subset_2):
        """
        Compute the matrix of occupation number diffs -(f_l1n1 - f_l2n2).

        Parameters
        ----------
        occnums : ndarray
            the (unweighted FD) KS occupation numbers
        orb_subset_1 : tuple
            the first subset of orbitals (eg valence)
        orb_subset_2 : tuple
            the second subset of orbitals (eg conduction)

        Returns
        -------
        occ_diff_mat : ndarray
            the occupation number difference matrix
        """
        nbands, nspin, lmax, nmax = np.shape(occnums)
        occ_diff_mat = np.zeros((nbands, lmax, nmax, lmax, nmax), dtype=np.float32)

        for l1, n1 in orb_subset_1:
            for l2, n2 in orb_subset_2:
                occ_diff = -(occnums[:, 0, l1, n1] - occnums[:, 0, l2, n2])
                # only terms with l1 = l2 +/- 1 will contribute to final answer
                if abs(l1 - l2) != 1:
                    continue
                # integral is one-sided over positive energy differences
                elif occ_diff < 0:
                    continue
                else:
                    occ_diff_mat[:, l1, n1, l2, n2] = occ_diff
        return occ_diff_mat

    @staticmethod
    def calc_eig_diff_mat(eigvals, orb_subset_1, orb_subset_2):
        """
        Compute the matrix of eigenvalue differences e_l1n1 - e_ln2n2

        Parameters
        ----------
        eigvals : ndarray
            the KS energy eigenvalues
        orb_subset_1 : tuple
            the first subset of orbitals (eg valence)
        orb_subset_2 : tuple
            the second subset of orbitals (eg conduction)

        Returns
        -------
        occ_diff_mat : ndarray
            the occupation number difference matrix
        """
        nbands, nspin, lmax, nmax = np.shape(eigvals)
        eig_diff_mat = np.zeros((nbands, lmax, nmax, lmax, nmax), dtype=np.float32)
        eig_diff_mat += 1e-6  # slight offset from zero since we divide by it eventually

        for l1, n1 in orb_subset_1:
            for l2, n2 in orb_subset_2:
                # only terms with l1 = l2 +/- 1 will contribute to final answer
                if abs(l1 - l2) != 1:
                    continue
                # integral is one-sided over positive energy differences
                elif eigvals[:, 0, l1, n1] - eigvals[:, 0, l2, n2] < 0:
                    continue
                else:
                    eig_diff_mat[:, l1, n1, l2, n2] = (
                        eigvals[:, 0, l1, n1] - eigvals[:, 0, l2, n2]
                    )
        return eig_diff_mat

    @staticmethod
    def calc_mel_grad_int(orb_l1n1, orb_l2n2, l1, n1, l2, n2, m, xgrid):
        r"""
        Calculate the matrix element :math:`|<\phi_{n1l1}|\nabla|\phi_{n1l2}>|^2`.

        Parameters
        ----------
        orb_l1n1 : ndarray
            l1,n1 radial KS orbital
        orb_l2n2 : ndarray
            l2,n2 radial KS orbital
        l1 : int
            1st angular momentum quantum number
        n1 : int
            1st principal quantum number
        l2 : int
            2nd angular momentum quantum number
        n2 : int
            2nd principal quantum number
        m : int
            magnetic quantum number
        xgrid : ndarray
            log grid

        Returns
        -------
        mel_grad_int : float
            the matrix element :math:`|<\phi_{n1l1}|\nabla|\phi_{n1l2}>|^2`.
        """
        R1_int = RadialInts.calc_R1_int(orb_l1n1, orb_l2n2, xgrid)
        R2_int = RadialInts.calc_R2_int(orb_l1n1, orb_l2n2, xgrid)

        mel_grad_int = R1_int * SphHamInts.P_int(
            2, l1, l2, m
        ) + R2_int * SphHamInts.P_int(4, l1, l2, m)

        return mel_grad_int

    @staticmethod
    def sig_to_N(sig, V):
        """
        Map the integrated conducivity to electron number.

        Parameters
        ----------

        sig : float
            integrated conducivity
        V : float
            volume of sphere

        Returns
        -------
        N_ele : float
            electron number
        """
        N_ele = sig * (2 * V / np.pi)

        return N_ele


class SphHamInts:
    @classmethod
    def P_mat_int(cls, func_int, lmax):
        """
        Compute the matrix of P function (angular) integrals (see notes).

        Parameters
        ----------
        func_int : int
            the desired P integral (can be 2 or 4)
        lmax : int
            the maximum value of angular momentum

        Returns
        -------
        P_mat : ndarray
            matrix of P func integrals for chosen func_int

        Notes
        -----
        See Refs. [7]_ and [8]_ (supplemental material) for the definitions of the
        P2 and P4 functions, ands the :func:`P2_func`, :func:`P4_func` and
        :func:`P_int` functions.
        """
        P_mat = np.zeros((lmax, lmax, 2 * lmax + 1))

        for l1 in range(lmax):
            for l2 in range(lmax):
                # sum rules mean all terms with l1!=l2 are zero
                if abs(l1 - l2) == 1:
                    # m cannot exceed either of l1 or l2
                    lsmall = min(l1, l2)
                    for m in range(-lsmall, lsmall + 1):
                        P_mat[l1, l2, lsmall + m] = cls.P_int(func_int, l1, l2, m)
                else:
                    continue
        return P_mat

    @classmethod
    def P_int(cls, func_int, l1, l2, m):
        r"""
        Integrate the P2 or P4 function (see notes).

        Parameters
        ----------
        func_int : int
            the desired P integral (can be 2 or 4)
        l1 : int
            1st angular quantum number
        l2 : int
            2nd angular quantum number
        m : int
            magnetic quantum number

        Returns
        -------
        P_int_ : float
            the integrated P2 or P4 function

        Notes
        -----
        The integrals are defined as

        .. math::
            \bar{P}^{(n)}_{ll'm} = 2\pi c_{lm}c_{l'm}\int_{-1}^1 dx f_p^{(n)}[l_1,l_2,m](x)

        With the functions :math:`f_p^{(n)}(x)` defined below (:func:`P2_func`
        and :func:`P4_func`).
        """
        if func_int == 2:
            integ = quad(cls.P2_func, -1, 1, args=(l1, l2, m))[0]
        elif func_int == 4:
            integ = quad(cls.P4_func, -1, 1, args=(l1, l2, m))[0]
        else:
            sys.exit("Error: func_int value not recognised, must be 2 or 4")

        P_int_ = 2 * np.pi * cls.sph_ham_coeff(l1, m) * cls.sph_ham_coeff(l2, m) * integ

        return P_int_

    @staticmethod
    def P2_func(x, l1, l2, m):
        r"""
        The 'P2' function (see notes).

        Parameters
        ----------
        x : float
            input for Legendre polynomial
        l1 : int
            1st angular quantum number
        l2 : int
            2nd angular quantum number
        m : int
            magnetic quantum number

        Returns
        -------
        P2_func_ : float
            the P2 function

        Notes
        -----
        The P2 function is defined as (see also Refs. [7]_ and [8]_)

        .. math::
            f_p^{(2)}[l_1,l_2,m](x) = x P_{l_1}^m (x) P_{l_2}^m (x)

        where P_{l}^m (x) are Legendre polynomial functions.
        """
        P2_func_ = x * lpmv(m, l1, x) * lpmv(m, l2, x)

        return P2_func_

    @staticmethod
    def P4_func(x, l1, l2, m):
        r"""
        The 'P4' function (see notes).

        Parameters
        ----------
        x : float
            input for Legendre polynomial
        l1 : int
            1st angular quantum number
        l2 : int
            2nd angular quantum number
        m : int
            magnetic quantum number

        Returns
        -------
        P4_func_ : float
            the P4 function

        Notes
        -----
        The P4 function is defined as (see also Refs. [7]_ and [8]_)

        .. math::
            f_p^{(4)}[l_1,l_2,m](x)&=-(1-x)^2 P^m_{l_1}(x) \frac{dP_{l_2}^m(x)}{dx}\\
                                    &= P^m_{l_1}(x) [(l_2+m)P_{l_2-1}^m(x)-xl_2 P_{l_2}^m(x)]

        where :math:`P_{l}^m(x)` are Legendre polynomial functions.
        """

        if (l2 + m) != 0:
            factor = (l2 + m) * lpmv(m, l2 - 1, x) - l2 * x * lpmv(m, l2, x)
        else:
            factor = -l2 * x * lpmv(m, l2, x)

        return lpmv(m, l1, x) * factor

    @staticmethod
    def sph_ham_coeff(l, m):
        r"""
        Compute coefficients of spherical harmonic functions.

        Parameters
        ----------
        l : int
           angular quantum number
        m : int
           magnetic quantum number

        Returns
        -------
        c_lm : float
            coefficient for spherical harmonic function (l,m) (see notes)

        Notes
        -----
        The spherical harmonic functions with coefficients :math:`c_{lm}` are defined as

        .. math::
            Y_m^l(\theta,\phi) &= c_{lm} P_l^m (\cos\theta) e^{im\phi}\\
            c_{lm} &= \sqrt{\frac{(2l+1)(l-m)!}{4\pi(l+m)!}}

        """
        c_lm = np.sqrt((2 * l + 1) * factorial(l - m) / (factorial(l + m) * 4 * np.pi))
        return c_lm


class RadialInts:
    @classmethod
    def calc_R1_int_mat(cls, eigfuncs, occnums, xgrid, orb_subset_1, orb_subset_2):
        r"""
        Compute the 'R1' integral matrix (see notes).

        Parameters
        ----------
        eigfuncs : ndarray
            the KS eigenfunctions
        occnums : ndarray
            the KS occupation numbers
        xgrid : ndarray
            the log grid
        orb_subset_1 : tuple
            the first subset of orbitals (eg valence)
        orb_subset_2 : tuple
            the second subset of orbitals (eg conduction)

        Returns
        -------
        R1_mat : ndarray
            the R1 integral matrix (see notes)

        Notes
        -----
        The definition of the R1 integral is (see Ref. [7]_ and supplementary of [8]_)

        .. math::
            R^{(1)}=4\pi\int_0^R dr r^2 X_{n_1 l_1}(r) \frac{dX_{n_2 l_2}(r)}{dr},

        where :math:`X_{nl}(r)` are the radial KS functions.
        """
        # take the derivative of orb2
        # compute the gradient of the orbitals
        deriv_orb2 = np.gradient(eigfuncs, xgrid, axis=-1, edge_order=2)

        # chain rule to convert from dP_dx to dX_dr
        grad_orb2 = np.exp(-1.5 * xgrid) * (deriv_orb2 - 0.5 * eigfuncs)

        # initiliaze the matrix
        nbands, nspin, lmax, nmax = np.shape(occnums)
        R1_mat = np.zeros((nbands, lmax, nmax, lmax, nmax), dtype=np.float32)

        # integrate over the sphere
        for l1, n1 in orb_subset_1:
            for l2, n2 in orb_subset_2:
                # only l1 = l2 +/- 1 terms are non-zero
                if abs(l1 - l2) != 1:
                    continue
                else:
                    R1_mat[:, l1, n1, l2, n2] = cls.R1_int_term(
                        eigfuncs[:, 0, l1, n1], grad_orb2[:, 0, l2, n2], xgrid
                    )
                    # non-symmetric term
                    if orb_subset_1 != orb_subset_2:
                        R1_mat[:, l2, n2, l1, n1] = cls.R1_int_term(
                            eigfuncs[:, 0, l2, n2], grad_orb2[:, 0, l1, n1], xgrid
                        )

        return R1_mat

    @staticmethod
    def R1_int_term(eigfunc, grad_orb2, xgrid):
        """
        Input function to the :func:`calc_R1_int_mat` function.

        Parameters
        ----------
        eigfunc : ndarray
            KS orbital l1,n1
        grad_orb2 : ndarray
            derivative of KS orbital l2,n2
        xgrid : ndarray
            log grid

        Returns
        -------
        R1_int : float
            the matrix element for the R1_int_mat function
        """
        func_int = eigfunc * np.exp(-xgrid / 2.0) * grad_orb2
        R1_int = 4 * np.pi * np.trapz(np.exp(3.0 * xgrid) * func_int, xgrid)

        return R1_int

    @classmethod
    def calc_R2_int_mat(cls, eigfuncs, occnums, xgrid, orb_subset_1, orb_subset_2):
        r"""
        Compute the 'R2' integral matrix (see notes).

        Parameters
        ----------
        eigfuncs : ndarray
            the KS eigenfunctions
        occnums : ndarray
            the KS occupation numbers
        xgrid : ndarray
            the log grid
        orb_subset_1 : tuple
            the first subset of orbitals (eg valence)
        orb_subset_2 : tuple
            the second subset of orbitals (eg conduction)

        Returns
        -------
        R2_mat : ndarray
            the R2 integral matrix (see notes)

        Notes
        -----
        The definition of the R2 integral is (see Ref. [7]_ and supplementary of [8]_)

        .. math::
            R^{(1)}=4\pi\int_0^R dr r X_{n_1 l_1}(r) X_{n_2 l_2}(r),

        where :math:`X_{nl}(r)` are the radial KS functions.
        """
        # initiliaze the matrix
        nbands, nspin, lmax, nmax = np.shape(occnums)
        R2_mat = np.zeros((nbands, lmax, nmax, lmax, nmax), dtype=np.float32)

        # integrate over the sphere
        for l1, n1 in orb_subset_1:
            for l2, n2 in orb_subset_2:
                if abs(l1 - l2) != 1:
                    continue
                else:
                    R2_mat[:, l1, n1, l2, n2] = cls.R2_int_term(
                        eigfuncs[:, 0, l1, n1], eigfuncs[:, 0, l2, n2], xgrid
                    )

                    if orb_subset_1 != orb_subset_2:

                        R2_mat[:, l2, n2, l1, n1] = cls.R2_int_term(
                            eigfuncs[:, 0, l2, n2], eigfuncs[:, 0, l1, n1], xgrid
                        )

        return R2_mat

    @staticmethod
    def R2_int_term(eigfunc_1, eigfunc_2, xgrid):
        """
        Input function to the :func:`calc_R2_int_mat` function.

        Parameters
        ----------
        eigfunc_1 : ndarray
            KS orbital l1,n1
        eigfunc_2 : ndarray
            KS orbital l2,n2
        xgrid : ndarray
            log grid

        Returns
        -------
        R2_int : float
            the matrix element for the R2_int_mat function
        """
        R2_int = 4 * np.pi * np.trapz(np.exp(xgrid) * eigfunc_1 * eigfunc_2, xgrid)

        return R2_int

    @staticmethod
    def calc_R1_int(orb1, orb2, xgrid):
        r"""
        Compute the R1 integral between two orbitals orb1 and orb2 (see notes).

        Parameters
        ----------
        orb1 : ndarray
            the first radial orbital
        orb2 : ndarray
            the second radial orbital

        Returns
        -------
        R1_int : ndarray
            the R1 integral

        Notes
        -----
        See :func:`calc_R1_int_mat` for definition of the integral
        """
        # take the derivative of orb2
        # compute the gradient of the orbitals
        deriv_orb2 = np.gradient(orb2, xgrid, axis=-1, edge_order=2)

        # chain rule to convert from dP_dx to dX_dr
        grad_orb2 = np.exp(-1.5 * xgrid) * (deriv_orb2 - 0.5 * orb2)

        # integrate over the sphere
        func_int = orb1 * np.exp(-xgrid / 2.0) * grad_orb2
        R1_int = np.trapz(np.exp(3.0 * xgrid) * func_int, xgrid)

        return R1_int

    @staticmethod
    def calc_R2_int(orb1, orb2, xgrid):
        r"""
        Compute the R2 integral between two orbitals orb1 and orb2 (see notes).

        Parameters
        ----------
        orb1 : ndarray
            the first radial orbital
        orb2 : ndarray
            the second radial orbital

        Returns
        -------
        R2_int : ndarray
            the R2 integral

        Notes
        -----
        See :func:`calc_R2_int_mat` for definition of the integral
        """
        func_int = np.exp(xgrid) * orb1 * orb2
        R2_int = np.trapz(func_int, xgrid)

        return R2_int
