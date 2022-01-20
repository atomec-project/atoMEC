"""Kubo-Greenwood conductivity etc"""

from math import pi, sqrt, factorial
from scipy.special import lpmv
from scipy.integrate import quad
import numpy as np
from numba import jit

from . import writeoutput


################################################################
# functions to compute various integrals of legendre functions #
################################################################


class KuboGreenwood:
    def __init__(self, orbitals, valence_orbs=[], nmax=0, lmax=0):

        self._orbitals = orbitals
        self._xgrid = orbitals._xgrid
        self._eigfuncs = orbitals._eigfuncs
        # self._eigfuncs_mod = gs_ortho(orbitals.eigfuncs, self._xgrid)
        self._eigvals = orbitals.eigvals
        self._occnums = orbitals.occnums
        self._dos = orbitals.DOS * orbitals.kpt_int_weight
        nbands, self._spindims, lmax_default, nmax_default = np.shape(self._eigvals)
        if nmax == 0:
            self._nmax = nmax_default
        else:
            self._nmax = nmax
        if lmax == 0:
            self._lmax = lmax_default
        else:
            self._lmax = lmax
        self.valence_orbs = valence_orbs
        self._all_orbs = None
        self._cond_orbs = None
        self._sig_cc = None
        self._sig_cv = None
        self._sig_vv = None
        self._sig_tot = None
        # self._cond_tot = None
        self._N_tot = None
        self._N_free = None
        self._R1_int = None
        self._R2_int = None
        self._P2_int = None
        self._P4_int = None
        self._occ_diff_mat = None
        self._eig_diff_mat = None
        self._dos_prod_mat = None

    # @property
    # def occnums(self):
    #     if np.all(self._occnums == 0.0):
    #         self._occnums = self.calc_occnums()
    #     return self._occnums

    # def calc_occnums(self):
    #     # convert the occupation numbers to be "pure" occnums
    #     lmat_inv = np.zeros_like(self._orbitals.lbound)
    #     occnums_tot = self._orbitals.occnums + self._orbitals._occnums_ub
    #     for l in range(self._lmax):
    #         lmat_inv[:, l] = 0.5 / (2 * l + 1.0)
    #     occnums = lmat_inv * occnums_tot

    #     return occnums

    @property
    def all_orbs(self):
        all_orbs_tmp = []
        for l in range(self._lmax):
            for n in range(self._nmax):
                all_orbs_tmp.append((l, n))
        self._all_orbs = all_orbs_tmp
        return self._all_orbs

    @property
    def cond_orbs(self):
        cond_orbs_tmp = self.all_orbs
        for val_orbs in self.valence_orbs:
            cond_orbs_tmp.remove(val_orbs)
        self._cond_orbs = cond_orbs_tmp
        return self._cond_orbs

    @property
    def sig_tot(self):
        self._sig_tot = self.calc_sig(
            self._eigfuncs,
            self._occnums,
            self._eigvals,
            self._dos,
            self._xgrid,
            self.all_orbs,
            self.all_orbs,
        )
        return self._sig_tot

    def cond_tot(self, gamma=0.01, maxfreq=50, nfreq=200):
        _cond_tot = self.calc_sig_func(
            self._eigfuncs,
            self._occnums,
            self._eigvals,
            self._dos,
            self._xgrid,
            self.all_orbs,
            self.all_orbs,
            omega_max=maxfreq,
            n_freq=nfreq,
            gamma=gamma,
        )
        return _cond_tot

    @property
    def sig_cc(self):
        self._sig_cc = self.calc_sig(
            self._eigfuncs,
            self._occnums,
            self._eigvals,
            self._dos,
            self._xgrid,
            self.cond_orbs,
            self.cond_orbs,
        )
        return self._sig_cc

    @property
    def sig_vv(self):
        self._sig_vv = self.calc_sig(
            self._eigfuncs,
            self._occnums,
            self._eigvals,
            self._dos,
            self._xgrid,
            self.valence_orbs,
            self.valence_orbs,
        )
        return self._sig_vv

    @property
    def sig_cv(self):
        self._sig_cv = self.calc_sig(
            self._eigfuncs,
            self._occnums,
            self._eigvals,
            self._dos,
            self._xgrid,
            self.valence_orbs,
            self.cond_orbs,
        )
        return self._sig_cv

    @property
    def N_tot(self):
        rmax = np.exp(self._xgrid)[-1]
        V = (4.0 / 3.0) * pi * rmax ** 3.0
        self._N_tot = self.sig_tot * (2 * V / pi)
        return self._N_tot

    @property
    def N_free(self):
        rmax = np.exp(self._xgrid)[-1]
        V = (4.0 / 3.0) * pi * rmax ** 3.0
        self._N_free = self.sig_cc * (2 * V / pi)
        return self._N_free

    @property
    @writeoutput.timing
    def R1_int(self):
        self._R1_int = calc_R1_int_mat(
            self._eigfuncs,
            self._occnums,
            self._xgrid,
            self.all_orbs,
            self.all_orbs,
        )
        return self._R1_int

    @property
    @writeoutput.timing
    def R2_int(self):
        self._R2_int = calc_R2_int_mat(
            self._eigfuncs,
            self._occnums,
            self._xgrid,
            self.all_orbs,
            self.all_orbs,
        )
        return self._R2_int

    @property
    @writeoutput.timing
    def P2_int(self):
        self._P2_int = P_mat_int(2, self._lmax)
        return self._P2_int

    @property
    @writeoutput.timing
    def P4_int(self):
        self._P4_int = P_mat_int(4, self._lmax)
        return self._P4_int

    @property
    @writeoutput.timing
    def occ_diff_mat(self):
        self._occ_diff_mat = calc_occ_diff_mat(
            self._occnums, self.all_orbs, self.all_orbs
        )
        return self._occ_diff_mat

    @property
    @writeoutput.timing
    def eig_diff_mat(self):
        self._eig_diff_mat = calc_eig_diff_mat(
            self._eigvals, self.all_orbs, self.all_orbs
        )
        return self._eig_diff_mat

    @property
    @writeoutput.timing
    def dos_prod_mat(self):
        self._dos_prod_mat = calc_dos_prod_mat(
            self._dos, self._occnums, self.all_orbs, self.all_orbs
        )
        return self._dos_prod_mat

    def check_sum_rule(self, l, n, m):
        sum_mom = 0.0
        new_orbs = self.all_orbs
        new_orbs.remove((l, n))
        nbands, nspin, lmax, nmax = np.shape(self._eigvals)

        for k in range(nbands):
            sum_mom = 0.0
            for l1, n1 in new_orbs:
                # the eigenvalue difference
                eig_diff = self._eigvals[k, 0, l1, n1] - self._eigvals[k, 0, l, n]
                if abs(l1 - l) != 1:
                    continue
                else:

                    # eigenfunctions
                    orb_l1n1 = sqrt(4 * pi) * self._eigfuncs[k, 0, l1, n1]
                    orb_ln = sqrt(4 * pi) * self._eigfuncs[k, 0, l, n]

                    # compute the matrix element
                    if abs(m) > l1:
                        mel_sq = 0
                    else:
                        mel = calc_mel_kgm(
                            orb_ln, orb_l1n1, l, n, l1, n1, m, self._xgrid
                        )
                        mel_cc = calc_mel_kgm(
                            orb_l1n1, orb_ln, l1, n1, l, n, m, self._xgrid
                        )
                        mel_sq = np.abs(mel_cc * mel)
                    sum_mom += mel_sq / eig_diff
                    print(l1, n1, sum_mom)
        return sum_mom

    @staticmethod
    @writeoutput.timing
    def calc_sig(
        eigfuncs,
        occnums,
        eigvals,
        dos,
        xgrid,
        orb_subset_1,
        orb_subset_2,
        gamma=0.0,
        eig_min_diff=1e-3,
        eig_max_diff=1e4,
        occ_min_diff=1e-4,
    ):

        sig = 0.0

        nbands, nspin, lmax, nmax = np.shape(occnums)

        R1_int = calc_R1_int_mat(eigfuncs, occnums, xgrid, orb_subset_1, orb_subset_2)
        R2_int = calc_R2_int_mat(eigfuncs, occnums, xgrid, orb_subset_1, orb_subset_2)
        P2_int = P_mat_int(2, lmax)
        P4_int = P_mat_int(4, lmax)

        tmp_mat_1 = np.einsum("kabcd,ace->kabcde", R1_int, P2_int)
        tmp_mat_2 = np.einsum("kabcd,ace->kabcde", R2_int, P4_int)
        tmp_mat_3 = np.einsum("kabcd,ace->kcdabe", R1_int, P2_int)
        tmp_mat_4 = np.einsum("kabcd,ace->kcdabe", R2_int, P4_int)

        mel_sq_mat = np.sum(
            np.abs((tmp_mat_1 + tmp_mat_2) * (tmp_mat_3 + tmp_mat_4)),
            axis=-1,
        )

        occ_diff_mat = calc_occ_diff_mat(occnums, orb_subset_1, orb_subset_2)
        eig_diff_mat = calc_eig_diff_mat(eigvals, orb_subset_1, orb_subset_2)
        dos_prod_mat = calc_dos_prod_mat(dos, occnums, orb_subset_1, orb_subset_2)

        # sig_bare = np.sum(dos_prod_mat * mel_sq_mat * occ_diff_mat / eig_diff_mat)
        sig_bare = np.einsum(
            "kln,klnpq->", dos_prod_mat, mel_sq_mat * occ_diff_mat / eig_diff_mat
        )

        rmax = np.exp(xgrid)[-1]
        V = (4.0 / 3.0) * pi * rmax ** 3.0

        # smooth_fac = eig_diff ** 2 / (gamma ** 2 + eig_diff ** 2)
        sig = 2 * pi * sig_bare / V

        # for l1, n1 in orb_subset_1:
        #     for l2, n2 in orb_subset_2:
        #         # the eigenvalue difference
        #         eig_diff = eig_diff_mat[l1, n1, l2, n2]
        #         # occupation number difference
        #         occnum_diff = abs(occnums[0, l1, n1] - occnums[0, l2, n2])
        #         if eig_diff < eig_min_diff:
        #             continue
        #         elif eig_diff > eig_max_diff:
        #             continue
        #         elif abs(l1 - l2) != 1:
        #             continue
        #         elif occnum_diff < occ_min_diff:
        #             continue
        #         else:
        #             occnum_diff = -occ_diff_mat[l1, n1, l2, n2]

        #             # compute the matrix element
        #             mel_sq = 0.0
        #             for m1 in range(-l1, l1 + 1):
        #                 for m2 in range(-l2, l2 + 1):
        #                     if m1 == m2:
        #                         mel = calc_mel_kgm_2(R1_int, R2_int, l2, n2, l1, n1, m1)
        #                         mel_cc = calc_mel_kgm_2(
        #                             R1_int, R2_int, l1, n1, l2, n2, m1
        #                         )
        #                         mel_sq += abs(mel * mel_cc)

        #             # mel_sq = np.sum(tmp_mat, tmp_mat_cc, axis=-1)
        #             mel_sq = mel_sq_mat[l1, n1, l2, n2]

        #             # compute the volume
        #             rmax = np.exp(xgrid)[-1]
        #             V = (4.0 / 3.0) * pi * rmax ** 3.0

        #             smooth_fac = eig_diff ** 2 / (gamma ** 2 + eig_diff ** 2)
        #             sig += 2 * pi * smooth_fac * occnum_diff * mel_sq / (eig_diff * V)

        return sig

    @staticmethod
    @writeoutput.timing
    def calc_sig_func(
        eigfuncs,
        occnums,
        eigvals,
        dos,
        xgrid,
        orb_subset_1,
        orb_subset_2,
        omega_max=50,
        n_freq=100,
        gamma=0.5,
        eig_min_diff=1e-3,
        eig_max_diff=1e4,
        occ_min_diff=1e-4,
    ):

        nbands, nspin, lmax, nmax = np.shape(occnums)

        R1_int = calc_R1_int_mat(eigfuncs, occnums, xgrid, orb_subset_1, orb_subset_2)
        R2_int = calc_R2_int_mat(eigfuncs, occnums, xgrid, orb_subset_1, orb_subset_2)
        P2_int = P_mat_int(2, lmax)
        P4_int = P_mat_int(4, lmax)

        tmp_mat_1 = np.einsum("kabcd,ace->kabcde", R1_int, P2_int)
        tmp_mat_2 = np.einsum("kabcd,ace->kabcde", R2_int, P4_int)
        tmp_mat_3 = np.einsum("kabcd,ace->kcdabe", R1_int, P2_int)
        tmp_mat_4 = np.einsum("kabcd,ace->kcdabe", R2_int, P4_int)

        occ_diff_mat = calc_occ_diff_mat(occnums, orb_subset_1, orb_subset_2)
        eig_diff_mat = calc_eig_diff_mat(eigvals, orb_subset_1, orb_subset_2)
        dos_prod_mat = calc_dos_prod_mat(dos, occnums, orb_subset_1, orb_subset_2)

        mel_sq_mat = np.sum(
            np.abs((tmp_mat_1 + tmp_mat_2) * (tmp_mat_3 + tmp_mat_4)),
            axis=-1,
        )

        # omega_arr = np.logspace(-5, np.log(omega_max), n_freq)
        omega_0 = 1e-5
        omega_arr = np.linspace(omega_0, np.sqrt(omega_max), n_freq) ** 2
        # omega_arr = np.linspace(omega_0, omega_max, n_freq)
        sig_omega = np.zeros((np.size(omega_arr), 2))
        omega_dummy_mat = np.ones((nbands, lmax, nmax, lmax, nmax, n_freq))
        eig_diff_omega_mat = np.einsum(
            "nijkl,nijklm->nijklm", eig_diff_mat, omega_dummy_mat
        )
        eig_diff_lorentz_mat = lorentzian(omega_arr, eig_diff_omega_mat, gamma)

        mat1 = np.einsum("kln,klnpq->klnpq", dos_prod_mat, mel_sq_mat * occ_diff_mat)
        mat2 = eig_diff_lorentz_mat / eig_diff_omega_mat

        rmax = np.exp(xgrid)[-1]
        V = (4.0 / 3.0) * pi * rmax ** 3.0

        sig_omega[:, 1] = np.einsum("nijkl,nijklm->m", mat1, mat2) * 2 * pi / V
        sig_omega[:, 0] = omega_arr

        # for i, omega in enumerate(omega_arr):
        #     sig_omega[i, 0] = omega
        #     for l1, n1 in orb_subset_1:
        #         for l2, n2 in orb_subset_2:
        #             # the eigenvalue difference
        #             eig_diff = eigvals[0, l1, n1] - eigvals[0, l2, n2]
        #             # occupation number difference
        #             occnum_diff = abs(occnums[0, l1, n1] - occnums[0, l2, n2])
        #             if eig_diff < eig_min_diff:
        #                 continue
        #             elif eig_diff > eig_max_diff:
        #                 continue
        #             elif abs(l1 - l2) != 1:
        #                 continue
        #             elif occnum_diff < occ_min_diff:
        #                 continue
        #             else:
        #                 occnum_diff = -(occnums[0, l1, n1] - occnums[0, l2, n2])

        #                 mel_sq = mel_sq_mat[l1, n1, l2, n2]
        #                 # compute the volume
        #                 rmax = np.exp(xgrid)[-1]
        #                 V = (4.0 / 3.0) * pi * rmax ** 3.0

        #                 lorentz = lorentzian(omega, eig_diff, gamma)
        #                 sig_omega[i, 1] += (
        #                     2 * pi * lorentz * occnum_diff * mel_sq / (V * omega)
        #                 )
        sig_tot = (2 * V / pi) * np.trapz(sig_omega[:, 1], x=omega_arr)

        return sig_omega, sig_tot


def sph_ham_coeff(l, m):
    r"""The coefficients of spherical harmonic functions"""
    c_lm = sqrt((2 * l + 1) * factorial(l - m) / (factorial(l + m) * 4 * pi))
    return c_lm


def P_mat_int(func_int, lmax):

    P_mat = np.zeros((lmax, lmax, 2 * lmax + 1))

    for l1 in range(lmax):
        for l2 in range(lmax):
            if abs(l1 - l2) == 1:
                lsmall = min(l1, l2)
                for m in range(-lsmall, lsmall + 1):
                    P_mat[l1, l2, m] = P_int(func_int, l1, l2, m)
            else:
                continue
    return P_mat


def P_int(func_int, l1, l2, m):
    r"""The P2 integral"""

    if func_int == 2:
        integ = quad(P2_func, -1, 1, args=(l1, l2, m))[0]
    elif func_int == 4:
        integ = quad(P4_func, -1, 1, args=(l1, l2, m))[0]

    return 2 * pi * sph_ham_coeff(l1, m) * sph_ham_coeff(l2, m) * integ


def P2_func(x, l1, l2, m):
    r"""Input functional for P2_int"""

    return x * lpmv(m, l1, x) * lpmv(m, l2, x)


def P4_func(x, l1, l2, m):
    r"""Input functional for P4_int"""

    if (l2 + m) != 0:
        factor = (l2 + m) * lpmv(m, l2 - 1, x) - l2 * x * lpmv(m, l2, x)
    else:
        factor = -l2 * x * lpmv(m, l2, x)

    return lpmv(m, l1, x) * factor


# jit
@writeoutput.timing
def calc_R1_int_mat(eigfuncs, occnums, xgrid, orb_subset_1, orb_subset_2):
    r"""Compute the R1 integral."""

    # take the derivative of orb2
    # compute the gradient of the orbitals
    deriv_orb2 = np.gradient(eigfuncs, xgrid, axis=-1, edge_order=2)

    # chain rule to convert from dP_dx to dX_dr
    grad_orb2 = np.exp(-1.5 * xgrid) * (deriv_orb2 - 0.5 * eigfuncs)

    # initiliaze the matrix
    nbands, nspin, lmax, nmax = np.shape(occnums)
    R1_mat = np.zeros((nbands, lmax, nmax, lmax, nmax))

    # integrate over the sphere
    for k in range(nbands):
        for l1, n1 in orb_subset_1:
            for l2, n2 in orb_subset_2:
                if abs(l1 - l2) != 1:
                    continue
                elif abs(occnums[k, 0, l2, n2] - occnums[k, 0, l1, n1]) < 0:
                    continue
                else:
                    func_int = (
                        eigfuncs[k, 0, l1, n1]
                        * np.exp(-xgrid / 2.0)
                        * grad_orb2[k, 0, l2, n2]
                    )

                    R1_mat[k, l1, n1, l2, n2] = (
                        4 * pi * np.trapz(np.exp(3.0 * xgrid) * func_int, xgrid)
                    )
    return R1_mat


# jit
@writeoutput.timing
def calc_occ_diff_mat(occnums, orb_subset_1, orb_subset_2):

    nbands, nspin, lmax, nmax = np.shape(occnums)
    occ_diff_mat = np.zeros((nbands, lmax, nmax, lmax, nmax))

    for k in range(nbands):
        for l1, n1 in orb_subset_1:
            for l2, n2 in orb_subset_2:
                occ_diff = -(occnums[k, 0, l1, n1] - occnums[k, 0, l2, n2])
                if abs(l1 - l2) != 1:
                    continue
                elif occ_diff < 0:
                    continue
                else:
                    occ_diff_mat[k, l1, n1, l2, n2] = occ_diff
    return occ_diff_mat


# jit
@writeoutput.timing
def calc_dos_prod_mat(dos, occnums, orb_subset_1, orb_subset_2):

    nbands, nspin, lmax, nmax = np.shape(dos)
    dos_prod_mat = np.zeros((nbands, lmax, nmax, lmax, nmax))

    for k in range(nbands):
        for l1, n1 in orb_subset_1:
            for l2, n2 in orb_subset_2:
                occ_diff = -(occnums[k, 0, l1, n1] - occnums[k, 0, l2, n2])
                if abs(l1 - l2) != 1:
                    continue
                elif occ_diff < 0:
                    continue
                else:
                    dos_prod_mat[k, l1, n1, l2, n2] = dos[
                        k, 0, l2, n2
                    ]  # * dos[k, 0, l2, n2]

    dos_prod_mat = dos[:, 0]
    return dos_prod_mat


# jit
@writeoutput.timing
def calc_eig_diff_mat(eigvals, orb_subset_1, orb_subset_2):

    nbands, nspin, lmax, nmax = np.shape(eigvals)
    eig_diff_mat = np.zeros((nbands, lmax, nmax, lmax, nmax))
    eig_diff_mat += 1e-6

    for k in range(nbands):
        for l1, n1 in orb_subset_1:
            for l2, n2 in orb_subset_2:
                if abs(l1 - l2) != 1:
                    continue
                elif eigvals[k, 0, l1, n1] - eigvals[k, 0, l2, n2] < 0:
                    continue
                else:
                    eig_diff_mat[k, l1, n1, l2, n2] = (
                        eigvals[k, 0, l1, n1] - eigvals[k, 0, l2, n2]
                    )
    return eig_diff_mat


# jit
@writeoutput.timing
def calc_R2_int_mat(eigfuncs, occnums, xgrid, orb_subset_1, orb_subset_2):
    r"""Compute the R2 integral."""

    # initiliaze the matrix
    nbands, nspin, lmax, nmax = np.shape(occnums)
    R2_mat = np.zeros((nbands, lmax, nmax, lmax, nmax))

    # integrate over the sphere
    for k in range(nbands):
        for l1, n1 in orb_subset_1:
            for l2, n2 in orb_subset_2:
                if abs(l1 - l2) != 1:
                    continue
                elif abs(occnums[k, 0, l2, n2] - occnums[k, 0, l1, n1]) < 0:
                    continue
                else:
                    func_int = (
                        eigfuncs[k, 0, l1, n1] * eigfuncs[k, 0, l2, n2] * np.exp(-xgrid)
                    )

                R2_mat[k, l1, n1, l2, n2] = (
                    4 * pi * np.trapz(np.exp(2.0 * xgrid) * func_int, xgrid)
                )

    return R2_mat


def calc_R1_int(orb1, orb2, xgrid):
    r"""Compute the R1 integral."""

    # take the derivative of orb2
    # compute the gradient of the orbitals
    deriv_orb2 = np.gradient(orb2, xgrid, axis=-1, edge_order=2)

    # chain rule to convert from dP_dx to dX_dr
    grad_orb2 = np.exp(-1.5 * xgrid) * (deriv_orb2 - 0.5 * orb2)

    # integrate over the sphere
    func_int = orb1 * np.exp(-xgrid / 2.0) * grad_orb2
    return np.trapz(np.exp(3.0 * xgrid) * func_int, xgrid)


def calc_R2_int(orb1, orb2, xgrid):
    r"""Compute the R2 integral."""

    func_int = np.exp(xgrid) * orb1 * orb2

    return np.trapz(func_int, xgrid)


def sig_contrib(orbitals, n1, l1, n2, l2):

    # the eigenvalue difference
    eig_diff = orbitals.eigvals[0, l1, n1] - orbitals.eigvals[0, l2, n2]

    # the occupation number difference
    # first sum bound and unbound comps
    occnums_tot = orbitals.occnums + orbitals.occnums_ub

    # convert the occupation numbers to be "pure" occnums
    lbound = orbitals.lbound + orbitals.lunbound
    lmax = np.shape(lbound)[1]
    lmat_inv = np.zeros_like(lbound)
    for l in range(lmax):
        lmat_inv[:, l] = 0.5 / (2 * l + 1.0)
    occnums = lmat_inv * occnums_tot

    occnum_diff = -(occnums[0, l1, n1] - occnums[0, l2, n2])

    # eigenfunctions
    orb_l1n1 = sqrt(4 * pi) * orbitals.eigfuncs[0, l1, n1]
    orb_l2n2 = sqrt(4 * pi) * orbitals.eigfuncs[0, l2, n2]

    xgrid = orbitals._xgrid

    # compute the matrix element
    mel = calc_mel_kg(orb_l1n1, orb_l2n2, l1, n1, l2, n2, xgrid)
    mel_sq = mel ** 2

    # compute the volume
    rmax = np.exp(xgrid)[-1]
    V = (4.0 / 3.0) * pi * rmax ** 3.0

    sig = 2 * pi * occnum_diff * mel_sq / (eig_diff * V)

    return sig


def calc_mel_kg(orb_l1n1, orb_l2n2, l1, n1, l2, n2, xgrid):

    R1_int = calc_R1_int(orb_l1n1, orb_l2n2, xgrid)
    R2_int = calc_R2_int(orb_l1n1, orb_l2n2, xgrid)

    lsmall = min(l1, l2)

    mel_tot = 0.0
    for m in range(-lsmall, lsmall + 1):
        mel_tot += R1_int * P_int(2, l1, l2, m)
        mel_tot += R2_int * P_int(4, l1, l2, m)

    return mel_tot


def calc_mel_kgm(orb_l1n1, orb_l2n2, l1, n1, l2, n2, m, xgrid):

    R1_int = calc_R1_int(orb_l1n1, orb_l2n2, xgrid)
    R2_int = calc_R2_int(orb_l1n1, orb_l2n2, xgrid)

    mel_tot = 0.0
    mel_tot += R1_int * P_int(2, l1, l2, m)
    mel_tot += R2_int * P_int(4, l1, l2, m)

    return mel_tot


def calc_mel_kgm_2(R1_int, R2_int, l1, n1, l2, n2, m):

    mel = R1_int[l1, n1, l2, n2] * P_int(2, l1, l2, m) + R2_int[l1, n1, l2, n2] * P_int(
        4, l1, l2, m
    )

    return mel


def lorentzian(x, x0, gamma):

    # prefac = x / (x ** 2 + gamma ** 2)
    prefac = 1.0
    # prefac = 1 / x
    return (gamma / pi) * (prefac / (gamma ** 2 + (x - x0) ** 2))


def prod_eigfuncs(phi0, phi1, xgrid):

    return 4 * pi * np.trapz(np.exp(2.0 * xgrid) * phi0 * phi1, xgrid)


def proj_eigfuncs(phi0, phi1, xgrid):

    return (prod_eigfuncs(phi0, phi1, xgrid) / prod_eigfuncs(phi0, phi0, xgrid)) * phi0


@writeoutput.timing
def gs_ortho(eigfuncs, xgrid):

    nbands, nspin, lmax, nmax, ngrid = np.shape(eigfuncs)
    eigfuncs_ortho = np.zeros_like(eigfuncs)
    norm = np.zeros_like(eigfuncs)

    for k in range(nbands):
        for sp in range(nspin):
            for l in range(lmax):
                for n1 in range(nmax):
                    eigfuncs_ortho[k, sp, l, n1] = eigfuncs[k, sp, l, n1]
                    for n2 in range(n1):
                        eigfuncs_ortho[k, sp, l, n1] -= proj_eigfuncs(
                            eigfuncs_ortho[k, sp, l, n2],
                            eigfuncs[k, sp, l, n1],
                            xgrid,
                        )
                    norm[k, sp, l, n1] = prod_eigfuncs(
                        eigfuncs_ortho[k, sp, l, n1],
                        eigfuncs_ortho[k, sp, l, n1],
                        xgrid,
                    )

    a = norm ** (-0.5)
    eigfuncs_ortho = eigfuncs_ortho * a

    return eigfuncs_ortho
