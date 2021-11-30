"""Kubo-Greenwood conductivity etc"""

from math import pi, sqrt, factorial
from scipy.special import lpmv
from scipy.integrate import quad
import numpy as np


################################################################
# functions to compute various integrals of legendre functions #
################################################################


class KuboGreenwood:
    def __init__(self, orbitals, valence_orbs=[]):

        self._orbitals = orbitals
        self._xgrid = orbitals._xgrid
        self._eigfuncs = orbitals.eigfuncs
        self._eigvals = orbitals.eigvals
        self._occnums = np.zeros_like(self._eigvals)
        self._spindims, self._lmax, self._nmax = np.shape(self._eigvals)
        self.valence_orbs = valence_orbs
        self._all_orbs = None
        self._cond_orbs = None
        self._sig_cc = None
        self._sig_cv = None
        self._sig_vv = None
        self._sig_tot = None
        self._N_tot = None
        self._N_free = None

    @property
    def occnums(self):
        if np.all(self._occnums == 0.0):
            self._occnums = self.calc_occnums()
        return self._occnums

    def calc_occnums(self):
        # convert the occupation numbers to be "pure" occnums
        lmat_inv = np.zeros_like(self._orbitals.lbound)
        occnums_tot = self._orbitals.occnums + self._orbitals._occnums_ub
        for l in range(self._lmax):
            lmat_inv[:, l] = 0.5 / (2 * l + 1.0)
        occnums = lmat_inv * occnums_tot

        return occnums

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
        if self._sig_tot is None:
            self._sig_tot = self.calc_sig(
                self._eigfuncs,
                self.occnums,
                self._eigvals,
                self._xgrid,
                self.all_orbs,
                self.all_orbs,
            )
        return self._sig_tot

    @property
    def sig_cc(self):
        if self._sig_cc is None:
            self._sig_cc = self.calc_sig(
                self._eigfuncs,
                self.occnums,
                self._eigvals,
                self._xgrid,
                self.cond_orbs,
                self.cond_orbs,
            )
        return self._sig_cc

    @property
    def sig_vv(self):
        if self._sig_vv is None:
            self._sig_vv = self.calc_sig(
                self._eigfuncs,
                self.occnums,
                self._eigvals,
                self._xgrid,
                self.valence_orbs,
                self.valence_orbs,
            )
        return self._sig_vv

    @property
    def sig_cv(self):
        if self._sig_cv is None:
            self._sig_cv = self.calc_sig(
                self._eigfuncs,
                self.occnums,
                self._eigvals,
                self._xgrid,
                self.cond_orbs,
                self.valence_orbs,
            )
        return self._sig_cv

    @property
    def N_tot(self):
        if self._N_tot is None:
            rmax = np.exp(self._xgrid)[-1]
            V = (4.0 / 3.0) * pi * rmax ** 3.0
            self._N_tot = self.sig_tot * (2 * V / pi)
        return self._N_tot

    @property
    def N_free(self):
        if self._N_free is None:
            rmax = np.exp(self._xgrid)[-1]
            V = (4.0 / 3.0) * pi * rmax ** 3.0
            self._N_free = self.sig_cc * (2 * V / pi)
        return self._N_free

    @staticmethod
    def calc_sig(
        eigfuncs, occnums, eigvals, xgrid, orb_subset_1, orb_subset_2, gamma=0.0
    ):

        sig = 0.0
        for l1, n1 in orb_subset_1:
            for l2, n2 in orb_subset_2:
                # the eigenvalue difference
                eig_diff = eigvals[0, l1, n1] - eigvals[0, l2, n2]
                if eig_diff <= 0:
                    continue
                elif abs(l1 - l2) != 1:
                    continue
                elif abs(occnums[0, l1, n1] - occnums[0, l2, n2]) < 1e-6:
                    continue
                else:
                    occnum_diff = -(occnums[0, l1, n1] - occnums[0, l2, n2])

                    # eigenfunctions
                    orb_l1n1 = sqrt(4 * pi) * eigfuncs[0, l1, n1]
                    orb_l2n2 = sqrt(4 * pi) * eigfuncs[0, l2, n2]

                    # compute the matrix element
                    mel = calc_mel_kg(orb_l1n1, orb_l2n2, l1, n1, l2, n2, xgrid)
                    mel_sq = mel ** 2

                    # compute the volume
                    rmax = np.exp(xgrid)[-1]
                    V = (4.0 / 3.0) * pi * rmax ** 3.0

                    smooth_fac = eig_diff ** 2 / (gamma ** 2 + eig_diff ** 2)
                    sig += 2 * pi * smooth_fac * occnum_diff * mel_sq / (eig_diff * V)

        return sig


def sph_ham_coeff(l, m):
    r"""The coefficients of spherical harmonic functions"""
    c_lm = sqrt((2 * l + 1) / (4 * pi) * factorial(l - abs(m)) / factorial(l + abs(m)))
    return c_lm


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


def calc_R1_int(orb1, orb2, xgrid):
    r"""Compute the R1 integral."""

    # take the derivative of orb2
    # compute the gradient of the orbitals
    deriv_orb2 = np.gradient(orb2, xgrid, axis=-1, edge_order=2)

    # chain rule to convert from dP_dx to dX_dr
    grad_orb2 = np.exp(-1.5 * xgrid) * (deriv_orb2 - 0.5 * orb2)

    # integrate over the sphere
    func_int = np.exp(3.0 * xgrid) * orb1 * np.exp(-xgrid / 2.0) * grad_orb2
    return np.trapz(func_int, xgrid)


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
