"""
Contains models used to compute properties of interest from the Atom object.

So far, the only model implemented is the ISModel. More models will be added in future releases.

Classes
-------
ISModel : Ion-sphere type model, static properties such as KS orbitals, density and energy are directly computed
"""

# import standard packages

# import external packages
import numpy as np
from mendeleev import element
from math import pi, log

# import internal packages
from . import check_inputs
from . import config
from . import staticKS
from . import convergence
from . import writeoutput
from . import xc


class ISModel:
    """
    The ISModel represents a particular family of AA models known as ion-sphere models.

    The implementation in atoMEC is based on the model described in the following pre-print:
    T. J. Callow, E. Kraisler, S. B. Hansen, and A. Cangi, (2021).
    First-principles derivation and properties of density-functional average-atom models. arXiv preprint arXiv:2103.09928.

    Parameter inputs for this model are related to particular choices of approximation, e.g. boundary conditions
    or exchange-correlation functional, rather than fundamental physical properties.

    Parameters
    ----------
    atom : :obj:`Atom`
        The :obj:`Atom` object
    xfunc_id : str or int, optional
        The exchange functional, can be the libxc code or string, or special internal value
        Default : "lda_x"
    cfunc_id : str or int, optional
        The correlation functional, can be the libxc code or string, or special internal value
        Default : "lda_c_pw"
    bc : str, optional
        The boundary condition, can be "dirichlet" or "neumann"
        Default : "dirichlet"
    spinpol : bool, optional
        Whether to run a spin-polarized calculation
        Default : False
    spinmag : int, optional
        The spin-magentization
        Default: 0 for nele even, 1 for nele odd
    unbound : str, optional
        The way in which the unbound electron density is computed
        Default : "ideal"
    write_info : bool, optional
        Writes information about the model parameters
        Default : True

    Attributes
    ----------
    nele_tot: int
        total number of electrons
    nele: array_like
        number of electrons per spin channel (or total if spin unpolarized)
    """

    def __init__(
        self,
        atom,
        xfunc_id=config.xfunc_id,
        cfunc_id=config.cfunc_id,
        bc=config.bc,
        spinpol=config.spinpol,
        spinmag=-1,
        unbound=config.unbound,
        write_info=True,
    ):

        # Input variables
        self.nele_tot = atom.nele
        self.spinpol = spinpol
        self.spinmag = spinmag
        self.xfunc_id = xfunc_id
        self.cfunc_id = cfunc_id
        self.bc = bc
        self.unbound = unbound

        # print the information
        if write_info:
            print(self.info)

    @property
    def spinpol(self):
        """bool: Whether calculation will be spin-polarized."""
        return self._spinpol

    @spinpol.setter
    def spinpol(self, spinpol):
        self._spinpol = check_inputs.ISModel.check_spinpol(spinpol)
        # define the leading dimension for KS quantities (density, orbs, pot)
        if self._spinpol:
            config.spindims = 2
        else:
            config.spindims = 1

        try:
            # compute the no of electrons in each spin channel
            self.nele = check_inputs.ISModel.calc_nele(
                self.spinmag, self.nele_tot, self.spinpol
            )
            config.nele = self.nele
        except AttributeError:
            pass

        # reset the x and c functionals if spinpol changes
        try:
            config.xfunc = xc.set_xc_func(self._xfunc_id)
            config.cfunc = xc.set_xc_func(self._cfunc_id)
        except AttributeError:
            pass

    @property
    def spinmag(self):
        """int: the spin magentization (difference in number of up/down spin electrons)."""
        return self._spinmag

    @spinmag.setter
    def spinmag(self, spinmag):
        self._spinmag = check_inputs.ISModel.check_spinmag(spinmag, self.nele_tot)
        try:
            # compute the no of electrons in each spin channel
            self.nele = check_inputs.ISModel.calc_nele(
                self.spinmag, self.nele_tot, self.spinpol
            )
            config.nele = self.nele
        except AttributeError:
            pass

    @property
    def xfunc_id(self):
        """str: exchange functional shorthand id."""
        return self._xfunc_id

    @xfunc_id.setter
    def xfunc_id(self, xfunc_id):
        self._xfunc_id = check_inputs.ISModel.check_xc(xfunc_id, "exchange")
        # define the exchange func object
        config.xfunc = xc.set_xc_func(self._xfunc_id)

    @property
    def cfunc_id(self):
        """str: correlation functional shorthand id."""
        return self._cfunc_id

    @cfunc_id.setter
    def cfunc_id(self, cfunc_id):
        self._cfunc_id = check_inputs.ISModel.check_xc(cfunc_id, "correlation")
        # define the correlation func object
        config.cfunc = xc.set_xc_func(self._cfunc_id)

    @property
    def bc(self):
        """str: the boundary condition for solving the KS equations in a finite sphere."""
        return self._bc

    @bc.setter
    def bc(self, bc):
        self._bc = check_inputs.ISModel.check_bc(bc)
        config.bc = self._bc

    @property
    def unbound(self):
        """str: the treatment of unbound (free) electrons."""
        return self._unbound

    @unbound.setter
    def unbound(self, unbound):
        self._unbound = check_inputs.ISModel.check_unbound(unbound)
        config.unbound = self._unbound

    @property
    def info(self):
        """str: formatted description of the ISModel attributes."""
        return writeoutput.write_ISModel_data(self)

    @writeoutput.timing
    def CalcEnergy(
        self,
        nmax,
        lmax,
        grid_params={},
        conv_params={},
        scf_params={},
        force_bound=[],
        write_info=True,
    ):
        r"""
        Run a self-consistent calculation to minimize the Kohn-Sham free energy functional.

        Parameters
        ----------
        nmax : int
            maximum no. eigenvalues to compute for each value of angular momentum
        lmax : int
            maximum no. angular momentum eigenfunctions to consider
        grid_params : dict, optional
            dictionary of grid parameters as follows:
            {
            `ngrid` (``int``)   : number of grid points,
            `x0`    (``float``) : LHS grid point takes form
            :math:`r_0=\exp(x_0)`; :math:`x_0` can be specified
            }
        conv_params : dict, optional
            dictionary of convergence parameters as follows:
            {
            `econv` (``float``) : convergence for total energy,
            `nconv` (``float``) : convergence for density,
            `vconv` (``float``) : convergence for electron number
            }
        scf_params : dict, optional
            dictionary for scf cycle parameters as follows:
            {
            `maxscf`  (``int``)   : maximum number of scf cycles,
            `mixfrac` (``float``) : density mixing fraction
            }
        write_info : bool, optional
            prints the scf cycle and final parameters
            defaults to True

        Returns
        -------
        output_dict : dict
            dictionary containing final KS quantities as follows:
            {
            `energy` (:obj:`staticKS.Energy`)       : total energy object,
            `density` (:obj:`staticKS.Density`)     : density object,
            `potential` (:obj:`staticKS.Potential`)  : potential object,
            `orbitals` (:obj:`staticKS.Orbitals`)    : orbitals object
            }
        """
        # boundary cond, unbound electrons, xc func objects
        config.bc = self.bc
        config.unbound = self.unbound

        # reset global parameters if they are changed
        config.nmax = nmax
        config.lmax = lmax
        config.grid_params = check_inputs.EnergyCalcs.check_grid_params(grid_params)
        config.conv_params = check_inputs.EnergyCalcs.check_conv_params(conv_params)
        config.scf_params = check_inputs.EnergyCalcs.check_scf_params(scf_params)

        # experimental change
        config.force_bound = force_bound

        # set up the xgrid and rgrid
        xgrid, rgrid = staticKS.log_grid(log(config.r_s))

        # initialize orbitals
        orbs = staticKS.Orbitals(xgrid)
        # use coulomb potential as initial guess
        v_init = staticKS.Potential.calc_v_en(xgrid)
        orbs.compute(v_init, init=True)

        # occupy orbitals
        orbs.occupy()

        # write the initial spiel
        scf_init = writeoutput.SCF.write_init()
        print(scf_init)

        # initialize the convergence object
        conv = convergence.SCF(xgrid)

        for iscf in range(config.scf_params["maxscf"]):

            # construct density
            rho = staticKS.Density(orbs)

            # construct potential
            pot = staticKS.Potential(rho)

            # compute energies
            energy = staticKS.Energy(orbs, rho)
            E_free = energy.F_tot

            # mix potential
            if iscf > 1:
                alpha = config.scf_params["mixfrac"]
                v_s = alpha * pot.v_s + (1 - alpha) * v_s_old
            else:
                v_s = pot.v_s

            # update the orbitals with the KS potential
            orbs.compute(v_s)
            orbs.occupy()

            # update old potential
            v_s_old = v_s

            # test convergence
            conv_vals = conv.check_conv(E_free, v_s, rho.total, iscf)

            # write scf output
            scf_string = writeoutput.SCF.write_cycle(iscf, E_free, conv_vals)
            print(scf_string)

            # exit if converged
            if conv_vals["complete"]:
                break

        # compute final density and energy
        rho = staticKS.Density(orbs)
        energy = staticKS.Energy(orbs, rho)

        # write final output
        scf_final = writeoutput.SCF().write_final(energy, orbs, rho, conv_vals)
        print(scf_final)

        # write the density to file
        writeoutput.density_to_csv(rgrid, rho)

        # write the potential to file
        writeoutput.potential_to_csv(rgrid, pot)

        output_dict = {
            "energy": energy,
            "density": rho,
            "potential": pot,
            "orbitals": orbs,
        }

        return output_dict
