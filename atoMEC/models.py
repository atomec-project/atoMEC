"""
Contains models used to compute properties of interest from the Atom object.

So far, the only model implemented is the ISModel. More models will be added in future
releases.

Classes
-------
* :class:`ISModel` : Ion-sphere type model, static properties such as KS orbitals, \
density and energy are directly computed
"""

# import standard packages

# import external packages
from math import log, pi

# import internal packages
from . import check_inputs
from . import config
from . import staticKS
from . import convergence
from . import writeoutput
from . import xc
from . import unitconv


class ISModel:
    """
    The ISModel represents a particular family of AA models known as ion-sphere models.

    The implementation in atoMEC is based on the model described in [1]_.

    Parameter inputs for this model are related to particular choices of approximation,
    e.g. boundary conditions or exchange-correlation functional, rather than
    fundamental physical properties.

    Parameters
    ----------
    atom : atoMEC.Atom
        The main atom object
    xfunc_id : str or int, optional
        The exchange functional, can be the libxc code or string,
        or special internal value
        Default : "lda_x"
    cfunc_id : str or int, optional
        The correlation functional, can be the libxc code or string, or special
        internal value
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
    v_shift : bool, optional
        Shifts the potential vertically by its value at the boundary, v(R_s)
        Default : True
    write_info : bool, optional
        Writes information about the model parameters
        Default : True

    Attributes
    ----------
    nele_tot: int
        total number of electrons
    nele: array_like
        number of electrons per spin channel (or total if spin unpolarized)

    References
    ----------
    .. [1] T. J. Callow, E. Kraisler, S. B. Hansen, and A. Cangi, (2021).
       First-principles derivation and properties of density-functional
       average-atom models, `arXiv:2103.09928 <https://arxiv.org/abs/2103.09928>`__.
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
        v_shift=config.v_shift,
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
        self.v_shift = v_shift

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
        """int: the spin magentization (difference in no. up/down spin electrons)."""
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
        """str: boundary condition for solving the KS equations in a finite sphere."""
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
    def v_shift(self):
        """bool: shift the potential vertically by a constant."""
        return self._v_shift

    @v_shift.setter
    def v_shift(self, v_shift):
        self._v_shift = check_inputs.ISModel.check_v_shift(v_shift)
        config.v_shift = self._v_shift

    @property
    def info(self):
        """str: formatted description of the ISModel attributes."""
        return writeoutput.write_ISModel_data(self)

    # @writeoutput.timing
    def CalcEnergy(
        self,
        nmax,
        lmax,
        grid_params={},
        conv_params={},
        scf_params={},
        force_bound=[],
        verbosity=0,
        write_info=True,
        write_density=True,
        write_potential=True,
        density_file="density.csv",
        potential_file="potential.csv",
        guess=False,
        guess_pot=0,
    ):
        r"""
        Run a self-consistent calculation to minimize the Kohn-Sham free energy.

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
        force_bound : list of list of ints, optional
            force certain levels to be bound, for example:
            `force_bound = [0, 1, 0]`
            forces the orbital with quantum numbers :math:`\sigma=0,\ l=1,\ n=0` to be
            always bound even if it has positive energy. This prevents convergence
            issues.
        verbosity : int, optional
            how much information is printed at each SCF cycle.
            `verbosity=0` prints the total energy and convergence values (default).
            `verbosity=1` prints the above and the KS eigenvalues and occupations.
        write_info : bool, optional
            prints the scf cycle and final parameters
            default: True
        write_density : bool, optional
            writes the density to a text file
            default: True
        write_potential : bool, optional
            writes the potential to a text file
            default: True
        density_file : str, optional
            name of the file to write the density to
            default: `density.csv`
        potential_file : str, optional
            name of the file to write the potential to
            default: `potential.csv`
        guess : bool, optional
            use coulomb pot (guess=False) or given pot (guess=True) as initial guess
        guess_pot : numpy array, optional
            initial guess for the potential

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
        # use coulomb potential or input given potential as initial guess
        if guess:
            v_init = guess_pot
        else:
            v_init = staticKS.Potential.calc_v_en(xgrid)
        v_s_old = v_init  # initialize the old potential
        orbs.compute(v_init, init=True)

        # occupy orbitals
        orbs.occupy()

        # write the initial spiel
        scf_init = writeoutput.SCF.write_init()
        if write_info:
            print(scf_init)

        # initialize the convergence object
        conv = convergence.SCF(xgrid)

        for iscf in range(config.scf_params["maxscf"]):

            # print orbitals and occupations
            if verbosity == 1:
                eigs, occs = writeoutput.SCF.write_orb_info(orbs)
                print("\n" + "Orbital eigenvalues (Ha) :" + "\n\n" + eigs)
                print("Orbital occupations (2l+1) * f_{nl} :" + "\n\n" + occs)

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

            # shift potential
            if config.v_shift:
                for i in range(config.spindims):
                    v_s[i, :] = v_s[i, :] - v_s[i, -1]

            # update the orbitals with the KS potential
            orbs.compute(v_s)
            orbs.occupy()

            # update old potential
            v_s_old = v_s

            # test convergence
            conv_vals = conv.check_conv(E_free, v_s, rho.total, iscf)

            # write scf output
            scf_string = writeoutput.SCF.write_cycle(iscf, E_free, conv_vals)
            if write_info:
                print(scf_string)

            # exit if converged
            if conv_vals["complete"]:
                break

        # compute final density and energy
        rho = staticKS.Density(orbs)
        energy = staticKS.Energy(orbs, rho)

        # write final output
        scf_final = writeoutput.SCF().write_final(energy, orbs, rho, conv_vals)
        if write_info:
            print(scf_final)

        # write the density to file
        if write_density:
            writeoutput.density_to_csv(rgrid, rho, density_file)

        # write the potential to file
        if write_potential:
            writeoutput.potential_to_csv(rgrid, pot, potential_file)

        output_dict = {
            "energy": energy,
            "density": rho,
            "potential": pot,
            "orbitals": orbs,
        }

        return output_dict

    def CalcPressure(
        self,
        atom,
        energy_output,
        nmax=None,
        lmax=None,
        grid_params={},
        conv_params={},
        scf_params={},
        force_bound=[],
        write_info=False,
        verbosity=0,
        dR=0.01,
    ):
        r"""
        Calculate the electronic pressure using the finite differences method.

        Parameters
        ----------
        atom : atoMEC.Atom
            The main atom object
        energy_output : dict
            output parameters of the function CalcEnergy
        nmax : int
            maximum no. eigenvalues to compute for each value of angular momentum
        lmax : int
            maximum no. angular momentum eigenfucntions to consider
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
        force_bound : list of list of ints, optional
            force certain levels to be bound, for example:
            `force_bound = [0, 1, 0]`
            forces the orbital quith quantum numbers :math:`\sigma=0,\ l=1,\ n=0` to be
            always bound even if it has positive energy. This prevents convergence
            issues.
        verbosity : int, optional
            how much information is printed at each SCF cycle.
            `verbosity=0` prints the total energy and convergence values (default)
            `verbosity=1` prints the above and the KS eigenvalues and occupations.
        write_info : bool, optional
            prints the scf cycle and final parameters
            defaults to False
        dR : float, optional
            radius difference for finite difference calculation
            defaults to 0.01

        Returns
        -------
        pressureHa : float
            electronic pressure in Ha
        """
        print("Pressure is being calculated. Please be patient!" + "\n")

        # set nmax and lmax to default config values if they aren't specified
        if nmax is None:
            nmax = config.nmax
        if lmax is None:
            lmax = config.lmax

        # initialize the main radius we are interested in
        main_rad = atom.radius

        # change main radius by +dR
        atom.radius = main_rad + dR

        # calculate free energy for new radius and store it
        output1 = self.CalcEnergy(
            nmax,
            lmax,
            grid_params=grid_params,
            write_info=write_info,
            guess=True,
            guess_pot=energy_output["potential"].v_s,
            write_density=False,
            write_potential=False,
        )
        F1 = output1["energy"].F_tot

        # change main radius by -dR
        atom.radius = main_rad - dR

        # calculate free energy for new radius and store it
        output2 = self.CalcEnergy(
            nmax,
            lmax,
            grid_params=grid_params,
            write_info=write_info,
            guess=True,
            guess_pot=energy_output["potential"].v_s,
            write_density=False,
            write_potential=False,
        )
        F2 = output2["energy"].F_tot

        dFdR = (F1 - F2) / (2 * dR)  # finite differences
        dRdV = 1 / (4 * pi * main_rad ** 2)  # V = sphere of radius R (main_rad) volume

        # calculate pressure by thermodynamic definition p = -dFdV and chain rule
        pressureHa = -dFdR * dRdV
        pressureGPa = pressureHa * unitconv.ha_to_gpa

        pressurestat = "{preamble:30s}: {p_ha:<.6g} Ha / {p_gpa:<.6g} GPa".format(
            preamble="Electronic pressure", p_ha=pressureHa, p_gpa=pressureGPa
        )
        spc = "\n"
        print(pressurestat + spc)

        return pressureHa
