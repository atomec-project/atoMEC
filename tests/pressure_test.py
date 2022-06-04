#!/usr/bin/env python3
"""
Pressure test.

Computes the pressure with ISModel.CalcPressure function and checks it returns
the expected value.
"""

from atoMEC import Atom, models, config
import numpy as np


# expected values and tolerance
pressure_expected = 0.013395564
accuracy = 1e-4


class TestPressure:
    """Test class for the CalcPressure function."""

    def test_pressure(self):
        """Check pressure is given by expected value."""
        # parallel
        config.numcores = -1

        assert np.isclose(self._run(), pressure_expected, atol=accuracy)

    @staticmethod
    def _run():
        """
        Compute the pressure for an Li atom.

        Returns
        -------
        pressure : float
            the pressure (in Hartree units)
        """
        # set up the atom and model
        Li_at = Atom("Li", 10, radius=2.5, units_temp="eV")
        model = models.ISModel(Li_at, unbound="quantum", v_shift=False)

        # run the SCF calculation
        output = model.CalcEnergy(
            3,
            3,
            scf_params={"maxscf": 3},
            grid_params={"ngrid": 1000},
            verbosity=1,
        )

        pressure = model.CalcPressure(Li_at, output)

        return pressure


if __name__ == "__main__":
    print(TestPressure._run())
