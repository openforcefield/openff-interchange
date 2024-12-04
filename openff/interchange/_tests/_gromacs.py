"""Helpers for testing GROMACS interoperability."""

from openff.utilities import temporary_cd

from openff.interchange import Interchange
from openff.interchange.drivers import get_gromacs_energies


def gmx_monolithic_vs_itp(state: Interchange):
    with temporary_cd():
        get_gromacs_energies(state, _monolithic=True).compare(get_gromacs_energies(state, _monolithic=False))

        # TODO: More helpful handling of failures, i.e.
        #         * Detect differences in positions
        #         * Detect differences in box vectors
        #         * Detect differences in non-bonded settings
