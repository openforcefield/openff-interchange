"""Helpers for testing GROMACS interoperability."""

from openff.utilities import temporary_cd

from openff.interchange import Interchange
from openff.interchange.components.mdconfig import get_smirnoff_defaults
from openff.interchange.drivers import get_gromacs_energies


def gmx_roundtrip(state: Interchange, apply_smirnoff_defaults: bool = False):
    with temporary_cd():
        state.to_gromacs(prefix="state", decimal=8)
        new_state = Interchange.from_gromacs(topology_file="state.top", gro_file="state.gro")

        get_smirnoff_defaults(periodic=True).apply(new_state)


def gmx_monolithic_vs_itp(state: Interchange):
    with temporary_cd():
        # TODO: More helpful handling of failures, i.e.
        #         * Detect differences in positions
        #         * Detect differences in box vectors
        #         * Detect differences in non-bonded settings
        get_gromacs_energies(state, _monolithic=True).compare(get_gromacs_energies(state, _monolithic=False))
