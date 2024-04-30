"""Test compatibility checks in OpenMM import."""

import re

import pytest
from openff.toolkit import Molecule, Quantity

from openff.interchange.exceptions import UnsupportedImportError
from openff.interchange.interop.openmm._import import from_openmm
from openff.interchange.warnings import MissingPositionsWarning


class TestUnsupportedCases:
    def test_error_topology_mismatch(self, monkeypatch, sage_unconstrained, ethanol):
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        topology = ethanol.to_topology()
        topology.box_vectors = Quantity([4, 4, 4], "nanometer")

        other_topology = Molecule.from_smiles("O").to_topology()
        other_topology.box_vectors = Quantity([4, 4, 4], "nanometer")

        system = sage_unconstrained.create_openmm_system(topology)

        # This should not error
        from_openmm(system=system, topology=topology.to_openmm())

        with pytest.raises(
            UnsupportedImportError,
            match=re.escape(
                "The number of particles in the system (9) and "
                "the number of atoms in the topology (3) do not match.",
            ),
        ):
            from_openmm(
                system=system,
                topology=other_topology.to_openmm(),
            )

    def test_found_virtual_sites(self, monkeypatch, tip4p, water):
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        topology = water.to_topology()
        topology.box_vectors = Quantity([4, 4, 4], "nanometer")

        system = tip4p.create_openmm_system(topology)

        with pytest.raises(
            UnsupportedImportError,
            match="A particle is a virtual site, which is not yet supported.",
        ):
            from_openmm(
                system=system,
                topology=topology.to_openmm(),
            )

    def test_missing_positions_warning(self, monkeypatch, sage, water):
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        topology = water.to_topology()
        topology.box_vectors = Quantity([4, 4, 4], "nanometer")

        with pytest.warns(
            MissingPositionsWarning,
            match="are you sure",
        ):
            from_openmm(
                system=sage.create_openmm_system(topology),
                topology=topology.to_openmm(),
            )
