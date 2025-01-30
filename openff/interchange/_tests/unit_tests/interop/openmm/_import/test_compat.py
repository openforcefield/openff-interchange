"""Test compatibility checks in OpenMM import."""

import re

import pytest
from openff.toolkit import Molecule, Quantity

from openff.interchange.exceptions import UnsupportedImportError
from openff.interchange.interop.openmm._import import from_openmm
from openff.interchange.warnings import MissingPositionsWarning


class TestUnsupportedCases:
    @pytest.mark.filterwarnings("ignore:.*are you sure you don't want to pass positions")
    def test_error_topology_mismatch(self, sage_unconstrained, ethanol):
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
                "The number of particles in the system (9) and the number of atoms in the topology (3) do not match.",
            ),
        ):
            from_openmm(
                system=system,
                topology=other_topology.to_openmm(),
            )

    def test_found_virtual_sites(self, tip4p, water):
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

    def test_missing_positions_warning(self, sage, water):
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
