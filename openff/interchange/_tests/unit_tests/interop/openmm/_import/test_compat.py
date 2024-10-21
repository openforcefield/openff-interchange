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
                "The number of particles in the system (9) and "
                "the number of atoms in the topology (3) do not match.",
            ),
        ):
            from_openmm(
                system=system,
                topology=other_topology.to_openmm(),
            )

    def test_found_out_of_plane_virtual_site(self, tip5p, water):
        topology = water.to_topology()
        topology.box_vectors = Quantity([4, 4, 4], "nanometer")

        system = tip5p.create_openmm_system(topology)

        with pytest.raises(
            UnsupportedImportError,
            match="A particle is an `outOfPlane` virtual site, which is not yet supported.",
        ):
            from_openmm(
                system=system,
                topology=topology.to_openmm(),
            )

    def test_found_two_particle_average_virtual_site(
        self,
        sage_with_bond_charge,
        default_integrator,
    ):
        simulation = sage_with_bond_charge.create_interchange(
            Molecule.from_smiles("CCl").to_topology(),
        ).to_openmm_simulation(integrator=default_integrator)

        with pytest.raises(
            UnsupportedImportError,
            match="A particle is a `TwoParticleAverage` virtual site, which is not yet supported.",
        ):
            from_openmm(
                system=simulation.system,
                topology=simulation.topology,
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
