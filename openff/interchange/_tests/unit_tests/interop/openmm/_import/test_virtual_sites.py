import pytest
from openff.toolkit import Topology

from openff.interchange import Interchange
from openff.interchange.components._packmol import solvate_topology
from openff.interchange.drivers.openmm import _get_openmm_energies, get_openmm_energies


class TestTIP4PVirtualSites:
    def test_dimer_energy_equals(self, tip4p, water_dimer):
        out: Interchange = tip4p.create_interchange(water_dimer)

        roundtripped = Interchange.from_openmm(
            system=out.to_openmm_system(),
            topology=out.to_openmm_topology(collate=False),
            positions=out.get_positions(include_virtual_sites=True).to_openmm(),
            box_vectors=out.box.to_openmm(),
        )

        assert get_openmm_energies(out) == _get_openmm_energies(roundtripped)

    def test_minimize_solvated_ligand(self, sage_with_tip4p, ethanol, default_integrator):
        topology = solvate_topology(ethanol.to_topology())

        simulation = sage_with_tip4p.create_simulation(
            topology,
        ).to_openmm_simulation(
            integrator=default_integrator,
        )

        roundtripped = Interchange.from_openmm(
            system=simulation.system,
            topology=simulation.topology,
            positions=simulation.context.getState(getPositions=True).getPositions(),
            box_vectors=simulation.system.getDefaultPeriodicBoxVectors(),
        )

        original_energy = get_openmm_energies(simulation)

        # TODO: Much more validation could be done here, but if a simulation
        #       can start and minimize at all, that should catch most problems
        roundtripped.minimize()

        assert get_openmm_energies(roundtripped) < original_energy

    def test_error_index_mismatch(self, tip4p, water):
        out: Interchange = tip4p.create_interchange(Topology.from_molecules([water, water]))

        with pytest.raises(
            ValueError,  # TODO: Make a different error
            match="The number of particles in the system and the number of atoms in the topology do not match.",
        ):
            Interchange.from_openmm(
                system=out.to_openmm_system(),
                topology=out.to_openmm_topology(collate=True),
            )
