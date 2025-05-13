import pytest
from openff.toolkit import Quantity, Topology

from openff.interchange import Interchange
from openff.interchange._tests import MoleculeWithConformer
from openff.interchange.components._packmol import solvate_topology
from openff.interchange.constants import kj_mol
from openff.interchange.drivers.openmm import _get_openmm_energies, _process, get_openmm_energies


class TestTIP4PVirtualSites:
    def test_tip4p_openmm_xml(self, water_dimer):
        """
        Prepare a TIP4P water dimer with OpenMM's style of 4-site water.

        Below is used as a guide
        https://openmm.github.io/openmm-cookbook/latest/notebooks/tutorials/Histone_methyltransferase_simulation_with_a_multisite_water_model_TIP4P-Ew.html
        """
        pytest.importorskip("openmm")

        import openmm.app

        modeller = openmm.app.Modeller(
            topology=water_dimer.to_openmm(),
            positions=water_dimer.get_positions().to("nanometer").to_openmm(),
        )

        forcefield = openmm.app.ForceField("tip4pew.xml")

        modeller.addExtraParticles(forcefield=forcefield)

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=openmm.app.PME,
            nonbondedCutoff=1.0 * openmm.unit.nanometers,
            constraints=openmm.app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0001,  # match Interchange.to_openmm_system default
        )

        imported = Interchange.from_openmm(
            topology=modeller.topology,
            system=system,
            positions=modeller.getPositions(),
            box_vectors=modeller.getTopology().getPeriodicBoxVectors(),
        )

        get_openmm_energies(
            imported,
            combine_nonbonded_forces=True,
        ).compare(
            _process(
                raw_energies=_get_openmm_energies(
                    system=system,
                    positions=modeller.getPositions(),
                    box_vectors=modeller.getTopology().getPeriodicBoxVectors(),
                    round_positions=None,
                    platform="Reference",
                ),
                combine_nonbonded_forces=True,
                detailed=False,
                system=system,
            ),
            tolerances={
                "Nonbonded": 1e-5 * kj_mol,
            },
        )

    @pytest.mark.skip(
        reason="Rewrite to use OpenMM or update from_openmm to support `LocalCoordinatesSite`s",
    )
    def test_dimer_energy_equals(self, tip4p, water_dimer):
        out: Interchange = tip4p.create_interchange(water_dimer)

        roundtripped = Interchange.from_openmm(
            system=out.to_openmm_system(),
            topology=out.to_openmm_topology(collate=False),
            positions=out.get_positions(include_virtual_sites=True).to_openmm(),
            box_vectors=out.box.to_openmm(),
        )

        assert get_openmm_energies(out) == _get_openmm_energies(roundtripped)

    @pytest.mark.skip(
        reason="Rewrite to use OpenMM or update from_openmm to support `LocalCoordinatesSite`s",
    )
    def test_minimize_solvated_ligand(self, sage_with_tip4p, default_integrator):
        topology = solvate_topology(
            topology=MoleculeWithConformer.from_smiles("CO").to_topology(),
            nacl_conc=Quantity(1.0, "mole / liter"),
            target_density=Quantity(0.6, "gram / milliliter"),
            working_directory=".",
        )

        simulation = sage_with_tip4p.create_interchange(topology).to_openmm_simulation(integrator=default_integrator)

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
            match="Particle ordering mismatch",
        ):
            Interchange.from_openmm(
                system=out.to_openmm_system(),
                topology=out.to_openmm_topology(collate=True),
            )
