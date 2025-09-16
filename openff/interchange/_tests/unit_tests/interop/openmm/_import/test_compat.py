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

    def test_found_out_of_plane_virtual_site(self, water_dimer):
        pytest.importorskip("openmm")

        import openmm.app

        modeller = openmm.app.Modeller(
            topology=water_dimer.to_openmm(),
            positions=water_dimer.get_positions().to("nanometer").to_openmm(),
        )

        forcefield = openmm.app.ForceField("tip5p.xml")

        modeller.addExtraParticles(forcefield=forcefield)

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=openmm.app.PME,
            nonbondedCutoff=1.0 * openmm.unit.nanometers,
            constraints=openmm.app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005,
        )

        with pytest.raises(
            UnsupportedImportError,
            match=r"A particle is a virtual site of type.*OutOfPlane.*which is not yet supported.",
        ):
            from_openmm(
                system=system,
                topology=modeller.topology,
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
