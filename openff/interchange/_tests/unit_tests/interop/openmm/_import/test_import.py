import copy
import random
from collections import defaultdict

import numpy
import pytest
from openff.toolkit import Molecule, Quantity, Topology, unit
from openff.utilities import get_data_file_path, has_package, skip_if_missing

from openff.interchange import Interchange
from openff.interchange._tests import needs_gmx
from openff.interchange.constants import kj_mol
from openff.interchange.drivers.openmm import get_openmm_energies
from openff.interchange.exceptions import UnsupportedImportError
from openff.interchange.interop.openmm._import._import import _convert_nonbonded_force

if has_package("openmm"):
    import openmm.app
    import openmm.unit


@skip_if_missing("openmm")
class TestFromOpenMM:
    def test_simple_roundtrip(self, monkeypatch, sage_unconstrained, ethanol):
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        ethanol.generate_conformers(n_conformers=1)

        interchange = Interchange.from_smirnoff(
            sage_unconstrained,
            [ethanol],
            box=Quantity([4, 4, 4], unit.nanometer),
        )

        system = interchange.to_openmm(combine_nonbonded_forces=True)

        converted = Interchange.from_openmm(
            system=system,
            topology=interchange.topology.to_openmm(),
            positions=interchange.positions,
            box_vectors=interchange.box,
        )

        get_openmm_energies(interchange).compare(
            get_openmm_energies(converted),
            tolerances={
                "Bond": 1e-6 * kj_mol,
                "Angle": 1e-6 * kj_mol,
                "Torsion": 1e-6 * kj_mol,
                "Nonbonded": 1e-3 * kj_mol,
            },
        )

        assert isinstance(converted.box.m, numpy.ndarray)

        # OpenMM seems to avoid using the built-in type
        assert converted.box.m.dtype in (float, numpy.float32, numpy.float64)

    @pytest.fixture
    def simple_system(self):
        return openmm.XmlSerializer.deserialize(
            open(
                get_data_file_path(
                    "system.xml",
                    "openff.interchange._tests.data",
                ),
            ).read(),
        )

    @pytest.mark.parametrize("as_argument", [False, True])
    def test_different_ways_to_process_box_vectors(
        self,
        monkeypatch,
        as_argument,
        simple_system,
    ):
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        topology = Molecule.from_smiles("C").to_topology()
        topology._molecule_virtual_site_map = defaultdict(list)
        topology._particle_map = {index: index for index in range(topology.n_atoms)}

        if as_argument:
            box = Interchange.from_openmm(
                system=simple_system,
                topology=topology,
                box_vectors=simple_system.getDefaultPeriodicBoxVectors(),
            ).box
        else:
            box = Interchange.from_openmm(
                system=simple_system,
                topology=topology,
            ).box

        assert box.shape == (3, 3)

        assert type(box.m[2][2]) in (float, numpy.float64, numpy.float32)
        assert type(box.m[1][1]) is not Quantity

    def test_topology_and_system_box_vectors_differ(
        self,
        monkeypatch,
        simple_system,
    ):
        """Ensure that, if box vectors specified in the topology and system differ, those in the topology are used."""
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        topology = Molecule.from_smiles("C").to_topology()
        topology.box_vectors = Quantity([4, 5, 6], unit.nanometer)

        box = Interchange.from_openmm(
            system=simple_system,
            topology=topology.to_openmm(),
        ).box

        assert numpy.diag(box.m_as(unit.nanometer)) == pytest.approx([4, 5, 6])

    def test_openmm_roundtrip_metadata(self, monkeypatch, sage):
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        # Make an example OpenMM Topology with metadata.
        # Here we use OFFTK to make the OpenMM Topology, but this could just as easily come from another source
        ethanol = Molecule.from_smiles("CCO")
        benzene = Molecule.from_smiles("c1ccccc1")

        for atom in ethanol.atoms:
            atom.metadata["chain_id"] = "1"
            atom.metadata["residue_number"] = "1"
            atom.metadata["insertion_code"] = ""
            atom.metadata["residue_name"] = "ETH"
        for atom in benzene.atoms:
            atom.metadata["chain_id"] = "1"
            atom.metadata["residue_number"] = "2"
            atom.metadata["insertion_code"] = "A"
            atom.metadata["residue_name"] = "BNZ"

        topology = Topology.from_molecules([ethanol, benzene])
        topology.box_vectors = Quantity([4, 4, 4], "nanometer")

        # Roundtrip the topology with metadata through openmm,
        # which requires a system even though it won't be used here
        # and requires PME for now
        interchange = Interchange.from_openmm(
            system=sage.create_openmm_system(topology),
            topology=topology.to_openmm(),
        )

        # Ensure that the metadata is the same
        for atom in interchange.topology.molecule(0).atoms:
            assert atom.metadata["chain_id"] == "1"
            assert atom.metadata["residue_number"] == "1"
            assert atom.metadata["insertion_code"] == ""
            assert atom.metadata["residue_name"] == "ETH"
        for atom in interchange.topology.molecule(1).atoms:
            assert atom.metadata["chain_id"] == "1"
            assert atom.metadata["residue_number"] == "2"
            assert atom.metadata["insertion_code"] == "A"
            assert atom.metadata["residue_name"] == "BNZ"

    @pytest.mark.slow
    def test_openmm_native_roundtrip_metadata(self, monkeypatch, sage):
        """
        Test that metadata is the same whether we load a PDB through OpenMM+Interchange vs. Topology.from_pdb.
        """
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        pdb = openmm.app.PDBFile(
            get_data_file_path(
                "ALA_GLY/ALA_GLY.pdb",
                "openff.interchange._tests.data",
            ),
        )

        topology = Topology.from_pdb(
            get_data_file_path("ALA_GLY/ALA_GLY.pdb", "openff.interchange._tests.data"),
        )

        interchange = Interchange.from_openmm(
            system=sage.create_openmm_system(topology),
            topology=pdb.topology,
        )

        for roundtrip_atom, off_atom in zip(interchange.topology.atoms, topology.atoms):
            # off_atom's metadata also includes a little info about how the chemistry was
            # assigned, so we remove this from the comparison
            off_atom_metadata = copy.deepcopy(off_atom.metadata)
            del off_atom_metadata["match_info"]
            assert roundtrip_atom.metadata == off_atom_metadata

    def test_electrostatics_cutoff_not_ignored(self, monkeypatch, ethanol):
        pytest.importorskip("openmmforcefields")

        import openmm.app
        import openmm.unit
        from openmmforcefields.generators import GAFFTemplateGenerator

        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        topology = ethanol.to_topology()
        topology.box_vectors = Quantity([4, 4, 4], "nanometer")

        gaff = GAFFTemplateGenerator(molecules=ethanol)
        force_field = openmm.app.ForceField()

        force_field.registerTemplateGenerator(gaff.generator)

        system = force_field.createSystem(
            topology=topology.to_openmm(),
            nonbondedMethod=openmm.app.PME,
            nonbondedCutoff=1.2345 * openmm.unit.nanometer,
        )

        interchange = Interchange.from_openmm(
            system=system,
            topology=topology.to_openmm(),
        )

        assert interchange["Electrostatics"].cutoff.m_as(
            unit.nanometer,
        ) == pytest.approx(1.2345)
        assert interchange["vdW"].cutoff.m_as(unit.nanometer) == pytest.approx(1.2345)

    @needs_gmx
    @pytest.mark.skip(reason="needs OpenMM -> Interchange -> GROMACS virtual sites implemented")
    def test_fill_in_rigid_water_parameters(self, water_dimer, monkeypatch):
        import openmm.app

        from openff.interchange.drivers import get_gromacs_energies

        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        openmm_force_field = openmm.app.ForceField("tip3p.xml")
        openmm_topology = water_dimer.to_openmm()

        imported = Interchange.from_openmm(
            system=openmm_force_field.createSystem(
                openmm_topology,
                constraints=openmm.app.HBonds,
                nonbondedMethod=openmm.app.PME,
            ),
            topology=openmm_topology,
            positions=water_dimer.get_positions().to_openmm(),
        )

        assert len(imported["Bonds"].key_map) == imported.topology.n_bonds == 4
        assert len(imported["Angles"].key_map) == imported.topology.n_angles == 2

        # Mostly just ensure GROMACS evaluation doesn't crash
        get_openmm_energies(imported, combine_nonbonded_forces=False).compare(
            get_gromacs_energies(imported),
            tolerances={
                "Electrostatics": Quantity(0.1, "kilojoule / mole"),
            },
        )


@skip_if_missing("openmm")
class TestProcessTopology:
    def test_with_openff_topology(self, monkeypatch, sage, basic_top):
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        system = sage.create_openmm_system(basic_top)

        basic_top._molecule_virtual_site_map = defaultdict(list)
        basic_top._particle_map = {index: index for index in range(basic_top.n_atoms)}

        with_openff = Interchange.from_openmm(
            system=system,
            topology=basic_top,
        )

        # positions are lost when making an openmm.app.Topology,
        # just feed them back in to make comparison easier
        with_openmm = Interchange.from_openmm(
            system=system,
            topology=basic_top.to_openmm(),
            positions=basic_top.get_positions().to_openmm(),
        )

        assert with_openff.topology.n_atoms == with_openmm.topology.n_atoms
        assert with_openff.topology.n_bonds == with_openmm.topology.n_bonds

        get_openmm_energies(
            with_openff,
            combine_nonbonded_forces=True,
        ).compare(
            get_openmm_energies(with_openmm),
            tolerances={
                "Angle": Quantity(0.001, "kilojoule / mole"),
                "Nonbonded": Quantity(0.001, "kilojoule / mole"),
            },
        )


@skip_if_missing("openmm")
class TestConvertNonbondedForce:
    def test_unsupported_method(self):
        # Cannot parametrize with a class in an optional module
        for method in (
            openmm.NonbondedForce.NoCutoff,
            openmm.NonbondedForce.CutoffNonPeriodic,
            openmm.NonbondedForce.CutoffPeriodic,
        ):
            force = openmm.NonbondedForce()
            force.setNonbondedMethod(method)

            with pytest.raises(UnsupportedImportError):
                _convert_nonbonded_force(force, dict())

    def test_parse_switching_distance(self):
        force = openmm.NonbondedForce()
        force.setNonbondedMethod(openmm.NonbondedForce.PME)

        cutoff = 1 + random.random() * 0.1
        switch_width = random.random() * 0.1

        force.setCutoffDistance(cutoff)
        force.setUseSwitchingFunction(True)
        force.setSwitchingDistance(cutoff - switch_width)

        vdw, _ = _convert_nonbonded_force(force=force, particle_map=dict())

        assert vdw.cutoff.m_as(unit.nanometer) == pytest.approx(cutoff)
        assert vdw.switch_width.m_as(unit.nanometer) == pytest.approx(switch_width)

    def test_parse_switching_distance_unused(self):
        force = openmm.NonbondedForce()
        force.setNonbondedMethod(openmm.NonbondedForce.PME)

        cutoff = 1 + random.random() * 0.1

        force.setCutoffDistance(cutoff)

        vdw, _ = _convert_nonbonded_force(force=force, particle_map=dict())

        assert vdw.cutoff.m_as(unit.nanometer) == pytest.approx(cutoff)
        assert vdw.switch_width.m_as(unit.nanometer) == 0.0


@skip_if_missing("openmm")
class TestConvertConstraints:
    def test_num_constraints(self, monkeypatch, sage, basic_top):
        """Test that the number of constraints is preserved when converting to and from OpenMM"""
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        interchange = sage.create_interchange(basic_top)

        converted = Interchange.from_openmm(
            topology=interchange.topology.to_openmm(),
            system=interchange.to_openmm(combine_nonbonded_forces=True),
        )

        assert "Constraints" in interchange.collections
        assert "Constraints" in converted.collections

        assert len(interchange["Constraints"].key_map) == len(
            converted["Constraints"].key_map,
        )
