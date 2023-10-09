import numpy
import pytest
from openff.toolkit import ForceField, Molecule
from openff.units import unit
from openff.utilities import has_package, requires_package, skip_if_missing

from openff.interchange import Interchange
from openff.interchange._tests import _BaseTest, get_test_file_path
from openff.interchange.interop.gromacs._import._import import from_files

if has_package("openmm"):
    import openmm.app
    import openmm.unit


class TestToGro(_BaseTest):
    def test_residue_names(self, sage):
        """Reproduce issue #642."""
        # This could maybe just test the behavior of _convert?
        ligand = Molecule.from_smiles("CCO")
        ligand.generate_conformers(n_conformers=1)

        for atom in ligand.atoms:
            atom.metadata["residue_name"] = "LIG"

        Interchange.from_smirnoff(
            sage,
            [ligand],
        ).to_gro("should_have_residue_names.gro")

        for line in open("should_have_residue_names.gro").readlines()[2:-2]:
            assert line[5:10] == "LIG  "

    @skip_if_missing("openmm")
    def test_tip4p_dimer(self, tip4p, water_dimer):
        tip4p.create_interchange(water_dimer).to_gro("_dimer.gro")

        positions = openmm.app.GromacsGroFile(
            "_dimer.gro",
        ).getPositions(asNumpy=True)

        assert positions.shape == (8, 3)

        assert not numpy.allclose(positions[3], positions[7])


class TestSettles(_BaseTest):
    def test_settles_units(self, monkeypatch, water):
        """Reproduce issue #720."""
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        water.name = "WAT"

        ForceField("openff-2.1.0.offxml").create_interchange(
            water.to_topology(),
        ).to_gromacs(
            prefix="settles",
        )

        system = from_files("settles.top", "settles.gro")

        settles = system.molecule_types["WAT"].settles[0]

        assert settles.oxygen_hydrogen_distance.m_as(unit.angstrom) == pytest.approx(
            0.981124525388,
        )
        assert settles.hydrogen_hydrogen_distance.m_as(unit.angstrom) == pytest.approx(
            1.513900654525,
        )


@pytest.mark.slow()
@requires_package("openmm")
class TestCommonBoxes(_BaseTest):
    @pytest.mark.parametrize(
        "pdb_file",
        [
            get_test_file_path("cube.pdb").as_posix(),
            get_test_file_path("dodecahedron.pdb").as_posix(),
            get_test_file_path("octahedron.pdb").as_posix(),
        ],
    )
    def test_common_boxes(self, pdb_file):
        original_box_vectors = openmm.app.PDBFile(
            pdb_file,
        ).topology.getPeriodicBoxVectors()

        from openff.toolkit import Topology

        # TODO: Regenerate test files to be simpler and smaller, no need to use a protein
        force_field = ForceField(
            "openff-2.1.0.offxml",
            "ff14sb_off_impropers_0.0.3.offxml",
        )

        topology = Topology.from_pdb(pdb_file)

        force_field.create_interchange(topology).to_gro(pdb_file.replace("pdb", "gro"))

        parsed_box_vectors = openmm.app.GromacsGroFile(
            pdb_file.replace("pdb", "gro"),
        ).getPeriodicBoxVectors()

        numpy.testing.assert_allclose(
            numpy.array(original_box_vectors.value_in_unit(openmm.unit.nanometer)),
            numpy.array(parsed_box_vectors.value_in_unit(openmm.unit.nanometer)),
        )


class TestVirtualSites:
    @skip_if_missing("openmm")
    def test_tip4p_charge_neutrality(self, tip4p, water_dimer):
        tip4p.create_interchange(water_dimer).to_top("_dimer.top")

        system = openmm.app.GromacsTopFile("_dimer.top").createSystem()

        assert system.getForce(0).getNumParticles() == 8

        charges = [
            system.getForce(0).getParticleParameters(i)[0]._value for i in range(4)
        ]

        assert sum(charges) == pytest.approx(0.0)
