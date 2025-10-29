import random
from importlib import resources
from math import exp

import numpy
import pytest
from openff.toolkit import ForceField, Molecule, Quantity, Topology, unit
from openff.toolkit.typing.engines.smirnoff import VirtualSiteHandler
from openff.units.openmm import ensure_quantity
from openff.utilities import (
    get_data_file_path,
    has_package,
    requires_package,
    skip_if_missing,
)

from openff.interchange import Interchange
from openff.interchange._tests import (
    MoleculeWithConformer,
    get_protein,
    get_test_file_path,
    needs_gmx,
)
from openff.interchange.components.nonbonded import BuckinghamvdWCollection
from openff.interchange.components.potentials import Potential
from openff.interchange.drivers import get_gromacs_energies, get_openmm_energies
from openff.interchange.exceptions import GMXMdrunError, UnsupportedExportError
from openff.interchange.interop.gromacs._import._import import (
    _read_box,
    _read_coordinates,
    from_files,
)
from openff.interchange.models import PotentialKey, TopologyKey

if has_package("openmm"):
    import openmm.app
    import openmm.unit


@needs_gmx
class _NeedsGROMACS:
    pass


def parse_residue_ids(file) -> list[str]:
    """Parse residue IDs, and only the residue IDs, from a GROMACS .gro file."""
    with open(file) as f:
        ids = [line[:5].strip() for line in f.readlines()[2:-1]]

    return ids


class TestToGro(_NeedsGROMACS):
    @pytest.mark.xfail(reason="Broken")
    def test_residue_names(self, sage):
        """Reproduce issue #642."""
        # This could maybe just test the behavior of _convert?
        ligand = Molecule.from_smiles("CCO")
        ligand.generate_conformers(n_conformers=1)

        topology = ligand.to_topology()
        topology.box_vectors = Quantity([4, 4, 4], "nanometer")

        for atom in ligand.atoms:
            atom.metadata["residue_name"] = "LIG"

        sage.create_interchange(topology).to_gro("should_have_residue_names.gro")

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


@skip_if_missing("mdtraj")
@skip_if_missing("openmm")
class TestGROMACSGROFile(_NeedsGROMACS):
    try:
        _INTERMOL_PATH = resources.files(
            "intermol.tests.gromacs.unit_tests",
        )
    except ModuleNotFoundError:
        _INTERMOL_PATH = None

    @skip_if_missing("intermol")
    def test_load_gro(self):
        file = self._INTERMOL_PATH / "angle10_vacuum/angle10_vacuum.gro"

        positions = _read_coordinates(file)
        box = _read_box(file)

        openmm_gro = openmm.app.GromacsGroFile(file)
        openmm_positions = ensure_quantity(
            openmm_gro.getPositions(),
            "openff",
        )
        openmm_box = ensure_quantity(
            openmm_gro.getPeriodicBoxVectors(),
            "openff",
        )

        assert numpy.allclose(positions, openmm_positions)
        assert numpy.allclose(box, openmm_box)

    @skip_if_missing("intermol")
    @pytest.mark.skip("don't run parmed tests")
    def test_load_gro_nonstandard_precision(self):
        pytest.importorskip("intermol")

        file = self._INTERMOL_PATH / "lj3_bulk/lj3_bulk.gro"

        coords = _read_coordinates(file).m_as(unit.nanometer)

        # OpenMM seems to assume a precision of 3. Use InterMol instead here.
        from intermol.gromacs.grofile_parser import GromacsGroParser

        intermol_gro = GromacsGroParser(file)
        intermol_gro.read()

        # InterMol stores positions as NumPy arrays _of openmm.unit.Quantity_
        # objects, so need to carefully convert everything through unitless
        # floats; converting through Pint-like quantities might produce a
        # funky double-wrapped thing
        def converter(x):
            return x.value_in_unit(openmm.unit.nanometer)

        other_coords = numpy.frompyfunc(converter, 1, 1)(intermol_gro.positions).astype(
            float,
        )

        assert numpy.allclose(coords, other_coords)

        # This file happens to have 12 digits of preicion; what really matters is that
        # the convential precision of 3 was not used.
        n_decimals = len(str(coords[0, 0]).split(".")[1])
        assert n_decimals == 12

    def test_vaccum_unsupported(self, sage):
        molecule = MoleculeWithConformer.from_smiles("CCO")

        out = Interchange.from_smirnoff(force_field=sage, topology=[molecule])

        assert out.box is None

        with pytest.raises(UnsupportedExportError, match="2020"):
            out.to_gro("tmp.gro")

    @pytest.mark.slow
    @pytest.mark.parametrize("offset_residue_ids", [True, False])
    @skip_if_missing("openmm")
    def test_residue_info(self, offset_residue_ids):
        """Test that residue information is passed through to .gro files."""
        import mdtraj

        protein = get_protein("MainChain_HIE")

        if offset_residue_ids:
            offset = random.randint(10, 20)

            for atom in protein.atoms:
                atom.metadata["residue_number"] = str(
                    int(atom.metadata["residue_number"]) + offset,
                )

            # Need to manually update residues _and_ atoms
            # https://github.com/openforcefield/openff-toolkit/issues/1921
            for residue in protein.residues:
                residue.residue_number = str(int(residue.residue_number) + offset)

        ff14sb = ForceField("ff14sb_off_impropers_0.0.3.offxml")

        out = Interchange.from_smirnoff(
            force_field=ff14sb,
            topology=[protein],
        )

        out.box = [4, 4, 4]

        out.to_gro("tmp.gro")

        mdtraj_topology = mdtraj.load("tmp.gro").topology

        for found_residue, original_residue in zip(
            mdtraj_topology.residues,
            out.topology.hierarchy_iterator("residues"),
        ):
            assert found_residue.name == original_residue.residue_name
            found_index = [*original_residue.atoms][0].metadata["residue_number"]
            assert str(found_residue.resSeq) == found_index

    def test_fill_in_residue_ids(self, sage):
        """Ensure that, if inputs have no residue_number, they are generated on-the-fly."""
        topology = Topology.from_molecules(
            [MoleculeWithConformer.from_smiles(smi) for smi in ["CC", "O", "C"]],
        )
        topology.box_vectors = Quantity([4, 4, 4], "nanometer")

        sage.create_interchange(topology).to_gro("fill.gro")

        expected_residue_ids = 8 * ["1"] + 3 * ["2"] + 5 * ["3"]

        found_residue_ids = parse_residue_ids("fill.gro")

        for expected, found in zip(expected_residue_ids, found_residue_ids):
            assert expected == found

    def test_atom_index_gt_100_000(self, water, sage):
        """Ensure that atom indices are written correctly for large numbers."""
        water.add_hierarchy_scheme(
            ("residue_name", "residue_number"),
            "residues",
        )

        topology = Topology.from_molecules(2 * [water])

        topology.box_vectors = unit.Quantity([4, 4, 4], unit.nanometer)

        # Can't just update atoms' metadata, neeed to create these scheme/element objects
        # and need to also modify the residue objects themselves
        for molecule_index, molecule in enumerate(topology.molecules):
            for atom in molecule.atoms:
                atom.metadata["residue_number"] = str(molecule_index + 123_456)

        for residue_index, residue in enumerate(topology.residues):
            residue.residue_number = str(residue_index + 123_456)

        interchange = sage.create_interchange(topology)

        interchange.to_gro("large.gro")

        expected_residue_ids = 3 * ["23456"] + 3 * ["23457"]

        found_residue_ids = parse_residue_ids("large.gro")

        for expected, found in zip(expected_residue_ids, found_residue_ids):
            assert expected == found

    @pytest.mark.slow
    def test_atom_names_pdb(self):
        topology = get_protein("MainChain_ALA_ALA").to_topology()
        topology.box_vectors = Quantity([4, 4, 4], "nanometer")

        ff14sb = ForceField("ff14sb_off_impropers_0.0.3.offxml")

        Interchange.from_smirnoff(ff14sb, topology).to_gro(
            "atom_names.gro",
        )

        pdb_object = openmm.app.PDBFile(
            get_data_file_path("proteins/MainChain_ALA_ALA.pdb", "openff.toolkit"),
        )
        pdb_atom_names = [atom.name for atom in pdb_object.topology.atoms()]

        openmm_atom_names = openmm.app.GromacsGroFile("atom_names.gro").atomNames

        assert openmm_atom_names == pdb_atom_names


class TestGROMACS(_NeedsGROMACS):
    @pytest.mark.parametrize(
        "monolithic",
        [
            True,
            False,
        ],
    )
    @pytest.mark.parametrize(
        "smiles",
        [
            "C",
            "O=C=O",  # Adds unconstrained bonds without torsion(s)
            "CC",  # Adds a proper torsion term(s)
            # "C=O",  # Simplest molecule with any improper torsion
            "OC=O",  # Simplest molecule with a multi-term torsion
            # "CCOC",  # This hits t86, which has a non-1.0 idivf
            # "C1COC(=O)O1",  # This adds an improper, i2
        ],
    )
    def test_simple_roundtrip(
        self,
        sage_unconstrained,
        smiles,
        monolithic,
        monkeypatch,
    ):
        # Skip if using RDKit conformer, which does weird stuff around 180 deg
        pytest.importorskip("openeye.oechem")

        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        molecule = MoleculeWithConformer.from_smiles(smiles)
        molecule.name = molecule.to_hill_formula()
        topology = molecule.to_topology()

        out = sage_unconstrained.create_interchange(topology)

        out.box = [4, 4, 4]
        out.positions = numpy.round(molecule.conformers[0], 2)

        out.to_gromacs("out", decimal=3, monolithic=monolithic)

        converted = Interchange.from_gromacs("out.top", "out.gro")

        assert numpy.allclose(out.positions, converted.positions, atol=1e-3)
        assert numpy.allclose(out.box, converted.box)

        get_gromacs_energies(out).compare(
            get_gromacs_energies(converted),
            tolerances={
                "Bond": 0.002 * molecule.n_bonds * unit.kilojoule / unit.mol,
                "Electrostatics": 0.05 * unit.kilojoule / unit.mol,
            },
        )

    def test_num_impropers(self, sage):
        parmed = pytest.importorskip("parmed")

        out = Interchange.from_smirnoff(
            sage,
            MoleculeWithConformer.from_smiles("CC1=CC=CC=C1").to_topology(),
        )

        out.box = unit.Quantity(4 * numpy.eye(3), units=unit.nanometer)
        out.to_top("tmp.top")

        # Sanity check; toluene should have some improper(s)
        assert len(out["ImproperTorsions"].key_map) > 0

        struct = parmed.load_file("tmp.top")
        n_impropers_parmed = len([d for d in struct.dihedrals if d.improper])
        assert n_impropers_parmed == len(out["ImproperTorsions"].key_map)

    @pytest.mark.slow
    @skip_if_missing("intermol")
    @pytest.mark.skip(reason="Re-implement when SMIRNOFF supports more mixing rules")
    def test_set_mixing_rule(self, ethanol_top, sage):
        from intermol.gromacs.gromacs_parser import GromacsParser

        interchange = Interchange.from_smirnoff(force_field=sage, topology=ethanol_top)
        interchange.positions = numpy.zeros((ethanol_top.n_atoms, 3))
        interchange.to_gro("tmp.gro")

        interchange.box = [4, 4, 4]
        interchange.to_top("lorentz.top")
        lorentz = GromacsParser("lorentz.top", "tmp.gro").read()
        assert lorentz.combination_rule == "Lorentz-Berthelot"

        interchange["vdW"].mixing_rule = "geometric"

        interchange.to_top("geometric.top")
        geometric = GromacsParser("geometric.top", "tmp.gro").read()
        assert geometric.combination_rule == "Multiply-Sigeps"

    @pytest.mark.skip(reason="Re-implement when SMIRNOFF supports more mixing rules")
    def test_unsupported_mixing_rule(self, ethanol_top, sage):
        # TODO: Update this test when the model supports more mixing rules than GROMACS does
        interchange = Interchange.from_smirnoff(force_field=sage, topology=ethanol_top)
        interchange["vdW"].mixing_rule = "kong"

        with pytest.raises(UnsupportedExportError, match="rule `geometric` not compat"):
            interchange.to_top("out.top")

    @pytest.mark.slow
    @skip_if_missing("openmm")
    def test_residue_info(self, sage):
        """Test that residue information is passed through to .top files."""
        from openff.units.openmm import from_openmm

        parmed = pytest.importorskip("parmed")

        protein = get_protein("MainChain_HIE")

        box_vectors = from_openmm(
            openmm.app.PDBFile(
                get_data_file_path(
                    "proteins/MainChain_HIE.pdb",
                    "openff.toolkit",
                ),
            ).topology.getPeriodicBoxVectors(),
        )

        ff14sb = ForceField("ff14sb_off_impropers_0.0.3.offxml")

        out = Interchange.from_smirnoff(
            force_field=ff14sb,
            topology=[protein],
            box=box_vectors,
        )

        out.to_top("tmp.top")

        parmed_structure = parmed.load_file("tmp.top")

        for found_residue, original_residue in zip(
            parmed_structure.residues,
            out.topology.hierarchy_iterator("residues"),
        ):
            assert found_residue.name == original_residue.residue_name
            assert str(found_residue.number + 1) == original_residue.residue_number

    @pytest.mark.slow
    @pytest.mark.skip(
        reason="Update when energy reports intentionally support non-vdW handlers",
    )
    def test_argon_buck(self):
        """Test that Buckingham potentials are supported and can be exported"""
        from openff.interchange.smirnoff import SMIRNOFFElectrostaticsCollection

        mol = MoleculeWithConformer.from_smiles("[#18]", name="Argon")

        top = Topology.from_molecules([mol, mol])

        # http://www.sklogwiki.org/SklogWiki/index.php/Argon#Buckingham_potential
        erg_mol = unit.erg / unit.mol * float(unit.avogadro_number)
        A = 1.69e-8 * erg_mol
        B = 1 / (0.273 * unit.angstrom)
        C = 102e-12 * erg_mol * unit.angstrom**6

        r = 0.3 * unit.nanometer

        buck = BuckinghamvdWCollection()
        coul = SMIRNOFFElectrostaticsCollection(method="pme")

        pot_key = PotentialKey(id="[#18]")
        pot = Potential(parameters={"A": A, "B": B, "C": C})

        for atom in top.atoms:
            top_key = TopologyKey(atom_indices=(top.atom_index(atom),))
            buck.key_map.update({top_key: pot_key})

            coul.key_map.update({top_key: pot_key})
            coul.potentials.update(
                {pot_key: Potential(parameters={"charge": 0 * unit.elementary_charge})},
            )

        for molecule in top.molecules:
            molecule.partial_charges = unit.Quantity(
                molecule.n_atoms * [0],
                unit.elementary_charge,
            )

        buck.potentials[pot_key] = pot

        out = Interchange()
        out.collections["Buckingham-6"] = buck
        out.collections["Electrostatics"] = coul
        out.topology = top
        out.box = [10, 10, 10] * unit.nanometer
        out.positions = [[0, 0, 0], [0.3, 0, 0]] * unit.nanometer
        out.to_gro("out.gro")
        out.to_top("out.top")

        omm_energies = get_openmm_energies(out, combine_nonbonded_forces=True)
        by_hand = A * exp(-B * r) - C * r**-6

        resid = omm_energies.energies["vdW"] - by_hand
        assert resid < 1e-5 * unit.kilojoule / unit.mol

        # TODO: Add back comparison to GROMACS energies once GROMACS 2020+
        # supports Buckingham potentials
        with pytest.raises(GMXMdrunError):
            get_gromacs_energies(out, mdp="cutoff_buck")

    @pytest.mark.skip("Broken, unclear if cases like these are worth supporting")
    def test_nonconsecutive_isomorphic_molecules(self, sage_unconstrained):
        molecules = [Molecule.from_smiles(smiles) for smiles in ["CC", "CCO", "CC"]]

        for index, molecule in enumerate(molecules):
            molecule.generate_conformers(n_conformers=1)
            molecule.conformers[0] += unit.Quantity(3 * [5 * index], unit.angstrom)

        topology = Topology.from_molecules(molecules)
        topology.box_vectors = unit.Quantity([4, 4, 4], unit.nanometer)

        out = Interchange.from_smirnoff(sage_unconstrained, topology)

        get_gromacs_energies(out).compare(
            get_openmm_energies(out, combine_nonbonded_forces=True),
            {"Nonbonded": 0.5 * unit.kilojoule_per_mole},
        )

    @pytest.mark.parametrize("name", ["MOL0", "MOL999", ""])
    def test_exisiting_mol0_names_overwritten(self, name, sage, ethanol, cyclohexane):
        parmed = pytest.importorskip("parmed")

        ethanol.name = name
        cyclohexane.name = name

        sage.create_interchange(
            Topology.from_molecules([ethanol, cyclohexane]),
        ).to_top("tmp.top")

        assert [*parmed.load_file("tmp.top").molecules.keys()] == ["MOL0", "MOL1"]

    @pytest.mark.filterwarnings("ignore:Setting positions to None")
    @pytest.mark.parametrize("name", ["MOL0", "MOL222", ""])
    def test_roundtrip_with_combine(
        self,
        name,
        sage_unconstrained,
        ethanol,
        cyclohexane,
        monkeypatch,
    ):
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        for molecule in [ethanol, cyclohexane]:
            molecule.name = name
            molecule.generate_conformers(n_conformers=1)

        # translate one by 1 nm
        cyclohexane._conformers = [
            cyclohexane.conformers[0] + Quantity([0, 0, 1], "nanometer"),
        ]

        ethanol_interchange = sage_unconstrained.create_interchange(
            ethanol.to_topology(),
        )
        ethanol_interchange.box = [4, 4, 4]

        cyclohexane_interchange = sage_unconstrained.create_interchange(
            cyclohexane.to_topology(),
        )
        cyclohexane_interchange.box = [4, 4, 4]

        combined = ethanol_interchange.combine(cyclohexane_interchange)

        ethanol_interchange.to_gromacs("ethanol", decimal=8)
        cyclohexane_interchange.to_gromacs("cyclohexane", decimal=8)
        combined.to_gromacs("combined", decimal=8)

        converted_ethanol = Interchange.from_gromacs(
            "ethanol.top",
            "ethanol.gro",
        )
        converted_cyclohexane = Interchange.from_gromacs(
            "cyclohexane.top",
            "cyclohexane.gro",
        )

        converted_combined = Interchange.from_gromacs(
            "combined.top",
            "combined.gro",
        )

        combined_from_converted = converted_ethanol.combine(converted_cyclohexane)
        combined_from_converted["vdW"].cutoff = combined["vdW"].cutoff
        converted_combined["vdW"].cutoff = combined["vdW"].cutoff

        assert numpy.allclose(converted_combined.positions, combined.positions)
        assert numpy.allclose(
            converted_combined.positions,
            combined_from_converted.positions,
        )
        assert numpy.allclose(converted_combined.box, combined.box)
        assert numpy.allclose(converted_combined.box, combined_from_converted.box)

        get_gromacs_energies(combined).compare(
            get_gromacs_energies(converted_combined),
            tolerances={
                "Electrostatics": 0.05 * unit.kilojoule / unit.mol,
            },
        )
        get_gromacs_energies(combined).compare(
            get_gromacs_energies(combined_from_converted),
            tolerances={
                "Electrostatics": 0.05 * unit.kilojoule / unit.mol,
            },
        )


class TestGROMACSMetadata(_NeedsGROMACS):
    @skip_if_missing("openmm")
    @skip_if_missing("mdtraj")
    @pytest.mark.slow
    def test_atom_names_pdb(self):
        topology = get_protein("MainChain_ALA_ALA").to_topology()
        topology.box_vectors = Quantity([4, 4, 4], "nanometer")

        ff14sb = ForceField("ff14sb_off_impropers_0.0.3.offxml")

        ff14sb.create_interchange(topology).to_gromacs("atom_names")

        pdb_object = openmm.app.PDBFile(
            get_data_file_path(
                "proteins/MainChain_ALA_ALA.pdb",
                "openff.toolkit",
            ),
        )
        openmm_object = openmm.app.GromacsTopFile("atom_names.top")

        pdb_atom_names = [atom.name for atom in pdb_object.topology.atoms()]

        openmm_atom_names = [atom.name for atom in openmm_object.topology.atoms()]

        assert openmm_atom_names == pdb_atom_names


class TestSettles(_NeedsGROMACS):
    def test_settles_units(self, monkeypatch, water):
        """Reproduce issue #720."""
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        water.name = "WAT"

        interchange = ForceField("openff-2.1.0.offxml").create_interchange(
            water.to_topology(),
        )
        interchange.box = [10, 10, 10]
        interchange.to_gromacs(
            prefix="settles",
        )

        system = from_files("settles.top", "settles.gro")

        settles = system.molecule_types["WAT"].settles[0]

        assert settles.oxygen_hydrogen_distance.m_as(unit.angstrom) == pytest.approx(
            0.9572,
        )
        assert settles.hydrogen_hydrogen_distance.m_as(unit.angstrom) == pytest.approx(
            1.513900654525,
        )


@pytest.mark.slow
@requires_package("openmm")
class TestCommonBoxes(_NeedsGROMACS):
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


class TestMergeAtomTypes(_NeedsGROMACS):
    @pytest.mark.parametrize(
        "smiles",
        [
            "CC",  # Two identical carbons
            "C1CCCCC1",  # Identical carbons in the ring
            "C1[C@@H](N)C[C@@H](N)CC1",  # Identical carbons and nitrogens in the ring
        ],
    )
    def test_energies_with_merging_atom_types(self, sage, smiles):
        """
        Tests #962
        """
        molecule = MoleculeWithConformer.from_smiles(smiles)
        molecule.name = molecule.to_hill_formula()
        topology = molecule.to_topology()

        out = Interchange.from_smirnoff(force_field=sage, topology=topology)
        out.box = [4, 4, 4]
        out.positions = molecule.conformers[0]

        get_gromacs_energies(out).compare(
            get_gromacs_energies(out, _merge_atom_types=True),
            tolerances={
                "Bond": 0.002 * molecule.n_bonds * unit.kilojoule / unit.mol,
                "Electrostatics": 0.05 * unit.kilojoule / unit.mol,
            },
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "smiles",
        [
            "CC",  # Two identical carbons
            "C1CCCCC1",  # Identical carbons in the ring
            "C1[C@@H](N)C[C@@H](N)CC1",  # Identical carbons and nitrogens in the ring
        ],
    )
    def test_simple_roundtrip_with_merging_atom_types(
        self,
        sage_unconstrained,
        smiles,
        monkeypatch,
    ):
        """
        Tests #962
        """
        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        molecule = MoleculeWithConformer.from_smiles(smiles)
        molecule.name = molecule.to_hill_formula()
        topology = molecule.to_topology()

        out = sage_unconstrained.create_interchange(topology)
        out.box = [4, 4, 4]
        out.positions = molecule.conformers[0]

        out.to_top("out.top")
        out.to_top("out_merged.top", _merge_atom_types=True)
        out.to_gro("out.gro", decimal=3)

        converted = Interchange.from_gromacs("out.top", "out.gro")
        converted_merged = Interchange.from_gromacs(
            "out_merged.top",
            "out.gro",
        )

        assert numpy.allclose(converted.positions, converted_merged.positions)
        assert numpy.allclose(converted.box, converted_merged.box)

        get_gromacs_energies(converted_merged).compare(
            get_gromacs_energies(converted),
            tolerances={
                "Bond": 0.002 * molecule.n_bonds * unit.kilojoule / unit.mol,
                "Electrostatics": 0.05 * unit.kilojoule / unit.mol,
            },
        )

    @pytest.mark.parametrize(
        "molecule_list",
        [
            ["CC", "CCO"],
            ["CC", "CCCC"],
            ["c1ccccc1", "CC", "CCO", "O"],
            ["c1ccncc1", "n1cnccc1", "[nH]1cccc1"],
        ],
    )
    def test_merge_atom_types_of_similar_molecules(
        self,
        molecule_list,
        sage,
    ):
        parmed = pytest.importorskip("parmed")

        topology = Topology.from_molecules(
            [Molecule.from_smiles(smi) for smi in molecule_list],
        )

        out = sage.create_interchange(topology)

        for merge in [True, False]:
            out.to_top(f"{merge}.top", _merge_atom_types=merge)

        not_merged = parmed.load_file("False.top")
        merged = parmed.load_file("True.top")

        # These are (n_atoms x 1) lists of parameters, which should be
        # read from [ atoms ] section and cross-referenced to [ atomtypes ]
        for attr in ["sigma", "epsilon", "charge"]:
            assert [getattr(atom, attr) for atom in not_merged.atoms] == [getattr(atom, attr) for atom in merged.atoms]


class TestGROMACSVirtualSites(_NeedsGROMACS):
    @pytest.fixture
    def sigma_hole_type(self, sage):
        """A handler with a bond charge virtual site on a C-Cl bond."""
        return VirtualSiteHandler.VirtualSiteBondChargeType(
            name="EP",
            smirks="[#6:1]-[#17:2]",
            distance=1.4 * unit.angstrom,
            type="BondCharge",
            match="once",
            charge_increment1=0.1 * unit.elementary_charge,
            charge_increment2=0.2 * unit.elementary_charge,
        )

    @pytest.fixture
    def sage_with_monovalent_lone_pair(self, sage):
        """Fixture that loads an SMIRNOFF XML for argon"""
        virtual_site_handler = VirtualSiteHandler(version=0.3)

        carbonyl_type = VirtualSiteHandler.VirtualSiteType(
            name="EP",
            smirks="[O:1]=[C:2]-[*:3]",
            distance=0.3 * unit.angstrom,
            type="MonovalentLonePair",
            match="all_permutations",
            outOfPlaneAngle=0.0 * unit.degree,
            inPlaneAngle=120.0 * unit.degree,
            charge_increment1=0.05 * unit.elementary_charge,
            charge_increment2=0.1 * unit.elementary_charge,
            charge_increment3=0.15 * unit.elementary_charge,
        )

        virtual_site_handler.add_parameter(parameter=carbonyl_type)
        sage.register_parameter_handler(virtual_site_handler)

        return sage

    @pytest.mark.xfail
    @skip_if_missing("parmed")
    def test_sigma_hole_example(self, sage_with_sigma_hole):
        """Test that a single-molecule sigma hole example runs"""
        parmed = pytest.importorskip("parmed")

        molecule = MoleculeWithConformer.from_smiles("CCl", name="Chloromethane")

        out = Interchange.from_smirnoff(
            force_field=sage_with_sigma_hole,
            topology=molecule.to_topology(),
        )
        out.box = [4, 4, 4]
        out.positions = molecule.conformers[0]

        # TODO: Sanity-check reported energies
        get_gromacs_energies(out)

        out.to_top("sigma.top")
        gmx_top = parmed.load_file("sigma.top")

        assert abs(numpy.sum([p.charge for p in gmx_top.atoms])) < 1e-3

    @pytest.mark.slow
    def test_carbonyl_example(self, sage_with_planar_monovalent_carbonyl, ethanol):
        """Test that a single-molecule planar carbonyl example can run 0 steps."""
        ethanol.generate_conformers(n_conformers=1)

        hexanal = MoleculeWithConformer.from_smiles("CCCCCC=O")
        hexanal._conformers[0] += Quantity("2 nanometer")

        topology = Topology.from_molecules([ethanol, hexanal])
        topology.box_vectors = Quantity([4, 4, 4], "nanometer")

        get_gromacs_energies(
            sage_with_planar_monovalent_carbonyl.create_interchange(topology),
        )

    @skip_if_missing("openmm")
    def test_tip4p_charge_neutrality(self, tip4p, water_dimer):
        tip4p.create_interchange(water_dimer).to_top("_dimer.top")

        system = openmm.app.GromacsTopFile("_dimer.top").createSystem()

        assert system.getForce(0).getNumParticles() == 8

        charges = [system.getForce(0).getParticleParameters(i)[0]._value for i in range(4)]

        assert sum(charges) == pytest.approx(0.0)
