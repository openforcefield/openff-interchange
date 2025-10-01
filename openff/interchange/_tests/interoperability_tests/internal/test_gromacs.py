from importlib import resources

import numpy
import pytest
from openff.toolkit import ForceField, Molecule, Quantity, Topology, unit
from openff.toolkit.typing.engines.smirnoff import VirtualSiteHandler
from openff.units.openmm import ensure_quantity
from openff.utilities import get_data_file_path, has_package, skip_if_missing

from openff.interchange import Interchange
from openff.interchange._tests import MoleculeWithConformer, get_protein, needs_gmx
from openff.interchange.drivers import get_gromacs_energies, get_openmm_energies
from openff.interchange.exceptions import (
    UnsupportedExportError,
    VirtualSiteTypeNotImplementedError,
)
from openff.interchange.interop.gromacs._import._import import (
    _read_box,
    _read_coordinates,
)

if has_package("openmm"):
    import openmm.app
    import openmm.unit



@skip_if_missing("intermol")
@skip_if_missing("mdtraj")
@skip_if_missing("openmm")
@needs_gmx
class TestGROMACSGROFile:
    try:
        _INTERMOL_PATH = resources.files(
            "intermol.tests.gromacs.unit_tests",
        )
    except ImportError:
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
    def test_load_gro_nonstandard_precision(self):
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


@needs_gmx
class TestGROMACS:
    @pytest.mark.slow
    @pytest.mark.skip("from_top is not yet refactored for new Topology API")
    @pytest.mark.parametrize("reader", ["intermol", "internal"])
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
    def test_simple_roundtrip(self, sage, smiles, reader):
        molecule = MoleculeWithConformer.from_smiles(smiles)
        molecule.name = molecule.to_hill_formula()
        topology = molecule.to_topology()

        out = Interchange.from_smirnoff(force_field=sage, topology=topology)
        out.box = [4, 4, 4]
        out.positions = molecule.conformers[0]

        out.to_top("out.top")
        out.to_gro("out.gro")

        converted = Interchange.from_gromacs("out.top", "out.gro", reader=reader)

        assert numpy.allclose(out.positions, converted.positions)
        assert numpy.allclose(out.box, converted.box)

        get_gromacs_energies(out).compare(
            get_gromacs_energies(converted),
            tolerances={
                "Bond": 0.002 * molecule.n_bonds * unit.kilojoule / unit.mol,
                "Electrostatics": 0.05 * unit.kilojoule / unit.mol,
            },
        )

    @skip_if_missing("parmed")
    def test_num_impropers(self, sage):
        out = Interchange.from_smirnoff(
            sage,
            MoleculeWithConformer.from_smiles("CC1=CC=CC=C1").to_topology(),
        )

        out.box = Quantity(4 * numpy.eye(3), units=unit.nanometer)
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

    @pytest.mark.skip("Broken, unclear if cases like these are worth supporting")
    def test_nonconsecutive_isomorphic_molecules(self, sage_unconstrained):
        molecules = [Molecule.from_smiles(smiles) for smiles in ["CC", "CCO", "CC"]]

        for index, molecule in enumerate(molecules):
            molecule.generate_conformers(n_conformers=1)
            molecule.conformers[0] += Quantity(3 * [5 * index], unit.angstrom)

        topology = Topology.from_molecules(molecules)
        topology.box_vectors = Quantity([4, 4, 4], unit.nanometer)

        out = Interchange.from_smirnoff(sage_unconstrained, topology)

        get_gromacs_energies(out).compare(
            get_openmm_energies(out, combine_nonbonded_forces=True),
            {"Nonbonded": 0.5 * unit.kilojoule_per_mole},
        )


@needs_gmx
@pytest.mark.skip("Needs rewrite")
class TestGROMACSVirtualSites:
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

    def test_carbonyl_example(self, sage_with_monovalent_lone_pair):
        """Test that a single-molecule DivalentLonePair example runs"""
        mol = MoleculeWithConformer.from_smiles("C=O", name="Carbon_monoxide")

        out = Interchange.from_smirnoff(
            force_field=sage_with_monovalent_lone_pair,
            topology=mol.to_topology(),
        )
        out.box = [4, 4, 4]
        out.positions = mol.conformers[0]

        with pytest.raises(
            VirtualSiteTypeNotImplementedError,
            match=r"MonovalentLonePair not implemented.",
        ):
            # TODO: Sanity-check reported energies
            get_gromacs_energies(out)
