from math import exp

import mdtraj as md
import numpy as np
import openmm
import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField, VirtualSiteHandler
from openff.units import unit
from openff.utilities.testing import skip_if_missing
from openmm import unit as openmm_unit
from pkg_resources import resource_filename

from openff.interchange.components.interchange import Interchange
from openff.interchange.components.mdtraj import _OFFBioTop
from openff.interchange.components.nonbonded import BuckinghamvdWHandler
from openff.interchange.components.potentials import Potential
from openff.interchange.components.smirnoff import SMIRNOFFVirtualSiteHandler
from openff.interchange.drivers import get_gromacs_energies, get_openmm_energies
from openff.interchange.exceptions import GMXMdrunError, UnsupportedExportError
from openff.interchange.interop.internal.gromacs import from_gro
from openff.interchange.models import PotentialKey, TopologyKey
from openff.interchange.testing import _BaseTest
from openff.interchange.testing.utils import needs_gmx
from openff.interchange.utils import get_test_file_path


@needs_gmx
class TestGROMACSGROFile(_BaseTest):
    @skip_if_missing("intermol")
    def test_load_gro(self):
        file = resource_filename(
            "intermol", "tests/gromacs/unit_tests/angle10_vacuum/angle10_vacuum.gro"
        )

        internal_coords = from_gro(file).positions.m_as(unit.nanometer)
        internal_box = from_gro(file).box.m_as(unit.nanometer)

        openmm_gro = openmm.app.GromacsGroFile(file)
        openmm_coords = np.array(
            openmm_gro.getPositions().value_in_unit(openmm_unit.nanometer)
        )
        openmm_box = np.array(
            openmm_gro.getPeriodicBoxVectors().value_in_unit(openmm_unit.nanometer)
        )

        assert np.allclose(internal_coords, openmm_coords)
        assert np.allclose(internal_box, openmm_box)

    @skip_if_missing("intermol")
    def test_load_gro_nonstandard_precision(self):
        file = resource_filename(
            "intermol", "tests/gromacs/unit_tests/lj3_bulk/lj3_bulk.gro"
        )
        internal_coords = from_gro(file).positions.m_as(unit.nanometer)

        # OpenMM seems to assume a precision of 3. Use InterMol instead here.
        from intermol.gromacs.grofile_parser import GromacsGroParser

        intermol_gro = GromacsGroParser(file)
        intermol_gro.read()

        # InterMol stores positions as NumPy arrays _of openmm.unit.Quantity_
        # objects, so need to carefully convert everything through unitless
        # floats; converting through Pint-like quantities might produce a
        # funky double-wrapped thing
        def converter(x):
            return x.value_in_unit(openmm_unit.nanometer)

        other_coords = np.frompyfunc(converter, 1, 1)(intermol_gro.positions).astype(
            float
        )

        assert np.allclose(internal_coords, other_coords)

        # This file happens to have 12 digits of preicion; what really matters is that
        # the convential precision of 3 was not used.
        n_decimals = len(str(internal_coords[0, 0]).split(".")[1])
        assert n_decimals == 12

    @pytest.mark.slow()
    def test_residue_names_in_gro_file(self):
        """Test that residue names > 5 characters don't break .gro file output"""
        benzene = Molecule.from_file(get_test_file_path("benzene.sdf"))
        benzene.name = "supercalifragilisticexpialidocious"
        top = _OFFBioTop.from_molecules(benzene)
        top.mdtop = md.Topology.from_openmm(top.to_openmm())

        # Populate an entire interchange because ...
        force_field = ForceField("openff-1.0.0.offxml")
        out = Interchange.from_smirnoff(force_field, top)
        out.box = [4, 4, 4]
        out.positions = benzene.conformers[0]

        # ... the easiest way to check the validity of the files
        # is to see if GROMACS can run them
        get_gromacs_energies(out)


@needs_gmx
class TestGROMACS(_BaseTest):
    @pytest.mark.slow()
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
    def test_simple_roundtrip(self, smiles, reader):
        parsley = ForceField("openff_unconstrained-1.0.0.offxml")

        molecule = Molecule.from_smiles(smiles)
        molecule.name = molecule.to_hill_formula(molecule)
        molecule.generate_conformers(n_conformers=1)
        topology = molecule.to_topology()

        out = Interchange.from_smirnoff(force_field=parsley, topology=topology)
        out.box = [4, 4, 4]
        out.positions = molecule.conformers[0]

        out.to_top("out.top")
        out.to_gro("out.gro")

        converted = Interchange.from_gromacs("out.top", "out.gro", reader=reader)

        assert np.allclose(out.positions, converted.positions)
        assert np.allclose(out.box, converted.box)

        get_gromacs_energies(out).compare(
            get_gromacs_energies(converted),
            custom_tolerances={
                "Bond": 0.002 * molecule.n_bonds * unit.kilojoule / unit.mol,
                "Electrostatics": 0.05 * unit.kilojoule / unit.mol,
            },
        )

    @skip_if_missing("parmed")
    def test_num_impropers(self, parsley):
        top = Molecule.from_smiles("CC1=CC=CC=C1").to_topology()
        out = Interchange.from_smirnoff(parsley, top)
        out.to_top("tmp.top")

        # Sanity check; toluene should have some improper(s)
        assert len(out["ImproperTorsions"].slot_map) > 0

        import parmed as pmd

        struct = pmd.load_file("tmp.top")
        n_impropers_parmed = len([d for d in struct.dihedrals if d.improper])
        assert n_impropers_parmed == len(out["ImproperTorsions"].slot_map)

    @skip_if_missing("intermol")
    def test_set_mixing_rule(self, ethanol_top, parsley):
        from intermol.gromacs.gromacs_parser import GromacsParser

        openff_sys = Interchange.from_smirnoff(
            force_field=parsley, topology=ethanol_top
        )
        openff_sys.positions = np.zeros((ethanol_top.n_topology_atoms, 3))
        openff_sys.to_gro("tmp.gro")

        openff_sys.to_top("lorentz.top")
        lorentz = GromacsParser("lorentz.top", "tmp.gro").read()
        assert lorentz.combination_rule == "Lorentz-Berthelot"

        openff_sys["vdW"].mixing_rule = "geometric"

        openff_sys.to_top("geometric.top")
        geometric = GromacsParser("geometric.top", "tmp.gro").read()
        assert geometric.combination_rule == "Multiply-Sigeps"

    @pytest.mark.xfail(
        reason="cannot test unsupported mixing rules in GROMACS with current SMIRNOFFvdWHandler model"
    )
    def test_unsupported_mixing_rule(self, ethanol_top, parsley):
        # TODO: Update this test when the model supports more mixing rules than GROMACS does
        openff_sys = Interchange.from_smirnoff(
            force_field=parsley, topology=ethanol_top
        )
        openff_sys["vdW"].mixing_rule = "kong"

        with pytest.raises(UnsupportedExportError, match="rule `geometric` not compat"):
            openff_sys.to_top("out.top")

    @pytest.mark.slow()
    def test_argon_buck(self):
        """Test that Buckingham potentials are supported and can be exported"""
        from openff.interchange.components.smirnoff import SMIRNOFFElectrostaticsHandler

        mol = Molecule.from_smiles("[#18]")
        mol.name = "Argon"
        top = _OFFBioTop.from_molecules([mol, mol])
        top.mdtop = md.Topology.from_openmm(top.to_openmm())

        # http://www.sklogwiki.org/SklogWiki/index.php/Argon#Buckingham_potential
        erg_mol = unit.erg / unit.mol * float(unit.avogadro_number)
        A = 1.69e-8 * erg_mol
        B = 1 / (0.273 * unit.angstrom)
        C = 102e-12 * erg_mol * unit.angstrom ** 6

        r = 0.3 * unit.nanometer

        buck = BuckinghamvdWHandler()
        coul = SMIRNOFFElectrostaticsHandler(method="pme")

        pot_key = PotentialKey(id="[#18]")
        pot = Potential(parameters={"A": A, "B": B, "C": C})

        for atom in top.mdtop.atoms:
            top_key = TopologyKey(atom_indices=(atom.index,))
            buck.slot_map.update({top_key: pot_key})

            coul.slot_map.update({top_key: pot_key})
            coul.potentials.update(
                {pot_key: Potential(parameters={"charge": 0 * unit.elementary_charge})}
            )

        buck.potentials[pot_key] = pot

        out = Interchange()
        out.handlers["Buckingham-6"] = buck
        out.handlers["Electrostatics"] = coul
        out.topology = top
        out.box = [10, 10, 10] * unit.nanometer
        out.positions = [[0, 0, 0], [0.3, 0, 0]] * unit.nanometer
        out.to_gro("out.gro", writer="internal")
        out.to_top("out.top", writer="internal")

        omm_energies = get_openmm_energies(out)
        by_hand = A * exp(-B * r) - C * r ** -6

        resid = omm_energies.energies["vdW"] - by_hand
        assert resid < 1e-5 * unit.kilojoule / unit.mol

        # TODO: Add back comparison to GROMACS energies once GROMACS 2020+
        # supports Buckingham potentials
        with pytest.raises(GMXMdrunError):
            get_gromacs_energies(out, mdp="cutoff_buck")


@needs_gmx
class TestGROMACSVirtualSites(_BaseTest):
    @pytest.fixture()
    def parsley_with_sigma_hole(self, parsley):
        """Fixture that loads an SMIRNOFF XML for argon"""
        virtual_site_handler = VirtualSiteHandler(version=0.3)

        sigma_type = VirtualSiteHandler.VirtualSiteBondChargeType(
            name="EP",
            smirks="[#6:1]-[#17:2]",
            distance=1.4 * openmm_unit.angstrom,
            type="BondCharge",
            match="once",
            charge_increment1=0.1 * openmm_unit.elementary_charge,
            charge_increment2=0.2 * openmm_unit.elementary_charge,
        )

        virtual_site_handler.add_parameter(parameter=sigma_type)
        parsley.register_parameter_handler(virtual_site_handler)

        return parsley

    @pytest.fixture()
    def parsley_with_monovalent_lone_pair(self, parsley):
        """Fixture that loads an SMIRNOFF XML for argon"""
        virtual_site_handler = VirtualSiteHandler(version=0.3)

        carbonyl_type = VirtualSiteHandler.VirtualSiteMonovalentLonePairType(
            name="EP",
            smirks="[O:1]=[C:2]-[*:3]",
            distance=0.3 * openmm_unit.angstrom,
            type="MonovalentLonePair",
            match="once",
            outOfPlaneAngle=0.0 * openmm_unit.degree,
            inPlaneAngle=120.0 * openmm_unit.degree,
            charge_increment1=0.05 * openmm_unit.elementary_charge,
            charge_increment2=0.1 * openmm_unit.elementary_charge,
            charge_increment3=0.15 * openmm_unit.elementary_charge,
        )

        virtual_site_handler.add_parameter(parameter=carbonyl_type)
        parsley.register_parameter_handler(virtual_site_handler)

        return parsley

    @skip_if_missing("parmed")
    def test_sigma_hole_example(self, parsley_with_sigma_hole):
        """Test that a single-molecule sigma hole example runs"""
        mol = Molecule.from_smiles("CCl")
        mol.name = "Chloromethane"
        mol.generate_conformers(n_conformers=1)

        out = Interchange.from_smirnoff(
            force_field=parsley_with_sigma_hole, topology=mol.to_topology()
        )
        out.box = [4, 4, 4]
        out.positions = mol.conformers[0]
        out.handlers["VirtualSites"] = SMIRNOFFVirtualSiteHandler._from_toolkit(
            parameter_handler=parsley_with_sigma_hole["VirtualSites"],
            topology=mol.to_topology(),
        )
        out["vdW"]._from_toolkit_virtual_sites(
            parameter_handler=parsley_with_sigma_hole["VirtualSites"],
            topology=mol.to_topology(),
        )
        out["Electrostatics"]._from_toolkit_virtual_sites(
            parameter_handler=parsley_with_sigma_hole["VirtualSites"],
            topology=mol.to_topology(),
        )

        # TODO: Sanity-check reported energies
        get_gromacs_energies(out)

        import numpy as np
        import parmed as pmd

        out.to_top("sigma.top")
        gmx_top = pmd.load_file("sigma.top")

        assert abs(np.sum([p.charge for p in gmx_top.atoms])) < 1e-3

    def test_carbonyl_example(self, parsley_with_monovalent_lone_pair):
        """Test that a single-molecule DivalentLonePair example runs"""
        mol = Molecule.from_smiles("C=O")
        mol.name = "Carbon_monoxide"
        mol.generate_conformers(n_conformers=1)

        out = Interchange.from_smirnoff(
            force_field=parsley_with_monovalent_lone_pair, topology=mol.to_topology()
        )
        out.box = [4, 4, 4]
        out.positions = mol.conformers[0]
        out.handlers["VirtualSites"] = SMIRNOFFVirtualSiteHandler._from_toolkit(
            parameter_handler=parsley_with_monovalent_lone_pair["VirtualSites"],
            topology=mol.to_topology(),
        )
        out["vdW"]._from_toolkit_virtual_sites(
            parameter_handler=parsley_with_monovalent_lone_pair["VirtualSites"],
            topology=mol.to_topology(),
        )
        out["Electrostatics"]._from_toolkit_virtual_sites(
            parameter_handler=parsley_with_monovalent_lone_pair["VirtualSites"],
            topology=mol.to_topology(),
        )

        # TODO: Sanity-check reported energies
        get_gromacs_energies(out)

        # ParmEd does not support 3fad, and cannot be used to sanity check charges
