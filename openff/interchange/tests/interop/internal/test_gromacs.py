from math import exp

import mdtraj as md
import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField, VirtualSiteHandler
from openff.units import unit
from openff.utilities.testing import skip_if_missing
from simtk import unit as simtk_unit

from openff.interchange.components.interchange import Interchange
from openff.interchange.components.mdtraj import OFFBioTop
from openff.interchange.components.nonbonded import BuckinghamvdWHandler
from openff.interchange.components.potentials import Potential
from openff.interchange.components.smirnoff import SMIRNOFFVirtualSiteHandler
from openff.interchange.drivers import get_gromacs_energies, get_openmm_energies
from openff.interchange.exceptions import GMXMdrunError, UnsupportedExportError
from openff.interchange.models import PotentialKey, TopologyKey
from openff.interchange.tests import BaseTest
from openff.interchange.tests.energy_tests.test_energies import needs_gmx
from openff.interchange.utils import get_test_file_path


@needs_gmx
class TestGROMACS(BaseTest):
    @skip_if_missing("parmed")
    def test_set_mixing_rule(self, ethanol_top, parsley):
        import parmed as pmd

        openff_sys = Interchange.from_smirnoff(
            force_field=parsley, topology=ethanol_top
        )

        openff_sys.to_top("lorentz.top")
        top_file = pmd.load_file("lorentz.top")
        assert top_file.combining_rule == "lorentz"

        openff_sys["vdW"].mixing_rule = "geometric"

        openff_sys.to_top("geometric.top")
        top_file = pmd.load_file("geometric.top")
        assert top_file.combining_rule == "geometric"

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
    def test_residue_names_in_gro_file(self):
        """Test that residue names > 5 characters don't break .gro file output"""
        benzene = Molecule.from_file(get_test_file_path("benzene.sdf"))
        benzene.name = "supercalifragilisticexpialidocious"
        top = OFFBioTop.from_molecules(benzene)
        top.mdtop = md.Topology.from_openmm(top.to_openmm())

        # Populate an entire interchange because ...
        force_field = ForceField("openff-1.0.0.offxml")
        out = Interchange.from_smirnoff(force_field, top)
        out.box = [4, 4, 4]
        out.positions = benzene.conformers[0]

        # ... the easiest way to check the validity of the files
        # is to see if GROMACS can run them
        get_gromacs_energies(out)

    def test_argon_buck(self):
        """Test that Buckingham potentials are supported and can be exported"""
        from openff.interchange.components.smirnoff import SMIRNOFFElectrostaticsHandler

        mol = Molecule.from_smiles("[#18]")
        top = OFFBioTop.from_molecules([mol, mol])
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

        resid = omm_energies.energies["Nonbonded"] - by_hand
        assert resid < 1e-5 * unit.kilojoule / unit.mol

        # TODO: Add back comparison to GROMACS energies once GROMACS 2020+
        # supports Buckingham potentials
        with pytest.raises(GMXMdrunError):
            get_gromacs_energies(out, mdp="cutoff_buck")


@needs_gmx
class TestGROMACSVirtualSites(BaseTest):
    @pytest.fixture()
    def parsley_with_sigma_hole(self, parsley):
        """Fixture that loads an SMIRNOFF XML for argon"""
        virtual_site_handler = VirtualSiteHandler(version=0.3)

        sigma_type = VirtualSiteHandler.VirtualSiteBondChargeType(
            name="EP",
            smirks="[#6:1]-[#17:2]",
            distance=1.4 * simtk_unit.angstrom,
            type="BondCharge",
            match="once",
            charge_increment1=0.1 * simtk_unit.elementary_charge,
            charge_increment2=0.2 * simtk_unit.elementary_charge,
        )

        virtual_site_handler.add_parameter(parameter=sigma_type)
        parsley.register_parameter_handler(virtual_site_handler)

        return parsley

    @skip_if_missing("parmed")
    def test_sigma_hole_example(self, parsley_with_sigma_hole):
        """Test that a single-molecule sigma hole example runs"""
        mol = Molecule.from_smiles("CCl")
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
