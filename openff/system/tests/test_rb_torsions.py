import numpy as np
import pytest
from openff.toolkit.topology.molecule import Molecule
from openff.units import unit
from openff.utilities.testing import skip_if_missing
from openff.utilities.utilities import has_package

from openff.system.components.foyer import RBTorsionHandler
from openff.system.components.potentials import Potential
from openff.system.models import PotentialKey, TopologyKey
from openff.system.stubs import ForceField
from openff.system.tests.base_test import BaseTest
from openff.system.tests.energy_tests.openmm import get_openmm_energies

kj_mol = unit.Unit("kilojoule / mol")

if has_package("gromacs"):
    from openff.system.tests.energy_tests.gromacs import (
        _get_mdp_file,
        _run_gmx_energy,
        get_gromacs_energies,
    )


class TestRBTorsions(BaseTest):
    @pytest.fixture(scope="class")
    def ethanol_with_rb_torsions(self):
        mol = Molecule.from_smiles("CC")
        mol.generate_conformers(n_conformers=1)
        top = mol.to_topology()
        parsley = ForceField("openff-1.0.0.offxml")
        out = parsley.create_openff_system(top)
        out.box = [4, 4, 4]
        out.positions = mol.conformers[0]
        out.positions = np.round(out.positions, 2)

        rb_torsions = RBTorsionHandler()
        smirks = "[#1:1]-[#6X4:2]-[#6X4:3]-[#1:4]"
        pot_key = PotentialKey(id=smirks)
        for proper in top.propers:
            top_key = TopologyKey(
                atom_indices=tuple(a.topology_atom_index for a in proper)
            )
            rb_torsions.slot_map.update({top_key: pot_key})

        # Values from HC-CT-CT-HC RB torsion
        # https://github.com/mosdef-hub/foyer/blob/7816bf53a127502520a18d76c81510f96adfdbed/foyer/forcefields/xml/oplsaa.xml#L2585
        pot = Potential(
            parameters={
                "C0": 0.6276 * kj_mol,
                "C1": 1.8828 * kj_mol,
                "C2": 0.0 * kj_mol,
                "C3": -2.5104 * kj_mol,
                "C4": 0.0 * kj_mol,
                "C5": 0.0 * kj_mol,
            }
        )

        rb_torsions.potentials.update({pot_key: pot})

        out.handlers.update({"RBTorsions": rb_torsions})
        out.handlers.pop("ProperTorsions")

        return out

    @skip_if_missing("gromacs")
    @pytest.mark.slow
    def test_rb_torsions(self, ethanol_with_rb_torsions):
        omm = get_openmm_energies(ethanol_with_rb_torsions, round_positions=3).energies[
            "Torsion"
        ]
        gmx = get_gromacs_energies(ethanol_with_rb_torsions).energies["Torsion"]

        assert (gmx - omm).m_as(kj_mol) < 1e-6

    @pytest.mark.slow
    @skip_if_missing("foyer")
    @skip_if_missing("mbuild")
    @skip_if_missing("gromacs")
    def test_rb_torsions_vs_foyer(self, ethanol_with_rb_torsions):
        # Given that these force constants are copied from Foyer's OPLS-AA file,
        # compare to processing through the current MoSDeF pipeline
        import foyer
        import mbuild

        comp = mbuild.load("CC", smiles=True)
        comp.xyz = ethanol_with_rb_torsions.positions.m_as(unit.nanometer)
        ff = foyer.Forcefield(name="oplsaa")
        from_foyer = ff.apply(comp)
        from_foyer.box = [40, 40, 40, 90, 90, 90]
        from_foyer.save("from_foyer.top")
        from_foyer.save("from_foyer.gro")

        rb_torsion_energy_from_foyer = _run_gmx_energy(
            top_file="from_foyer.top",
            gro_file="from_foyer.gro",
            mdp_file=_get_mdp_file("default"),
        ).energies["Torsion"]

        # GROMACS vs. OpenMM was already compared, so just use one
        omm = get_gromacs_energies(ethanol_with_rb_torsions).energies["Torsion"]

        assert (omm - rb_torsion_energy_from_foyer).m_as(kj_mol) < 1e-6
