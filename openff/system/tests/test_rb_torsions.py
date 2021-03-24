from openff.toolkit.topology.molecule import Molecule
from simtk import unit as omm_unit

from openff.system import unit
from openff.system.components.misc import RBTorsionHandler
from openff.system.components.potentials import Potential
from openff.system.models import PotentialKey, TopologyKey
from openff.system.stubs import ForceField
from openff.system.tests.energy_tests.gromacs import get_gromacs_energies
from openff.system.tests.energy_tests.openmm import get_openmm_energies

kj_mol = unit.Unit("kilojoule / mol")


def test_ethanol_opls():
    mol = Molecule.from_smiles("CC")
    mol.generate_conformers(n_conformers=1)
    top = mol.to_topology()
    parsley = ForceField("openff-1.0.0.offxml")
    out = parsley.create_openff_system(top)
    out.box = [4, 4, 4]
    out.positions = mol.conformers[0]

    rb_torsions = RBTorsionHandler()
    smirks = "[#1:1]-[#6X4:2]-[#6X4:3]-[#1:4]"
    pot_key = PotentialKey(id=smirks)
    for proper in top.propers:
        top_key = TopologyKey(atom_indices=tuple(a.topology_atom_index for a in proper))
        rb_torsions.slot_map.update({top_key: pot_key})

    # Values from HC-CT-CT-HC RB torsion
    # https://github.com/mosdef-hub/foyer/blob/7816bf53a127502520a18d76c81510f96adfdbed/foyer/forcefields/xml/oplsaa.xml#L2585
    pot = Potential(
        parameters={
            "c0": 1.6276 * kj_mol,
            "c1": 1.8828 * kj_mol,
            "c2": 0.0 * kj_mol,
            "c3": -2.5104 * kj_mol,
            "c4": 0.0 * kj_mol,
            "c5": 0.0 * kj_mol,
        }
    )

    rb_torsions.potentials.update({pot_key: pot})

    out.handlers.update({"RBTorsions": rb_torsions})
    out.handlers.pop("ProperTorsions")

    gmx = get_openmm_energies(out).energies["Torsion"]
    omm = get_gromacs_energies(out).energies["Torsion"]

    assert (gmx - omm).value_in_unit(omm_unit.kilojoule_per_mole) < 1e-3
