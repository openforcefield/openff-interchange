from openff.toolkit.topology import Molecule
from simtk import unit as omm_unit

from openff.system.stubs import ForceField
from openff.system.tests.energy_tests.amber import get_amber_energies
from openff.system.tests.energy_tests.gromacs import get_gromacs_energies


def test_amber_energy():
    """Basic test to see if the amber energy driver is functional"""
    mol = Molecule.from_smiles("CCO")
    mol.generate_conformers(n_conformers=1)
    top = mol.to_topology()

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")
    off_sys = parsley.create_openff_system(top)

    off_sys.box = [4, 4, 4]
    off_sys.positions = mol.conformers[0]

    omm_energies = get_gromacs_energies(off_sys, mdp="cutoff_hbonds")
    amb_energies = get_amber_energies(off_sys)
    omm_energies.compare(
        amb_energies,
        custom_tolerances={
            "Bond": 3.6 * omm_unit.kilojoule_per_mole,
            "Angle": 0.2 * omm_unit.kilojoule_per_mole,
            "Torsion": 1.9 * omm_unit.kilojoule_per_mole,
            "Nonbonded": 34.9 * omm_unit.kilojoule_per_mole,
        },
    )
