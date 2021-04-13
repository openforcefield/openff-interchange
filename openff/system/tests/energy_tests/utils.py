import mbuild as mb
from openff.toolkit.topology import Molecule
from simtk import unit as omm_unit

from openff.system.tests.energy_tests.report import EnergyReport


def compare_gromacs_openmm(
    gmx_report: EnergyReport,
    omm_report: EnergyReport,
    # custom_tolerances: Dict[str, float] = None,
):

    # TODO: Tighten differences
    # TODO: Nonbonded components
    # np.testing doesn't work on Quantity

    gmx_report.compare(omm_report)


def compare_gromacs(
    report1: EnergyReport,
    report2: EnergyReport,
    #    custom_tolerances: Dict[str, float] = None,
):
    # TODO: Tighten differences
    # TODO: Nonbonded components
    # np.testing doesn't work on Quantity

    report1.compare(report2)
    # TODO: Fix constraints and other issues around GROMACS non-bonded energies


def compare_openmm(
    report1: EnergyReport,
    report2: EnergyReport,
    #    custom_tolerances: Dict[str, float] = None,
):

    report1.compare(report2)


def offmol_to_compound(off_mol: Molecule) -> mb.Compound:

    assert len(off_mol.conformers) > 0

    comp = mb.Compound()

    for a in off_mol.atoms:
        atom_comp = mb.Particle(name=a.element.symbol)
        comp.add(atom_comp, label=a.name)

    for b in off_mol.bonds:
        comp.add_bond((comp[b.atom1_index], comp[b.atom2_index]))

    comp.xyz = off_mol.conformers[0].value_in_unit(omm_unit.nanometer)

    return comp
