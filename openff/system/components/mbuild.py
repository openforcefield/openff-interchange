import mbuild as mb
from openff.toolkit.topology import Molecule, Topology
from simtk import unit


def offmol_to_compound(off_mol: Molecule) -> mb.Compound:

    if not off_mol.has_unique_atom_names:
        off_mol.generate_unique_atom_names()

    if off_mol.n_conformers == 0:
        off_mol.generate_conformers(n_conformers=1)

    comp = mb.Compound()
    comp.name = off_mol.name

    for a in off_mol.atoms:
        atom_comp = mb.Particle(name=a.element.symbol)
        comp.add(atom_comp, label=a.name)

    for b in off_mol.bonds:
        comp.add_bond((comp[b.atom1_index], comp[b.atom2_index]))

    comp.xyz = off_mol.conformers[0].value_in_unit(unit.nanometer)

    return comp


def offtop_to_compound(off_top: Topology) -> mb.Compound:

    sub_comps = []

    for top_mol in off_top.topology_molecules:
        # TODO: This could have unintended consequences if the TopologyMolecule
        # has atoms in a different order than the reference Molecule
        this_comp = offmol_to_compound(top_mol.reference_molecule)
        sub_comps.append(this_comp)

    comp = mb.Compound(subcompounds=sub_comps)
    return comp
