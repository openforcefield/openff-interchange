from typing import TYPE_CHECKING, List

from openff.interchange.models import VirtualSiteKey

if TYPE_CHECKING:
    from openmm import app

    from openff.interchange import Interchange


def to_openmm_topology(
    interchange: "Interchange", ensure_unique_atom_names: bool = True
) -> "app.Topology":
    """Create an OpenMM Topology containing some virtual site information (if appropriate)."""
    # Heavily cribbed from the toolkit
    # https://github.com/openforcefield/openff-toolkit/blob/0.11.0rc2/openff/toolkit/topology/topology.py

    from collections import defaultdict

    from openff.toolkit.topology.molecule import Bond
    from openmm import app

    from openff.interchange.interop._virtual_sites import (
        _virtual_site_parent_molecule_mapping,
    )

    topology = interchange.topology

    virtual_site_molecule_map = _virtual_site_parent_molecule_mapping(interchange)

    molecule_virtual_site_map = defaultdict(list)

    for virtual_site, molecule in virtual_site_molecule_map.items():
        molecule_virtual_site_map[topology.molecule_index(molecule)].append(
            virtual_site
        )

    has_virtual_sites = len(virtual_site_molecule_map) > 0

    virtual_site_element = app.element.Element.getByMass(0)

    openmm_topology = app.Topology()

    if ensure_unique_atom_names:
        for ref_mol in topology.reference_molecules:
            if not ref_mol.has_unique_atom_names:
                ref_mol.generate_unique_atom_names()

    # Go through atoms in OpenFF to preserve the order.
    omm_atoms = []

    # For each atom in each molecule, determine which chain/residue it should be a part of
    for molecule in topology.molecules:
        molecule_index = topology.molecule_index(molecule)

        # No chain or residue can span more than one OFF molecule, so reset these to None for the first
        # atom in each molecule.
        last_chain = None
        last_residue = None
        for atom in molecule.atoms:
            # If the residue name is undefined, assume a default of "UNK"
            if "residue_name" in atom.metadata:
                atom_residue_name = atom.metadata["residue_name"]
            else:
                atom_residue_name = "UNK"

            # If the residue number is undefined, assume a default of "0"
            if "residue_number" in atom.metadata:
                atom_residue_number = atom.metadata["residue_number"]
            else:
                atom_residue_number = "0"

            # If the chain ID is undefined, assume a default of "X"
            if "chain_id" in atom.metadata:
                atom_chain_id = atom.metadata["chain_id"]
            else:
                atom_chain_id = "X"

            # Determine whether this atom should be part of the last atom's chain, or if it
            # should start a new chain
            if last_chain is None:
                chain = openmm_topology.addChain(atom_chain_id)
            elif last_chain.id == atom_chain_id:
                chain = last_chain
            else:
                chain = openmm_topology.addChain(atom_chain_id)
            # Determine whether this atom should be a part of the last atom's residue, or if it
            # should start a new residue
            if last_residue is None:
                residue = openmm_topology.addResidue(atom_residue_name, chain)
                residue.id = atom_residue_number
            elif all(
                (
                    (last_residue.name == atom_residue_name),
                    (int(last_residue.id) == int(atom_residue_number)),
                    (chain.id == last_chain.id),
                )
            ):
                residue = last_residue
            else:
                residue = openmm_topology.addResidue(atom_residue_name, chain)
                residue.id = atom_residue_number

            # Add atom.
            element = app.Element.getByAtomicNumber(atom.atomic_number)
            omm_atom = openmm_topology.addAtom(atom.name, element, residue)

            # Make sure that OpenFF and OpenMM Topology atoms have the same indices.
            # assert topology.atom_index(atom) == int(omm_atom.id) - 1
            omm_atoms.append(omm_atom)

            last_chain = chain
            last_residue = residue

        if has_virtual_sites:
            virtual_sites_in_this_molecule: List[
                VirtualSiteKey
            ] = molecule_virtual_site_map[molecule_index]
            for this_virtual_site in virtual_sites_in_this_molecule:
                virtual_site_name = this_virtual_site.name

                # For now, assume that the residue of the last atom in the molecule is the same
                # residue as the entire molecule - this in unsafe for (bio)polymers/macromolecules
                virtual_site_residue = residue

                openmm_topology.addAtom(
                    virtual_site_name,
                    virtual_site_element,
                    virtual_site_residue,
                )

        # Add all bonds.
        bond_types = {1: app.Single, 2: app.Double, 3: app.Triple}
        for bond in molecule.bonds:
            atom1, atom2 = bond.atoms
            atom1_idx, atom2_idx = topology.atom_index(atom1), topology.atom_index(
                atom2
            )
            if isinstance(bond, Bond):
                if bond.is_aromatic:
                    bond_type = app.Aromatic
                else:
                    bond_type = bond_types[bond.bond_order]
                bond_order = bond.bond_order
            else:
                raise RuntimeError(
                    "Unexpected bond type found while iterating over Topology.bonds."
                    f"Found {type(bond)}, allowed is Bond."
                )

            openmm_topology.addBond(
                omm_atoms[atom1_idx],
                omm_atoms[atom2_idx],
                type=bond_type,
                order=bond_order,
            )

    if interchange.box is not None:
        from openff.units.openmm import to_openmm

        openmm_topology.setPeriodicBoxVectors(to_openmm(interchange.box))

    return openmm_topology
