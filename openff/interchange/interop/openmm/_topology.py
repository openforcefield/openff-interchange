"""
Helper functions for exporting the topology to OpenMM.
"""

from typing import TYPE_CHECKING

from openff.utilities.utilities import has_package

from openff.interchange import Interchange
from openff.interchange.models import VirtualSiteKey

if has_package("openmm") or TYPE_CHECKING:
    import openmm.app


def to_openmm_topology(
    interchange: "Interchange",
    collate: bool = False,
    ensure_unique_atom_names: str | bool = "residues",
) -> "openmm.app.Topology":
    """
    Create an OpenMM Topology containing some virtual site information (if appropriate).

    Parameters
    ----------
    interchange
        The Interchange object to convert to an OpenMM Topology.
    collate
        If False, the default, virtual sites will be added to residues at the end of the topology.
        If True, virtual sites will be collated with their associated molecule and added to the residue of the last
        atom in the molecule they belong to.

    """
    # Heavily cribbed from the toolkit
    # https://github.com/openforcefield/openff-toolkit/blob/0.11.0rc2/openff/toolkit/topology/topology.py

    from collections import defaultdict

    from openff.toolkit import Topology
    from openff.toolkit.topology._mm_molecule import _SimpleBond
    from openff.toolkit.topology.molecule import Bond

    from openff.interchange.interop._virtual_sites import (
        _virtual_site_parent_molecule_mapping,
    )

    # Copy topology to avoid modifying input (eg, when generating atom names)
    topology = Topology(interchange.topology)

    virtual_site_molecule_map = _virtual_site_parent_molecule_mapping(interchange)

    molecule_virtual_site_map = defaultdict(list)

    for virtual_site, molecule_index in virtual_site_molecule_map.items():
        molecule_virtual_site_map[molecule_index].append(virtual_site)

    has_virtual_sites = len(virtual_site_molecule_map) > 0

    virtual_site_element = openmm.app.element.Element.getByMass(0)

    openmm_topology = openmm.app.Topology()

    # Create unique atom names (as requested)
    if ensure_unique_atom_names:
        for molecule in topology._molecules:
            if isinstance(ensure_unique_atom_names, str) and hasattr(
                molecule,
                ensure_unique_atom_names,
            ):
                for hier_elem in getattr(molecule, ensure_unique_atom_names):
                    if not hier_elem.has_unique_atom_names:
                        hier_elem.generate_unique_atom_names()
            elif not molecule.has_unique_atom_names:
                molecule.generate_unique_atom_names()

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
            # If the these are undefined, assume a default of
            # residue name/number "UNK" / "0", chain ID "X"
            atom_residue_name = atom.metadata.get("residue_name", "UNK")
            atom_residue_number = atom.metadata.get("residue_number", "0")
            atom_chain_id = atom.metadata.get("chain_id", "X")

            # Determine whether this atom should be part of the last atom's chain, or if it
            # should start a new chain
            if last_chain is None:
                chain = openmm_topology.addChain(id=atom_chain_id)
            elif last_chain.id == atom_chain_id:
                chain = last_chain
            else:
                chain = openmm_topology.addChain(id=atom_chain_id)
            # Determine whether this atom should be a part of the last atom's residue, or if it
            # should start a new residue
            if last_residue is None:
                residue = openmm_topology.addResidue(name=atom_residue_name, chain=chain)
                residue.id = atom_residue_number
            elif all(
                (
                    (last_residue.name == atom_residue_name),
                    (int(last_residue.id) == int(atom_residue_number)),
                    (chain.id == last_chain.id),
                ),
            ):
                residue = last_residue
            else:
                residue = openmm_topology.addResidue(name=atom_residue_name, chain=chain)
                residue.id = atom_residue_number

            # Add atom.
            element = openmm.app.Element.getByAtomicNumber(atom.atomic_number)
            omm_atom = openmm_topology.addAtom(atom.name, element, residue)

            # Make sure that OpenFF and OpenMM Topology atoms have the same indices.
            # assert topology.atom_index(atom) == int(omm_atom.id) - 1
            omm_atoms.append(omm_atom)

            last_chain = chain
            last_residue = residue

        if has_virtual_sites and collate:
            virtual_sites_in_this_molecule: list[VirtualSiteKey] = molecule_virtual_site_map[molecule_index]
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
        bond_types = {1: openmm.app.Single, 2: openmm.app.Double, 3: openmm.app.Triple}
        for bond in molecule.bonds:
            atom1, atom2 = bond.atoms
            atom1_idx, atom2_idx = (
                topology.atom_index(atom1),
                topology.atom_index(
                    atom2,
                ),
            )
            if isinstance(bond, Bond):
                if bond.is_aromatic:
                    bond_type = openmm.app.Aromatic
                else:
                    bond_type = bond_types[bond.bond_order]
                bond_order = bond.bond_order
            elif isinstance(bond, _SimpleBond):
                bond_type = None
                bond_order = None
            else:
                raise RuntimeError(
                    "Unexpected bond type found while iterating over Topology.bonds."
                    f"Found {type(bond)}, allowed is Bond.",
                )

            openmm_topology.addBond(
                omm_atoms[atom1_idx],
                omm_atoms[atom2_idx],
                type=bond_type,
                order=bond_order,
            )

    if has_virtual_sites and not collate:
        chain = None
        residue = None
        for virtual_site_key in interchange["VirtualSites"].key_map:
            assert isinstance(virtual_site_key, VirtualSiteKey)
            parent_atom_index = virtual_site_key.orientation_atom_indices[0]
            parent_atom = omm_atoms[parent_atom_index]

            if chain is None or chain.id != parent_atom.residue.chain.id:
                chain = openmm_topology.addChain(id=parent_atom.residue.chain.id)

            if residue is None or not (
                residue.name == parent_atom.residue.name
                and residue.id == parent_atom.residue.id
                and residue.insertionCode == parent_atom.residue.insertionCode
                and residue.chain.id == parent_atom.residue.chain.id
            ):
                residue = openmm_topology.addResidue(
                    name=parent_atom.residue.name,
                    chain=chain,
                    id=parent_atom.residue.id,
                    insertionCode=parent_atom.residue.id,
                )

            virtual_site_name = virtual_site_key.name

            openmm_topology.addAtom(
                name=virtual_site_name,
                element=virtual_site_element,
                residue=residue,
            )
    if interchange.box is not None:
        from openff.units.openmm import to_openmm

        openmm_topology.setPeriodicBoxVectors(to_openmm(interchange.box))

    return openmm_topology
