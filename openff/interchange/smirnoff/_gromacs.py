from typing import TYPE_CHECKING, Dict, List

from openff.units import unit
from openff.units.elements import MASSES, SYMBOLS

from openff.interchange.components.interchange import Interchange
from openff.interchange.exceptions import UnsupportedExportError
from openff.interchange.interop.gromacs.models.models import (
    GROMACSAngle,
    GROMACSAtom,
    GROMACSBond,
    GROMACSMolecule,
    GROMACSSystem,
    LennardJonesAtomType,
)
from openff.interchange.models import TopologyKey

if TYPE_CHECKING:
    from openff.toolkit.topology.molecule import Atom, Molecule


def _convert(interchange: Interchange) -> GROMACSSystem:
    """Convert an `Interchange` object to `GROMACSSystem`."""
    if "vdW" in interchange.collections:
        nonbonded_function = 1
        scale_lj = interchange["vdW"].scale_14
        _combination_rule = interchange["vdW"].mixing_rule.lower()
        gen_pairs = "yes"
    else:
        raise UnsupportedExportError(
            "Could not find a handler for short-ranged vdW interactions that is compatible "
            "with GROMACS.",
        )

    if _combination_rule == "lorentz-berthelot":
        combination_rule = 2
    elif _combination_rule == "geometric":
        combination_rule = 3
    else:
        raise UnsupportedExportError(
            f"Could not find a GROMACS-compatible combination rule for mixing rule "
            f"{_combination_rule}.",
        )

    scale_electrostatics = interchange["Electrostatics"].scale_14

    system = GROMACSSystem(
        nonbonded_function=nonbonded_function,
        combination_rule=combination_rule,
        gen_pairs=gen_pairs,
        vdw_14=scale_lj,
        coul_14=scale_electrostatics,
    )

    unique_molecule_map: Dict[
        int,
        List,
    ] = interchange.topology.identical_molecule_groups

    # Give each atom in each unique molecule a unique name so that can act like an atom type

    # TODO: Virtual sites
    _atom_atom_type_map: Dict["Atom", str] = dict()

    try:
        vdw_collection = interchange["vdW"]
        electrostatics_collection = interchange["Electrostatics"]
    except KeyError:
        raise UnsupportedExportError("Plugins not supported.")

    for unique_molecule_index in unique_molecule_map:
        unique_molecule = interchange.topology.molecule(unique_molecule_index)

        if unique_molecule.name == "":
            unique_molecule.name = "MOL" + str(unique_molecule_index)

        for atom in unique_molecule.atoms:
            atom_type_name = f"{unique_molecule.name}{unique_molecule.atom_index(atom)}"
            _atom_atom_type_map[atom] = atom_type_name

            topology_index = interchange.topology.atom_index(atom)
            key = TopologyKey(atom_indices=(topology_index,))
            vdw_parameters = vdw_collection.potentials[
                vdw_collection.key_map[key]
            ].parameters
            charge = electrostatics_collection.charges[key]

            # Build atom types
            system.atom_types[atom_type_name] = LennardJonesAtomType(
                name=_atom_atom_type_map[atom],
                bonding_type="",
                atomic_number=atom.atomic_number,
                mass=MASSES[atom.atomic_number],
                charge=unit.Quantity(0.0, unit.elementary_charge),
                particle_type="A",
                sigma=vdw_parameters["sigma"].to(unit.nanometer),
                epsilon=vdw_parameters["epsilon"].to(unit.kilojoule_per_mole),
            )

            _atom_atom_type_map[atom] = atom_type_name

    for unique_molecule_index in unique_molecule_map:
        unique_molecule = interchange.topology.molecule(unique_molecule_index)

        if unique_molecule.name == "":
            unique_molecule.name = "MOL" + str(unique_molecule_index)

        molecule = GROMACSMolecule(name=unique_molecule.name)

        for atom in unique_molecule.atoms:
            name = SYMBOLS[atom.atomic_number] if atom.name == "" else atom.name
            charge = (
                unit.Quantity(0.0, unit.elementary_charge)
                if atom.partial_charge is None
                else atom.partial_charge
            )

            molecule.atoms.append(
                GROMACSAtom(
                    index=unique_molecule.atom_index(atom) + 1,
                    name=name,
                    atom_type=_atom_atom_type_map[atom],
                    residue_index=atom.metadata.get("residue_number", 1),
                    residue_name=atom.metadata.get("residue_name", "RES"),
                    charge_group_number=1,
                    charge=charge,
                    mass=MASSES[atom.atomic_number],
                ),
            )

        _convert_bonds(molecule, unique_molecule, interchange)
        _convert_angles(molecule, unique_molecule, interchange)
        # pairs
        # dihedrals
        # settles?
        # constraints?

        system.molecule_types[unique_molecule.name] = molecule

        system.molecules[unique_molecule.name] = len(
            [
                molecule
                for molecule in interchange.topology.molecules
                if molecule.is_isomorphic_with(
                    [*interchange.topology.unique_molecules][0],
                )
            ],
        )

    system.positions = interchange.positions
    system.box = interchange.box

    return system


def _convert_bonds(
    molecule: GROMACSMolecule,
    unique_molecule: "Molecule",
    interchange: Interchange,
):
    collection = interchange["Bonds"]

    for bond in unique_molecule.bonds:
        molecule_indices = tuple(
            sorted(unique_molecule.atom_index(a) for a in bond.atoms),
        )
        topology_indices = tuple(
            sorted(interchange.topology.atom_index(atom) for atom in bond.atoms),
        )

        found_match = False
        for top_key in collection.key_map:
            top_key: TopologyKey  # type: ignore[no-redef]
            if top_key.atom_indices == topology_indices:
                pot_key = collection.key_map[top_key]
                found_match = True
                break
            elif top_key.atom_indices == topology_indices[::-1]:
                pot_key = collection.key_map[top_key]
                found_match = True
                break
            else:
                found_match = False

        if not found_match:
            print(
                f"Failed to find parameters for bond with topology indices {topology_indices}",
            )
            continue

        params = collection.potentials[pot_key].parameters

        molecule.bonds.append(
            GROMACSBond(
                atom1=molecule_indices[0] + 1,
                atom2=molecule_indices[1] + 1,
                function=1,
                length=params["length"].to(unit.nanometer),
                k=params["k"].to(unit.kilojoule_per_mole / unit.nanometer**2),
            ),
        )


def _convert_angles(
    molecule: GROMACSMolecule,
    unique_molecule: "Molecule",
    interchange: Interchange,
):
    collection = interchange["Angles"]

    for angle in unique_molecule.angles:
        topology_indices = tuple(interchange.topology.atom_index(a) for a in angle)
        molecule_indices = tuple(unique_molecule.atom_index(a) for a in angle)

        found_match = False
        for top_key in collection.key_map:
            top_key: TopologyKey  # type: ignore[no-redef]
            if top_key.atom_indices == topology_indices:
                pot_key = collection.key_map[top_key]
                found_match = True
                break
            elif top_key.atom_indices == topology_indices[::-1]:
                pot_key = collection.key_map[top_key]
                found_match = True
                break
            else:
                found_match = False

        if not found_match:
            print(
                f"Failed to find parameters for angle with topology indices {topology_indices}",
            )
            continue

        params = collection.potentials[pot_key].parameters

        molecule.angles.append(
            GROMACSAngle(
                atom1=molecule_indices[0] + 1,
                atom2=molecule_indices[1] + 1,
                atom3=molecule_indices[2] + 1,
                angle=params["angle"].to(unit.degree),
                k=params["k"].to(unit.kilojoule_per_mole / unit.radian**2),
            ),
        )
