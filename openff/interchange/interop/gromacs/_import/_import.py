from typing import Dict, List, Literal, Optional, Tuple

from openff.models.models import DefaultModel
from openff.models.types import FloatQuantity
from openff.units import unit
from pydantic import Field, PositiveInt


class GROMACSAtomType(DefaultModel):
    """Base class for GROMACS atom types."""

    name: str
    bonding_type: str
    atomic_number: Optional[PositiveInt]
    mass: unit.Quantity
    charge: unit.Quantity
    particle_type: str


class LennardJonesAtomType(GROMACSAtomType):
    """A Lennard-Jones atom type."""

    sigma: unit.Quantity
    epsilon: unit.Quantity


class GROMACSAtom(DefaultModel):
    """Base class for GROMACS atoms."""

    index: PositiveInt
    name: str
    atom_type: str
    residue_index: PositiveInt
    residue_name: str
    charge_group_number: PositiveInt
    charge: unit.Quantity
    mass: unit.Quantity


class GROMACSBond(DefaultModel):
    """A GROMACS bond."""

    atom1: PositiveInt = Field(
        description="The GROMACS index of the first atom in the bond.",
    )
    atom2: PositiveInt = Field(
        description="The GROMACS index of the second atom in the bond.",
    )
    function: Literal[1]
    length: unit.Quantity
    k: unit.Quantity


class GROMACSPair(DefaultModel):
    """A GROMACS pair."""

    atom1: PositiveInt = Field(
        description="The GROMACS index of the first atom in the pair.",
    )
    atom2: PositiveInt = Field(
        description="The GROMACS index of the second atom in the pair.",
    )


class GROMACSSettles(DefaultModel):
    """A settles-style constraint for water."""

    first_atom: PositiveInt = Field(
        description="The GROMACS index of the first atom in the water.",
    )

    oxygen_hydrogen_distance: FloatQuantity = Field(
        description="The fixed distance between the oxygen and hydrogen.",
    )

    hydrogen_hydrogen_distance: FloatQuantity = Field(
        description="The fixed distance between the oxygen and hydrogen.",
    )


class GROMACSAngle(DefaultModel):
    """A GROMACS angle."""

    atom1: PositiveInt = Field(
        description="The GROMACS index of the first atom in the angle.",
    )
    atom2: PositiveInt = Field(
        description="The GROMACS index of the second atom in the angle.",
    )
    atom3: PositiveInt = Field(
        description="The GROMACS index of the third atom in the angle.",
    )
    angle: unit.Quantity
    k: unit.Quantity


class GROMACSDihedral(DefaultModel):
    """A GROMACS dihedral."""

    atom1: PositiveInt = Field(
        description="The GROMACS index of the first atom in the dihedral.",
    )
    atom2: PositiveInt = Field(
        description="The GROMACS index of the second atom in the dihedral.",
    )
    atom3: PositiveInt = Field(
        description="The GROMACS index of the third atom in the dihedral.",
    )
    atom4: PositiveInt = Field(
        description="The GROMACS index of the fourth atom in the dihedral.",
    )
    phi: unit.Quantity
    k: unit.Quantity
    multiplicity: PositiveInt


class GROMACSMolecule(DefaultModel):
    """Base class for GROMACS molecules."""

    name: str
    nrexcl: Literal[3] = Field(
        3,
        description="The farthest neighbor distance whose interactions should be excluded.",
    )

    atoms: List[GROMACSAtom] = Field(
        list(),
        description="The atoms in this molecule.",
    )
    pairs: List[GROMACSPair] = Field(
        list(),
        description="The pairs in this molecule.",
    )
    settles: List[GROMACSSettles] = Field(
        list(),
        description="The settles in this molecule.",
    )
    bonds: List[GROMACSBond] = Field(
        list(),
        description="The bonds in this molecule.",
    )
    angles: List[GROMACSAngle] = Field(
        list(),
        description="The angles in this molecule.",
    )
    dihedrals: List[GROMACSDihedral] = Field(
        list(),
        description="The dihedrals in this molecule.",
    )


class GROMACSSystem(DefaultModel):
    """A GROMACS system. Adapted from Intermol."""

    name: str = ""
    nonbonded_function: int = Field(
        1,
        ge=1,
        le=2,
        description="The nonbonded function.",
    )
    combination_rule: int = Field(
        1,
        ge=1,
        le=3,
        description="The combination rule.",
    )
    gen_pairs: bool = Field(True, description="Whether or not to generate pairs.")
    vdw_14: float = Field(
        0.5,
        description="The 1-4 scaling factor for dispersion interactions.",
    )
    coul_14: float = Field(
        0.833333,
        description="The 1-4 scaling factor for electrostatic interactions.",
    )
    atom_types: Dict[str, GROMACSAtomType] = Field(
        dict(),
        description="Atom types, keyed by name.",
    )
    molecule_types: Dict[str, GROMACSMolecule] = Field(
        dict(),
        description="Molecule types, keyed by name.",
    )
    molecules: Dict[str, int] = Field(
        dict(),
        description="The number of each molecule type in the system, keyed by the name of each molecule.",
    )

    @classmethod
    def from_top(cls, file):
        """
        Parse a GROMACS topology file. Adapted from Intermol.

        https://github.com/shirtsgroup/InterMol/blob/v0.1.2/intermol/gromacs/gromacs_parser.py
        """
        with open(file) as f:
            for line in f:
                stripped = line.split(";")[0].strip()

                if len(stripped) == 0:
                    continue

                if stripped.startswith(";"):
                    continue

                if stripped.startswith("["):
                    if not len(stripped.split()) == 3 and stripped.endswith("]"):
                        raise ValueError("Invalid GROMACS topology file")

                    current_directive = stripped[1:-1].strip()

                    continue

                if current_directive == "defaults":
                    (
                        nonbonded_function,
                        combination_rule,
                        gen_pairs,
                        vdw_14,
                        coul_14,
                    ) = _process_defaults(line)

                    system = cls(
                        nonbonded_function=nonbonded_function,
                        combination_rule=combination_rule,
                        gen_pairs=gen_pairs,
                        vdw_14=vdw_14,
                        coul_14=coul_14,
                    )

                elif current_directive == "atomtypes":
                    atom_type = _process_atomtype(line)
                    system.atom_types[atom_type.name] = atom_type

                elif current_directive == "moleculetype":
                    molecule_type = _process_moleculetype(line)
                    system.molecule_types[molecule_type.name] = molecule_type

                    current_molecule = molecule_type.name

                elif current_directive == "atoms":
                    system.molecule_types[current_molecule].atoms.append(
                        _process_atom(line),
                    )

                elif current_directive == "pairs":
                    pair = _process_pair(line)  # noqa

                elif current_directive == "settles":
                    system.molecule_types[current_molecule].settles.append(
                        _process_settles(line),
                    )

                elif current_directive == "bonds":
                    system.molecule_types[current_molecule].bonds.append(
                        _process_bond(line),
                    )

                elif current_directive == "angles":
                    system.molecule_types[current_molecule].angles.append(
                        _process_angle(line),
                    )

                elif current_directive == "dihedrals":
                    system.molecule_types[current_molecule].dihedrals.append(
                        _process_dihedral(line),
                    )

                elif current_directive == "system":
                    system.name = _process_system(line)

                elif current_directive == "molecules":
                    molecule_name, number_of_copies = _process_molecule(line)

                    system.molecules[molecule_name] = number_of_copies

                elif current_directive in ["exclusions"]:
                    pass

                else:
                    raise ValueError(f"Invalid directive {current_directive}")

        return system


def _process_defaults(line: str) -> Tuple[int, int, str, float, float]:
    split = line.split()

    nonbonded_function = int(split[0])

    if nonbonded_function != 1:
        raise ValueError("Only LJ nonbonded functions are supported.")

    combination_rule = int(split[1])

    if combination_rule != 2:
        raise ValueError("Only Lorentz-Berthelot combination rules are supported.")

    gen_pairs = split[2]
    lj_14 = float(split[3])
    coul_14 = float(split[4])

    return nonbonded_function, combination_rule, gen_pairs, lj_14, coul_14


def _process_atomtype(
    line: str,
) -> GROMACSAtomType:
    split = line.split()

    # The second (bonding type) and third (atomic number) columns are optional. Handle these cases
    # explicitly by assuming that the ptype column is always a 1-length string.
    # See https://github.com/shirtsgroup/InterMol/blob/v0.1.2/intermol/gromacs/gromacs_parser.py#L1357-L1368

    if len(split[3]) == 1:
        # Bonded type and atomic number are both missing.
        split.insert(1, None)
        split.insert(1, None)

    elif len(split[4]) == 1 and len(split[5]) >= 1:
        if split[1][0].isalpha():
            # Atomic number is missing.
            split.insert(2, None)
        else:
            # Bonded type is missing.
            split.insert(1, None)

    atom_type = split[0]
    bonding_type = split[1]

    atomic_number = int(split[2]) if split[2] is not None else None
    mass = unit.Quantity(float(split[3]), unit.dalton)

    charge = unit.Quantity(float(split[4]), unit.elementary_charge)

    particle_type = split[5]

    if particle_type == "A":
        sigma = unit.Quantity(float(split[6]), unit.nanometer)
        epsilon = unit.Quantity(float(split[7]), unit.kilojoule_per_mole)
    else:
        raise ValueError(f"Particle type must be A, parsed {particle_type}.")

    return LennardJonesAtomType(
        name=atom_type,
        bonding_type=bonding_type,
        atomic_number=atomic_number,
        mass=mass,
        charge=charge,
        particle_type=particle_type,
        sigma=sigma,
        epsilon=epsilon,
    )


def _process_moleculetype(line: str) -> GROMACSMolecule:
    split = line.split()

    molecule_type = split[0]
    nrexcl = int(split[1])

    return GROMACSMolecule(name=molecule_type, nrexcl=nrexcl)


def _process_atom(
    line: str,
) -> GROMACSAtom:
    split = line.split()

    atom_number = int(split[0])
    atom_type = split[1]
    residue_number = int(split[2])
    residue_name = split[3]
    atom_name = split[4]
    charge_group_number = int(split[5])
    charge = unit.Quantity(float(split[6]), unit.elementary_charge)
    mass = unit.Quantity(float(split[7]), unit.amu)

    return GROMACSAtom(
        index=atom_number,
        atom_type=atom_type,
        name=atom_name,
        residue_index=residue_number,
        residue_name=residue_name,
        charge_group_number=charge_group_number,
        charge=charge,
        mass=mass,
    )


def _process_pair(line: str) -> GROMACSPair:
    split = line.split()

    atom1 = int(split[0])
    atom2 = int(split[1])

    nonbonded_function = int(split[2])

    if nonbonded_function == 1:
        return GROMACSPair(
            atom1=atom1,
            atom2=atom2,
        )
    else:
        raise ValueError(
            f"Nonbonded function must be 1, parsed a pair with {nonbonded_function}.",
        )


def _process_settles(line: str) -> GROMACSSettles:
    split = line.split()

    first_atom = int(split[0])

    oxygen_hydrogen_distance = unit.Quantity(float(split[2]), unit.nanometer)
    hydrogen_hydrogen_distance = unit.Quantity(float(split[3]), unit.nanometer)

    return GROMACSSettles(
        first_atom=first_atom,
        oxygen_hydrogen_distance=oxygen_hydrogen_distance,
        hydrogen_hydrogen_distance=hydrogen_hydrogen_distance,
    )


def _process_bond(line: str) -> GROMACSBond:
    split = line.split()

    atom1 = int(split[0])
    atom2 = int(split[1])

    bond_function = int(split[2])

    if bond_function == 1:
        bond_length = unit.Quantity(float(split[3]), unit.nanometer)
        bond_k = unit.Quantity(
            float(split[4]),
            unit.kilojoule_per_mole / unit.nanometer**2,
        )

        return GROMACSBond(
            atom1=atom1,
            atom2=atom2,
            function=bond_function,
            length=bond_length,
            k=bond_k,
        )

    else:
        raise ValueError(f"Bond function must be 1, parsed {bond_function}.")


def _process_angle(
    line: str,
) -> GROMACSAngle:
    split = line.split()

    atom1 = int(split[0])
    atom2 = int(split[1])
    atom3 = int(split[2])

    angle_function = int(split[3])

    if angle_function == 1:
        angle = unit.Quantity(float(split[4]), unit.degrees)
        k = unit.Quantity(float(split[5]), unit.kilojoule_per_mole)
    else:
        raise ValueError(f"Angle function must be 1, parsed {angle_function}.")

    return GROMACSAngle(
        atom1=atom1,
        atom2=atom2,
        atom3=atom3,
        angle=angle,
        k=k,
    )


def _process_dihedral(
    line: str,
) -> GROMACSDihedral:
    split = line.split()

    atom1 = int(split[0])
    atom2 = int(split[1])
    atom3 = int(split[2])
    atom4 = int(split[3])

    dihedral_function = int(split[4])

    if dihedral_function == 1:
        phi = unit.Quantity(float(split[5]), unit.degrees)
        k = unit.Quantity(float(split[6]), unit.kilojoule_per_mole)
        multiplicity = int(float(split[7]))

        return GROMACSDihedral(
            atom1=atom1,
            atom2=atom2,
            atom3=atom3,
            atom4=atom4,
            phi=phi,
            k=k,
            multiplicity=multiplicity,
        )

    else:
        # raise ValueError(f"Dihedral function must be 1, parsed {dihedral_function}.")
        print(f"Dihedral function must be 1, parsed {dihedral_function}.")
        return  # type: ignore[return-value]


def _process_molecule(line: str) -> Tuple[str, int]:
    split = line.split()

    molecule_name = split[0]
    number_of_molecules = int(split[1])

    return molecule_name, number_of_molecules


def _process_system(line: str) -> str:
    split = line.split()

    system_name = split[0]

    return system_name
