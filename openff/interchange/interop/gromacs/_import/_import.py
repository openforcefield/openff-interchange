from typing import Tuple

from openff.models.models import DefaultModel
from openff.units import unit
from pydantic import Field, conint


class GROMACSSystem(DefaultModel):
    """A GROMACS system. Adapted from Intermol."""

    nonbonded_function: conint(ge=1, le=2) = Field(
        1,
        description="The nonbonded function.",
    )
    combination_rule: conint(ge=1, le=3) = Field(1, description="The combination rule.")
    gen_pairs: bool = Field(True, description="Whether or not to generate pairs.")
    vdw_14: float = Field(
        0.5,
        description="The 1-4 scaling factor for dispersion interactions.",
    )
    coul_14: float = Field(
        0.833333,
        description="The 1-4 scaling factor for electrostatic interactions.",
    )

    @classmethod
    def from_top(cls, file):
        """
        Parse a GROMACS topology file. Adapted from Intermol.

        https://github.com/shirtsgroup/InterMol/blob/v0.1.2/intermol/gromacs/gromacs_parser.py
        """
        with open(file) as f:
            for line in f:
                stripped = line.strip()

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

                elif current_directive == "atomtypes":
                    atom_type = _process_atomtype(line)  # noqa

                elif current_directive == "moleculetype":
                    molecule_type = _process_moleculetype(line)  # noqa

                elif current_directive == "atoms":
                    atom = _process_atom(line)  # noqa

                elif current_directive == "pairs":
                    pair = _process_pair(line)  # noqa

                elif current_directive == "bonds":
                    bond = _process_bond(line)  # noqa

                elif current_directive == "angles":
                    angle = _process_angle(line)  # noqa

                elif current_directive == "dihedrals":
                    dihedral = _process_dihedral(line)  # noqa

                elif current_directive == "molecules":
                    molecule = _process_molecule(line)  # noqa

                elif current_directive in ["settles", "system", "exclusions"]:
                    pass

                else:
                    raise ValueError(f"Invalid directive {current_directive}")

        system = cls(
            nonbonded_function=nonbonded_function,
            combination_rule=combination_rule,
            gen_pairs=gen_pairs,
            vdw_14=vdw_14,
            coul_14=coul_14,
        )

        return system


def _process_defaults(line: str) -> Tuple[int, int, str, float, float]:
    split = line.split()

    nonbonded_function = int(split[0])
    combination_rule = int(split[1])
    gen_pairs = split[2]
    lj_14 = float(split[3])
    coul_14 = float(split[4])

    return nonbonded_function, combination_rule, gen_pairs, lj_14, coul_14


def _process_atomtype(
    line: str,
) -> Tuple[str, str, int, float, unit.Quantity, str, unit.Quantity, unit.Quantity]:
    split = line.split()

    atom_type = split[0]
    bonding_type = split[1]

    atomic_number = int(split[2])
    mass = float(split[3])

    charge = unit.Quantity(float(split[4]), unit.elementary_charge)

    particle_type = split[5]

    if particle_type == "A":
        sigma = unit.Quantity(float(split[6]), unit.nanometer)
        epsilon = unit.Quantity(float(split[7]), unit.kilojoule_per_mole)
    else:
        raise ValueError(f"Particle type must be A, parsed {particle_type}.")

    return (
        atom_type,
        bonding_type,
        atomic_number,
        mass,
        charge,
        particle_type,
        sigma,
        epsilon,
    )


def _process_moleculetype(line: str) -> Tuple[str, int]:
    split = line.split()

    molecule_type = split[0]
    nrexcl = int(split[1])

    return molecule_type, nrexcl


def _process_atom(
    line: str,
) -> Tuple[int, str, int, str, str, int, unit.Quantity, unit.Quantity]:
    split = line.split()

    atom_number = int(split[0])
    atom_type = split[1]
    residue_number = int(split[2])
    residue_name = split[3]
    atom_name = split[4]
    charge_group_number = int(split[5])
    charge = unit.Quantity(float(split[6]), unit.elementary_charge)
    mass = unit.Quantity(float(split[7]), unit.amu)

    return (
        atom_number,
        atom_type,
        residue_number,
        residue_name,
        atom_name,
        charge_group_number,
        charge,
        mass,
    )


def _process_pair(line: str):
    pass


def _process_bond(line: str) -> Tuple[int, int, int, unit.Quantity, unit.Quantity]:
    #    ;   ai     aj funct  r               k
    #      1       2 1       1.21830000e-01    5.33627360e+05
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
    else:
        raise ValueError(f"Bond function must be 1, parsed {bond_function}.")

    return atom1, atom2, bond_function, bond_length, bond_k


def _process_angle(
    line: str,
) -> Tuple[int, int, int, int, unit.Quantity, unit.Quantity]:
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

    return atom1, atom2, atom3, angle_function, angle, k


def _process_dihedral(
    line: str,
) -> Tuple[int, int, int, int, int, unit.Quantity, unit.Quantity]:
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
        return atom1, atom2, atom3, atom4, dihedral_function, phi, k, multiplicity
    else:
        # raise ValueError(f"Dihedral function must be 1, parsed {dihedral_function}.")
        print(f"Dihedral function must be 1, parsed {dihedral_function}.")
        return


def _process_molecule(line: str) -> Tuple[str, int]:
    split = line.split()

    molecule_name = split[0]
    number_of_molecules = int(split[1])

    return molecule_name, number_of_molecules
