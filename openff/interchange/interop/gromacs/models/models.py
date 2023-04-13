"""Classes used to represent GROMACS state."""
from typing import Dict, List, Optional

from openff.models.models import DefaultModel
from openff.models.types import ArrayQuantity, FloatQuantity
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
    atom_type: str  # Maybe this should point to an object
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
    function: int = Field(1, const=True, description="The GROMACS bond function type.")
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


class GROMACSExclusion(DefaultModel):
    """An Exclusion between an atom and other(s)."""

    # Extra exclusions within a molecule can be added manually in a [ exclusions ] section. Each
    # line should start with one atom index, followed by one or more atom indices. All non-bonded
    # interactions between the first atom and the other atoms will be excluded.

    first_atom: PositiveInt

    other_atoms: List[PositiveInt]


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


# TODO: Subclasses could define their allowed "function type" as an extra runtime safeguard?
class PeriodicProperDihedral(GROMACSDihedral):
    """A type 1 dihedral in GROMACS."""

    phi: unit.Quantity
    k: unit.Quantity
    multiplicity: PositiveInt


class RyckaertBellemansDihedral(GROMACSDihedral):
    """A type 3 dihedral in GROMACS."""

    c0: unit.Quantity
    c1: unit.Quantity
    c2: unit.Quantity
    c3: unit.Quantity
    c4: unit.Quantity
    c5: unit.Quantity


class PeriodicImproperDihedral(GROMACSDihedral):
    """A type 4 dihedral in GROMACS."""

    phi: unit.Quantity
    k: unit.Quantity
    multiplicity: PositiveInt


class GROMACSMolecule(DefaultModel):
    """Base class for GROMACS molecules."""

    name: str
    nrexcl: int = Field(
        3,
        const=True,
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
    exclusions: List[GROMACSExclusion] = Field(
        list(),
        description="The exclusions in this molecule.",
    )


class GROMACSSystem(DefaultModel):
    """A GROMACS system. Adapted from Intermol."""

    positions: Optional[ArrayQuantity] = None
    box: Optional[ArrayQuantity] = None

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
    def from_files(cls, top_file, gro_file):
        """Parse a GROMACS topology file."""
        from openff.interchange.interop.gromacs._import._import import from_files

        return from_files(top_file, gro_file, cls=cls)

    def to_top(self, file):
        """Write a GROMACS topology file."""
        from openff.interchange.interop.gromacs.export._export import to_top

        return to_top(self, file)
