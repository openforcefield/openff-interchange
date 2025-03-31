"""Classes used to represent GROMACS state."""

from typing import Annotated

from openff.toolkit import Quantity
from pydantic import (
    Field,
    PositiveInt,
    PrivateAttr,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)

from openff.interchange._annotations import _DistanceQuantity
from openff.interchange.pydantic import _BaseModel


def validate_particle_type(
    value: str,
    handler: ValidatorFunctionWrapHandler,
    info: ValidationInfo,
) -> str:
    """Validate the particle_type field."""
    # info.data is like the extra values argument in v1
    values = info.data

    if values["mass"].m == 0.0:
        assert value in ("D", "V"), 'Particle type must be "D" or "V" if massless'
    elif values["mass"].m > 0.0:
        assert value == "A", 'Particle type must be "A" if it has mass'

    return value


_ParticleType = Annotated[
    str,
    WrapValidator(validate_particle_type),
]


class GROMACSAtomType(_BaseModel):
    """Base class for GROMACS atom types."""

    name: str
    bonding_type: str = ""
    atomic_number: int
    mass: Quantity
    charge: Quantity
    particle_type: _ParticleType


class LennardJonesAtomType(GROMACSAtomType):
    """A Lennard-Jones atom type."""

    sigma: Quantity
    epsilon: Quantity


class GROMACSAtom(_BaseModel):
    """Base class for GROMACS atoms."""

    index: PositiveInt
    name: str
    atom_type: str  # Maybe this should point to an object
    residue_index: PositiveInt
    residue_name: str
    charge_group_number: PositiveInt
    charge: Quantity
    mass: Quantity


# Should the physical values (distance/angles) be float or Quantity?
class GROMACSVirtualSite(_BaseModel):
    """Base class for storing GROMACS virtual sites."""

    type: str
    name: str
    header_tag: Annotated[int, Field(strict=True, ge=2)]
    site: PositiveInt
    func: PositiveInt
    orientation_atoms: list[int]


class GROMACSVirtualSite2(GROMACSVirtualSite):
    """GROMACS virtual site type 2."""

    type: str = "2"
    header_tag: int = 2
    func: int = 1
    a: float


class GROMACSVirtualSite3(GROMACSVirtualSite):
    """GROMACS virtual site type 3."""

    type: str = "3"
    header_tag: int = 3
    func: int = 1
    a: float
    b: float


class GROMACSVirtualSite3fd(GROMACSVirtualSite):
    """GROMACS virtual site type 3fd."""

    type: str = "3fd"
    header_tag: int = 3
    func: int = 2
    a: float
    d: float


class GROMACSVirtualSite3fad(GROMACSVirtualSite):
    """GROMACS virtual site type 3fad."""

    type: str = "3fad"
    header_tag: int = 3
    func: int = 3
    theta: float
    d: float


class GROMACSVirtualSite3out(GROMACSVirtualSite):
    """GROMACS virtual site type 3out."""

    type: str = "3out"
    header_tag: int = 3
    func: int = 4
    a: float
    b: float
    c: float


class GROMACSVirtualSite4fdn(GROMACSVirtualSite):
    """GROMACS virtual site type 4fdn."""

    type: str = "4fdn"
    header_tag: int = 4
    func: int = 2
    a: float
    b: float
    c: float


class GROMACSBond(_BaseModel):
    """A GROMACS bond."""

    atom1: PositiveInt = Field(
        description="The GROMACS index of the first atom in the bond.",
    )
    atom2: PositiveInt = Field(
        description="The GROMACS index of the second atom in the bond.",
    )
    function: int = Field(1, description="The GROMACS bond function type.")
    length: Quantity
    k: Quantity


class GROMACSPair(_BaseModel):
    """A GROMACS pair."""

    atom1: PositiveInt = Field(
        description="The GROMACS index of the first atom in the pair.",
    )
    atom2: PositiveInt = Field(
        description="The GROMACS index of the second atom in the pair.",
    )


class GROMACSSettles(_BaseModel):
    """A settles-style constraint for water."""

    first_atom: PositiveInt = Field(
        description="The GROMACS index of the first atom in the water.",
    )

    oxygen_hydrogen_distance: _DistanceQuantity = Field(
        description="The fixed distance between the oxygen and hydrogen.",
    )

    hydrogen_hydrogen_distance: _DistanceQuantity = Field(
        description="The fixed distance between the oxygen and hydrogen.",
    )


class GROMACSExclusion(_BaseModel):
    """An Exclusion between an atom and other(s)."""

    # Extra exclusions within a molecule can be added manually in a [ exclusions ] section. Each
    # line should start with one atom index, followed by one or more atom indices. All non-bonded
    # interactions between the first atom and the other atoms will be excluded.

    first_atom: PositiveInt

    other_atoms: list[PositiveInt]


class GROMACSAngle(_BaseModel):
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
    angle: Quantity
    k: Quantity


class GROMACSDihedral(_BaseModel):
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

    phi: Quantity
    k: Quantity
    multiplicity: PositiveInt


class RyckaertBellemansDihedral(GROMACSDihedral):
    """A type 3 dihedral in GROMACS."""

    c0: Quantity
    c1: Quantity
    c2: Quantity
    c3: Quantity
    c4: Quantity
    c5: Quantity


class PeriodicImproperDihedral(GROMACSDihedral):
    """A type 4 dihedral in GROMACS."""

    phi: Quantity
    k: Quantity
    multiplicity: PositiveInt


class GROMACSMolecule(_BaseModel):
    """Base class for GROMACS molecules."""

    name: str
    nrexcl: int = Field(
        3,
        description="The farthest neighbor distance whose interactions should be excluded.",
    )

    atoms: list[GROMACSAtom] = Field(
        list(),
        description="The atoms in this molecule.",
    )
    pairs: list[GROMACSPair] = Field(
        list(),
        description="The pairs in this molecule.",
    )
    settles: list[GROMACSSettles] = Field(
        list(),
        description="The settles in this molecule.",
    )
    bonds: list[GROMACSBond] = Field(
        list(),
        description="The bonds in this molecule.",
    )
    angles: list[GROMACSAngle] = Field(
        list(),
        description="The angles in this molecule.",
    )
    dihedrals: list[GROMACSDihedral] = Field(
        list(),
        description="The dihedrals in this molecule.",
    )
    virtual_sites: list[GROMACSVirtualSite] = Field(
        list(),
        description="The virtual sites in this molecule.",
    )
    exclusions: list[GROMACSExclusion] = Field(
        list(),
        description="The exclusions in this molecule.",
    )

    # TODO: This can desync between system- and molecule-level data, it might be better
    #       to not store atom types at the system level, instead storing them at the
    #       molecule and grouping up to system level at write time
    _contained_atom_types: dict[str, LennardJonesAtomType] = PrivateAttr()


class GROMACSSystem(_BaseModel):
    """A GROMACS system. Adapted from Intermol."""

    positions: _DistanceQuantity | None = None
    box: _DistanceQuantity | None = None

    name: str = ""
    nonbonded_function: int = Field(
        1,
        ge=1,
        le=1,
        description="The nonbonded function.",
    )
    combination_rule: int = Field(
        2,
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
    atom_types: dict[str, LennardJonesAtomType] = Field(
        dict(),
        description="Atom types, keyed by name.",
    )
    molecule_types: dict[str, GROMACSMolecule] = Field(
        dict(),
        description="Molecule types, keyed by name.",
    )
    molecules: list[tuple[str, int]] = Field(
        list(),
        description="The number of each molecule type in the system, ordered topologically.",
    )

    @classmethod
    def from_files(cls, top_file, gro_file):
        """Parse a GROMACS topology file."""
        from openff.interchange.interop.gromacs._import._import import from_files

        return from_files(top_file, gro_file, cls=cls)

    def to_files(self, prefix: str, decimal: int = 3):
        """Write a GROMACS topology file."""
        from openff.interchange.interop.gromacs.export._export import GROMACSWriter

        writer = GROMACSWriter(
            system=self,
            top_file=f"{prefix}.top",
            gro_file=f"{prefix}.gro",
        )

        writer.to_top()
        writer.to_gro(decimal=decimal)

    def to_top(self, file: str):
        """Write a GROMACS topology file."""
        from openff.interchange.interop.gromacs.export._export import GROMACSWriter

        GROMACSWriter(
            system=self,
            top_file=file,
            gro_file="_.gro",
        ).to_top()

    def to_gro(self, file: str, decimal: int = 3):
        """Write a GROMACS coordinate file."""
        from openff.interchange.interop.gromacs.export._export import GROMACSWriter

        GROMACSWriter(
            system=self,
            top_file="_.gro",
            gro_file=file,
        ).to_gro(decimal=decimal)

    def remove_molecule_type(self, molecule_name: str, n_copies: int = 1):
        """Remove a molecule type from the system."""
        import numpy

        if molecule_name not in self.molecule_types:
            raise ValueError(
                f"The molecule type {molecule_name} is not present in this system.",
            )

        if n_copies != 1 or sum(_n_copies for name, _n_copies in self.molecules if name == molecule_name) != 1:
            raise NotImplementedError()

        n_atoms_before = sum(
            len(self.molecule_types[molecule_name].atoms) * n_copies
            for molecule_name, n_copies in self.molecules[
                : self.molecules.index(
                    (molecule_name, 1),
                )
            ]
        )

        if self.positions is not None:
            row_indices_to_delete = [
                *range(
                    n_atoms_before,
                    n_atoms_before + len(self.molecule_types[molecule_name].atoms),
                ),
            ]

            # Pint lacks __array_function__ needed here, so strip and then tag units
            self.positions = Quantity(
                numpy.delete(self.positions.m, row_indices_to_delete, axis=0),
                self.positions.units,
            )

        self.molecule_types.pop(molecule_name)
        self.molecules = [(name, n_copies) for name, n_copies in self.molecules if name != molecule_name]

    def add_molecule_type(self, molecule: GROMACSMolecule, n_copies: int):
        """Add a molecule type to the system."""
        if molecule.name in self.molecule_types:
            raise ValueError(
                f"The molecule type {molecule.name} is already present in this system.",
            )

        if len(molecule._contained_atom_types) == 0:
            raise ValueError(
                f"The molecule type {molecule.name} does not contain any atom types.",
            )

        for atom_type_name, atom_type in molecule._contained_atom_types.items():
            if atom_type_name in self.atom_types:
                raise ValueError(
                    f"An atom type {atom_type_name} is already present in this system.",
                )

            self.atom_types[atom_type_name] = atom_type

        self.molecule_types[molecule.name] = molecule
        self.molecules.append((molecule.name, n_copies))
