"""Classes used to represent GROMACS state."""

from openff.models.models import DefaultModel
from openff.models.types import ArrayQuantity, FloatQuantity
from openff.toolkit import Quantity

from openff.interchange._pydantic import (
    Field,
    PositiveInt,
    PrivateAttr,
    conint,
    validator,
)


class GROMACSAtomType(DefaultModel):
    """Base class for GROMACS atom types."""

    name: str
    bonding_type: str = ""
    atomic_number: int
    mass: Quantity
    charge: Quantity
    particle_type: str

    @validator("particle_type")
    def validate_particle_type(
        cls,
        v: str,
        values,
    ) -> str:
        if values["mass"].m == 0.0:
            assert v in ("D", "V"), 'Particle type must be "D" or "V" if massless'
        elif values["mass"].m > 0.0:
            assert v == "A", 'Particle type must be "A" if it has mass'

        return v


class LennardJonesAtomType(GROMACSAtomType):
    """A Lennard-Jones atom type."""

    sigma: Quantity
    epsilon: Quantity


class GROMACSAtom(DefaultModel):
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
class GROMACSVirtualSite(DefaultModel):
    """Base class for storing GROMACS virtual sites."""

    type: str
    name: str
    header_tag: conint(ge=2)
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


class GROMACSBond(DefaultModel):
    """A GROMACS bond."""

    atom1: PositiveInt = Field(
        description="The GROMACS index of the first atom in the bond.",
    )
    atom2: PositiveInt = Field(
        description="The GROMACS index of the second atom in the bond.",
    )
    function: int = Field(1, const=True, description="The GROMACS bond function type.")
    length: Quantity
    k: Quantity


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

    other_atoms: list[PositiveInt]


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
    angle: Quantity
    k: Quantity


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


class GROMACSMolecule(DefaultModel):
    """Base class for GROMACS molecules."""

    name: str
    nrexcl: int = Field(
        3,
        const=True,
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


class GROMACSSystem(DefaultModel):
    """A GROMACS system. Adapted from Intermol."""

    positions: ArrayQuantity | None = None
    box: ArrayQuantity | None = None

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
    molecules: dict[str, int] = Field(
        dict(),
        description="The number of each molecule type in the system, keyed by the name of each molecule.",
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

        if n_copies > self.molecules[molecule_name]:
            raise ValueError(
                f"Cannot remove {n_copies} copies of {molecule_name} from this system "
                f"because only {self.molecules[molecule_name]} are present.",
            )

        if n_copies != 1 or self.molecules[molecule_name] != 1:
            raise NotImplementedError()

        molecule_names = [*self.molecules.keys()]
        molecules_before = molecule_names[: molecule_names.index(molecule_name)]
        n_atoms_before = sum(
            len(self.molecule_types[name].atoms) * self.molecules[name]
            for name in molecules_before
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
        self.molecules[molecule_name] -= n_copies

        if self.molecules[molecule_name] == 0:
            self.molecules.pop(molecule_name)

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
        self.molecules[molecule.name] = n_copies
