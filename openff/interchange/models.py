"""Custom Pydantic models."""

import abc
from typing import Any, Literal, cast

from pydantic import Field

from openff.interchange.pydantic import _BaseModel


class TopologyKey(_BaseModel, abc.ABC):
    """
    A unique identifier of a segment of a chemical topology.

    These refer to a single portion of a chemical graph, i.e. a single valence term,
    (a bond, angle, or dihedral) or a single atom. These target only the information in
    the chemical graph and do not store physics parameters. For example, a TopologyKey
    corresponding to a bond would store the indices of the two atoms that compose the
    bond, but not the force constant or equilibrium bond length as determined by the
    force field.

    Topology keys compare equal to (and hash the same as) tuples of their atom
    indices as long as their other fields are `None`.

    Examples
    --------
    Create a ``TopologyKey`` identifying some specific angle

    .. code-block:: pycon

        >>> from openff.interchange.models import TopologyKey
        >>> this_angle = TopologyKey(atom_indices=(2, 1, 3))
        >>> this_angle
        TopologyKey with atom indices (2, 1, 3)

    Create a ``TopologyKey`` indentifying just one atom

    .. code-block:: pycon

        >>> this_atom = TopologyKey(atom_indices=(4,))
        >>> this_atom
        TopologyKey with atom indices (4,)

    Compare a ``TopologyKey`` to a tuple containing the atom indices

    .. code-block:: pycon

        >>> key = TopologyKey(atom_indices=(0, 1))
        >>> key == (0, 1)
        True

    Index into a dictionary with a tuple

    .. code-block:: pycon

        >>> d = {TopologyKey(atom_indices=(0, 1)): "some_bond"}
        >>> d[0, 1]
        'some_bond'

    """

    atom_indices: tuple[int, ...] = Field(
        description="The indices of the atoms occupied by this interaction",
    )

    def _tuple(self) -> tuple:
        """Tuple representation of this key, which is overriden by most child classes."""
        return tuple(self.atom_indices)

    def __hash__(self) -> int:
        return hash(self._tuple())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple):
            return self._tuple() == other
        elif isinstance(other, TopologyKey):
            return self._tuple() == other._tuple()
        else:
            return NotImplemented

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} with atom indices {self.atom_indices}"


class BondKey(TopologyKey):
    """
    A unique identifier of the atoms associated in a bond potential.

    Examples
    --------
    Index into a dictionary with a tuple

    .. code-block:: pycon

        >>> d = {
        ...     BondKey(atom_indices=(0, 1)): "some_bond",
        ...     BondKey(atom_indices=(1, 2), bond_order=1.5): "some_other_bond",
        ... }
        >>> d[0, 1]
        'some_bond'
        >>> d[(1, 2), 1.5]
        'some_other_bond'

    """

    atom_indices: tuple[int, int] = Field(
        description="The indices of the atoms occupied by this interaction",
    )

    bond_order: float | None = Field(
        None,
        description=(
            "If this key represents as topology component subject to interpolation between "
            "multiple parameters(s), the bond order determining the coefficients of the wrapped "
            "potentials."
        ),
    )

    def _tuple(self) -> tuple[int, int] | tuple[tuple[int, int], float]:
        if self.bond_order is None:
            return cast(tuple[int, int], self.atom_indices)
        else:
            return (
                cast(
                    tuple[int, int],
                    self.atom_indices,
                ),
                float(self.bond_order),
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} with atom indices {self.atom_indices}"
            f"{'' if self.bond_order is None else ', bond order ' + str(self.bond_order)}"
        )


class AngleKey(TopologyKey):
    """
    A unique identifier of the atoms associated in an angle potential.

    Examples
    --------
    Index into a dictionary with a tuple

    .. code-block:: pycon

        >>> d = {AngleKey(atom_indices=(0, 1, 2)): "some_angle"}
        >>> d[0, 1, 2]
        'some_angle'

    """

    atom_indices: tuple[int, int, int] = Field(
        description="The indices of the atoms occupied by this interaction",
    )

    def _tuple(self) -> tuple[int, int, int]:
        return cast(tuple[int, int, int], self.atom_indices)


class ProperTorsionKey(TopologyKey):
    """
    A unique identifier of the atoms associated in a proper torsion potential.

    Examples
    --------
    Index into a dictionary with a tuple

    .. code-block:: pycon

        >>> d = {
        ...     ProperTorsionKey(atom_indices=(0, 1, 2, 3)): "torsion1",
        ...     ProperTorsionKey(atom_indices=(0, 1, 2, 3), mult=2): "torsion2",
        ...     ProperTorsionKey(atom_indices=(5, 6, 7, 8), mult=2, phase=0.78, bond_order=1.5): "torsion3",
        ... }
        >>> d[0, 1, 2, 3]
        'torsion1'
        >>> d[(0, 1, 2, 3), 2, None, None]
        'torsion2'
        >>> d[(5, 6, 7, 8), 2, 0.78, 1.5]
        'torsion3'

    """

    atom_indices: tuple[int, int, int, int] = Field(
        description="The indices of the atoms occupied by this interaction",
    )

    mult: int | None = Field(
        None,
        description="The index of this duplicate interaction",
    )

    phase: float | None = Field(
        None,
        description="If this key represents as topology component subject to interpolation between "
        "multiple parameters(s), the phase determining the coefficients of the wrapped "
        "potentials.",
    )

    bond_order: float | None = Field(
        None,
        description=(
            "If this key represents as topology component subject to interpolation between "
            "multiple parameters(s), the bond order determining the coefficients of the wrapped "
            "potentials."
        ),
    )

    def _tuple(
        self,
    ) -> (
        tuple[int, int, int, int]
        | tuple[
            tuple[int, int, int, int],
            int | None,
            float | None,
            float | None,
        ]
    ):
        if self.mult is None and self.phase is None and self.bond_order is None:
            return cast(tuple[int, int, int, int], self.atom_indices)
        else:
            return (
                cast(
                    tuple[int, int, int, int],
                    self.atom_indices,
                ),
                self.mult,
                self.phase,
                self.bond_order,
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} with atom indices {self.atom_indices}"
            f"{'' if self.mult is None else ', mult ' + str(self.mult)}"
            f"{'' if self.bond_order is None else ', bond order ' + str(self.bond_order)}"
        )


class ImproperTorsionKey(ProperTorsionKey):
    """
    A unique identifier of the atoms associated in an improper torsion potential.

    The central atom is the second atom in the `atom_indices` tuple, or accessible via `get_central_atom_index`.

    Examples
    --------
    Index into a dictionary with a tuple

    .. code-block:: pycon

        >>> d = {
        ...     ImproperTorsionKey(atom_indices=(0, 1, 2, 3)): "torsion1",
        ...     ImproperTorsionKey(atom_indices=(0, 1, 2, 3), mult=2): "torsion2",
        ...     ImproperTorsionKey(atom_indices=(5, 6, 7, 8), mult=2, phase=0.78, bond_order=1.5): "torsion3",
        ... }
        >>> d[0, 1, 2, 3]
        'torsion1'
        >>> d[(0, 1, 2, 3), 2, None, None]
        'torsion2'
        >>> d[(5, 6, 7, 8), 2, 0.78, 1.5]
        'torsion3'

    """

    def get_central_atom_index(self) -> int:
        """Get the index of the central atom of this improper torsion."""
        return self.atom_indices[1]


class LibraryChargeTopologyKey(_BaseModel):
    """
    A unique identifier of the atoms associated with a library charge.
    """

    # TODO: Store all atoms associated with this charge?
    # TODO: Is there an upper bound on the number of atoms that can be associated with a LibraryChargeType?
    # TODO: Eventually rename this for coherence with `TopologyKey`
    this_atom_index: int

    @property
    def atom_indices(self) -> tuple[int]:
        """Alias for `this_atom_index`."""
        return (self.this_atom_index,)

    def __hash__(self) -> int:
        return hash((self.this_atom_index,))

    def __eq__(self, other) -> bool:
        return super().__eq__(other) or other == self.this_atom_index


class SingleAtomChargeTopologyKey(LibraryChargeTopologyKey):
    """
    Shim class for storing the result of charge_from_molecules.
    """

    extras: dict = dict()  # noqa: RUF012


class ChargeModelTopologyKey(_BaseModel):
    """Subclass of `TopologyKey` for use with charge models only."""

    this_atom_index: int
    partial_charge_method: str

    @property
    def atom_indices(self) -> tuple[int]:
        """Alias for `this_atom_index`."""
        return (self.this_atom_index,)

    def __hash__(self) -> int:
        return hash((self.this_atom_index, self.partial_charge_method))


class ChargeIncrementTopologyKey(_BaseModel):
    """Subclass of `TopologyKey` for use with charge increments only."""

    # TODO: Eventually rename this for coherence with `TopologyKey`
    this_atom_index: int
    other_atom_indices: tuple[int, ...]

    @property
    def atom_indices(self) -> tuple[int]:
        """Alias for `this_atom_index`."""
        return (self.this_atom_index,)

    def __hash__(self) -> int:
        return hash((self.this_atom_index, self.other_atom_indices))


class BaseVirtualSiteKey(TopologyKey):
    # TODO: Overriding the attribute of a parent class is clumsy, but less grief than
    #       having this not inherit from `TopologyKey`. It might be useful to just have
    #       orientation_atom_indices point to the same thing.
    atom_indices: tuple[int] | None = None  # type: ignore[assignment]

    orientation_atom_indices: tuple[int, ...] = Field(
        description="The indices of the 'orientation atoms' which are used to define the position "
        "of this virtual site. The first atom is the 'parent atom' which defines which atom the "
        "virtual site is 'attached' to.",
    )
    type: str = Field(description="The type of this virtual site parameter.")
    name: str = Field(description="The name of this virtual site parameter.")

    def __hash__(self) -> int:
        return hash(
            (
                self.orientation_atom_indices,
                self.name,
                self.type,
            ),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} with orientation atom indices {self.orientation_atom_indices}"


class ImportedVirtualSiteKey(BaseVirtualSiteKey):
    """
    A unique identifier of a virtual site in the scope of a chemical topology.

    Meant to be used with data imported from OpenMM or other engines.

    Use the engine-specific identifier, like `openmm.ThreeParticleAverageSite`, in the "type" field.
    """

    pass


class SMIRNOFFVirtualSiteKey(BaseVirtualSiteKey):
    """A unique identifier of a SMIRNOFF virtual site in the scope of a chemical topology."""

    match: Literal["once", "all_permutations"] = Field(
        description="The `match` attribute of the associated virtual site type",
    )

    def __hash__(self) -> int:
        return hash(
            (
                self.orientation_atom_indices,
                self.name,
                self.type,
                self.match,
            ),
        )


VirtualSiteKey = SMIRNOFFVirtualSiteKey


class PotentialKey(_BaseModel):
    """
    A unique identifier of an instance of physical parameters as applied to a segment of a chemical topology.

    These refer to a single term in a force field as applied to a single segment of a chemical
    topology, i.e. a single atom or dihedral. For example, a PotentialKey corresponding to a
    bond would store the the force constant and the equilibrium bond length as determined by
    the force field. These keys to not have direct knowledge of where in a topology they have been
    applied.

    Examples
    --------
    Create a PotentialKey corresponding to the parameter with id `b55` in OpenFF "Parsley" 1.0.0

    .. code-block:: pycon

        >>> from openff.interchange.models import PotentialKey
        >>> from openff.toolkit import ForceField
        >>> parsley = ForceField("openff-1.0.0.offxml")
        >>> param = parsley["Bonds"].get_parameter({"id": "b55"})[0]
        >>> bond_55 = PotentialKey(id=param.smirks)
        >>> bond_55
        PotentialKey associated with handler 'None' with id '[#16X4,#16X3:1]-[#8X2:2]'

    Create a PotentialKey corresponding to the angle parameters in OPLS-AA defined
    between atom types opls_135, opls_135, and opls_140

    .. code-block:: pycon

        >>> oplsaa_angle = PotentialKey(id="opls_135-opls_135-opls_140")
        >>> oplsaa_angle
        PotentialKey associated with handler 'None' with id 'opls_135-opls_135-opls_140'

    """

    id: str = Field(
        ...,
        description="A unique identifier of this potential, i.e. a SMARTS pattern or an atom type",
    )
    mult: int | None = Field(
        None,
        description="The index of this duplicate interaction",
    )
    associated_handler: str | None = Field(
        None,
        description="The type of handler this potential key is associated with, "
        "i.e. 'Bonds', 'vdW', or 'LibraryCharges",
    )
    bond_order: float | None = Field(
        None,
        description="If this is a key to a WrappedPotential interpolating multiple parameter(s), "
        "the bond order determining the coefficients of the wrapped potentials.",
    )
    virtual_site_type: str | None = Field(
        None,
        description="The 'type' of virtual site (i.e. `BondCharge`) this parameter is associated with.",
    )
    cosmetic_attributes: dict[str, Any] = Field(
        dict(),
        description="A dictionary of cosmetic attributes associated with this potential key.",
    )

    def __hash__(self) -> int:
        return hash((self.id, self.mult, self.associated_handler, self.bond_order))

    def __repr__(self) -> str:
        return (
            f"PotentialKey associated with handler '{self.associated_handler}' with id '{self.id}'"
            f"{'' if self.mult is None else ', mult ' + str(self.mult)}"
            f"{'' if self.bond_order is None else ', bond order ' + str(self.bond_order)}"
        )
