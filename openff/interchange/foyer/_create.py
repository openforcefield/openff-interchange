from openff.toolkit import Topology
from openff.units import Quantity, unit
from openff.utilities.utilities import has_package

from openff.interchange import Interchange
from openff.interchange.common._positions import _infer_positions
from openff.interchange.components.potentials import Collection
from openff.interchange.foyer._nonbonded import (
    FoyerElectrostaticsHandler,
    FoyerVDWHandler,
)
from openff.interchange.foyer._valence import (
    FoyerHarmonicAngleHandler,
    FoyerHarmonicBondHandler,
    FoyerPeriodicImproperHandler,
    FoyerPeriodicProperHandler,
    FoyerRBImproperHandler,
    FoyerRBProperHandler,
)
from openff.interchange.models import LibraryChargeTopologyKey, TopologyKey

if has_package("foyer"):
    from foyer.forcefield import Forcefield

_CollectionAlias = type[Collection]


def get_handlers_callable() -> dict[str, _CollectionAlias]:
    """Map Foyer-style handlers from string identifiers."""
    return {
        "vdW": FoyerVDWHandler,
        "Electrostatics": FoyerElectrostaticsHandler,
        "Bonds": FoyerHarmonicBondHandler,
        "Angles": FoyerHarmonicAngleHandler,
        "RBTorsions": FoyerRBProperHandler,
        "RBImpropers": FoyerRBImproperHandler,
        "ProperTorsions": FoyerPeriodicProperHandler,
        "ImproperTorsions": FoyerPeriodicImproperHandler,
    }


def _create_interchange(
    force_field: "Forcefield",
    topology: Topology,
    box: Quantity | None = None,
    positions: Quantity | None = None,
) -> Interchange:
    interchange = Interchange(topology=Topology(topology))

    interchange.positions = _infer_positions(interchange.topology, positions)

    interchange.box = interchange.topology.box_vectors if box is None else box

    # This block is from a mega merge, unclear if it's still needed
    for name, handler_class in get_handlers_callable().items():
        interchange.collections[name] = handler_class(type=name)

    vdw_handler = interchange["vdW"]
    vdw_handler.scale_14 = force_field.lj14scale
    vdw_handler.store_matches(force_field, topology=interchange.topology)
    vdw_handler.store_potentials(force_field=force_field)

    atom_slots = vdw_handler.key_map

    electrostatics = interchange["Electrostatics"]
    electrostatics.scale_14 = force_field.coulomb14scale
    electrostatics.store_charges(
        atom_slots=atom_slots,
        force_field=force_field,
    )

    for name, handler in interchange.collections.items():
        if name not in ["vdW", "Electrostatics"]:
            handler.store_matches(atom_slots, topology=interchange.topology)
            handler.store_potentials(force_field)

    # TODO: Populate .mdconfig, but only after a reasonable number of state mutations have been tested

    charges: dict[TopologyKey | LibraryChargeTopologyKey, Quantity] = (
        electrostatics.charges
    )

    for molecule in interchange.topology.molecules:
        molecule_charges = [
            charges[
                TopologyKey(atom_indices=(interchange.topology.atom_index(atom),))
            ].m
            for atom in molecule.atoms
        ]

        # Quantity(list[Quantity]) works ... but is a big magical to mypy
        molecule.partial_charges = Quantity(  # type: ignore[call-overload]
            molecule_charges,
            unit.elementary_charge,
        )

    interchange.collections["vdW"] = vdw_handler
    interchange.collections["Electrostatics"] = electrostatics

    return interchange
