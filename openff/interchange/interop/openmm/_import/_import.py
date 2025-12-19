import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Union

from openff.toolkit import Quantity, Topology
from openff.units import ensure_quantity
from openff.units.openmm import from_openmm as from_openmm_
from openff.utilities.utilities import has_package, requires_package
from pydantic import ValidationError

from openff.interchange.common._nonbonded import ElectrostaticsCollection, vdWCollection
from openff.interchange.common._topology import InterchangeTopology
from openff.interchange.common._valence import (
    AngleCollection,
    BondCollection,
    ConstraintCollection,
    ProperTorsionCollection,
)
from openff.interchange.exceptions import NoPositionsError, UnsupportedImportError
from openff.interchange.interop.openmm._import._virtual_sites import _convert_virtual_sites
from openff.interchange.interop.openmm._import.compat import _check_compatible_inputs
from openff.interchange.models import ImportedVirtualSiteKey
from openff.interchange.warnings import MissingPositionsWarning

if has_package("openmm"):
    import openmm
    import openmm.app
    import openmm.unit

if TYPE_CHECKING:
    import openmm
    import openmm.app
    import openmm.unit

    from openff.interchange import Interchange


@requires_package("openmm")
def from_openmm(
    *,
    system: "openmm.System",
    topology: Union["openmm.app.Topology", Topology],
    positions: Quantity | None = None,
    box_vectors: Quantity | None = None,
) -> "Interchange":
    """Create an Interchange object from OpenMM data."""
    from openff.interchange import Interchange

    _check_compatible_inputs(system=system, topology=topology)

    if isinstance(topology, openmm.app.Topology):
        from openff.interchange.components.toolkit import _simple_topology_from_openmm

        openff_topology = _simple_topology_from_openmm(topology, system)

        if topology.getPeriodicBoxVectors() is not None:
            openff_topology.box_vectors = from_openmm_(topology.getPeriodicBoxVectors())

        # OpenMM topologies do not store positions

        particle_map = openff_topology._particle_map

    elif isinstance(topology, (Topology, InterchangeTopology)):
        # could split this out into separate cases, but behavior is similar
        openff_topology = topology

        particle_map = {index: index for index in range(topology.n_atoms)}
        try:
            positions = openff_topology.get_positions()
        except NoPositionsError:
            # fall pack to positions argument,
            # InterchangeTopology.get_positions() raises this
            pass

    else:
        raise ValueError(
            f"Could not process `topology` argument of type {type(topology)=}.",
        )

    interchange = Interchange(topology=openff_topology)

    try:
        interchange.topology._molecule_virtual_site_map = openff_topology._molecule_virtual_site_map
    except AttributeError:
        interchange.topology._molecule_virtual_site_map = defaultdict(list)

    # TODO: Actually build up the VirtualSiteCollection, maybe using _molecule_virtual_site_map

    constraints = _convert_constraints(system, particle_map)

    if constraints is not None:
        interchange.collections["Constraints"] = constraints

    virtual_sites = _convert_virtual_sites(system, openff_topology, particle_map)

    if virtual_sites is not None:
        interchange.collections["VirtualSites"] = virtual_sites

    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            vdw, coul = _convert_nonbonded_force(force, particle_map)
            interchange.collections["vdW"] = vdw
            interchange.collections["Electrostatics"] = coul
        elif isinstance(force, openmm.HarmonicBondForce):
            bonds = _convert_harmonic_bond_force(force)
            interchange.collections["Bonds"] = bonds
        elif isinstance(force, openmm.HarmonicAngleForce):
            angles = _convert_harmonic_angle_force(force)
            interchange.collections["Angles"] = angles
        elif isinstance(force, openmm.PeriodicTorsionForce):
            proper_torsions = _convert_periodic_torsion_force(force)
            interchange.collections["ProperTorsions"] = proper_torsions
        elif isinstance(force, openmm.CMMotionRemover):
            pass
        else:
            raise UnsupportedImportError(
                f"Unsupported OpenMM Force type ({type(force)}) found.",
            )

    if positions is None:
        warnings.warn(
            "Nothing was passed to the `positions` argument, are you sure you don't want to pass positions?",
            MissingPositionsWarning,
        )

    else:
        assert len(positions) == len(particle_map)

        interchange.positions = ensure_quantity(positions, "openff")[  # type: ignore[index]
            [key for key, val in particle_map.items() if isinstance(val, int)]
        ]

    if box_vectors is not None:
        _box_vectors = box_vectors

    elif interchange.topology.box_vectors is not None:
        _box_vectors = interchange.topology.box_vectors

    else:
        # If there is no box argument passed and the topology is non-periodic
        # and the system does not have default box vectors, it'll end up as None
        _box_vectors = from_openmm_(system.getDefaultPeriodicBoxVectors())

    # TODO: Does this run through the Interchange.box validator?
    interchange.box = _box_vectors

    try:
        num_physics_bonds = len(interchange["Bonds"].key_map)
    except LookupError:
        num_physics_bonds = 0

    if interchange.topology.n_bonds > num_physics_bonds:
        # There are probably missing (physics) bonds from rigid waters. The topological
        # bonds are probably processed correctly.
        _fill_in_rigid_water_bonds(interchange)

    return interchange


def _convert_constraints(
    system: "openmm.System",
    particle_map: dict[int, int | ImportedVirtualSiteKey],
) -> ConstraintCollection | None:
    from openff.interchange.components.potentials import Potential
    from openff.interchange.models import BondKey, PotentialKey

    if system.getNumConstraints() == 0:
        return None

    constraints = ConstraintCollection()

    # Map the unique distances (float, implicitly nanometer) to indices used for deduplication
    unique_distances: dict[float, int] = {
        distance: index
        for index, distance in enumerate(
            {
                system.getConstraintParameters(index)[2].value_in_unit(
                    openmm.unit.nanometer,
                )
                for index in range(system.getNumConstraints())
            },
        )
    }

    _keys: dict[float, PotentialKey] = dict()

    for distance, index in unique_distances.items():
        potential_key = PotentialKey(id=f"Constraint{index}")
        _keys[distance] = potential_key
        constraints.potentials[potential_key] = Potential(
            parameters={"distance": Quantity(distance, "nanometer")},
        )

    for index in range(system.getNumConstraints()):
        atom1, atom2, _distance = system.getConstraintParameters(index)

        distance = _distance.value_in_unit(openmm.unit.nanometer)

        constraints.key_map[BondKey(atom_indices=(particle_map[atom1], particle_map[atom2]))] = _keys[distance]

    return constraints


def _convert_nonbonded_force(
    force: "openmm.NonbondedForce",
    particle_map: dict[int, int | ImportedVirtualSiteKey],
) -> tuple[vdWCollection, ElectrostaticsCollection]:
    from openff.units.openmm import from_openmm as from_openmm_quantity

    from openff.interchange.components.potentials import Potential
    from openff.interchange.models import PotentialKey, TopologyKey

    if force.getNonbondedMethod() != 4:
        raise UnsupportedImportError(
            "Importing from OpenMM only currently supported with `openmm.NonbondedForce.PME`.",
        )

    vdw = vdWCollection()
    electrostatics = ElectrostaticsCollection(version=0.4, scale_14=0.833333)

    n_parametrized_particles = force.getNumParticles()

    for idx in range(n_parametrized_particles):
        charge, sigma, epsilon = force.getParticleParameters(idx)

        try:
            top_key = TopologyKey(atom_indices=(particle_map[idx],))
        except ValidationError:
            top_key: ImportedVirtualSiteKey = particle_map[idx]  # type: ignore[no-redef]

        pot = Potential(
            parameters={
                "sigma": from_openmm_quantity(sigma),
                "epsilon": from_openmm_quantity(epsilon),
            },
        )

        pot_key = PotentialKey(id=f"{idx}", associated_handler="vdW")
        vdw.key_map.update({top_key: pot_key})
        vdw.potentials.update({pot_key: pot})

        # This quacks like it's from a library charge, but tracks that it's
        # not actually coming from a source
        pot_key = PotentialKey(id=f"{idx}", associated_handler="ExternalSource")
        electrostatics.key_map.update({top_key: pot_key})
        electrostatics.potentials.update(
            {pot_key: Potential(parameters={"charge": from_openmm_quantity(charge)})},
        )

    if force.getNonbondedMethod() == 4:
        vdw.cutoff = force.getCutoffDistance()
        electrostatics.cutoff = force.getCutoffDistance()
    else:
        raise UnsupportedImportError(
            f"Parsing a non-bonded force of type {type(force)} with {force.getNonbondedMethod()} not yet supported.",
        )

    if force.getUseSwitchingFunction():
        vdw.switch_width = vdw.cutoff - from_openmm_quantity(
            force.getSwitchingDistance(),
        )
    else:
        vdw.switch_width = 0.0 * vdw.cutoff.units

    return vdw, electrostatics


def _convert_harmonic_bond_force(
    force: "openmm.HarmonicBondForce",
) -> BondCollection:
    from openff.units.openmm import from_openmm as from_openmm_quantity

    from openff.interchange.common._valence import BondCollection
    from openff.interchange.components.potentials import Potential
    from openff.interchange.models import BondKey, PotentialKey

    bonds = BondCollection()

    n_parametrized_bonds = force.getNumBonds()

    for idx in range(n_parametrized_bonds):
        atom1, atom2, length, k = force.getBondParameters(idx)
        top_key = BondKey(atom_indices=(atom1, atom2))
        pot_key = PotentialKey(id=f"{atom1}-{atom2}", associated_handler="Bonds")
        pot = Potential(
            parameters={
                "length": from_openmm_quantity(length),
                "k": from_openmm_quantity(k),
            },
        )

        bonds.key_map.update({top_key: pot_key})
        bonds.potentials.update({pot_key: pot})

    return bonds


def _convert_harmonic_angle_force(
    force: "openmm.HarmonicAngleForce",
) -> AngleCollection:
    from openff.units.openmm import from_openmm as from_openmm_quantity

    from openff.interchange.common._valence import AngleCollection
    from openff.interchange.components.potentials import Potential
    from openff.interchange.models import AngleKey, PotentialKey

    angles = AngleCollection()

    n_parametrized_angles = force.getNumAngles()

    for idx in range(n_parametrized_angles):
        atom1, atom2, atom3, angle, k = force.getAngleParameters(idx)
        top_key = AngleKey(atom_indices=(atom1, atom2, atom3))
        pot_key = PotentialKey(
            id=f"{atom1}-{atom2}-{atom3}",
            associated_handler="Angles",
        )
        pot = Potential(
            parameters={
                "angle": from_openmm_quantity(angle),
                "k": from_openmm_quantity(k),
            },
        )

        angles.key_map.update({top_key: pot_key})
        angles.potentials.update({pot_key: pot})

    return angles


def _convert_periodic_torsion_force(
    force: "openmm.PeriodicTorsionForce",
) -> ProperTorsionCollection:
    # TODO: Can impropers be separated out from a PeriodicTorsionForce?
    # Maybe by seeing if a quartet is in mol/top.propers or .impropers
    from openff.units.openmm import from_openmm as from_openmm_quantity

    from openff.interchange.common._valence import ProperTorsionCollection
    from openff.interchange.components.potentials import Potential
    from openff.interchange.models import PotentialKey, ProperTorsionKey

    proper_torsions = ProperTorsionCollection()

    n_parametrized_torsions = force.getNumTorsions()

    for idx in range(n_parametrized_torsions):
        atom1, atom2, atom3, atom4, per, phase, k = force.getTorsionParameters(idx)
        # TODO: Process layered torsions
        # TODO: Check if this torsion is an improper
        top_key = ProperTorsionKey(atom_indices=(atom1, atom2, atom3, atom4), mult=0)
        while top_key in proper_torsions.key_map:
            top_key.mult = top_key.mult + 1  # type: ignore[operator]

        pot_key = PotentialKey(
            id=f"{atom1}-{atom2}-{atom3}-{atom4}",
            mult=top_key.mult,
            associated_handler="ProperTorsions",
        )
        pot = Potential(
            parameters={
                "periodicity": Quantity(int(per), "dimensionless"),
                "phase": from_openmm_quantity(phase),
                "k": from_openmm_quantity(k),
                "idivf": Quantity(1, "dimensionless"),
            },
        )

        proper_torsions.key_map.update({top_key: pot_key})
        proper_torsions.potentials.update({pot_key: pot})

    return proper_torsions


def _fill_in_rigid_water_bonds(interchange: "Interchange"):
    from openff.toolkit.topology._mm_molecule import Molecule, _SimpleMolecule

    from openff.interchange.components.potentials import Potential
    from openff.interchange.models import AngleKey, BondKey, PotentialKey

    if "Bonds" not in interchange.collections:
        interchange.collections.update({"Bonds": BondCollection()})

    if "Angles" not in interchange.collections:
        interchange.collections.update({"Angles": AngleCollection()})

    simple_water = _SimpleMolecule.from_molecule(Molecule.from_smiles("O"))

    rigid_water_bond_key = PotentialKey(id="rigid_water", associated_handler="Bonds")
    rigid_water_bond = Potential(
        parameters={
            "length": Quantity("1.0 angstrom"),
            "k": Quantity("50,000.0 kcal/mol/angstrom**2"),
        },
    )

    rigid_water_angle_key = PotentialKey(id="rigid_water", associated_handler="Angles")
    rigid_water_angle = Potential(
        parameters={
            "angle": Quantity("104.5 degree"),
            "k": Quantity("1.0 kcal/mol/rad**2"),
        },
    )

    interchange["Bonds"].potentials.update(
        {PotentialKey(id="rigid_water", associated_handler="Bonds"): rigid_water_bond},
    )

    interchange["Angles"].potentials.update(
        {
            PotentialKey(
                id="rigid_water",
                associated_handler="Angles",
            ): rigid_water_angle,
        },
    )

    for molecule in interchange.topology.molecules:
        if not molecule.is_isomorphic_with(simple_water):
            continue

        for bond in molecule.bonds:
            bond_key = BondKey(
                atom_indices=(
                    interchange.topology.atom_index(bond.atom1),
                    interchange.topology.atom_index(bond.atom2),
                ),
            )

            if bond_key not in interchange["Bonds"].key_map:
                # add 1 A / 50,000 kcal/mol/A2 force constant
                interchange["Bonds"].key_map.update({bond_key: rigid_water_bond_key})

        for angle in molecule.angles:
            angle_key = AngleKey(
                atom_indices=(
                    interchange.topology.atom_index(angle[0]),
                    interchange.topology.atom_index(angle[1]),
                    interchange.topology.atom_index(angle[2]),
                ),
            )

            if angle_key not in interchange["Angles"].key_map:
                # add very flimsy force constant, since equilibrium angles differ
                # across models
                interchange["Angles"].key_map.update({angle_key: rigid_water_angle_key})
