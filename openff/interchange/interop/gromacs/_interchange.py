import networkx
from openff.toolkit import Topology
from openff.units import unit

from openff.interchange import Interchange
from openff.interchange.common._nonbonded import ElectrostaticsCollection, vdWCollection
from openff.interchange.common._valence import (
    AngleCollection,
    BondCollection,
    ImproperTorsionCollection,
    ProperTorsionCollection,
    RyckaertBellemansTorsionCollection,
)
from openff.interchange.components.potentials import Potential
from openff.interchange.interop.gromacs.models.models import (
    GROMACSSystem,
    PeriodicImproperDihedral,
    PeriodicProperDihedral,
    RyckaertBellemansDihedral,
)
from openff.interchange.models import (
    AngleKey,
    BondKey,
    ImproperTorsionKey,
    PotentialKey,
    ProperTorsionKey,
    TopologyKey,
)


def to_interchange(
    system: GROMACSSystem,
) -> Interchange:
    """
    Convert a GROMACS system to an Interchange object.

    Parameters
    ----------
    system: GROMACSSystem
        The GROMACS system to convert.

    Returns
    -------
    interchange
        The converted Interchange object.

    """
    if (system.nonbonded_function, system.gen_pairs) != (1, True):
        raise NotImplementedError()

    if system.combination_rule == 2:
        mixing_rule = "lorentz-berthelot"
    elif system.combination_rule == 3:
        mixing_rule = "geometric"

    vdw = vdWCollection(scale14=system.vdw_14, mixing_rule=mixing_rule)
    electrostatics = ElectrostaticsCollection(version=0.4, scale_14=system.coul_14)

    bonds = BondCollection()
    angles = AngleCollection()
    # TODO: Split out periodic and RB/other styles?
    periodic_propers = ProperTorsionCollection()
    rb_torsions = RyckaertBellemansTorsionCollection()
    impropers = ImproperTorsionCollection()

    vdw.potentials = {
        PotentialKey(id=f"{atom_type.name}"): Potential(
            parameters={"sigma": atom_type.sigma, "epsilon": atom_type.epsilon},
        )
        for atom_type in system.atom_types.values()
    }

    molecule_start_index = 0

    for molecule_name, molecule_type in system.molecule_types.items():
        for _ in range(system.molecules[molecule_name]):
            for atom in molecule_type.atoms:
                topology_atom_index = molecule_start_index + atom.index - 1
                topology_key = TopologyKey(
                    atom_indices=(topology_atom_index,),
                )

                vdw.key_map.update(
                    {topology_key: PotentialKey(id=f"{atom.atom_type}")},
                )

                # GROMACS does NOT necessarily tie partial charges to atom types, so need a new key for each atom
                electrostatics_key = PotentialKey(id=f"{topology_key.atom_indices[0]}")
                electrostatics.key_map.update(
                    {topology_key: electrostatics_key},
                )

                electrostatics.potentials.update(
                    {electrostatics_key: Potential(parameters={"charge": atom.charge})},
                )

            for bond in molecule_type.bonds:
                topology_key = BondKey(
                    atom_indices=(
                        bond.atom1 + molecule_start_index - 1,
                        bond.atom2 + molecule_start_index - 1,
                    ),
                )

                potential_key = PotentialKey(
                    id="-".join(map(str, topology_key.atom_indices)),
                )

                potential = Potential(
                    parameters={
                        "k": bond.k,
                        "length": bond.length,
                    },
                )

                bonds.key_map.update({topology_key: potential_key})
                bonds.potentials.update({potential_key: potential})

            for angle in molecule_type.angles:
                topology_key = AngleKey(
                    atom_indices=(
                        angle.atom1 + molecule_start_index - 1,
                        angle.atom2 + molecule_start_index - 1,
                        angle.atom3 + molecule_start_index - 1,
                    ),
                )

                potential_key = PotentialKey(
                    id="-".join(map(str, topology_key.atom_indices)),
                )

                potential = Potential(
                    parameters={
                        "k": angle.k,
                        "angle": angle.angle,
                    },
                )

                angles.key_map.update({topology_key: potential_key})
                angles.potentials.update({potential_key: potential})

            for dihedral in molecule_type.dihedrals:
                if "Improper" in type(dihedral).__name__:
                    key_type = ImproperTorsionKey
                else:
                    key_type = ProperTorsionKey  # type: ignore[assignment]

                if isinstance(dihedral, PeriodicProperDihedral):
                    collection = periodic_propers
                elif isinstance(dihedral, RyckaertBellemansDihedral):
                    collection = rb_torsions
                elif isinstance(dihedral, PeriodicImproperDihedral):
                    collection = impropers
                else:
                    raise NotImplementedError(
                        f"Dihedral type {type(dihedral)} not implemented.",
                    )

                topology_key = key_type(
                    atom_indices=(
                        dihedral.atom1 + molecule_start_index - 1,
                        dihedral.atom2 + molecule_start_index - 1,
                        dihedral.atom3 + molecule_start_index - 1,
                        dihedral.atom3 + molecule_start_index - 1,
                    ),
                )

                potential_key = PotentialKey(
                    id="-".join(map(str, topology_key.atom_indices)),
                )

                if isinstance(
                    dihedral,
                    (PeriodicProperDihedral, PeriodicImproperDihedral),
                ):
                    potential = Potential(
                        parameters={
                            "periodicity": unit.Quantity(
                                dihedral.multiplicity,
                                unit.dimensionless,
                            ),
                            "phase": dihedral.phi,
                            "k": dihedral.k,
                            "idivf": 1 * unit.dimensionless,
                        },
                    )

                elif isinstance(dihedral, RyckaertBellemansDihedral):
                    potential = Potential(
                        parameters={
                            "c0": dihedral.c0,
                            "c1": dihedral.c1,
                            "c2": dihedral.c2,
                            "c3": dihedral.c3,
                            "c4": dihedral.c4,
                            "c5": dihedral.c5,
                        },
                    )

                else:
                    raise NotImplementedError()

                collection.key_map.update({topology_key: potential_key})
                collection.potentials.update({potential_key: potential})

            molecule_start_index += len(molecule_type.atoms)

    interchange = Interchange()

    interchange.collections.update(
        {
            "vdW": vdw,
            "Electrostatics": electrostatics,
            "Bonds": bonds,
            "Angles": angles,
            "ProperTorsions": periodic_propers,
            "RBTorsions": rb_torsions,
            "ImproperTorsions": impropers,
        },
    )

    interchange.topology = _convert_topology(system)
    interchange.positions = system.positions
    interchange.box = system.box

    return interchange


def _convert_topology(
    system: GROMACSSystem,
) -> Topology:
    from openff.toolkit.topology._mm_molecule import _SimpleMolecule

    topology = Topology()

    for molecule_name, molecule_type in system.molecule_types.items():
        graph = networkx.Graph()

        n_molecules = system.molecules[molecule_name]

        for atom in molecule_type.atoms:
            graph.add_node(
                atom.index - 1,
                atomic_number=system.atom_types[atom.atom_type].atomic_number,
            )

        if [
            system.atom_types[atom.atom_type].atomic_number
            for atom in molecule_type.atoms
        ] == [8, 1, 1]:
            graph.add_edge(0, 1)
            graph.add_edge(0, 2)
        else:
            for bond in molecule_type.bonds:
                graph.add_edge(
                    bond.atom1 - 1,
                    bond.atom2 - 1,
                )

        molecule = _SimpleMolecule._from_subgraph(graph)

        for _ in range(n_molecules):
            topology.add_molecule(molecule)

    return topology
