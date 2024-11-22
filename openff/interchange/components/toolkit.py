"""Utilities for processing and interfacing with the OpenFF Toolkit."""

from collections import defaultdict
from typing import TYPE_CHECKING

import networkx
import numpy
from openff.toolkit import ForceField, Molecule, Quantity, Topology
from openff.toolkit.topology._mm_molecule import _SimpleMolecule
from openff.toolkit.typing.engines.smirnoff.parameters import VirtualSiteHandler
from openff.toolkit.utils.collections import ValidatedList
from openff.utilities.utilities import has_package

from openff.interchange.models import ImportedVirtualSiteKey

if has_package("openmm") or TYPE_CHECKING:
    import openmm.app


def _get_num_h_bonds(topology: "Topology") -> int:
    """Get the number of (covalent) bonds containing a hydrogen atom."""
    n_bonds_containing_hydrogen = 0

    for bond in topology.bonds:
        if 1 in (bond.atom1.atomic_number, bond.atom2.atomic_number):
            n_bonds_containing_hydrogen += 1

    return n_bonds_containing_hydrogen


def _get_14_pairs(topology_or_molecule: Topology | Molecule):
    """Generate tuples of atom pairs, including symmetric duplicates."""
    # TODO: A replacement of Topology.nth_degree_neighbors in the toolkit
    #       may implement this in the future.
    for bond in topology_or_molecule.bonds:
        atom_i = bond.atom1
        atom_j = bond.atom2
        for atom_i_partner in atom_i.bonded_atoms:
            for atom_j_partner in atom_j.bonded_atoms:
                if atom_i_partner == atom_j_partner:
                    continue

                if atom_i_partner in atom_j_partner.bonded_atoms:
                    continue

                if atom_j_partner in atom_i_partner.bonded_atoms:
                    continue

                if {*atom_i_partner.bonded_atoms}.intersection(
                    {*atom_j_partner.bonded_atoms},
                ):
                    continue

                else:
                    atom_i_partner_index = topology_or_molecule.atom_index(
                        atom_i_partner,
                    )
                    atom_j_partner_index = topology_or_molecule.atom_index(
                        atom_j_partner,
                    )
                    if atom_i_partner_index > atom_j_partner_index:
                        yield (atom_j_partner, atom_i_partner)
                    else:
                        yield (atom_i_partner, atom_j_partner)


def _validated_list_to_array(validated_list: "ValidatedList") -> Quantity:
    unit_ = validated_list[0].units
    return Quantity(numpy.asarray([val.m for val in validated_list]), unit_)


def _combine_topologies(topology1: Topology, topology2: Topology) -> Topology:
    topology1_ = Topology(other=topology1)
    topology2_ = Topology(other=topology2)

    for molecule in topology2_.molecules:
        topology1_.add_molecule(molecule)

    return topology1_


def _check_electrostatics_handlers(force_field: "ForceField") -> bool:
    """
    Return whether or not this ForceField should have an Electrostatics tag.
    """
    # Manually-curated list of names of ParameterHandler classes that are expected
    # to assign/modify partial charges
    partial_charge_handlers = [
        "LibraryCharges",
        "ToolkitAM1BCC",
        "ChargeIncrementModel",
        "VirtualSites",
        "GBSA",
    ]

    # A more robust solution would probably take place late in the parameterization
    # process, but this solution should cover a vast majority of cases with minimal
    # complexity. Most notably this will *not* behave well with
    #   * parameter handler plugins
    #   * future additions to -built-in handlers
    #   * handlers that _could_ assign partial charges but happen to not assign
    #       any for some particular topology

    for parameter_handler_name in force_field.registered_parameter_handlers:
        if parameter_handler_name in partial_charge_handlers:
            return True

    return False


def _simple_topology_from_openmm(
    openmm_topology: "openmm.app.Topology",
    system: openmm.System | None = None,
) -> Topology:
    """
    Convert an OpenMM Topology into an OpenFF Topology consisting **only** of so-called `_SimpleMolecule`s.

    Arguments
    ---------
    openmm_topology: openmm.app.Topology
        The OpenMM Topology to convert.
    system: openmm.System, optional
        The OpenMM System associated with the topology. Only needed if there are virtual sites in the topology.
    """
    # TODO: Splice in fully-defined OpenFF `Molecule`s?

    graph = networkx.Graph()

    virtual_sites: list[ImportedVirtualSiteKey] = list()

    # map indices of OpenMM system with virtual site to topology indices, or virtual site keys,
    # in associated OpenFF topology, since stripping out virtual sites offsets particle indices
    openmm_openff_particle_map: dict[int, int | ImportedVirtualSiteKey] = dict()

    # TODO: This is nearly identical to Topology._openmm_topology_to_networkx.
    #  Should this method be replaced with a direct call to that?
    for atom in openmm_topology.atoms():
        if atom.element is None:
            assert isinstance(
                system,
                openmm.System,
            ), "`system` argument required if virtual sites are present in the topology"
            try:
                # assume ThreeParticleAverageSite for now
                orientation_atom_indices = [
                    openmm_openff_particle_map[system.getVirtualSite(atom.index).getParticle(i)] for i in range(3)
                ]
            except openmm.OpenMMException as error:
                if "This particle is not a virtual site" in str(error):
                    raise ValueError(
                        "Particle ordering mismatch between OpenMM system and topology. "
                        f"Look at particle {atom.index} in the topology.",
                    ) from error
            virtual_sites.append(
                ImportedVirtualSiteKey(
                    orientation_atom_indices=orientation_atom_indices,
                    name=atom.name,
                    type="ThreeParticleAverageSite",
                ),
            )

            # TODO: This will break if virtual sites are not after their parent/orientation atoms
            openmm_openff_particle_map[atom.index] = virtual_sites[-1]

        else:
            graph.add_node(
                atom.index,
                atomic_number=getattr(atom.element, "atomic_number", 0),
                name=atom.name,
                residue_name=atom.residue.name,
                # Note that residue number is mapped to residue.id here. The use of id vs. number
                # varies in other packages and the convention for the OpenFF-OpenMM interconversion
                # is recorded at
                # https://docs.openforcefield.org/projects/toolkit/en/0.15.1/users/molecule_conversion.html
                residue_number=atom.residue.id,
                insertion_code=atom.residue.insertionCode,
                chain_id=atom.residue.chain.id,
            )

            openmm_openff_particle_map[atom.index] = atom.index - len(virtual_sites)

    for bond in openmm_topology.bonds():
        graph.add_edge(
            bond.atom1.index,
            bond.atom2.index,
        )

    topology = _simple_topology_from_graph(graph)

    topology._molecule_virtual_site_map = defaultdict(list)

    # TODO: This iteration strategy scales horribly with system size - need to refactor -
    #       since looking up topology atom indices is slow. It's probably repetitive to
    #       look up the molecule index over and over again
    for particle in virtual_sites:
        molecule_index = topology.molecule_index(topology.atom(particle.orientation_atom_indices[0]).molecule)

        topology._molecule_virtual_site_map[molecule_index].append(particle)

    topology._particle_map = openmm_openff_particle_map

    return topology


def _simple_topology_from_graph(graph: networkx.Graph) -> Topology:
    """Convert a networkx Graph into an OpenFF Topology consisting only of `_SimpleMolecule`s."""
    topology = Topology()

    for component in networkx.connected_components(graph):
        subgraph = _reorder_subgraph(graph.subgraph(component))

        # Attempt to safeguard against the possibility that
        # the subgraphs are returned out of "atom order", like
        # if atoms in an later molecule have lesser atom indices
        # than this molecule
        #
        # Oct 2024 - need to add a test case for above?
        # these values are not necessarily equal because of virtual sites
        assert topology.n_atoms <= next(iter(subgraph.nodes))

        topology.add_molecule(_SimpleMolecule._from_subgraph(subgraph))

    return topology


def _reorder_subgraph(graph: networkx.Graph) -> networkx.Graph:
    """Ensure that the graph is ordered with ascending atoms."""
    new_graph = networkx.Graph()
    new_graph.add_nodes_from(sorted(graph.nodes(data=True)))
    new_graph.add_edges_from(graph.edges(data=True))

    return new_graph


# This is to re-implement:
#   https://github.com/openforcefield/openff-toolkit/blob/60014820e6a333bed04e8bf5181d177da066da4d/
#   openff/toolkit/typing/engines/smirnoff/parameters.py#L2509-L2515
# It doesn't seem ideal to assume that matching SMILES === isomorphism?
class _HashedMolecule(Molecule):
    def __hash__(self):
        return hash(self.to_smiles(mapped=True, explicit_hydrogens=True, isomeric=True))


def _assert_all_isomorphic(molecule_list: list[Molecule]) -> bool:
    hashed_molecules = {_HashedMolecule(molecule) for molecule in molecule_list}

    return len(hashed_molecules) == len(molecule_list)


def _lookup_virtual_site_parameter(
    parameter_handler: VirtualSiteHandler,
    smirks: str,
    name: str,
    match: str,
) -> VirtualSiteHandler.VirtualSiteType:
    """
    Given some attributes, look up a virtual site parameter.

    The toolkit does not reliably look up `VirtualSiteType`s when SMIRKS are not unique,
    which is valid for some virtual site use cases.
    https://github.com/openforcefield/openff-toolkit/issues/1847

    """
    if not isinstance(parameter_handler, VirtualSiteHandler):
        raise NotImplementedError("Only VirtualSiteHandler is currently supported.")

    for virtual_site_type in parameter_handler.parameters:
        if virtual_site_type.smirks == smirks and virtual_site_type.name == name and virtual_site_type.match == match:
            return virtual_site_type
    else:
        raise ValueError(
            f"No VirtualSiteType found with {smirks=}, name={name=}, and match={match=}.",
        )
