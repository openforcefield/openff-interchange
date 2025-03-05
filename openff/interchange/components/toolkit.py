"""Utilities for processing and interfacing with the OpenFF Toolkit."""

from functools import lru_cache
from typing import TYPE_CHECKING, Union

import networkx
import numpy
from openff.toolkit import ForceField, Molecule, Quantity, Topology
from openff.toolkit.topology._mm_molecule import _SimpleMolecule
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterHandler, VirtualSiteHandler
from openff.toolkit.utils.collections import ValidatedList
from openff.utilities.utilities import has_package

from openff.interchange.models import (
    PotentialKey,
)

if has_package("openmm") or TYPE_CHECKING:
    import openmm.app


_IDIVF_1 = Quantity(1.0, "dimensionless")
_PERIODICITIES = {
    1: Quantity(1, "dimensionless"),
    2: Quantity(2, "dimensionless"),
    3: Quantity(3, "dimensionless"),
    4: Quantity(4, "dimensionless"),
    5: Quantity(5, "dimensionless"),
    6: Quantity(6, "dimensionless"),
}


def _get_num_h_bonds(topology: "Topology") -> int:
    """Get the number of (covalent) bonds containing a hydrogen atom."""
    n_bonds_containing_hydrogen = 0

    for bond in topology.bonds:
        if 1 in (bond.atom1.atomic_number, bond.atom2.atomic_number):
            n_bonds_containing_hydrogen += 1

    return n_bonds_containing_hydrogen


def _get_14_pairs(topology_or_molecule: Union["Topology", "Molecule"]):
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


def _simple_topology_from_openmm(openmm_topology: "openmm.app.Topology") -> Topology:
    """Convert an OpenMM Topology into an OpenFF Topology consisting **only** of so-called `_SimpleMolecule`s."""
    # TODO: Splice in fully-defined OpenFF `Molecule`s?

    graph = networkx.Graph()

    # TODO: This is nearly identical to Topology._openmm_topology_to_networkx.
    #  Should this method be replaced with a direct call to that?
    for atom in openmm_topology.atoms():
        graph.add_node(
            atom.index,
            atomic_number=atom.element.atomic_number,
            name=atom.name,
            residue_name=atom.residue.name,
            # Note that residue number is mapped to residue.id here. The use of id vs. number varies in other packages
            # and the convention for the OpenFF-OpenMM interconversion is recorded at
            # https://docs.openforcefield.org/projects/toolkit/en/0.15.1/users/molecule_conversion.html
            residue_number=atom.residue.id,
            insertion_code=atom.residue.insertionCode,
            chain_id=atom.residue.chain.id,
        )

    for bond in openmm_topology.bonds():
        graph.add_edge(
            bond.atom1.index,
            bond.atom2.index,
        )

    return _simple_topology_from_graph(graph)


def _simple_topology_from_graph(graph: networkx.Graph) -> Topology:
    """Convert a networkx Graph into an OpenFF Topology consisting only of `_SimpleMolecule`s."""
    topology = Topology()

    for component in networkx.connected_components(graph):
        subgraph = _reorder_subgraph(graph.subgraph(component))

        # Attempt to safeguard against the possibility that
        # the subgraphs are returned out of "atom order", like
        # if atoms in an later molecule have lesser atom indices
        # than this molecule
        assert topology.n_atoms == next(iter(subgraph.nodes))

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


@lru_cache
def _cache_angle_parameter_lookup(
    potential_key: PotentialKey,
    parameter_handler: ParameterHandler,
) -> dict[str, Quantity]:
    parameter = parameter_handler.parameters[potential_key.id]

    return {parameter_name: getattr(parameter, parameter_name) for parameter_name in ["k", "angle"]}


@lru_cache
def _cache_torsion_parameter_lookup(
    potential_key: PotentialKey,
    parameter_handler: ParameterHandler,
    idivf: float | None = None,
) -> dict[str, Quantity]:
    smirks = potential_key.id
    n = potential_key.mult
    parameter = parameter_handler.parameters[smirks]

    if idivf is not None:
        # case of non-standard default_idivf in impropers
        _idivf = idivf
    elif parameter.idivf is None:
        # This appears to only come from imports
        _idivf = _IDIVF_1
    elif parameter.idivf[n] == 1.0:
        _idivf = _IDIVF_1
    else:
        _idivf = Quantity(parameter.idivf[n], "dimensionless")

    return {
        "k": parameter.k[n],
        "periodicity": _PERIODICITIES[parameter.periodicity[n]],
        "phase": parameter.phase[n],
        "idivf": _idivf,
    }
