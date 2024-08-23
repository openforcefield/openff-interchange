import abc
from typing_extensions import Self
from typing import TypeVar

from openff.toolkit import Topology
from openff.toolkit.typing.engines.smirnoff.parameters import (
    AngleHandler,
    BondHandler,
    ImproperTorsionHandler,
    ParameterHandler,
    ProperTorsionHandler,
)

from openff.interchange.components.potentials import Collection
from openff.interchange.exceptions import (
    InvalidParameterHandlerError,
    SMIRNOFFParameterAttributeNotImplementedError,
    UnassignedAngleError,
    UnassignedBondError,
    UnassignedTorsionError,
)
from openff.interchange.models import (
    LibraryChargeTopologyKey,
    PotentialKey,
    TopologyKey,
)

TP = TypeVar("TP", bound="ParameterHandler")


# Coped from the toolkit, see
# https://github.com/openforcefield/openff-toolkit/blob/0133414d3ab51e1af0996bcebe0cc1bdddc6431b/
# openff/toolkit/typing/engines/smirnoff/parameters.py#L2318
def _check_all_valence_terms_assigned(
    handler,
    assigned_terms,
    topology,
    valence_terms,
):
    """Check that all valence terms have been assigned."""
    if len(assigned_terms) == len(valence_terms):
        return

    # Convert the valence term to a valence dictionary to make sure
    # the order of atom indices doesn't matter for comparison.
    valence_terms_dict = assigned_terms.__class__()
    for atoms in valence_terms:
        atom_indices = (topology.atom_index(a) for a in atoms)
        valence_terms_dict[atom_indices] = atoms

    # Check that both valence dictionaries have the same keys (i.e. terms).
    assigned_terms_set = set(assigned_terms.keys())
    valence_terms_set = set(valence_terms_dict.keys())
    unassigned_terms = valence_terms_set.difference(assigned_terms_set)
    not_found_terms = assigned_terms_set.difference(valence_terms_set)

    # Raise an error if there are unassigned terms.
    err_msg = ""

    if len(unassigned_terms) > 0:
        unassigned_atom_tuples = []

        unassigned_str = ""
        for unassigned_tuple in unassigned_terms:
            unassigned_str += "\n- Topology indices " + str(unassigned_tuple)
            unassigned_str += ": names and elements "

            unassigned_atoms = []

            # Pull and add additional helpful info on missing terms
            for atom_idx in unassigned_tuple:
                atom = topology.atom(atom_idx)
                unassigned_atoms.append(atom)
                unassigned_str += f"({atom.name} {atom.symbol}), "
            unassigned_atom_tuples.append(tuple(unassigned_atoms))
        err_msg += (
            f"{handler.__class__.__name__} was not able to find parameters for the "
            f"following valence terms:\n{unassigned_str}"
        )
    if len(not_found_terms) > 0:
        if err_msg != "":
            err_msg += "\n"
        not_found_str = "\n- ".join([str(x) for x in not_found_terms])
        err_msg += (
            f"{handler.__class__.__name__} assigned terms that were not found in the topology:\n" f"- {not_found_str}"
        )
    if err_msg:
        err_msg += "\n"

        if isinstance(handler, BondHandler):
            exception_class = UnassignedBondError
        elif isinstance(handler, AngleHandler):
            exception_class = UnassignedAngleError
        elif isinstance(handler, (ProperTorsionHandler, ImproperTorsionHandler)):
            exception_class = UnassignedTorsionError
        else:
            raise RuntimeError(
                f"Could not find an exception class for handler {handler}",
            )

        exception = exception_class(err_msg)
        exception.unassigned_topology_atom_tuples = unassigned_atom_tuples
        exception.handler_class = handler.__class__
        raise exception


class SMIRNOFFCollection(Collection, abc.ABC):
    """Base class for handlers storing potentials produced by SMIRNOFF force fields."""

    type: str

    is_plugin: bool = False

    def modify_openmm_forces(self, *args, **kwargs):
        """Optionally modify, create, or delete forces. Currently only available to plugins."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def supported_parameters(cls):
        """Return a list of parameter attributes supported by this handler."""
        raise NotImplementedError()

    @classmethod
    def potential_parameters(cls):
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        raise NotImplementedError()

    @classmethod
    def check_supported_parameters(cls, parameter_handler: ParameterHandler):
        """Verify that a parameter handler is in an allowed list of handlers."""
        for parameter in parameter_handler.parameters:
            for parameter_attribute in parameter._get_defined_parameter_attributes():
                if parameter_attribute == "parent_id":
                    continue
                if parameter_attribute not in cls.supported_parameters():
                    raise SMIRNOFFParameterAttributeNotImplementedError(
                        parameter_attribute,
                    )

    @classmethod
    def check_openmm_requirements(cls, combine_nonbonded_forces: bool) -> None:
        """Run through a list of assertions about what is compatible when exporting this to OpenMM."""

    def store_matches(
        self,
        parameter_handler: ParameterHandler,
        topology: "Topology",
    ) -> None:
        """Populate self.key_map with key-val pairs of [TopologyKey, PotentialKey]."""
        if self.key_map:
            # TODO: Should the key_map always be reset, or should we be able to partially
            # update it? Also Note the duplicated code in the child classes
            self.key_map: dict[
                TopologyKey | LibraryChargeTopologyKey,
                PotentialKey,
            ] = dict()

        matches = parameter_handler.find_matches(topology)

        for key, val in matches.items():
            parameter: ParameterHandler.ParameterType = val.parameter_type

            cosmetic_attributes = {
                cosmetic_attribute: getattr(
                    parameter,
                    f"_{cosmetic_attribute}",
                )
                for cosmetic_attribute in parameter._cosmetic_attribs
            }

            topology_key = TopologyKey(atom_indices=key)

            potential_key = PotentialKey(
                id=parameter.smirks,
                associated_handler=parameter_handler.TAGNAME,
                cosmetic_attributes=cosmetic_attributes,
            )

            self.key_map[topology_key] = potential_key

        if self.__class__.__name__ in [
            "SMIRNOFFBondCollection",
            "SMIRNOFFAngleCollection",
        ]:
            valence_terms = self.valence_terms(topology)  # type: ignore[attr-defined]

            _check_all_valence_terms_assigned(
                handler=parameter_handler,
                assigned_terms=matches,
                topology=topology,
                valence_terms=valence_terms,
            )

    def store_potentials(self, parameter_handler: TP):
        """
        Populate self.potentials with key-val pairs of [PotentialKey, Potential].
        """
        raise NotImplementedError()

    @classmethod
    def create(
        cls,
        parameter_handler: TP,
        topology: "Topology",
    ) -> Self:
        """
        Create a SMIRNOFFCOllection from toolkit data.

        """
        if type(parameter_handler) not in cls.allowed_parameter_handlers():
            raise InvalidParameterHandlerError(type(parameter_handler))

        collection = cls()  # type: ignore[call-arg]
        if hasattr(collection, "fractional_bondorder_method"):
            if getattr(parameter_handler, "fractional_bondorder_method", None):
                collection.fractional_bond_order_method = (  # type: ignore[attr-defined]
                    parameter_handler.fractional_bondorder_method
                )
                collection.fractional_bond_order_interpolation = (  # type: ignore[attr-defined]
                    parameter_handler.fractional_bondorder_interpolation
                )
        collection.store_matches(parameter_handler=parameter_handler, topology=topology)
        collection.store_potentials(parameter_handler=parameter_handler)

        return collection

    def __repr__(self) -> str:
        return (
            f"Handler '{self.type}' with expression '{self.expression}', {len(self.key_map)} mapping keys, "
            f"and {len(self.potentials)} potentials"
        )
