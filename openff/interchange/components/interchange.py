"""An object for storing, manipulating, and converting molecular mechanics data."""
import warnings
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import mdtraj as md
import numpy as np
from openff.toolkit.topology.topology import Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.utilities.utilities import has_package, requires_package
from pydantic import Field, validator

from openff.interchange.components.mdtraj import _OFFBioTop
from openff.interchange.components.potentials import PotentialHandler
from openff.interchange.components.smirnoff import (
    SMIRNOFF_POTENTIAL_HANDLERS,
    SMIRNOFFBondHandler,
    SMIRNOFFConstraintHandler,
)
from openff.interchange.exceptions import (
    InternalInconsistencyError,
    InvalidBoxError,
    InvalidTopologyError,
    MissingParameterHandlerError,
    MissingPositionsError,
    SMIRNOFFHandlersNotImplementedError,
    UnsupportedExportError,
)
from openff.interchange.models import DefaultModel
from openff.interchange.types import ArrayQuantity

if TYPE_CHECKING:
    if has_package("foyer"):
        from foyer.forcefield import Forcefield as FoyerForcefield

_SUPPORTED_SMIRNOFF_HANDLERS = {
    "Constraints",
    "Bonds",
    "Angles",
    "ProperTorsions",
    "ImproperTorsions",
    "vdW",
    "Electrostatics",
    "LibraryCharges",
    "ChargeIncrementModel",
    "VirtualSites",
}


class Interchange(DefaultModel):
    """
    A object for storing, manipulating, and converting molecular mechanics data.

    .. warning :: This object is in an early and experimental state and unsuitable for production.
    .. warning :: This API is experimental and subject to change.
    """

    class InnerSystem(DefaultModel):
        """Inner representation of Interchange components."""

        # TODO: Ensure these fields are hidden from the user as intended
        handlers: Dict[str, PotentialHandler] = dict()
        topology: Optional[_OFFBioTop] = Field(None)
        box: ArrayQuantity["nanometer"] = Field(None)  # type: ignore
        positions: ArrayQuantity["nanometer"] = Field(None)  # type: ignore

        @validator("box")
        def validate_box(cls, val):
            if val is None:
                return val
            if val.shape == (3, 3):
                return val
            elif val.shape == (3,):
                val = val * np.eye(3)
                return val
            else:
                raise InvalidBoxError

    def __init__(self):
        self._inner_data = self.InnerSystem()

    @property
    def handlers(self):
        """Get the PotentialHandler objects in this Interchange object."""
        return self._inner_data.handlers

    def add_handler(self, handler_name: str, handler):
        """Add a ParameterHandler to this Interchange object."""
        self._inner_data.handlers.update({handler_name: handler})

    def remove_handler(self, handler_name: str):
        """Remove a PotentialHandler in this Interchange object."""
        self._inner_data.handlers.pop(handler_name)

    @property
    def topology(self):
        """Get the OpenFF Topology object in this Interchange object."""
        return self._inner_data.topology

    @topology.setter
    def topology(self, value):
        self._inner_data.topology = value

    @property
    def positions(self):
        """Get the positions of all particles."""
        return self._inner_data.positions

    @positions.setter
    def positions(self, value):
        self._inner_data.positions = value

    @property
    def box(self):
        """If periodic, an array representing the periodic boundary conditions."""
        return self._inner_data.box

    @box.setter
    def box(self, value):
        self._inner_data.box = value

    @classmethod
    def _check_supported_handlers(cls, force_field: ForceField):

        unsupported = list()

        for handler_name in force_field.registered_parameter_handlers:
            if handler_name in {"ToolkitAM1BCC"}:
                continue
            if handler_name not in _SUPPORTED_SMIRNOFF_HANDLERS:
                unsupported.append(handler_name)

        if unsupported:
            raise SMIRNOFFHandlersNotImplementedError(unsupported)

    @classmethod
    def from_smirnoff(
        cls,
        force_field: ForceField,
        topology: _OFFBioTop,
        box=None,
    ) -> "Interchange":
        """
        Create a new object by parameterizing a topology with a SMIRNOFF force field.

        Parameters
        ----------
        force_field
            The force field to parameterize the topology with.
        topology
            The topology to parameterize.
        box
            The box vectors associated with the interchange.

        Examples
        --------
        Generate an Interchange object from a single-molecule (OpenFF) topology and
        OpenFF 1.0.0 "Parsley"

        .. code-block:: pycon

            >>> from openff.interchange.components.interchange import Interchange
            >>> from openff.interchange.components.mdtraj import _OFFBioTop
            >>> from openff.toolkit.topology import Molecule
            >>> from openff.toolkit.typing.engines.smirnoff import ForceField
            >>> import mdtraj as md
            >>> mol = Molecule.from_smiles("CC")
            >>> mol.generate_conformers(n_conformers=1)
            >>> top = _OFFBioTop.from_molecules([mol])
            >>> top.mdtop = md.Topology.from_openmm(top.to_openmm())
            >>> parsley = ForceField("openff-1.0.0.offxml")
            >>> interchange = Interchange.from_smirnoff(topology=top, force_field=parsley)
            >>> interchange
            Interchange with 8 atoms, non-periodic topology

        """
        sys_out = Interchange()

        cls._check_supported_handlers(force_field)

        if isinstance(topology, _OFFBioTop):
            # TODO: See if Topology(topology) is fixed
            # https://github.com/openforcefield/openff-toolkit/issues/946
            sys_out.topology = deepcopy(topology)
            sys_out.topology.mdtop = topology.mdtop
        elif isinstance(topology, Topology):
            sys_out.topology = _OFFBioTop(other=topology)
            sys_out.topology.mdtop = md.Topology.from_openmm(topology.to_openmm())
        else:
            raise InvalidTopologyError(
                "Could not process topology argument, expected Topology or _OFFBioTop. "
                f"Found object of type {type(topology)}."
            )

        parameter_handlers_by_type = {
            force_field[parameter_handler_name].__class__: force_field[
                parameter_handler_name
            ]
            for parameter_handler_name in force_field.registered_parameter_handlers
        }

        if len(parameter_handlers_by_type) != len(
            force_field.registered_parameter_handlers
        ):

            raise NotImplementedError(
                "Only force fields that contain one instance of each parameter handler "
                "type are currently supported."
            )

        for potential_handler_type in SMIRNOFF_POTENTIAL_HANDLERS:

            parameter_handlers = [
                parameter_handlers_by_type[allowed_type]
                for allowed_type in potential_handler_type.allowed_parameter_handlers()
                if allowed_type in parameter_handlers_by_type
            ]

            if len(parameter_handlers) == 0:
                continue

            # TODO: Might be simpler to rework the bond handler to be self-contained and
            #       move back to the constraint handler dealing with the logic (and
            #       depending on the bond handler)
            if potential_handler_type == SMIRNOFFBondHandler:
                SMIRNOFFBondHandler.check_supported_parameters(force_field["Bonds"])
                potential_handler = SMIRNOFFBondHandler._from_toolkit(
                    parameter_handler=force_field["Bonds"],
                    topology=topology,
                    # constraint_handler=constraint_handler,
                )
                sys_out.handlers.update({"Bonds": potential_handler})
            elif potential_handler_type == SMIRNOFFConstraintHandler:
                bond_handler = force_field._parameter_handlers.get("Bonds", None)
                constraint_handler = force_field._parameter_handlers.get(
                    "Constraints", None
                )
                if constraint_handler is None:
                    continue
                constraints = SMIRNOFFConstraintHandler._from_toolkit(
                    parameter_handler=[
                        val
                        for val in [bond_handler, constraint_handler]
                        if val is not None
                    ],
                    topology=topology,
                )
                sys_out.handlers.update({"Constraints": constraints})
                continue
            elif len(potential_handler_type.allowed_parameter_handlers()) > 1:
                potential_handler = potential_handler_type._from_toolkit(  # type: ignore
                    parameter_handler=parameter_handlers,
                    topology=topology,
                )
            else:
                potential_handler_type.check_supported_parameters(parameter_handlers[0])
                potential_handler = potential_handler_type._from_toolkit(  # type: ignore
                    parameter_handler=parameter_handlers[0],
                    topology=topology,
                )
            sys_out.handlers.update({potential_handler.type: potential_handler})

        # `box` argument is only overriden if passed `None` and the input topology
        # has box vectors
        if box is None and topology.box_vectors is not None:
            sys_out.box = topology.box_vectors
        else:
            sys_out.box = box

        return sys_out

    def to_gro(self, file_path: Union[Path, str], writer="internal", decimal: int = 8):
        """Export this Interchange object to a .gro file."""
        if self.positions is None:
            raise MissingPositionsError(
                "Positions are required to write a `.gro` file but found None."
            )
        elif np.allclose(self.positions, 0):
            warnings.warn(
                "Positions seem to all be zero. Result coordinate file may be non-physical.",
                UserWarning,
            )

        # TODO: Enum-style class for handling writer arg?
        if writer == "parmed":
            from openff.interchange.interop.external import ParmEdWrapper

            ParmEdWrapper().to_file(self, file_path)

        elif writer == "internal":
            from openff.interchange.interop.internal.gromacs import to_gro

            to_gro(self, file_path, decimal=decimal)

    def to_top(self, file_path: Union[Path, str], writer="internal"):
        """Export this interchange to a .top file."""
        if writer == "parmed":
            from openff.interchange.interop.external import ParmEdWrapper

            ParmEdWrapper().to_file(self, file_path)

        elif writer == "internal":
            from openff.interchange.interop.internal.gromacs import to_top

            to_top(self, file_path)

    def to_lammps(self, file_path: Union[Path, str], writer="internal"):
        """Export this Interchange to a LAMMPS data file."""
        if writer != "internal":
            raise UnsupportedExportError

        from openff.interchange.interop.internal.lammps import to_lammps

        to_lammps(self, file_path)

    def to_openmm(self, combine_nonbonded_forces: bool = False):
        """Export this interchange to an OpenMM System."""
        from openff.interchange.interop.openmm import to_openmm as to_openmm_

        return to_openmm_(self, combine_nonbonded_forces=combine_nonbonded_forces)

    def _to_prmtop(self, file_path: Union[Path, str], writer="parmed"):
        """Export this interchange to an Amber .prmtop file."""
        if writer == "parmed":
            from openff.interchange.interop.external import ParmEdWrapper

            ParmEdWrapper().to_file(self, file_path)

        else:
            raise UnsupportedExportError

    def _to_crd(self, file_path: Union[Path, str], writer="parmed"):
        """Export this interchange to an Amber .crd file."""
        if writer == "parmed":
            from openff.interchange.interop.external import ParmEdWrapper

            ParmEdWrapper().to_file(self, file_path)

        else:
            raise UnsupportedExportError

    def _to_parmed(self):
        """Export this interchange to a ParmEd Structure."""
        from openff.interchange.interop.parmed import _to_parmed

        return _to_parmed(self)

    @classmethod
    def _from_parmed(cls, structure):
        from openff.interchange.interop.parmed import _from_parmed

        return _from_parmed(cls, structure)

    @classmethod
    @requires_package("foyer")
    def from_foyer(
        cls, topology: "_OFFBioTop", force_field: "FoyerForcefield", **kwargs
    ) -> "Interchange":
        """
        Create an Interchange object from a Foyer force field and an OpenFF topology.

        Examples
        --------
        Generate an Interchange object from a single-molecule (OpenFF) topology and
        the Foyer implementation of OPLS-AA

        .. code-block:: pycon

            >>> from openff.interchange.components.interchange import Interchange
            >>> from openff.interchange.components.mdtraj import _OFFBioTop
            >>> from openff.toolkit.topology import Molecule
            >>> from foyer import Forcefield
            >>> import mdtraj as md
            >>> mol = Molecule.from_smiles("CC")
            >>> mol.generate_conformers(n_conformers=1)
            >>> top = _OFFBioTop.from_molecules([mol])
            >>> top.mdtop = md.Topology.from_openmm(top.to_openmm())
            >>> oplsaa = Forcefield(name="oplsaa")
            >>> interchange = Interchange.from_foyer(topology=top, force_field=oplsaa)
            >>> interchange
            Interchange with 8 atoms, non-periodic topology

        """
        from openff.interchange.components.foyer import get_handlers_callable

        system = cls()
        system.topology = topology

        for name, Handler in get_handlers_callable().items():
            system.handlers[name] = Handler()

        system.handlers["vdW"].store_matches(force_field, topology=topology)
        system.handlers["vdW"].store_potentials(force_field=force_field)

        atom_slots = system.handlers["vdW"].slot_map

        system.handlers["Electrostatics"].store_charges(
            atom_slots=atom_slots,
            force_field=force_field,
        )

        system.handlers["vdW"].scale_14 = force_field.lj14scale
        system.handlers["Electrostatics"].scale_14 = force_field.coulomb14scale

        for name, handler in system.handlers.items():
            if name not in ["vdW", "Electrostatics"]:
                handler.store_matches(atom_slots, topology=topology)
                handler.store_potentials(force_field)

        return system

    def _get_parameters(self, handler_name: str, atom_indices: Tuple[int]) -> Dict:
        """
        Get parameter values of a specific potential.

        Here, parameters are expected to be uniquely dfined by the name of
        its associated handler and a tuple of atom indices.

        Note: This method only checks for equality of atom indices and will likely fail on complex cases
        involved layered parameters with multiple topology keys sharing identical atom indices.
        """
        for handler in self.handlers:
            if handler == handler_name:
                return self[handler_name]._get_parameters(atom_indices=atom_indices)
        raise MissingParameterHandlerError(
            f"Could not find parameter handler of name {handler_name}"
        )

    def _get_nonbonded_methods(self):
        if "vdW" in self.handlers:
            nonbonded_handler = "vdW"
        elif "Buckingham-6" in self.handlers:
            nonbonded_handler = "Buckingham-6"
        else:
            raise InternalInconsistencyError("Found no non-bonded handlers")

        nonbonded_ = {
            "electrostatics_method": self.handlers["Electrostatics"].method,
            "vdw_method": self.handlers[nonbonded_handler].method,
            "periodic_topology": self.box is not None,
        }

        return nonbonded_

    # TODO: Does this cause any strange behaviors with Pydantic?
    # Taken from https://stackoverflow.com/a/4017638/4248961
    _aliases = {"box_vectors": "x", "coordinates": "positions", "top": "topology"}

    def __setattr__(self, name, value):
        name = self._aliases.get(name, name)
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        name = self._aliases.get(name, name)
        return object.__getattribute__(self, name)

    def __getitem__(self, item: str):
        """Syntax sugar for looking up potential handlers or other components."""
        if type(item) != str:
            raise LookupError(
                "Only str arguments can be currently be used for lookups.\n"
                f"Found item {item} of type {type(item)}"
            )
        if item == "positions":
            return self.positions
        elif item in {"box", "box_vectors"}:
            return self.box
        elif item in self.handlers:
            return self.handlers[item]
        else:
            raise LookupError(
                f"Could not find component {item}. This object has the following "
                f"potential handlers registered:\n\t{[*self.handlers.keys()]}"
            )

    def __add__(self, other):
        """Combine two Interchange objects. This method is unstable and likely unsafe."""
        import mdtraj as md

        from openff.interchange.models import TopologyKey

        warnings.warn(
            "Iterchange object combination is experimental and likely to produce "
            "strange results. Use with caution!"
        )

        self_copy = Interchange()
        self_copy._inner_data = deepcopy(self._inner_data)

        atom_offset = self_copy.topology.mdtop.n_atoms

        other_top = deepcopy(other.topology)

        for top_mol in other_top.topology_molecules:
            self_copy.topology.add_molecule(top_mol.reference_molecule)

        self_copy.topology.mdtop = md.Topology.from_openmm(
            self_copy.topology.to_openmm()
        )

        for handler_name, handler in other.handlers.items():

            self_handler = self_copy.handlers[handler_name]

            for top_key, pot_key in handler.slot_map.items():

                new_atom_indices = tuple(
                    idx + atom_offset for idx in top_key.atom_indices
                )
                new_top_key = TopologyKey(
                    atom_indices=new_atom_indices,
                    mult=top_key.mult,
                )

                self_handler.slot_map.update({new_top_key: pot_key})
                self_handler.potentials.update({pot_key: handler.potentials[pot_key]})

        new_positions = np.vstack([self_copy.positions, other.positions])
        self_copy.positions = new_positions

        if not np.all(self_copy.box == other.box):
            raise NotImplementedError(
                "Combination with unequal box vectors is not curretnly supported"
            )

        return self_copy

    def __repr__(self):
        periodic = self.box is not None
        try:
            n_atoms = self.topology.mdtop.n_atoms
        except AttributeError:
            n_atoms = "unknown number of"
        except NameError:
            n_atoms = self.topology.n_topology_atoms
        return f"Interchange with {n_atoms} atoms, {'' if periodic else 'non-'}periodic topology"
