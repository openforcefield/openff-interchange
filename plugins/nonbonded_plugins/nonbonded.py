"""Custom classes exposed as plugins."""

import math
from collections.abc import Iterable
from typing import Literal

from openff.toolkit import Quantity, Topology, unit
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
    VirtualSiteHandler,
    _allow_only,
)

from openff.interchange._annotations import _DimensionlessQuantity, _DistanceQuantity
from openff.interchange.components.potentials import Potential
from openff.interchange.exceptions import InvalidParameterHandlerError
from openff.interchange.smirnoff._nonbonded import _SMIRNOFFNonbondedCollection

_HandlerIterable = Iterable[type[ParameterHandler]]
_CollectionAlias = type[_SMIRNOFFNonbondedCollection]


class BuckinghamHandler(ParameterHandler):
    """A custom SMIRNOFF handler for Buckingham interactions."""

    class BuckinghamType(ParameterType):
        """A custom SMIRNOFF type for Buckingham interactions."""

        _ELEMENT_NAME = "Buckingham"

        a = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole)
        b = ParameterAttribute(default=None, unit=unit.nanometer**-1)
        c = ParameterAttribute(
            default=None,
            unit=unit.kilojoule_per_mole * unit.nanometer**6,
        )

    _TAGNAME = "Buckingham"
    _INFOTYPE = BuckinghamType

    scale12 = ParameterAttribute(default=0.0, converter=float)
    scale13 = ParameterAttribute(default=0.0, converter=float)
    scale14 = ParameterAttribute(default=0.5, converter=float)
    scale15 = ParameterAttribute(default=1.0, converter=float)

    cutoff = ParameterAttribute(default=Quantity("9.0 angstrom"), unit=unit.angstrom)
    switch_width = ParameterAttribute(
        default=Quantity("1.0 angstrom"),
        unit=unit.angstrom,
    )

    periodic_method = ParameterAttribute(
        default="cutoff",
        converter=_allow_only(["cutoff"]),
    )
    nonperiodic_method = ParameterAttribute(
        default="no-cutoff",
        converter=_allow_only(["no-cutoff"]),
    )

    combining_rules = ParameterAttribute(
        default="Lorentz-Berthelot",
        converter=_allow_only(["Lorentz-Berthelot"]),
    )


class DoubleExponentialHandler(ParameterHandler):
    """A custom SMIRNOFF handler for double exponential interactions."""

    class DoubleExponentialType(ParameterType):
        """A custom SMIRNOFF type for double exponential interactions."""

        _ELEMENT_NAME = "Atom"

        r_min = ParameterAttribute(default=None, unit=unit.nanometers)
        epsilon = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole)

    # Give this a different name than the class provided in smirnoff-plugins
    # since the toolkit forbids two handlers from sharing a _TAGNAME
    _TAGNAME = "OtherDoubleExponential"
    _INFOTYPE = DoubleExponentialType

    scale12 = ParameterAttribute(default=0.0, converter=float)
    scale13 = ParameterAttribute(default=0.0, converter=float)
    scale14 = ParameterAttribute(default=0.5, converter=float)
    scale15 = ParameterAttribute(default=1.0, converter=float)

    # These are defined as dimensionless, we should consider enforcing global parameters
    # as being unit-bearing even if that means using `unit.dimensionless`
    alpha = ParameterAttribute(default=18.7)
    beta = ParameterAttribute(default=3.3)

    cutoff = ParameterAttribute(default=9.0 * unit.angstroms, unit=unit.angstrom)
    switch_width = ParameterAttribute(default=1.0 * unit.angstroms, unit=unit.angstrom)

    periodic_method = ParameterAttribute(
        default="cutoff",
        converter=_allow_only(["cutoff"]),
    )
    nonperiodic_method = ParameterAttribute(
        default="no-cutoff",
        converter=_allow_only(["no-cutoff"]),
    )

    combining_rules = ParameterAttribute(
        default="Lorentz-Berthelot",
        converter=_allow_only(["Lorentz-Berthelot"]),
    )


class C4IonHandler(ParameterHandler):
    """A custom SMIRNOFF handler adding C4 interactions from 10.1021/jp505875v."""

    class C4IonType(ParameterType):
        """A custom SMIRNOFF type for C4 ion interactions."""

        _ELEMENT_NAME = "Atom"

        c = ParameterAttribute(
            default=None,
            unit=unit.kilojoule_per_mole * unit.nanometer**4,
        )

    _TAGNAME = "C4Ion"
    _INFOTYPE = C4IonType


class SMIRNOFFBuckinghamCollection(_SMIRNOFFNonbondedCollection):
    """Handler storing vdW potentials as produced by a SMIRNOFF force field."""

    type: Literal["Buckingham"] = "Buckingham"

    is_plugin: bool = True

    acts_as: str = "vdW"

    expression: str = (
        "a*exp(-b*r)-c*r^-6;" "a=sqrt(a1*a2);" "b=2/(1/b1+1/b2);" "c=sqrt(c1*c2);"
    )

    periodic_method: str = "cutoff"
    nonperiodic_method: str = "no-cutoff"

    mixing_rule: str = "Buckingham"

    switch_width: _DistanceQuantity = Quantity(1.0, unit.angstrom)

    @classmethod
    def allowed_parameter_handlers(cls) -> _HandlerIterable:
        """Return a list of allowed types of ParameterHandler classes."""
        return (BuckinghamHandler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return a list of supported parameter attributes."""
        return "smirks", "id", "a", "b", "c"

    @classmethod
    def default_parameter_values(cls) -> Iterable[float]:
        """Per-particle parameter values passed to Force.addParticle()."""
        return 0.0, 0.0, 0.0

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        return "a", "b", "c"

    @classmethod
    def global_parameters(cls) -> Iterable[str]:
        """Return a list of global parameters, i.e. not per-potential parameters."""
        return tuple()

    def pre_computed_terms(self) -> dict[str, float]:
        """Return a dictionary of pre-computed terms for use in the expression."""
        return dict()

    @classmethod
    def check_openmm_requirements(cls, combine_nonbonded_forces: bool) -> None:
        """Run through a list of assertions about what is compatible when exporting this to OpenMM."""
        assert (
            not combine_nonbonded_forces
        ), "Custom non-bonded functional forms require `combine_nonbonded_forces=False`."

    def store_potentials(self, parameter_handler: BuckinghamHandler) -> None:
        """
        Populate self.potentials with key-val pairs of [TopologyKey, PotentialKey].

        """
        self.periodic_method = parameter_handler.periodic_method.lower()
        self.nonperiodic_method = parameter_handler.nonperiodic_method.lower()
        self.cutoff = parameter_handler.cutoff

        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            force_field_parameters = parameter_handler.parameters[smirks]

            potential = Potential(
                parameters={
                    parameter: getattr(force_field_parameters, parameter)
                    for parameter in self.potential_parameters()
                },
            )

            self.potentials[potential_key] = potential

    @classmethod
    def create(  # type: ignore[override]
        cls: _CollectionAlias,
        parameter_handler: BuckinghamHandler,
        topology: Topology,
    ) -> _SMIRNOFFNonbondedCollection:
        """
        Create a SMIRNOFFvdWCollection from toolkit data.

        """
        if type(parameter_handler) not in cls.allowed_parameter_handlers():
            raise InvalidParameterHandlerError(
                f"Found parameter handler type {type(parameter_handler)}, which is not "
                f"supported by potential type {type(cls)}",
            )

        handler = cls(
            scale_13=parameter_handler.scale13,
            scale_14=parameter_handler.scale14,
            scale_15=parameter_handler.scale15,
            cutoff=parameter_handler.cutoff,
            mixing_rule=parameter_handler.combining_rules.lower(),
            periodic_method=parameter_handler.periodic_method.lower(),
            nonperiodic_method=parameter_handler.nonperiodic_method.lower(),
            switch_width=parameter_handler.switch_width,
        )
        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler

    @classmethod
    def parameter_handler_precedence(cls) -> Iterable[str]:
        """
        Return the order in which parameter handlers take precedence when computing charges.
        """
        return "vdw", "VirtualSites"

    def create_virtual_sites(
        self,
        parameter_handler: VirtualSiteHandler,
        topology: Topology,
    ):
        """create() but with virtual sites."""
        raise NotImplementedError()


class SMIRNOFFDoubleExponentialCollection(_SMIRNOFFNonbondedCollection):
    """Handler storing vdW potentials as produced by a SMIRNOFF force field."""

    type: Literal["DoubleExponential"] = "DoubleExponential"

    is_plugin: bool = True

    acts_as: str = "vdW"

    expression: str = (
        "CombinedEpsilon*RepulsionFactor*RepulsionExp-CombinedEpsilon*AttractionFactor*AttractionExp;"
        "CombinedEpsilon=epsilon1*epsilon2;"
        "RepulsionExp=exp(-alpha*ExpDistance);"
        "AttractionExp=exp(-beta*ExpDistance);"
        "ExpDistance=r/CombinedR;"
        "CombinedR=r_min1+r_min2;"
    )

    periodic_method: str = "cutoff"
    nonperiodic_method: str = "no-cutoff"

    mixing_rule: str = ""

    switch_width: _DistanceQuantity = Quantity("1.0 angstrom")

    alpha: _DimensionlessQuantity
    beta: _DimensionlessQuantity

    @classmethod
    def allowed_parameter_handlers(cls) -> _HandlerIterable:
        """Return a list of allowed types of ParameterHandler classes."""
        return (DoubleExponentialHandler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return a list of supported parameter attributes."""
        return "smirks", "id", "r_min", "epsilon"

    @classmethod
    def default_parameter_values(cls) -> Iterable[float]:
        """Per-particle parameter values passed to Force.addParticle()."""
        return 0.0, 0.0

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        return "r_min", "epsilon"

    @classmethod
    def global_parameters(cls) -> Iterable[str]:
        """Return a list of global parameters, i.e. not per-potential parameters."""
        return "alpha", "beta"

    def pre_computed_terms(self) -> dict[str, float]:
        """Return a dictionary of pre-computed terms for use in the expression."""
        alpha_min_beta = self.alpha - self.beta

        return {
            "AlphaMinBeta": alpha_min_beta,
            "RepulsionFactor": self.beta * math.exp(self.alpha) / alpha_min_beta,
            "AttractionFactor": self.alpha * math.exp(self.beta) / alpha_min_beta,
        }

    def modify_parameters(
        self,
        original_parameters: dict[str, Quantity],
    ) -> dict[str, float]:
        """Optionally modify parameters prior to their being stored in a force."""
        # It's important that these keys are in the order of self.potential_parameters(),
        # consider adding a check somewhere that this is the case.
        _units = {"r_min": unit.nanometer, "epsilon": unit.kilojoule_per_mole}
        return {
            "r_min": original_parameters["r_min"].m_as(_units["r_min"]) * 0.5,
            "epsilon": math.sqrt(
                original_parameters["epsilon"].m_as(_units["epsilon"]),
            ),
        }

    @classmethod
    def check_openmm_requirements(cls, combine_nonbonded_forces: bool) -> None:
        """Run through a list of assertions about what is compatible when exporting this to OpenMM."""
        assert (
            not combine_nonbonded_forces
        ), "Custom non-bonded functional forms require `combine_nonbonded_forces=False`."

    def store_potentials(self, parameter_handler: DoubleExponentialHandler) -> None:
        """
        Populate self.potentials with key-val pairs of [TopologyKey, PotentialKey].

        """
        self.periodic_method = parameter_handler.periodic_method.lower()
        self.nonperiodic_method = parameter_handler.nonperiodic_method.lower()
        self.cutoff = parameter_handler.cutoff

        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            force_field_parameters = parameter_handler.parameters[smirks]

            potential = Potential(
                parameters={
                    parameter: getattr(force_field_parameters, parameter)
                    for parameter in self.potential_parameters()
                },
            )

            self.potentials[potential_key] = potential

    @classmethod
    def create(  # type: ignore[override]
        cls: _CollectionAlias,
        parameter_handler: DoubleExponentialHandler,
        topology: Topology,
    ) -> _SMIRNOFFNonbondedCollection:
        """
        Create a SMIRNOFFvdWCollection from toolkit data.

        """
        if type(parameter_handler) not in cls.allowed_parameter_handlers():
            raise InvalidParameterHandlerError(
                f"Found parameter handler type {type(parameter_handler)}, which is not "
                f"supported by potential type {type(cls)}",
            )

        handler = cls(
            alpha=parameter_handler.alpha,
            beta=parameter_handler.beta,
            scale_13=parameter_handler.scale13,
            scale_14=parameter_handler.scale14,
            scale_15=parameter_handler.scale15,
            cutoff=parameter_handler.cutoff,
            mixing_rule=parameter_handler.combining_rules.lower(),
            periodic_method=parameter_handler.periodic_method.lower(),
            nonperiodic_method=parameter_handler.nonperiodic_method.lower(),
            switch_width=parameter_handler.switch_width,
        )
        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler

    @classmethod
    def parameter_handler_precedence(cls) -> Iterable[str]:
        """
        Return the order in which parameter handlers take precedence when computing charges.
        """
        return "vdw", "VirtualSites"

    def create_virtual_sites(
        self,
        parameter_handler: VirtualSiteHandler,
        topology: Topology,
    ):
        """create() but with virtual sites."""
        raise NotImplementedError()


class SMIRNOFFC4IonCollection(_SMIRNOFFNonbondedCollection):
    """Handler storing vdW potentials as produced by a SMIRNOFF force field."""

    type: Literal["C4Ion"] = "C4Ion"

    is_plugin: bool = True

    expression: str = "c*r^-4;c=sqrt(c1*c2);"

    periodic_method: str = "cutoff"
    nonperiodic_method: str = "no-cutoff"

    @classmethod
    def allowed_parameter_handlers(cls) -> _HandlerIterable:
        """Return a list of allowed types of ParameterHandler classes."""
        return (C4IonHandler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return a list of supported parameter attributes."""
        return "smirks", "id", "c"

    @classmethod
    def default_parameter_values(cls) -> Iterable[float]:
        """Per-particle parameter values passed to Force.addParticle()."""
        return (0.0,)

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        return "c"

    @classmethod
    def check_openmm_requirements(cls, combine_nonbonded_forces: bool) -> None:
        """Run through a list of assertions about what is compatible when exporting this to OpenMM."""
        assert (
            combine_nonbonded_forces
        ), "The r ** -4 term is only implemented with a single `NonbondedForce`."

    def store_potentials(self, parameter_handler: DoubleExponentialHandler) -> None:
        """
        Populate self.potentials with key-val pairs of [TopologyKey, PotentialKey].

        """
        for potential_key in self.slot_map.values():
            smirks = potential_key.id
            force_field_parameters = parameter_handler.parameters[smirks]

            potential = Potential(
                parameters={
                    parameter: getattr(force_field_parameters, parameter)
                    for parameter in self.potential_parameters()
                },
            )

            self.potentials[potential_key] = potential

    def modify_openmm_forces(
        self,
        interchange,
        system,
        add_constrained_forces,
        constrained_pairs,
        particle_map,
    ):
        """Add a `openmm.CustomNonbondedForce` to handle the C/r^4 term. See 10.1021/jp505875v."""
        import openmm

        from openff.interchange.models import TopologyKey

        non_bonded_forces = [
            force
            for force in system.getForces()
            if isinstance(force, (openmm.NonbondedForce, openmm.CustomNonbondedForce))
        ]

        if len(non_bonded_forces) != 1:
            raise NotImplementedError(
                "This handler is only compatible with a single NonbondedForce/CustomNonbondedForce.",
            )

        if type(non_bonded_forces[0]) is openmm.CustomNonbondedForce:
            raise NotImplementedError(
                "This handler is only compatible with a single NonbondedForce/CustomNonbondedForce.",
            )

        non_bonded_force = non_bonded_forces[0]

        c4_force = openmm.CustomNonbondedForce(self.expression)

        system.addForce(c4_force)

        c4_force.addPerParticleParameter("c")

        for molecule in interchange.topology.molecules:
            for atom in molecule.atoms:
                atom_index = interchange.topology.atom_index(atom)

                top_key = TopologyKey(atom_indices=(atom_index,))

                if top_key not in self.potentials:
                    # This handler only adds interactions to highly-charged ions, so many
                    # atom indices will not be found in the key map. OpenMM requires all
                    # forces have as many particles as are found in the system, so just add
                    # this particle with a zeroed (numerator) parameters.
                    c4_force.addParticle([0.0])

                else:
                    c4_force.addParticle(
                        [self.potentials[top_key].parameters["c"]],
                    )

        # OpenMM requires "All Forces must have identical exclusions", so just copy them over
        for exception_index in range(non_bonded_force.getNumExceptions()):
            p1, p2, *_ = non_bonded_force.getExceptionParameters(exception_index)
            c4_force.addExclusion(p1, p2)

        # Copy non-bonded settings from the main `NonbondedForce`
        c4_force.setNonbondedMethod(non_bonded_force.getNonbondedMethod())
        c4_force.setCutoffDistance(non_bonded_force.getCutoffDistance())
        c4_force.setUseSwitchingFunction(non_bonded_force.getUseSwitchingFunction())
        c4_force.setSwitchingDistance(non_bonded_force.getSwitchingDistance())
        c4_force.setUseLongRangeCorrection(
            non_bonded_force.getUseDispersionCorrection(),
        )
