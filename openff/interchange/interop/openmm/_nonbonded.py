from collections import defaultdict
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Union

import openmm
from openff.units import unit as off_unit
from openff.units.openmm import to_openmm as to_openmm_quantity
from openmm import unit

from openff.interchange.constants import _PME
from openff.interchange.exceptions import (
    InternalInconsistencyError,
    UnsupportedCutoffMethodError,
    UnsupportedExportError,
)
from openff.interchange.interop.parmed import _lj_params_from_potential
from openff.interchange.models import TopologyKey, VirtualSiteKey

if TYPE_CHECKING:
    from openff.toolkit import Molecule

    from openff.interchange import Interchange


_LORENTZ_BERTHELOT = "sigma=(sigma1+sigma2)/2; epsilon=sqrt(epsilon1*epsilon2); "


def _process_nonbonded_forces(
    openff_sys,
    openmm_sys,
    combine_nonbonded_forces=False,
) -> Dict[Union[int, VirtualSiteKey], int]:
    """
    Process the non-bonded handlers in an Interchange into corresponding openmm objects.

    This typically involves processing the vdW and Electrostatics sections of an Interchange object
    into a corresponding openmm.NonbondedForce (if `combine_nonbonded_forces=True`) or a
    collection of other forces (NonbondedForce, CustomNonbondedForce, CustomBondForce) if
    `combine_nonbondoed_forces=False`.

    """
    from openff.interchange.components.smirnoff import _SMIRNOFFNonbondedHandler

    for handler in openff_sys.handlers.values():
        if isinstance(handler, _SMIRNOFFNonbondedHandler):
            break
    else:
        return dict()

    has_virtual_sites = "VirtualSites" in openff_sys.handlers

    if has_virtual_sites:
        from openff.interchange.interop._virtual_sites import (
            _virtual_site_parent_molecule_mapping,
        )
        from openff.interchange.interop.openmm._virtual_sites import (
            _check_virtual_site_exclusion_policy,
        )

        virtual_site_handler = openff_sys["VirtualSites"]

        _check_virtual_site_exclusion_policy(virtual_site_handler)

        virtual_site_molecule_map: Dict[
            VirtualSiteKey,
            "Molecule",
        ] = _virtual_site_parent_molecule_mapping(openff_sys)

        molecule_virtual_site_map: Dict[int, List[VirtualSiteKey]] = defaultdict(list)

        for virtual_site_key, molecule in virtual_site_molecule_map.items():

            molecule_index = openff_sys.topology.molecule_index(molecule)
            molecule_virtual_site_map[molecule_index].append(virtual_site_key)

    else:
        molecule_virtual_site_map = defaultdict(list)

    # Mapping between OpenFF "particles" and OpenMM particles (via inddex). OpenFF objects
    # (keys) are either atom indices (if atoms) or `VirtualSitesKey`s if virtual sites
    # openff_openmm_particle_map: Dict[Union[int, VirtualSiteKey], int] = dict()

    openff_openmm_particle_map = _add_particles_to_system(
        openff_sys,
        openmm_sys,
        molecule_virtual_site_map,
    )

    # TODO: Process ElectrostaticsHandler.exception_potential
    if "vdW" in openff_sys.handlers or "Electrostatics" in openff_sys.handlers:

        _data = _prepare_input_data(openff_sys)

        if combine_nonbonded_forces:
            _func = _create_single_nonbonded_force
        else:
            _func = _create_multiple_nonbonded_forces

        _func(
            _data,
            openff_sys,
            openmm_sys,
            molecule_virtual_site_map,
            openff_openmm_particle_map,
        )

    elif "Buckingham-6" in openff_sys.handlers:
        if has_virtual_sites:
            raise UnsupportedExportError(
                "Virtual sites with Buckingham-6 potential not supported. If this use case is important to you, "
                "please raise an issue describing the functionality you wish to see.",
            )

        buck_handler = openff_sys["Buckingham-6"]

        non_bonded_force = openmm.CustomNonbondedForce(
            "A * exp(-B * r) - C * r ^ -6; A = sqrt(A1 * A2); B = 2 / (1 / B1 + 1 / B2); C = sqrt(C1 * C2)",
        )
        non_bonded_force.addPerParticleParameter("A")
        non_bonded_force.addPerParticleParameter("B")
        non_bonded_force.addPerParticleParameter("C")
        openmm_sys.addForce(non_bonded_force)

        for molecule in openff_sys.topology.molecules:
            for _ in molecule.atoms:
                non_bonded_force.addParticle([0.0, 0.0, 0.0])

        if openff_sys.box is None:
            non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        else:
            non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
            non_bonded_force.setCutoffDistance(buck_handler.cutoff * unit.angstrom)

        for top_key, pot_key in buck_handler.slot_map.items():
            atom_idx = top_key.atom_indices[0]

            # TODO: Add electrostatics
            params = buck_handler.potentials[pot_key].parameters
            a = to_openmm_quantity(params["A"])
            b = to_openmm_quantity(params["B"])
            c = to_openmm_quantity(params["C"])
            non_bonded_force.setParticleParameters(atom_idx, [a, b, c])

        # TODO: In principle this is an i-i mapping between OpenFF and OpenMM atom indices
        return dict()

    else:
        # Here we assume there are no vdW interactions in any handlers
        # vdw_handler = None

        if has_virtual_sites:
            raise UnsupportedExportError(
                "Virtual sites with no vdW handler not currently supported. If this use case is important to you, "
                "please raise an issue describing the functionality you wish to see.",
            )

        try:
            electrostatics_handler = openff_sys["Electrostatics"]
        except LookupError:
            raise InternalInconsistencyError(
                "In a confused state, could not find any vdW interactions but also failed to find "
                "any electrostatics handler. This is a supported use case but should have been caught "
                "earlier in this function. Please file an issue with a minimal reproducing example.",
            )

        electrostatics_method = (
            electrostatics_handler.periodic_potential
            if electrostatics_handler
            else None
        )

        non_bonded_force = openmm.NonbondedForce()
        openmm_sys.addForce(non_bonded_force)

        for molecule in openff_sys.topology.molecules:

            for _ in molecule.atoms:
                non_bonded_force.addParticle(0.0, 1.0, 0.0)

        if electrostatics_method in ["Coulomb", None]:
            non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
            # TODO: Would setting the dispersion correction here have any impact?
        elif electrostatics_method == _PME:
            non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.LJPME)
            non_bonded_force.setEwaldErrorTolerance(1.0e-4)
        else:
            raise UnsupportedCutoffMethodError(
                f"Found no vdW interactions but an electrostatics method of {electrostatics_method}. "
                "This is either unsupported or ambiguous. If you believe this exception has been raised "
                "in error, please file an issue with a minimally reproducing example and your motivation "
                "for this use case.",
            )

    return openff_openmm_particle_map


def _add_particles_to_system(
    openff_sys: "Interchange",
    openmm_sys: openmm.System,
    molecule_virtual_site_map,
) -> Dict[Union[int, VirtualSiteKey], int]:

    has_virtual_sites = molecule_virtual_site_map not in (None, dict())

    openff_openmm_particle_map: Dict[Union[int, VirtualSiteKey], int] = dict()

    for molecule in openff_sys.topology.molecules:

        for atom in molecule.atoms:
            atom_index = openff_sys.topology.atom_index(atom)

            # Skip unit check for speed, trust that the toolkit reports mass in Dalton
            openmm_index = openmm_sys.addParticle(mass=atom.mass.m)

            openff_openmm_particle_map[atom_index] = openmm_index

        if has_virtual_sites:
            molecule_index = openff_sys.topology.molecule_index(molecule)

            for virtual_site_key in molecule_virtual_site_map[molecule_index]:
                openmm_index = openmm_sys.addParticle(mass=0.0)

                openff_openmm_particle_map[virtual_site_key] = openmm_index

    return openff_openmm_particle_map


def _prepare_input_data(openff_sys):
    try:
        vdw_handler = openff_sys["vdW"]
    except LookupError:
        vdw_handler = None

    if vdw_handler:
        vdw_cutoff = vdw_handler.cutoff
        vdw_method = vdw_handler.method.lower()
        mixing_rule = getattr(vdw_handler, "mixing_rule", None)
        vdw_expression = vdw_handler.expression
        vdw_expression = vdw_expression.replace("**", "^")
    else:
        vdw_cutoff = None
        vdw_method = None
        mixing_rule = None
        vdw_expression = "(no handler found)"

    try:
        electrostatics_handler = openff_sys["Electrostatics"]
    except LookupError:
        electrostatics_handler = None

    if electrostatics_handler is None:
        electrostatics_method = None
    else:
        if openff_sys.box is None:
            electrostatics_method = electrostatics_handler.nonperiodic_potential
        else:
            electrostatics_method = electrostatics_handler.periodic_potential

    return {
        "vdw_handler": vdw_handler,
        "vdw_cutoff": vdw_cutoff,
        "vdw_method": vdw_method,
        "vdw_expression": vdw_expression,
        "mixing_rule": mixing_rule,
        "mixing_rule_expression": _LORENTZ_BERTHELOT,
        "electrostatics_handler": electrostatics_handler,
        "electrostatics_method": electrostatics_method,
        "periodic": openff_sys.box is None,
    }


def _create_single_nonbonded_force(
    data: Dict,
    openff_sys: "Interchange",
    openmm_sys: openmm.System,
    molecule_virtual_site_map: Dict["Molecule", List[VirtualSiteKey]],
    openff_openmm_particle_map: Dict[Union[int, VirtualSiteKey], int],
):
    """Create a single openmm.NonbondedForce from vdW/electrostatics/virtual site handlers."""
    if data["mixing_rule"] not in ("lorentz-berthelot", None):
        raise UnsupportedExportError(
            "OpenMM's default NonbondedForce only supports Lorentz-Berthelot mixing rules."
            "Try setting `combine_nonbonded_forces=False`.",
        )

    has_virtual_sites = molecule_virtual_site_map not in (None, dict())

    non_bonded_force = openmm.NonbondedForce()
    openmm_sys.addForce(non_bonded_force)

    if openff_sys.box is None:
        if (data["vdw_method"] in ("cutoff", None)) and (
            data["electrostatics_method"] in ("Coulomb", None)
        ):
            non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
            non_bonded_force.setUseDispersionCorrection(True)
            if data["vdw_cutoff"]:
                non_bonded_force.setCutoffDistance(
                    to_openmm_quantity(data["vdw_cutoff"]),
                )
        else:
            raise UnsupportedCutoffMethodError(
                f"Combination of non-bonded cutoff methods {data['vdw_method']} (vdW) and "
                f"{data['electrostatics_method']} (Electrostatics) not currently supported or "
                f"invalid with `combine_nonbonded_forces=True` and `.box={openff_sys.box}`.",
            )

    else:
        if data["vdw_method"] in ("cutoff", None) and data["electrostatics_method"] in (
            _PME,
            None,
        ):
            non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
            non_bonded_force.setEwaldErrorTolerance(1.0e-4)
            non_bonded_force.setUseDispersionCorrection(True)
            if not data["vdw_cutoff"]:
                # With no vdW handler and/or ambiguous cutoff, cannot set it,
                # thereforce silently fall back to OpenMM's default. It's not
                # clear if this value matters with only (PME) charges and no
                # vdW interactions in the system.
                pass
            else:
                non_bonded_force.setCutoffDistance(
                    to_openmm_quantity(data["vdw_cutoff"]),
                )
        elif data["vdw_method"] == "pme" and data["electrostatics_method"] == _PME:
            non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.LJPME)
            non_bonded_force.setEwaldErrorTolerance(1.0e-4)
        else:
            raise UnsupportedCutoffMethodError(
                f"Combination of non-bonded cutoff methods {data['vdw_method']} (vdW) and "
                "{data['electrostatics_method']} (Electrostatics) not currently supported or invalid with "
                f"`combine_nonbonded_forces=True` and `.box={openff_sys.box}`.",
            )

    if data["electrostatics_handler"] is not None:
        try:
            partial_charges = data["electrostatics_handler"].charges_with_virtual_sites
        except AttributeError:
            partial_charges = data["electrostatics_handler"].charges

    # mapping between (openmm) index of each atom and the (openmm) index of each virtual particle
    #   of that parent atom (if any)
    # if no virtual sites at all, this remains an empty dict
    parent_virtual_particle_mapping: DefaultDict[int, List[int]] = defaultdict(list)

    for molecule in openff_sys.topology.molecules:

        for atom in molecule.atoms:

            non_bonded_force.addParticle(0.0, 1.0, 0.0)

            atom_index = openff_sys.topology.atom_index(atom)

            top_key = TopologyKey(atom_indices=(atom_index,))

            if data["electrostatics_handler"] is not None:
                partial_charge = partial_charges[top_key].m_as(off_unit.e)
            else:
                partial_charge = 0.0

            if data["vdw_handler"] is not None:
                pot_key = data["vdw_handler"].slot_map[top_key]
                sigma, epsilon = _lj_params_from_potential(
                    data["vdw_handler"].potentials[pot_key],
                )
                sigma = sigma.m_as(off_unit.nanometer)
                epsilon = epsilon.m_as(off_unit.kilojoule / off_unit.mol)
            else:
                sigma = unit.Quantity(0.0, unit.nanometer)
                epsilon = unit.Quantity(0.0, unit.kilojoules_per_mole)

            openmm_atom_index = openff_openmm_particle_map[atom_index]

            non_bonded_force.setParticleParameters(
                openmm_atom_index,
                partial_charge,
                sigma,
                epsilon,
            )

        if has_virtual_sites:
            molecule_index = openff_sys.topology.molecule_index(molecule)
        else:
            continue

        for virtual_site_key in molecule_virtual_site_map[molecule_index]:
            # TODO: Move this function to openff/interchange/interop/_particles.py ?
            from openff.interchange.interop.openmm._virtual_sites import (
                _create_openmm_virtual_site,
                _create_virtual_site_object,
            )

            _potential_key = openff_sys["VirtualSites"].slot_map[virtual_site_key]
            virtual_site_potential = openff_sys["VirtualSites"].potentials[
                _potential_key
            ]
            virtual_site_object = _create_virtual_site_object(
                virtual_site_key,
                virtual_site_potential,
            )

            openmm_particle = _create_openmm_virtual_site(
                virtual_site_object,
                openff_openmm_particle_map,
            )

            vdw_handler = openff_sys["vdW"]
            coul_handler = openff_sys["Electrostatics"]

            vdw_key = vdw_handler.slot_map.get(virtual_site_key)  # type: ignore[call-overload]
            coul_key = coul_handler.slot_map.get(virtual_site_key)  # type: ignore[call-overload]
            if vdw_key is None or coul_key is None:
                raise Exception(
                    f"Virtual site {virtual_site_key} is not associated with any "
                    "vdW and/or electrostatics interactions",
                )

            charge_increments = coul_handler.potentials[coul_key].parameters[
                "charge_increments"
            ]
            charge = to_openmm_quantity(-sum(charge_increments))

            vdw_parameters = vdw_handler.potentials[vdw_key].parameters
            sigma = to_openmm_quantity(vdw_parameters["sigma"])
            epsilon = to_openmm_quantity(vdw_parameters["epsilon"])

            index_system: int = openff_openmm_particle_map[virtual_site_key]
            index_force = non_bonded_force.addParticle(charge, sigma, epsilon)

            if index_system != index_force:
                raise InternalInconsistencyError(
                    "Mismatch in system and force indexing",
                )

            parent_atom_index = openff_openmm_particle_map[
                virtual_site_object.orientations[0]
            ]

            parent_virtual_particle_mapping[parent_atom_index].append(index_force)

            openmm_sys.setVirtualSite(index_system, openmm_particle)

    _create_exceptions(
        data,
        non_bonded_force,
        openff_sys,
        openmm_sys,
        openff_openmm_particle_map,
        parent_virtual_particle_mapping,
    )

    _apply_switching_function(data["vdw_handler"], non_bonded_force)


def _create_exceptions(
    data: Dict,
    non_bonded_force: openmm.NonbondedForce,
    openff_sys: "Interchange",
    openmm_sys,
    openff_openmm_particle_map: Dict,
    parent_virtual_particle_mapping: DefaultDict[int, List[int]],
):
    # The topology indices reported by toolkit methods must be converted to openmm indices
    bonds = [
        sorted(
            openff_openmm_particle_map[openff_sys.topology.atom_index(a)]
            for a in bond.atoms
        )
        for bond in openff_sys.topology.bonds
    ]

    coul_14 = getattr(data["electrostatics_handler"], "scale_14", 1.0)
    vdw_14 = getattr(data["vdw_handler"], "scale_14", 1.0)

    # First, create all atom-atom exceptions according to the conventional pattern
    non_bonded_force.createExceptionsFromBonds(
        bonds=bonds,
        coulomb14Scale=coul_14,
        lj14Scale=vdw_14,
    )

    # Faster to loop through exceptions and look up parents than opposite
    if parent_virtual_particle_mapping not in (None, dict()):
        # First add exceptions between each virtual particle and parent atom
        for (
            parent,
            virtual_particles_of_this_parent,
        ) in parent_virtual_particle_mapping.items():
            for virtual_particle in virtual_particles_of_this_parent:
                non_bonded_force.addException(
                    parent,
                    virtual_particle,
                    0.0,
                    0.0,
                    0.0,
                )

        for exception_index in range(non_bonded_force.getNumExceptions()):
            # These particles should only be atoms in this loop
            (
                p1,
                p2,
                charge_prod,
                _,
                epsilon,
            ) = non_bonded_force.getExceptionParameters(exception_index)
            for virtual_particle_of_p1 in parent_virtual_particle_mapping[p1]:
                # If this iterable is not empty, add an exception between p1's virtual
                # particle and the "other" atom in p1's exception
                if virtual_particle_of_p1 == p2:
                    continue

                if charge_prod._value == epsilon._value == 0.0:
                    non_bonded_force.addException(
                        virtual_particle_of_p1, p2, 0.0, 0.0, 0.0, replace=True
                    )
                else:
                    # TODO: Decide on best logic for inheriting scaled 1-4 interactions
                    raise Exception
            for virtual_particle_of_p2 in parent_virtual_particle_mapping[p2]:
                # If this iterable is not empty, add an exception between p1's virtual
                # particle and the "other" atom in p1's exception
                if virtual_particle_of_p2 == p1:
                    continue

                if charge_prod._value == epsilon._value == 0.0:
                    non_bonded_force.addException(
                        virtual_particle_of_p2, p1, 0.0, 0.0, 0.0, replace=True
                    )
                else:
                    # TODO: Decide on best logic for inheriting scaled 1-4 interactions
                    raise Exception


def _create_multiple_nonbonded_forces(
    data: Dict,
    openff_sys: "Interchange",
    openmm_sys: openmm.System,
    molecule_virtual_site_map: Dict,
    openff_openmm_particle_map: Dict[Union[int, VirtualSiteKey], int],
):
    vdw_force = openmm.CustomNonbondedForce(
        data["vdw_expression"] + "; " + data["mixing_rule_expression"],
    )
    vdw_force.addPerParticleParameter("sigma")
    vdw_force.addPerParticleParameter("epsilon")

    openmm_sys.addForce(vdw_force)

    has_virtual_sites = molecule_virtual_site_map not in (None, dict())

    for molecule in openff_sys.topology.molecules:
        for _ in molecule.atoms:
            vdw_force.addParticle([1.0, 0.0])

        if has_virtual_sites:
            molecule_index = openff_sys.topology.molecule_index(molecule)
            for _ in molecule_virtual_site_map[molecule_index]:
                vdw_force.addParticle(0.0, 1.0, 0.0)

    if data["vdw_method"] == "cutoff":
        if openff_sys.box is None:
            vdw_force.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
        else:
            vdw_force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        vdw_force.setUseLongRangeCorrection(True)
        vdw_force.setCutoffDistance(to_openmm_quantity(data["vdw_cutoff"]))

        _apply_switching_function(data["vdw_handler"], vdw_force)

    elif data["vdw_method"] == "pme":
        if openff_sys.box is None:
            raise UnsupportedCutoffMethodError(
                "vdW method pme/ljpme is not valid for non-periodic systems.",
            )
        else:
            raise UnsupportedCutoffMethodError(
                "LJ-PME with split non-bonded forces is not supported due to openmm.CustomNonbondedForce "
                "not supporting PME. If also using PME electrostatics, try `combine_nonbonded_forces=True`,  "
                "which should produce a single force with NonbondedForce.LJPME, which uses PME for both "
                "electrostatics and LJ forces tersm. If your use case would benenfit from split non-bonded "
                "forces with LJPME, please file an feature request.",
            )

    electrostatics_force = openmm.NonbondedForce()

    openmm_sys.addForce(electrostatics_force)

    for molecule in openff_sys.topology.molecules:
        for _ in molecule.atoms:
            electrostatics_force.addParticle(0.0, 1.0, 0.0)

        if has_virtual_sites:
            molecule_index = openff_sys.topology.molecule_index(molecule)
            for _ in molecule_virtual_site_map[molecule_index]:
                vdw_force.addParticle(0.0, 1.0, 0.0)

    if data["electrostatics_method"] == "reaction-field":
        raise UnsupportedExportError(
            "Reaction field electrostatics not supported. If this use case is important to you, "
            "please raise an issue describing the scope of functionality you would like to use.",
        )

    elif data["electrostatics_method"] == _PME:
        electrostatics_force.setNonbondedMethod(openmm.NonbondedForce.PME)
        electrostatics_force.setEwaldErrorTolerance(1.0e-4)
        electrostatics_force.setUseDispersionCorrection(True)
        if data["vdw_cutoff"] is not None:
            # All nonbonded forces must use the same cutoff, even though PME doesn't have a cutoff
            electrostatics_force.setCutoffDistance(
                to_openmm_quantity(data["vdw_cutoff"]),
            )
    elif data["electrostatics_method"] == "Coulomb":
        if openff_sys.box is None:
            electrostatics_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        else:
            raise UnsupportedCutoffMethodError(
                f"Electrostatics method {data['electrostatics_method']} ambiguous with a periodic system.",
            )

    elif data["electrostatics_method"] == "cutoff":
        raise UnsupportedCutoffMethodError(
            "OpenMM does not clearly support cut-off electrostatics with no reaction-field attenuation.",
        )
    else:
        raise UnsupportedCutoffMethodError(
            f"Electrostatics method {data['electrostatics_method']} not supported",
        )

    if data["electrostatics_handler"] is not None:
        try:
            partial_charges = data["electrostatics_handler"].charges_with_virtual_sites
        except AttributeError:
            partial_charges = data["electrostatics_handler"].charges

    for molecule in openff_sys.topology.molecules:
        for atom in molecule.atoms:
            atom_index = openff_sys.topology.atom_index(atom)
            # TODO: Actually process virtual site vdW parameters here

            top_key = TopologyKey(atom_indices=(atom_index,))

            if data["electrostatics_handler"] is not None:
                partial_charge = partial_charges[top_key].m_as(off_unit.e)
            else:
                partial_charge = 0.0

            if data["vdw_handler"] is not None:
                pot_key = data["vdw_handler"].slot_map[top_key]
                sigma, epsilon = _lj_params_from_potential(
                    data["vdw_handler"].potentials[pot_key],
                )
                sigma = sigma.m_as(off_unit.nanometer)
                epsilon = epsilon.m_as(off_unit.kilojoule / off_unit.mol)
            else:
                sigma = unit.Quantity(0.0, unit.nanometer)
                epsilon = unit.Quantity(0.0, unit.kilojoules_per_mole)

            vdw_force.setParticleParameters(atom_index, [sigma, epsilon])
            electrostatics_force.setParticleParameters(
                atom_index,
                partial_charge,
                0.0,
                0.0,
            )

    # Attempting to match the value used internally by OpenMM; The source of this value is likely
    # https://github.com/openmm/openmm/issues/1149#issuecomment-250299854
    # 1 / * (4pi * eps0) * elementary_charge ** 2 / nanometer ** 2
    coul_const = 138.935456  # kJ/nm

    vdw_14_force = openmm.CustomBondForce("4*epsilon*((sigma/r)^12-(sigma/r)^6)")
    vdw_14_force.addPerBondParameter("sigma")
    vdw_14_force.addPerBondParameter("epsilon")
    vdw_14_force.setUsesPeriodicBoundaryConditions(True)
    coul_14_force = openmm.CustomBondForce(f"{coul_const}*qq/r")
    coul_14_force.addPerBondParameter("qq")
    coul_14_force.setUsesPeriodicBoundaryConditions(True)

    openmm_sys.addForce(vdw_14_force)
    openmm_sys.addForce(coul_14_force)

    bonds = [
        sorted(openff_sys.topology.atom_index(a) for a in bond.atoms)
        for bond in openff_sys.topology.bonds
    ]

    coul_14 = (
        data["electrostatics_handler"].scale_14
        if "electrostatics_handler" in data
        else 1.0
    )
    vdw_14 = data["vdw_handler"].scale_14 if "vdw_handler" in data else 1.0

    electrostatics_force.createExceptionsFromBonds(
        bonds=bonds,
        coulomb14Scale=coul_14,
        lj14Scale=vdw_14,
    )

    for i in range(electrostatics_force.getNumExceptions()):
        (p1, p2, q, sig, eps) = electrostatics_force.getExceptionParameters(i)

        # If the interactions are both zero, assume this is a 1-2 or 1-3 interaction
        if q._value == 0 and eps._value == 0:
            pass
        else:
            # Assume this is a 1-4 interaction
            # Look up the vdW parameters for each particle
            sig1, eps1 = vdw_force.getParticleParameters(p1)
            sig2, eps2 = vdw_force.getParticleParameters(p2)
            q1, _, _ = electrostatics_force.getParticleParameters(p1)
            q2, _, _ = electrostatics_force.getParticleParameters(p2)

            # manually compute and set the 1-4 interactions
            sig_14 = (sig1 + sig2) * 0.5
            eps_14 = (eps1 * eps2) ** 0.5 * vdw_14
            qq = q1 * q2 * coul_14

            vdw_14_force.addBond(p1, p2, [sig_14, eps_14])
            coul_14_force.addBond(p1, p2, [qq])
        vdw_force.addExclusion(p1, p2)
        # electrostatics_force.addExclusion(p1, p2)
        electrostatics_force.setExceptionParameters(i, p1, p2, 0.0, 0.0, 0.0)
        # vdw_force.setExceptionParameters(i, p1, p2, 0.0, 0.0, 0.0)


def _apply_switching_function(vdw_handler, force: openmm.NonbondedForce):
    if not hasattr(force, "setUseSwitchingFunction"):
        raise ValueError(
            "Attempting to set switching funcntion on an OpenMM force that does nont support it."
            f"Passed force of type {type(force)}.",
        )
    if getattr(vdw_handler, "switch_width", None) is None:
        force.setUseSwitchingFunction(False)
    elif vdw_handler.switch_width.m == 0.0:
        force.setUseSwitchingFunction(False)
    else:
        switching_distance = to_openmm_quantity(
            vdw_handler.cutoff - vdw_handler.switch_width,
        )

        if switching_distance._value < 0:
            raise UnsupportedCutoffMethodError(
                "Found a `switch_width` greater than the cutoff distance. It's not clear "
                "what this means and it's probably invalid. Found "
                f"switch_width{vdw_handler.switch_width} and cutoff {vdw_handler.cutoff}",
            )

        force.setUseSwitchingFunction(True)
        force.setSwitchingDistance(switching_distance)
