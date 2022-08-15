"""Interfaces with OpenMM."""
from collections import defaultdict
from pathlib import Path
from typing import Union

import openmm
from openff.toolkit.topology import Topology
from openff.units import unit as off_unit
from openff.units.openmm import from_openmm as from_openmm_quantity
from openff.units.openmm import to_openmm as to_openmm_quantity
from openmm import unit

from openff.interchange.components.potentials import Potential
from openff.interchange.constants import _PME
from openff.interchange.exceptions import (
    InternalInconsistencyError,
    UnsupportedImportError,
)
from openff.interchange.interop.openmm._nonbonded import _process_nonbonded_forces
from openff.interchange.interop.openmm._positions import to_openmm_positions  # noqa
from openff.interchange.interop.openmm._topology import to_openmm_topology  # noqa
from openff.interchange.interop.openmm._valence import (
    _process_angle_forces,
    _process_bond_forces,
    _process_constraints,
    _process_improper_torsion_forces,
    _process_torsion_forces,
)
from openff.interchange.models import PotentialKey, TopologyKey, VirtualSiteKey


def to_openmm(
    openff_sys,
    combine_nonbonded_forces: bool = False,
    add_constrained_forces: bool = False,
) -> openmm.System:
    """
    Convert an Interchange to an OpenmM System.

    Parameters
    ----------
    openff_sys : openff.interchange.Interchange
        An OpenFF Interchange object
    combine_nonbonded_forces : bool, default=False
        If True, an attempt will be made to combine all non-bonded interactions into a single openmm.NonbondedForce.
        If False, non-bonded interactions will be split across multiple forces.
    add_constrained_forces : bool, default=False,
        If True, add valence forces that might be overridden by constraints, i.e. call `addBond` or `addAngle`
        on a bond or angle that is fully constrained.

    Returns
    -------
    openmm_sys : openmm.System
        The corresponding OpenMM System object

    """
    openmm_sys = openmm.System()

    # OpenFF box stored implicitly as nm, and that happens to be what
    # OpenMM casts box vectors to if provided only an np.ndarray
    if openff_sys.box is not None:
        box = openff_sys.box.m_as(off_unit.nanometer)
        openmm_sys.setDefaultPeriodicBoxVectors(*box)

    particle_map = _process_nonbonded_forces(
        openff_sys, openmm_sys, combine_nonbonded_forces=combine_nonbonded_forces
    )
    constrained_pairs = _process_constraints(openff_sys, openmm_sys, particle_map)
    _process_torsion_forces(openff_sys, openmm_sys)
    _process_improper_torsion_forces(openff_sys, openmm_sys)
    _process_angle_forces(
        openff_sys,
        openmm_sys,
        add_constrained_forces=add_constrained_forces,
        constrained_pairs=constrained_pairs,
    )
    _process_bond_forces(
        openff_sys,
        openmm_sys,
        add_constrained_forces=add_constrained_forces,
        constrained_pairs=constrained_pairs,
    )

    return openmm_sys


def _process_virtual_sites(openff_sys, openmm_sys):
    from openff.interchange.components._particles import (
        _BondChargeVirtualSite,
        _create_openmm_virtual_site,
        _DivalentLonePairVirtualSite,
        _MonovalentLonePairVirtualSite,
        _TrivalentLonePairVirtualSite,
    )

    try:
        virtual_site_handler = openff_sys["VirtualSites"]
    except LookupError:
        return

    from openff.interchange.interop.openmm._virtual_sites import (
        _check_virtual_site_exclusion_policy,
    )

    _check_virtual_site_exclusion_policy(virtual_site_handler)

    vdw_handler = openff_sys["vdW"]
    coul_handler = openff_sys["Electrostatics"]

    # TODO: Handle case of split-out non-bonded forces
    non_bonded_force = [
        f for f in openmm_sys.getForces() if type(f) == openmm.NonbondedForce
    ][0]

    virtual_site_key: VirtualSiteKey

    # Later, we will need to add exclusions between virtual site particles, which unfortunately
    # requires tracking which added virtual site particle is associated with which atom
    parent_particle_mapping = defaultdict(list)

    for (
        virtual_site_key,
        virtual_site_potential_key,
    ) in virtual_site_handler.slot_map.items():
        orientations = virtual_site_key.orientation_atom_indices
        parent_atom = orientations[0]

        virutal_site_potential_object = virtual_site_handler.potentials[
            virtual_site_potential_key
        ]

        if virtual_site_key.type == "BondCharge":
            virtual_site_object = _BondChargeVirtualSite(
                type="BondCharge",
                distance=virutal_site_potential_object.parameters["distance"],
                orientations=orientations,
            )
        elif virtual_site_key.type == "MonovalentLonePair":
            virtual_site_object = _MonovalentLonePairVirtualSite(
                type="MonovalentLonePair",
                distance=virutal_site_potential_object.parameters["distance"],
                out_of_plane_angle=virutal_site_potential_object.parameters[
                    "outOfPlaneAngle"
                ],
                in_plane_angle=virutal_site_potential_object.parameters["inPlaneAngle"],
                orientations=orientations,
            )
        elif virtual_site_key.type == "DivalentLonePair":
            virtual_site_object = _DivalentLonePairVirtualSite(
                type="DivalentLonePair",
                distance=virutal_site_potential_object.parameters["distance"],
                out_of_plane_angle=virutal_site_potential_object.parameters[
                    "outOfPlaneAngle"
                ],
                orientations=orientations,
            )
        elif virtual_site_key.type == "TrivalentLonePair":
            virtual_site_object = _TrivalentLonePairVirtualSite(
                type="TrivalentLonePair",
                distance=virutal_site_potential_object.parameters["distance"],
                orientations=orientations,
            )

        else:
            raise NotImplementedError(virtual_site_key.type)

        openmm_particle = _create_openmm_virtual_site(
            virtual_site_object,
            orientations,
        )

        vdw_key = vdw_handler.slot_map.get(virtual_site_key)
        coul_key = coul_handler.slot_map.get(virtual_site_key)
        if vdw_key is None or coul_key is None:
            raise Exception(
                f"Virtual site {virtual_site_key} is not associated with any "
                "vdW and/or electrostatics interactions"
            )

        charge_increments = coul_handler.potentials[coul_key].parameters[
            "charge_increments"
        ]
        charge = to_openmm_quantity(-sum(charge_increments))

        vdw_parameters = vdw_handler.potentials[vdw_key].parameters
        sigma = to_openmm_quantity(vdw_parameters["sigma"])
        epsilon = to_openmm_quantity(vdw_parameters["epsilon"])

        index_system = openmm_sys.addParticle(mass=0.0)
        index_force = non_bonded_force.addParticle(charge, sigma, epsilon)

        if index_system != index_force:
            raise InternalInconsistencyError("Mismatch in system and force indexing")

        parent_particle_mapping[parent_atom].append(index_force)

        openmm_sys.setVirtualSite(index_system, openmm_particle)

        if virtual_site_handler.exclusion_policy == "parents":
            for orientation_atom_index in orientations:
                non_bonded_force.addException(
                    orientation_atom_index, index_force, 0.0, 0.0, 0.0, replace=True
                )

                # This dict only contains `orientation_atom_index` if that orientation atom (of this
                # particle) is coincidentally a parent atom of another particle. This is probably
                # not common but must be dealt with separately.
                for other_particle_index in parent_particle_mapping[parent_atom]:
                    if other_particle_index in orientations:
                        continue
                    non_bonded_force.addException(
                        other_particle_index, index_force, 0.0, 0.0, 0.0, replace=True
                    )


def from_openmm(topology=None, system=None, positions=None, box_vectors=None):
    """Create an Interchange object from OpenMM data."""
    from openff.interchange import Interchange

    openff_sys = Interchange()

    if system:
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                vdw, coul = _convert_nonbonded_force(force)
                openff_sys.handlers["vdW"] = vdw
                openff_sys.handlers["Electrostatics"] = coul
            elif isinstance(force, openmm.HarmonicBondForce):
                bond_handler = _convert_harmonic_bond_force(force)
                openff_sys.handlers["Bonds"] = bond_handler
            elif isinstance(force, openmm.HarmonicAngleForce):
                angle_handler = _convert_harmonic_angle_force(force)
                openff_sys.handlers["Angles"] = angle_handler
            elif isinstance(force, openmm.PeriodicTorsionForce):
                proper_torsion_handler = _convert_periodic_torsion_force(force)
                openff_sys.handlers["ProperTorsions"] = proper_torsion_handler
            else:
                raise UnsupportedImportError(
                    "Unsupported OpenMM Force type ({type(force)}) found."
                )

    if topology is not None:
        from openff.interchange.components.toolkit import _simple_topology_from_openmm

        openff_topology = _simple_topology_from_openmm(topology)

        openff_sys.topology = openff_topology

    if positions is not None:
        openff_sys.positions = positions

    if box_vectors is not None:
        openff_sys.box = box_vectors

    return openff_sys


def _convert_nonbonded_force(force):
    from openff.interchange.components.smirnoff import (
        SMIRNOFFElectrostaticsHandler,
        SMIRNOFFvdWHandler,
    )

    vdw_handler = SMIRNOFFvdWHandler()
    electrostatics = SMIRNOFFElectrostaticsHandler(version=0.4, scale_14=0.833333)

    n_parametrized_particles = force.getNumParticles()

    for idx in range(n_parametrized_particles):
        charge, sigma, epsilon = force.getParticleParameters(idx)
        top_key = TopologyKey(atom_indices=(idx,))
        pot_key = PotentialKey(id=f"{idx}")
        pot = Potential(
            parameters={
                "sigma": from_openmm_quantity(sigma),
                "epsilon": from_openmm_quantity(epsilon),
            }
        )
        vdw_handler.slot_map.update({top_key: pot_key})
        vdw_handler.potentials.update({pot_key: pot})

        electrostatics.slot_map.update({top_key: pot_key})
        electrostatics.potentials.update(
            {pot_key: Potential(parameters={"charge": from_openmm_quantity(charge)})}
        )

    if force.getNonbondedMethod() == openmm.NonbondedForce.PME:
        electrostatics.periodic_potential = _PME
        vdw_handler.method = "cutoff"
    if force.getNonbondedMethod() == openmm.NonbondedForce.LJPME:
        electrostatics.periodic_potential = _PME
        vdw_handler.method = "PME"
    elif force.getNonbondedMethod() in {
        openmm.NonbondedForce.CutoffPeriodic,
        openmm.NonbondedForce.CutoffNonPeriodic,
    }:
        # TODO: Store reaction-field dielectric
        electrostatics.periodic_potential = "reaction-field"
        vdw_handler.method = "cutoff"
    elif force.getNonbondedMethod() == openmm.NonbondedForce.NoCutoff:
        electrostatics.periodic_potential = "Coulomb"
        vdw_handler.method = "Coulomb"

    if vdw_handler.method == "cutoff":
        vdw_handler.cutoff = force.getCutoffDistance()
    electrostatics.cutoff = force.getCutoffDistance()

    return vdw_handler, electrostatics


def _convert_harmonic_bond_force(force):
    from openff.interchange.components.smirnoff import SMIRNOFFBondHandler

    bond_handler = SMIRNOFFBondHandler()

    n_parametrized_bonds = force.getNumBonds()

    for idx in range(n_parametrized_bonds):
        atom1, atom2, length, k = force.getBondParameters(idx)
        top_key = TopologyKey(atom_indices=(atom1, atom2))
        pot_key = PotentialKey(id=f"{atom1}-{atom2}")
        pot = Potential(
            parameters={
                "length": from_openmm_quantity(length),
                "k": from_openmm_quantity(k),
            }
        )

        bond_handler.slot_map.update({top_key: pot_key})
        bond_handler.potentials.update({pot_key: pot})

    return bond_handler


def _convert_harmonic_angle_force(force):
    from openff.interchange.components.smirnoff import SMIRNOFFAngleHandler

    angle_handler = SMIRNOFFAngleHandler()

    n_parametrized_angles = force.getNumAngles()

    for idx in range(n_parametrized_angles):
        atom1, atom2, atom3, angle, k = force.getAngleParameters(idx)
        top_key = TopologyKey(atom_indices=(atom1, atom2, atom3))
        pot_key = PotentialKey(id=f"{atom1}-{atom2}-{atom3}")
        pot = Potential(
            parameters={
                "angle": from_openmm_quantity(angle),
                "k": from_openmm_quantity(k),
            }
        )

        angle_handler.slot_map.update({top_key: pot_key})
        angle_handler.potentials.update({pot_key: pot})

    return angle_handler


def _convert_periodic_torsion_force(force):
    # TODO: Can impropers be separated out from a PeriodicTorsionForce?
    # Maybe by seeing if a quartet is in mol/top.propers or .impropers
    from openff.interchange.components.smirnoff import SMIRNOFFProperTorsionHandler

    proper_torsion_handler = SMIRNOFFProperTorsionHandler()

    n_parametrized_torsions = force.getNumTorsions()

    for idx in range(n_parametrized_torsions):
        atom1, atom2, atom3, atom4, per, phase, k = force.getTorsionParameters(idx)
        # TODO: Process layered torsions
        top_key = TopologyKey(atom_indices=(atom1, atom2, atom3, atom4), mult=0)
        while top_key in proper_torsion_handler.slot_map:
            top_key.mult: int = top_key.mult + 1

        pot_key = PotentialKey(id=f"{atom1}-{atom2}-{atom3}-{atom4}", mult=top_key.mult)
        pot = Potential(
            parameters={
                "periodicity": int(per) * unit.dimensionless,
                "phase": from_openmm_quantity(phase),
                "k": from_openmm_quantity(k),
                "idivf": 1 * unit.dimensionless,
            }
        )

        proper_torsion_handler.slot_map.update({top_key: pot_key})
        proper_torsion_handler.potentials.update({pot_key: pot})

    return proper_torsion_handler


def _to_pdb(file_path: Union[Path, str], topology: Topology, positions):
    from openff.units.openmm import to_openmm
    from openmm import app

    openmm_topology = topology.to_openmm(ensure_unique_atom_names=False)

    positions = to_openmm(positions)

    with open(file_path, "w") as outfile:
        app.PDBFile.writeFile(openmm_topology, positions, outfile)


def get_nonbonded_force_from_openmm_system(omm_system):
    """Get a single NonbondedForce object with an OpenMM System."""
    for force in omm_system.getForces():
        if type(force) == openmm.NonbondedForce:
            return force


def get_partial_charges_from_openmm_system(omm_system):
    """Get partial charges from an OpenMM interchange as a unit.Quantity array."""
    # TODO: deal with virtual sites
    n_particles = omm_system.getNumParticles()
    force = get_nonbonded_force_from_openmm_system(omm_system)
    # TODO: don't assume the partial charge will always be parameter 0
    # partial_charges = [openmm_to_pint(force.getParticleParameters(idx)[0]) for idx in range(n_particles)]
    partial_charges = [
        force.getParticleParameters(idx)[0] / unit.elementary_charge
        for idx in range(n_particles)
    ]

    return partial_charges
