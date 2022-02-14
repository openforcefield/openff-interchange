"""Interfaces with InterMol."""
from typing import List, Union

from intermol.forces import (
    HarmonicAngle,
    HarmonicBond,
    TrigDihedral,
    convert_dihedral_from_trig_to_proper,
)
from intermol.system import System
from openff.toolkit.topology.topology import Topology
from openff.units import unit
from openff.units.openmm import from_openmm

from openff.interchange import Interchange
from openff.interchange.components.base import (
    BaseAngleHandler,
    BaseBondHandler,
    BaseElectrostaticsHandler,
    BaseImproperTorsionHandler,
    BaseProperTorsionHandler,
    BasevdWHandler,
)
from openff.interchange.components.potentials import Potential
from openff.interchange.models import PotentialKey, TopologyKey


def from_intermol_system(intermol_system: System) -> Interchange:
    """Convert and Intermol `System` to an `Interchange` object."""
    interchange = Interchange()

    interchange.box = intermol_system.box_vector
    interchange.positions = from_openmm([a.position for a in intermol_system.atoms])

    vdw_handler = BasevdWHandler(
        scale_14=intermol_system.lj_correction,
        mixing_rule=intermol_system.combination_rule,
    )

    if vdw_handler.mixing_rule == "Multiply-Sigeps":
        vdw_handler.mixing_rule = "geometric"

    electrostatics_handler = BaseElectrostaticsHandler(
        scale_14=intermol_system.coulomb_correction
    )

    bond_handler = BaseBondHandler()
    angle_handler = BaseAngleHandler()
    proper_handler = BaseProperTorsionHandler()
    improper_handler = BaseImproperTorsionHandler()

    # TODO: Store atomtypes on a minimal topology, not as a list
    atomtypes: List = [atom.atomtype[0] for atom in intermol_system.atoms]

    topology = Topology()

    # TODO: Either add molecule-by-molecule or splice into molecules later
    for atom in intermol_system.atoms:
        topology.add_atom(
            atomic_number=atom.atomic_number,
        )
        topology_key = TopologyKey(atom_indices=(atom.index - 1,))
        vdw_key = PotentialKey(id=atom.atomtype[0], associated_handler="vdW")
        electrostatics_key = PotentialKey(
            id=atom.atomtype[0], associated_handler="Electrostatics"
        )

        # Intermol has an abstraction layer for multiple states, though only one is implemented
        charge = from_openmm(atom.charge[0])
        sigma = atom.sigma[0]
        epsilon = atom.epsilon[0]

        vdw_handler.slot_map[topology_key] = vdw_key
        electrostatics_handler.slot_map[topology_key] = electrostatics_key

        vdw_handler.potentials[vdw_key] = Potential(
            parameters={"sigma": sigma, "epsilon": epsilon}
        )
        electrostatics_handler.potentials[electrostatics_key] = Potential(
            parameters={"charge": charge}
        )

    for molecule_type in intermol_system.molecule_types.values():
        for bond_force in molecule_type.bond_forces:
            if type(bond_force) != HarmonicBond:
                raise Exception

            topology.add_bond(
                atom1=topology._atoms[bond_force.atom1 - 1],  # type: ignore[attr-defined]
                atom2=topology._atoms[bond_force.atom2 - 1],  # type: ignore[attr-defined]
            )

            topology_key = TopologyKey(
                atom_indices=tuple(
                    val - 1 for val in [bond_force.atom1, bond_force.atom2]
                ),
            )
            potential_key = PotentialKey(
                id=f"{atomtypes[bond_force.atom1-1]}-{atomtypes[bond_force.atom2-1]}",
                associated_handler="Bonds",
            )

            bond_handler.slot_map[topology_key] = potential_key

            if potential_key not in bond_handler:
                potential = Potential(
                    parameters={
                        "k": from_openmm(bond_force.k),
                        "length": from_openmm(bond_force.length),
                    }
                )

                bond_handler.potentials[potential_key] = potential

        for angle_force in molecule_type.angle_forces:
            if type(angle_force) != HarmonicAngle:
                raise Exception

            topology_key = TopologyKey(
                atom_indices=(
                    tuple(
                        val - 1
                        for val in [
                            angle_force.atom1,
                            angle_force.atom2,
                            angle_force.atom3,
                        ]
                    )
                ),
            )
            potential_key = PotentialKey(
                id=(
                    f"{atomtypes[angle_force.atom1-1]}-{atomtypes[angle_force.atom2-1]}-"
                    f"{atomtypes[angle_force.atom3-1]}"
                ),
                associated_handler="Angles",
            )

            angle_handler.slot_map[topology_key] = potential_key

            if potential_key not in angle_handler.potentials:
                potential = Potential(
                    parameters={
                        "k": from_openmm(angle_force.k),
                        "angle": from_openmm(angle_force.theta),
                    }
                )

                angle_handler.potentials[potential_key] = potential

        for dihedral_force in molecule_type.dihedral_forces:
            if dihedral_force.improper:
                handler = improper_handler
            else:
                handler = proper_handler  # type: ignore[assignment]

            if type(dihedral_force) == TrigDihedral:
                dihedral_parameters = convert_dihedral_from_trig_to_proper(
                    {
                        "fc0": dihedral_force.fc0,
                        "fc1": dihedral_force.fc1,
                        "fc2": dihedral_force.fc2,
                        "fc3": dihedral_force.fc3,
                        "fc4": dihedral_force.fc4,
                        "fc5": dihedral_force.fc5,
                        "fc6": dihedral_force.fc6,
                        "phi": dihedral_force.phi,
                    }
                )

                if len(dihedral_parameters) != 1:
                    raise RuntimeError

                dihedral_parameters = dihedral_parameters[0]

            topology_key = TopologyKey(
                atom_indices=(
                    tuple(
                        val - 1
                        for val in [
                            dihedral_force.atom1,
                            dihedral_force.atom2,
                            dihedral_force.atom3,
                            dihedral_force.atom4,
                        ]
                    )
                ),
                mult=0,
            )

            def ensure_unique_key(
                handler: Union[BaseProperTorsionHandler, BaseImproperTorsionHandler],
                key: TopologyKey,
            ) -> None:
                if key in handler.slot_map:
                    key.mult += 1  # type: ignore[operator]
                    ensure_unique_key(handler, key)

            ensure_unique_key(handler, topology_key)

            potential_key = PotentialKey(
                id=(
                    f"{atomtypes[dihedral_force.atom1 - 1]}-{atomtypes[dihedral_force.atom2 - 1]}-"
                    f"{atomtypes[dihedral_force.atom3 - 1]}-{atomtypes[dihedral_force.atom4 - 1]}-"
                    f"{topology_key.mult}"
                ),
                associated_handler="ImproperTorsions"
                if dihedral_force.improper
                else "ProperTorsions",
            )

            handler.slot_map[topology_key] = potential_key

            if potential_key not in handler.potentials:

                potential = Potential(
                    parameters={
                        "phase": dihedral_parameters["phi"],
                        "periodicity": dihedral_parameters["multiplicity"],
                        "weight": dihedral_parameters["weight"],
                        "k": dihedral_parameters["k"],
                    }
                )

                if dihedral_force.improper:
                    potential.parameters["idivf"] = 1 * unit.dimensionless

                handler.potentials[potential_key] = potential

    interchange.handlers["vdW"] = vdw_handler
    interchange.handlers["Electrostatics"] = electrostatics_handler
    interchange.handlers["Bonds"] = bond_handler
    interchange.handlers["Angles"] = angle_handler
    interchange.handlers["ProperTorsions"] = proper_handler
    interchange.handlers["ImproperTorsions"] = improper_handler

    interchange.topology = topology

    return interchange
