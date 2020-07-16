import numpy as np

from ... import unit
from ...types import UnitArray
from ...utils import unwrap_list_of_pint_quantities


def get_distance(a, b):
    """
    get distance between two UnitArray quantities since __array_method__ isn't implemented
    for np.linalg.norm on pint.Quantity
    """
    return np.sqrt(np.sum((b - a) ** 2))


def build_distance_matrix(system_in):
    """
    Hack to generate an n_atoms x n_atoms matrix of distances, intended only
    for use on small systems that do not need neighbor lists
    """
    positions = system_in.positions

    n_atoms = system_in.topology.n_topology_atoms

    distances = UnitArray(np.zeros((n_atoms, n_atoms)), units=system_in.positions.units)

    for i in range(n_atoms):
        for j in range(n_atoms):
            # TODO: Here may be a place to drop in the bonded exceptions, or maybe
            # it would be worth worth carrying this array and a mask
            r = get_distance(positions[i, :], positions[j, :])
            distances[i, j] = UnitArray(r.magnitude, units=r.units)

    return distances


def compute_vdw(system_in):
    """
    Compute the vdW contribution to the potential energy function.
    This is mean to serve as a stand-in for a something more performant with a similar signature
    """
    slots = system_in.slot_smirks_map["vdW"].keys()
    term = system_in.term_collection.terms["vdW"]

    distances = build_distance_matrix(system_in)

    energy = 0
    for i in slots:
        for j in slots:
            if i == j:
                continue

            r = distances[i[0], j[0]]
            sig1 = term.potentials[term.smirks_map[i]].parameters["sigma"]
            eps1 = term.potentials[term.smirks_map[i]].parameters["epsilon"]
            sig2 = term.potentials[term.smirks_map[j]].parameters["sigma"]
            eps2 = term.potentials[term.smirks_map[j]].parameters["epsilon"]

            # TODO: Encode mixing rules somewhere?
            sig = (sig1 + sig2) * 0.5
            eps = (eps1 * eps2) ** 0.5

            ener = 4 * eps * ((sig / r) ** 12 - (sig / r) ** 6)
            energy += ener

    return energy


def compute_bonds(system_in):
    """
    Compute the bond contribution to the potential energy function.
    This is mean to serve as a stand-in for a something more performant with a similar signature
    """
    slots = system_in.slot_smirks_map["Bonds"].keys()
    term = system_in.term_collection.terms["Bonds"]

    def get_r(slot):
        """
        in: slot as a tuple of atom indicies, i.e. (4, 7)
        out: bond length
        """
        id1, id2 = slot[0], slot[1]
        pos1 = system_in.positions[id1, :]
        pos2 = system_in.positions[id2, :]

        return get_distance(pos2, pos1)

    energy = 0
    for slot in slots:
        r = get_r(slot)
        k = term.potentials[term.smirks_map[slot]].parameters["k"]
        length = term.potentials[term.smirks_map[slot]].parameters["length"]

        ener = 0.5 * k * (length - r) ** 2
        energy += ener

    return energy


def compute_electrostatics(system_in):
    """
    Compute the electrostatics contribution to the potential energy function.
    This is mean to serve as a stand-in for a something more performant with a similar signature
    """

    # From NIST CODATA 2014, see Table 3 in 10.1007/s10822-016-9977-1
    COUL = 332.0637130232 * unit.Unit(
        "kilocalorie / mole  * angstrom / elementary_charge ** 2"
    )
    slots = system_in.slot_smirks_map["Electrostatics"].keys()
    term = system_in.term_collection.terms["Electrostatics"]

    distances = build_distance_matrix(system_in)

    energy = 0
    for i in slots:
        for j in slots:
            if i == j:
                continue

            r = distances[i[0], j[0]]
            q1 = term.potentials[term.smirks_map[i]]
            q2 = term.potentials[term.smirks_map[j]]

            ener = q1 * q2 * COUL / r

            energy += ener

    return energy


SUPPORTED_HANDLERS = {
    "vdW": compute_vdw,
    "Bonds": compute_bonds,
    "Electrostatics": compute_electrostatics,
}


def compute_potential_energy(system_in, handlers=None):
    if not handlers:
        handlers = [*system_in.term_collection.terms.keys()]

    partial_potential_energies = dict()
    for handler in handlers:
        try:
            partial_energy = SUPPORTED_HANDLERS[handler](system_in)
            partial_potential_energies[handler] = partial_energy
        except KeyError as e:
            print("handler not supported yet")
            raise KeyError from e

    # TODO: Do this summation without recasting

    return np.sum(
        unwrap_list_of_pint_quantities([*partial_potential_energies.values()])
    )
