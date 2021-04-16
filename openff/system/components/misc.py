from typing import Dict, Set

from openff.toolkit.topology import Topology

from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.models import PotentialKey, TopologyKey


class BuckinghamvdWHandler(PotentialHandler):
    name = "Buckingham-6"
    expression = "A * exp(-B *r) - C * r ** -6"
    independent_variables: Set[str] = {"r"}
    method: str = "cutoff"
    cutoff: float = 9.0
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()
    scale_13: float = 0.0
    scale_14: float = 0.5
    scale_15: float = 1.0


class RBTorsionHandler(PotentialHandler):
    name = "Ryckaert-Bellemans"
    expression = (
        "C0 + C1 * (cos(phi - 180)) "
        "C2 * (cos(phi - 180)) ** 2 + C3 * (cos(phi - 180)) ** 3 "
        "C4 * (cos(phi - 180)) ** 4 + C5 * (cos(phi - 180)) ** 5 "
    )
    independent_variables: Set[str] = {"C0", "C1", "C2", "C3", "C4", "C5"}
    slot_map: Dict[TopologyKey, PotentialKey] = dict()
    potentials: Dict[PotentialKey, Potential] = dict()


class OFFBioTop(Topology):
    def __init__(self, mdtop=None, *args, **kwargs):
        self.mdtop = mdtop
        super().__init__(*args, **kwargs)


def _store_bond_partners(mdtop):
    for atom in mdtop.atoms:
        atom._bond_partners = []
    for bond in mdtop.bonds:
        bond.atom1._bond_partners.append(bond.atom2)
        bond.atom2._bond_partners.append(bond.atom1)


def _iterate_angles(mdtop):
    for atom1 in mdtop.atoms:
        for atom2 in atom1._bond_partners:
            for atom3 in atom2._bond_partners:
                if atom1 == atom3:
                    continue
                if atom1.index < atom3.index:
                    yield (atom1, atom2, atom3)
                else:
                    # Do no duplicate
                    pass  # yield (atom3, atom2, atom1)


def _iterate_propers(mdtop):
    for atom1 in mdtop.atoms:
        for atom2 in atom1._bond_partners:
            for atom3 in atom2._bond_partners:
                if atom1 == atom3:
                    continue
                for atom4 in atom3._bond_partners:
                    if atom4 in (atom1, atom2):
                        continue

                    if atom1.index < atom4.index:
                        yield (atom1, atom2, atom3, atom4)
                    else:
                        # Do no duplicate
                        pass  # yield (atom4, atom3, atom2, atom1)


def _iterate_impropers(mdtop):
    for atom1 in mdtop.atoms:
        for atom2 in atom1._bond_partners:
            for atom3 in atom2._bond_partners:
                if atom1 == atom3:
                    continue
                for atom4 in atom2._bond_partners:
                    if atom4 in (atom3, atom1):
                        continue

                    yield (atom1, atom2, atom3, atom4)
