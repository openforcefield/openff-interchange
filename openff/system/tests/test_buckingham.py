from math import exp

import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.units import unit
from scipy.constants import Avogadro

from openff.system.components.misc import BuckinghamvdWHandler
from openff.system.components.potentials import Potential
from openff.system.components.smirnoff import ElectrostaticsMetaHandler
from openff.system.exceptions import GMXMdrunError
from openff.system.models import PotentialKey, TopologyKey
from openff.system.tests.energy_tests.gromacs import get_gromacs_energies
from openff.system.tests.energy_tests.openmm import get_openmm_energies
from openff.system.utils import simtk_to_pint


def test_argon_buck():
    """Test that Buckingham potentials are supported and can be exported"""
    mol = Molecule.from_smiles("[#18]")
    top = Topology.from_molecules([mol, mol])

    A = 1.69e-8 * unit.Unit("erg / mol") * Avogadro
    B = 1 / (0.273 * unit.angstrom)
    C = 102e-12 * unit.Unit("erg / mol") * unit.angstrom ** 6 * Avogadro

    A = A.to(unit.Unit("kilojoule/mol"))
    B = B.to(unit.Unit("1 / nanometer"))
    C = C.to(unit.Unit("kilojoule / mol * nanometer ** 6"))

    r = 0.3 * unit.nanometer

    buck = BuckinghamvdWHandler()
    coul = ElectrostaticsMetaHandler()  # Just to pass compatibility checks

    pot_key = PotentialKey(id="[#18]")
    pot = Potential(parameters={"A": A, "B": B, "C": C})

    for top_atom in top.topology_atoms:
        top_key = TopologyKey(atom_indices=(top_atom.topology_atom_index,))
        buck.slot_map.update({top_key: pot_key})
        coul.charges.update({top_key: 0 * unit.elementary_charge})

    buck.potentials[pot_key] = pot

    from openff.system.components.system import System

    out = System()
    out.handlers["Buckingham-6"] = buck
    out.handlers["Electrostatics"] = coul
    out.topology = top
    out.box = [10, 10, 10] * unit.nanometer
    out.positions = [[0, 0, 0], [0.3, 0, 0]] * unit.nanometer
    out.to_gro("out.gro", writer="internal")
    out.to_top("out.top", writer="internal")

    omm_energies = get_openmm_energies(out)
    by_hand = A * exp(-B * r) - C * r ** -6

    resid = simtk_to_pint(omm_energies.energies["Nonbonded"]) - by_hand
    assert resid < 1e-5 * unit.kilojoule / unit.mol

    # TODO: Add back comparison to GROMACS energies once GROMACS 2020+
    # supports Buckingham potentials
    with pytest.raises(GMXMdrunError):
        get_gromacs_energies(out, mdp="cutoff_buck")
