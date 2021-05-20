from math import exp

import mdtraj as md
import pytest
from openff.toolkit.topology import Molecule
from openff.units import unit
from openff.utilities.testing import skip_if_missing

from openff.system.components.mdtraj import OFFBioTop
from openff.system.components.nonbonded import BuckinghamvdWHandler
from openff.system.components.potentials import Potential
from openff.system.components.smirnoff import ElectrostaticsMetaHandler
from openff.system.exceptions import GMXMdrunError
from openff.system.models import PotentialKey, TopologyKey
from openff.system.tests.energy_tests.openmm import get_openmm_energies


@skip_if_missing("gromacs")
def test_argon_buck():
    """Test that Buckingham potentials are supported and can be exported"""
    from openff.system.tests.energy_tests.gromacs import get_gromacs_energies

    mol = Molecule.from_smiles("[#18]")
    top = OFFBioTop.from_molecules([mol, mol])
    top.mdtop = md.Topology.from_openmm(top.to_openmm())

    erg_mol = unit.erg / unit.mol
    A = 1.69e-8 * erg_mol
    B = 1 / (0.273 * unit.angstrom)
    C = 102e-12 * erg_mol * unit.angstrom ** 6

    r = 0.3 * unit.nanometer

    buck = BuckinghamvdWHandler()
    coul = ElectrostaticsMetaHandler(method="pme")

    pot_key = PotentialKey(id="[#18]")
    pot = Potential(parameters={"A": A, "B": B, "C": C})

    for atom in top.mdtop.atoms:
        top_key = TopologyKey(atom_indices=(atom.index,))
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

    resid = omm_energies.energies["Nonbonded"] - by_hand
    assert resid < 1e-5 * unit.kilojoule / unit.mol

    # TODO: Add back comparison to GROMACS energies once GROMACS 2020+
    # supports Buckingham potentials
    with pytest.raises(GMXMdrunError):
        get_gromacs_energies(out, mdp="cutoff_buck")
