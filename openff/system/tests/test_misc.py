import numpy as np
from openff.toolkit.topology import Molecule, Topology
from simtk import unit

from openff.system.misc import offmol_to_compound, offtop_to_compound


def test_basic_mol_to_compound():
    """Test basic behavior of conversion to mBuild Compound"""
    offmol = Molecule.from_smiles("CCO")
    offmol.generate_conformers(n_conformers=1)

    comp = offmol_to_compound(offmol)

    assert comp.n_particles == offmol.n_atoms
    assert comp.n_bonds == offmol.n_bonds

    np.testing.assert_equal(
        offmol.conformers[0].value_in_unit(unit.nanometer),
        comp.xyz,
    )


def test_mbuild_conversion_generate_conformers():
    """Test that a single conformer is automatically generated"""
    offmol = Molecule.from_smiles("CCO")

    comp = offmol_to_compound(offmol)

    assert comp.n_particles == offmol.n_atoms
    assert comp.n_bonds == offmol.n_bonds

    offmol.generate_conformers(n_conformers=1)
    expected_conf = offmol.conformers[0]

    np.testing.assert_equal(
        expected_conf.value_in_unit(unit.nanometer),
        comp.xyz,
    )


def test_mbuild_conversion_first_conformer_used():
    """Test that only the first conformer in an OFFMol is used"""
    offmol = Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    offmol.generate_conformers(n_conformers=3)

    comp = offmol_to_compound(offmol)

    np.testing.assert_equal(
        offmol.conformers[0].value_in_unit(unit.nanometer),
        comp.xyz,
    )

    with np.testing.assert_raises(AssertionError):
        np.testing.assert_equal(
            offmol.conformers[1].value_in_unit(unit.nanometer),
            comp.xyz,
        )

    with np.testing.assert_raises(AssertionError):
        np.testing.assert_equal(
            offmol.conformers[2].value_in_unit(unit.nanometer),
            comp.xyz,
        )


def test_mbuild_conversion_element_names():
    """Test that the generated Compound has particle names that can be
    interpreted as elements"""
    offmol = Molecule.from_smiles(
        "CSC(CO[P]([O-])([O-])=O)c1cc(Br)c(F)c(Cl)n1",
        allow_undefined_stereo=True,
    )
    comp = offmol_to_compound(offmol)

    known_elements = {"C", "H", "O", "N", "Cl", "Br", "F", "S", "P"}

    for particle in comp.particles():
        assert particle.name in known_elements


def test_multi_mol_topology_to_compound():
    """Test the basic behavior of a (multi-mol) OFFTop"""
    ethanol = Molecule.from_smiles("CCO")
    ethanol.name = "ETH"
    methane = Molecule.from_smiles("C")
    methane.name = "MET"

    top = Topology.from_molecules([ethanol, ethanol, methane, methane])

    comp = offtop_to_compound(top)

    assert comp.n_particles == 28  # 2 * (9 + 5)
    assert comp.n_bonds == 24  # 2 * (8 + 4)
    assert len(comp.children) == 4  # 4 "molecules"
