import numpy
import pytest
from openff.toolkit import ForceField, Molecule
from openff.toolkit.tests.utils import get_14_scaling_factors
from openff.units import unit
from openff.units.openmm import ensure_quantity
from openmm import (
    HarmonicAngleForce,
    HarmonicBondForce,
    NonbondedForce,
    PeriodicTorsionForce,
)

from openff.interchange import Interchange
from openff.interchange.tests import _BaseTest


class TestOpenMM(_BaseTest):
    def test_no_nonbonded_force(self):
        """
        Ensure a SMIRNOFF-style force field can be exported to OpenMM even if no nonbonded handlers are present. For
        context, see https://github.com/openforcefield/openff-toolkit/issues/1102
        """

        sage = ForceField("openff_unconstrained-2.0.0.offxml")
        del sage._parameter_handlers["ToolkitAM1BCC"]
        del sage._parameter_handlers["LibraryCharges"]
        del sage._parameter_handlers["Electrostatics"]
        del sage._parameter_handlers["vdW"]

        water = Molecule.from_smiles("C")

        openmm_system = Interchange.from_smirnoff(sage, [water]).to_openmm()

        for force in openmm_system.getForces():
            if isinstance(force, NonbondedForce):
                pytest.fail("A NonbondedForce was found in the OpenMM system.")
            elif isinstance(force, PeriodicTorsionForce):
                assert force.getNumTorsions() == 0
            elif isinstance(force, HarmonicBondForce):
                assert force.getNumBonds() == 4
            elif isinstance(force, HarmonicAngleForce):
                assert force.getNumAngles() == 6
            else:
                pytest.fail(f"Unexpected force found, type: {type(force)}")

    def test_14_scale_factors_missing_electrostatics(self):
        # Ported from the toolkit after #1276
        top = Molecule.from_smiles("CCCC").to_topology()

        ff_no_electrostatics = ForceField("test_forcefields/test_forcefield.offxml")
        ff_no_electrostatics.deregister_parameter_handler("Electrostatics")
        ff_no_electrostatics.deregister_parameter_handler("ToolkitAM1BCC")

        out = Interchange.from_smirnoff(
            ff_no_electrostatics,
            top,
        ).to_openmm(combine_nonbonded_forces=True)

        numpy.testing.assert_almost_equal(
            actual=get_14_scaling_factors(out)[1],
            desired=ff_no_electrostatics["vdW"].scale14,
            decimal=8,
        )

    def test_14_scale_factors_missing_vdw(self):
        # Ported from the toolkit after #1276
        top = Molecule.from_smiles("CCCC").to_topology()

        ff_no_vdw = ForceField("test_forcefields/test_forcefield.offxml")
        ff_no_vdw.deregister_parameter_handler("vdW")

        out = Interchange.from_smirnoff(
            ff_no_vdw,
            top,
        ).to_openmm(combine_nonbonded_forces=True)

        numpy.testing.assert_almost_equal(
            actual=get_14_scaling_factors(out)[0],
            desired=ff_no_vdw["Electrostatics"].scale14,
            decimal=8,
        )

    def test_to_pdb_box_vectors(self, sage):
        """Reproduce https://github.com/openforcefield/openff-interchange/issues/548."""
        from openmm.app import PDBFile

        molecule = Molecule.from_smiles("CC")
        molecule.generate_conformers(n_conformers=1)
        box_vectors = unit.Quantity(
            10.0 * numpy.eye(3),
            unit.angstrom,
        )

        interchange = Interchange.from_smirnoff(
            topology=[molecule],
            force_field=sage,
            box=box_vectors,
        )

        interchange.to_pdb("temp.pdb")

        parsed_box_vectors = PDBFile("temp.pdb").topology.getPeriodicBoxVectors()

        numpy.testing.assert_allclose(
            box_vectors.m_as(unit.angstrom),
            ensure_quantity(parsed_box_vectors, "openff").m_as(unit.angstrom),
        )
