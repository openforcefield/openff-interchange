import numpy
import pytest
from openff.toolkit import Quantity
from openff.units import ensure_quantity
from openff.utilities import has_package, skip_if_missing

from openff.interchange import Interchange
from openff.interchange._tests import MoleculeWithConformer
from openff.interchange._tests._openmm import get_14_scaling_factors

if has_package("openmm"):
    from openmm import (
        CMMotionRemover,
        HarmonicAngleForce,
        HarmonicBondForce,
        NonbondedForce,
        PeriodicTorsionForce,
    )


@skip_if_missing("openmm")
class TestOpenMM:
    def test_no_nonbonded_force(self, fresh_sage):
        """
        Ensure a SMIRNOFF-style force field can be exported to OpenMM even if no nonbonded handlers are present. For
        context, see https://github.com/openforcefield/openff-toolkit/issues/1102
        """
        del fresh_sage._parameter_handlers["Constraints"]
        del fresh_sage._parameter_handlers["NAGLCharges"]
        del fresh_sage._parameter_handlers["LibraryCharges"]
        del fresh_sage._parameter_handlers["Electrostatics"]
        del fresh_sage._parameter_handlers["vdW"]

        methane = MoleculeWithConformer.from_smiles("C")

        openmm_system = Interchange.from_smirnoff(fresh_sage, [methane]).to_openmm()

        for force in openmm_system.getForces():
            if isinstance(force, NonbondedForce):
                pytest.fail("A NonbondedForce was found in the OpenMM system.")
            elif isinstance(force, PeriodicTorsionForce):
                assert force.getNumTorsions() == 0
            elif isinstance(force, HarmonicBondForce):
                assert force.getNumBonds() == 4
            elif isinstance(force, HarmonicAngleForce):
                assert force.getNumAngles() == 6
            elif isinstance(force, CMMotionRemover):
                pass
            else:
                pytest.fail(f"Unexpected force found, type: {type(force)}")

    def test_14_scale_factors_missing_electrostatics(self, fresh_sage):
        # Ported from the toolkit after #1276
        topology = MoleculeWithConformer.from_smiles("CCCC").to_topology()

        fresh_sage.deregister_parameter_handler("Electrostatics")
        fresh_sage.deregister_parameter_handler("NAGLCharges")
        fresh_sage.deregister_parameter_handler("LibraryCharges")

        system = fresh_sage.create_interchange(topology).to_openmm()

        numpy.testing.assert_almost_equal(
            actual=get_14_scaling_factors(system)[1],
            desired=fresh_sage["vdW"].scale14,
            decimal=8,
        )

    def test_14_scale_factors_missing_vdw(self, fresh_sage):
        # Ported from the toolkit after #1276
        topology = MoleculeWithConformer.from_smiles("CCCC").to_topology()

        fresh_sage.deregister_parameter_handler("vdW")

        system = fresh_sage.create_interchange(topology).to_openmm()

        numpy.testing.assert_almost_equal(
            actual=get_14_scaling_factors(system)[0],
            desired=fresh_sage["Electrostatics"].scale14,
            decimal=8,
        )

    def test_to_pdb_box_vectors(self, fresh_sage):
        """Reproduce https://github.com/openforcefield/openff-interchange/issues/548."""
        from openmm.app import PDBFile

        topology = MoleculeWithConformer.from_smiles("CC").to_topology()
        topology.box_vectors = Quantity(
            10.0 * numpy.eye(3),
            "angstrom",
        )

        interchange = Interchange.from_smirnoff(fresh_sage, topology)

        interchange.to_pdb("temp.pdb")

        parsed_box_vectors = PDBFile("temp.pdb").topology.getPeriodicBoxVectors()

        numpy.testing.assert_allclose(
            topology.box_vectors.m_as("angstrom"),
            ensure_quantity(parsed_box_vectors, "openff").m_as("angstrom"),
        )


@skip_if_missing("openmm")
class TestOpenMMMissingHandlers:
    def test_missing_vdw_combine_energies(self, fresh_sage):
        from openff.interchange.drivers import get_openmm_energies

        topology = MoleculeWithConformer.from_smiles("CC").to_topology()

        fresh_sage.deregister_parameter_handler("vdW")

        out = Interchange.from_smirnoff(fresh_sage, topology)

        energy1 = get_openmm_energies(out, combine_nonbonded_forces=True).total_energy
        energy2 = get_openmm_energies(out, combine_nonbonded_forces=False).total_energy

        assert abs(energy2 - energy1) < Quantity(
            1e-6,
            "kilojoule_per_mole",
        )
