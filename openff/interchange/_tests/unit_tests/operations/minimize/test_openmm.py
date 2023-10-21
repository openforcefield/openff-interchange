from openff.utilities import has_package, skip_if_missing

if has_package("openmm"):
    import numpy

    from openff.interchange._tests import MoleculeWithConformer
    from openff.interchange.drivers import get_openmm_energies
    from openff.interchange.operations.minimize import (
        _DEFAULT_ENERGY_MINIMIZATION_TOLERANCE,
    )
    from openff.interchange.operations.minimize.openmm import minimize_openmm


@skip_if_missing("openmm")
class TestOpenMMMinimization:
    def test_minimization_decreases_energy(self, sage):
        system = sage.create_interchange(
            MoleculeWithConformer.from_smiles("CNC(=O)C1=C(N)C=C(F)C=C1").to_topology(),
        )

        original_energy = get_openmm_energies(system).total_energy

        minimized_positions = minimize_openmm(
            system,
            tolerance=_DEFAULT_ENERGY_MINIMIZATION_TOLERANCE,
            max_iterations=10_000,
        )

        assert not numpy.allclose(minimized_positions, system.positions)

        system.positions = minimized_positions
        minimied_energy = get_openmm_energies(system).total_energy

        assert minimied_energy < original_energy
