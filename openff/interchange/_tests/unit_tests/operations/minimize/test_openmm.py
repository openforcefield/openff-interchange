from openff.utilities import has_package, skip_if_missing

if has_package("openmm"):
    import numpy
    import pytest
    from openff.toolkit import Molecule

    from openff.interchange._tests import MoleculeWithConformer
    from openff.interchange.drivers import get_openmm_energies
    from openff.interchange.exceptions import MissingPositionsError
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

    def test_missing_positions_error(self, tip3p):
        with pytest.raises(MissingPositionsError, match="positions=None"):
            tip3p.create_interchange(Molecule.from_smiles("O").to_topology()).minimize()

    def test_minimization_does_not_add_virtual_sites_as_atoms(self, tip4p, water_tip4p):
        system = tip4p.create_interchange(water_tip4p.to_topology())

        original_positions = system.positions

        system.minimize()

        assert system.positions.shape == original_positions.shape
