import numpy
from openff.toolkit import Quantity, Topology
from openff.units import unit

from openff.interchange._tests import _rng
from openff.interchange.common._positions import _infer_positions


class TestInferPositions:
    def test_short_circuit(self, methane):
        positions = Quantity(
            _rng.random((methane.n_atoms, 3)),
            unit.angstrom,
        )

        assert numpy.array_equal(
            _infer_positions(methane.to_topology(), positions).m_as(unit.angstrom),
            positions.m_as(unit.angstrom),
        )

    def test_basic(self, methane_with_conformer):
        assert numpy.array_equal(
            _infer_positions(methane_with_conformer.to_topology(), None).m_as(
                unit.angstrom,
            ),
            methane_with_conformer.conformers[0].m_as(unit.angstrom),
        )

    def test_multimolecule(self, methane_with_conformer, ethanol_with_conformer):
        topology = Topology.from_molecules(
            [methane_with_conformer, ethanol_with_conformer],
        )
        expected_positions = Quantity(
            numpy.concatenate(
                [
                    methane_with_conformer.conformers[0].m,
                    ethanol_with_conformer.conformers[0].m,
                ],
            ),
            unit.angstrom,
        )

        assert numpy.array_equal(
            _infer_positions(topology).m_as(unit.angstrom),
            expected_positions.m_as(unit.angstrom),
        )

    def test_mixed_conformers(self, methane_with_conformer, ethanol):
        topology = Topology.from_molecules([methane_with_conformer, ethanol])
        assert _infer_positions(topology) is None
