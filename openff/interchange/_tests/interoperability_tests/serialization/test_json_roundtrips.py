import pytest
from openff.toolkit import Quantity
from openff.units.openmm import from_openmm

from openff.interchange import Interchange
from openff.interchange.drivers import get_openmm_energies

tolerance = Quantity("0.001 kilojoules/mole")


class TestJSONRoundtrips:
    def test_after_from_openmm(self, basic_top, sage, default_integrator):
        pytest.importorskip("openmm")

        simulation = sage.create_interchange(basic_top).to_openmm_simulation(
            integrator=default_integrator,
        )
        state = simulation.context.getState(getPositions=True)

        interchange1 = Interchange.from_openmm(
            system=simulation.system,
            topology=simulation.topology,
            positions=from_openmm(state.getPositions(asNumpy=True)),
            box_vectors=from_openmm(state.getPeriodicBoxVectors(asNumpy=True)),
        )

        interchange2 = Interchange.model_validate_json(interchange1.model_dump_json())

        get_openmm_energies(interchange1).compare(
            get_openmm_energies(interchange2),
            tolerances={key: tolerance for key in ["Bond", "Angle", "Torsion", "Nonbonded"]},
        )
