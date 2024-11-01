import pytest

from openff.interchange import Interchange
from openff.interchange.drivers import get_openmm_energies


class TestJSONRoundtrips:
    def test_after_from_openmm(self, basic_top, sage, default_integrator):
        pytest.importorskip("openmm")

        simulation = sage.create_interchange(basic_top).to_openmm_simulation(
            integrator=default_integrator,
        )

        interchange1 = Interchange.from_openmm(
            system=simulation.system,
            topology=simulation.topology,
        )

        interchange2 = Interchange.model_validate_json(interchange1.model_dump_json())

        get_openmm_energies(interchange1).compare(get_openmm_energies(interchange2))
