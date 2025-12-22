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

    def test_charge_increments(self, sage_charge_increment_handler, carbonyl_planar):
        """Test JSON roundtrip when charges are assigned with a ChargeIncrementModelHandler."""
        pytest.importorskip("openmm")

        out = sage_charge_increment_handler.create_interchange(carbonyl_planar.to_topology())

        roundtripped = Interchange.model_validate_json(out.model_dump_json())

        get_openmm_energies(
            out,
            combine_nonbonded_forces=False,
        ).compare(
            get_openmm_energies(
                roundtripped,
                combine_nonbonded_forces=False,
            ),
        )

    def test_nagl_charges(self, sage_230, carbonyl_planar):
        """Test JSON roundtrip when charges are assigned with a ChargeIncrementModelHandler."""

        pytest.importorskip("openmm")
        pytest.importorskip("openff.nagl")

        out = sage_230.create_interchange(carbonyl_planar.to_topology())

        roundtripped = Interchange.model_validate_json(out.model_dump_json())

        get_openmm_energies(
            out,
            combine_nonbonded_forces=False,
        ).compare(
            get_openmm_energies(
                roundtripped,
                combine_nonbonded_forces=False,
            ),
        )
