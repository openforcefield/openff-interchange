from openff.toolkit import ForceField
from openff.units import unit


def test_lj_14_handler():
    force_field = ForceField(load_plugins=True)
    lj_14_handler = force_field.get_parameter_handler("LennardJones14")

    lj_14_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
            "epsilon": 0.0 * unit.kilojoule_per_mole,
            "sigma": 1.0 * unit.angstrom,
        }
    )

    lj_14_handler.add_parameter(
        {
            "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
            "epsilon": 0.15 * unit.kilocalorie_per_mole,
            "sigma": 3.1 * unit.angstrom,
        }
    )

    constraint_handler = force_field.get_parameter_handler("Constraints")
    # Keep the H-O bond length fixed at 0.9572 angstroms.
    constraint_handler.add_parameter(
        {"smirks": "[#1:1]-[#8X2H2+0:2]-[#1]", "distance": 0.9572 * unit.angstrom}
    )
    # Keep the H-O-H angle fixed at 104.52 degrees.
    constraint_handler.add_parameter(
        {"smirks": "[#1:1]-[#8X2H2+0]-[#1:2]", "distance": 1.5139 * unit.angstrom}
    )

    force_field.get_parameter_handler("Electrostatics")

    force_field.get_parameter_handler(
        "ChargeIncrementModel",
        {"version": "0.3", "partial_charge_method": "am1bcc"},
    )
