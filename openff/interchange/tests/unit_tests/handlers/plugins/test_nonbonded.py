import openmm
import pytest
from openff.toolkit import ForceField, Molecule
from openff.units import unit

from openff.interchange import Interchange
from openff.interchange.exceptions import UnsupportedExportError


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

    out = Interchange.from_smirnoff(
        force_field=force_field,
        topology=Molecule.from_smiles("O").to_topology(),
    )

    assert "LennardJones14" in out.handlers
    assert len(out.handlers["LennardJones14"].slot_map) == 3

    with pytest.raises(UnsupportedExportError, match="Custom vdW"):
        out.to_openmm(combine_nonbonded_forces=True)

    openmm_system = out.to_openmm(combine_nonbonded_forces=False)

    expression = "4*epsilon*((sigma/r)^14-(sigma/r)^6); sigma=(sigma1+sigma2)/2; epsilon=sqrt(epsilon1*epsilon2); "
    for force in openmm_system.getForces():
        if isinstance(force, openmm.CustomNonbondedForce):
            assert force.getEnergyFunction() == expression
            break
    else:
        raise Exception("Could not find custom non-bonded force")
