import pytest
from openff.toolkit import ForceField, Molecule


def test_impropers_can_exist_with_no_propers(sage):
    """See issue #1441"""
    openmm = pytest.importorskip("openmm")

    ff = ForceField()
    ff.register_parameter_handler(sage.get_parameter_handler("ImproperTorsions"))

    topology = Molecule.from_smiles("c1ccccc1").to_topology()
    assert len(ff.label_molecules(topology)[0]["ImproperTorsions"])

    system = ff.create_interchange(topology).to_openmm(0)
    assert system.getNumParticles() == 12

    force_names = set(
        [system.getForce(i).__class__.__name__ for i in range(system.getNumForces())],
    )

    assert system.getNumForces() == 2, f"Expected 2 forces, got {system.getNumForces()}: {force_names}"

    assert force_names == {"CMMotionRemover", "PeriodicTorsionForce"}

    for force in system.getForces():
        if type(force) is not openmm.PeriodicTorsionForce:
            continue

        # probably don't need to inspect parameters themselves, just there
        # being a force and it having the right number of torsions is probably enough
        assert force.getNumTorsions() == 18
