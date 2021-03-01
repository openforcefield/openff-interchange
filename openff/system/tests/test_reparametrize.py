import pytest
from openff.toolkit.topology import Molecule

from openff.system.exceptions import HandlerNotFoundError
from openff.system.stubs import ForceField
from openff.system.tests.energy_tests.openmm import get_openmm_energies


def test_full_reparametrize():
    mol = Molecule.from_smiles("OCCc1ccn2cnccc12")
    top = mol.to_topology()

    mol.generate_conformers(n_conformers=1)

    parsley = ForceField("openff-1.0.0.offxml")
    try:
        s99 = ForceField("smirnoff99frosst-1.1.0.offxml")
    except OSError:
        # Drop when below PR is in a release (0.9.2)
        # https://github.com/openforcefield/openff-toolkit/pull/816
        s99 = ForceField("smirnoff99Frosst-1.1.0.offxml")

    original = parsley.create_openff_system(top)
    reference = s99.create_openff_system(top)

    for out in [original, reference]:
        out.positions = mol.conformers[0]
        out.box = [4, 4, 4]

    s99_energies = get_openmm_energies(reference)

    with pytest.raises(HandlerNotFoundError):
        original.reparametrize(other_force_field=s99, handler_name="LibraryCharges")

    for energy_key, handler_names in zip(
        ["Bond", "Angle", "Torsion", "Nonbonded"],
        [["Bonds"], ["Angles"], ["ProperTorsions", "ImproperTorsions"], ["vdW"]],
    ):
        for handler_name in handler_names:
            original.reparametrize(other_force_field=s99, handler_name=handler_name)
        tmp_energies = get_openmm_energies(original)
        assert tmp_energies.energies[energy_key] == s99_energies.energies[energy_key]

    new_energies = get_openmm_energies(original)
    new_energies.compare(s99_energies)
