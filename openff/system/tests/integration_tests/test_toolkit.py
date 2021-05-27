import pytest
from openff.toolkit.topology.molecule import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import (
    UnassignedBondParameterException,
    UnassignedProperTorsionParameterException,
)
from openff.units import unit
from simtk import unit as simtk_unit

from openff.system.components.system import System
from openff.system.drivers.openmm import _get_openmm_energies, get_openmm_energies
from openff.system.utils import get_test_file_path

_parsley = ForceField("openff-1.0.0.offxml")
_box_vectors = simtk_unit.Quantity(
    [[4, 0, 0], [0, 4, 0], [0, 0, 4]], unit=simtk_unit.nanometer
)


@pytest.mark.slot
@pytest.mark.parametrize(
    "mol",
    Molecule.from_file(get_test_file_path("chosen.smi"), allow_undefined_stereo=True),
)
def test_energy_vs_toolkit(mol):
    try:
        mol.generate_conformers(n_conformers=1)
    except Exception as e:
        if str(e) == "OpenEye Omega conformer generation failed":
            pytest.xfail("Failed because of conformer generation")

    top = mol.to_topology()
    top.box_vectors = [4, 4, 4] * simtk_unit.nanometer

    try:
        toolkit_sys = _parsley.create_openmm_system(top)
    except (
        UnassignedBondParameterException,
        UnassignedProperTorsionParameterException,
    ):
        pytest.xfail("Failed because of missing bond or torsion parameters")

    toolkit_energy = _get_openmm_energies(
        toolkit_sys, box_vectors=_box_vectors, positions=mol.conformers[0]
    )

    openff_sys = System.from_smirnoff(force_field=_parsley, topology=top)
    openff_sys.positions = mol.conformers[0]
    system_energy = get_openmm_energies(openff_sys, combine_nonbonded_forces=True)

    kj_mol = unit.kilojoule / unit.mol

    toolkit_energy.compare(
        system_energy,
        custom_tolerances={
            "Bond": 1e-6 * kj_mol,
            "Angle": 1e-6 * kj_mol,
            "Torsion": 1.6e-5 * kj_mol,
            "Nonbonded": 1e-6 * kj_mol,
        },
    )
