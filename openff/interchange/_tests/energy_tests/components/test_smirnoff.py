import numpy as np
import pytest
from openff.toolkit import ForceField, Molecule
from openff.units import unit
from openff.utilities import get_data_file_path, has_package, skip_if_missing

from openff.interchange import Interchange
from openff.interchange._tests.unit_tests.smirnoff.test_valence import (
    TestBondOrderInterpolation,
)
from openff.interchange.drivers.openmm import (
    _get_openmm_energies,
    _process,
    get_openmm_energies,
)

if has_package("openmm"):
    import openmm
    import openmm.app
    import openmm.unit


@skip_if_missing("openmm")
class TestBondOrderInterpolationEnergies(TestBondOrderInterpolation):
    @pytest.mark.slow()
    def test_basic_bond_order_interpolation_energies(self, xml_ff_bo_bonds):
        forcefield = ForceField(
            get_data_file_path(
                "test_forcefields/test_forcefield.offxml",
                "openff.toolkit",
            ),
            xml_ff_bo_bonds,
        )

        mol = Molecule.from_file(
            get_data_file_path("molecules/CID20742535_anion.sdf", "openff.toolkit"),
        )
        mol.generate_conformers(n_conformers=1)
        top = mol.to_topology()

        out = Interchange.from_smirnoff(forcefield, top)
        out.box = [4, 4, 4] * unit.nanometer
        out.positions = mol.conformers[0]

        interchange_bond_energy = get_openmm_energies(
            out,
            combine_nonbonded_forces=True,
        ).energies["Bond"]

        system = forcefield.create_openmm_system(top)
        toolkit_bond_energy = _process(
            _get_openmm_energies(
                system=system,
                box_vectors=[[4, 0, 0], [0, 4, 0], [0, 0, 4]] * openmm.unit.nanometer,
                positions=mol.conformers[0].to_openmm(),
                round_positions=None,
                platform="Reference",
            ),
            system=system,
            combine_nonbonded_forces=True,
            detailed=False,
        ).energies["Bond"]

        assert abs(interchange_bond_energy - toolkit_bond_energy).m < 1e-2

        new = out.to_openmm(combine_nonbonded_forces=True)
        ref = forcefield.create_openmm_system(top)

        new_k = []
        new_length = []
        for force in new.getForces():
            if type(force) is openmm.HarmonicBondForce:
                for i in range(force.getNumBonds()):
                    new_k.append(force.getBondParameters(i)[3]._value)
                    new_length.append(force.getBondParameters(i)[2]._value)

        ref_k = []
        ref_length = []
        for force in ref.getForces():
            if type(force) is openmm.HarmonicBondForce:
                for i in range(force.getNumBonds()):
                    ref_k.append(force.getBondParameters(i)[3]._value)
                    ref_length.append(force.getBondParameters(i)[2]._value)

        np.testing.assert_allclose(ref_k, new_k, rtol=3e-5)
