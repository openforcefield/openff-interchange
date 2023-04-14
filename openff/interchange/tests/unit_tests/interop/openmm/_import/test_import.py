from openff.toolkit.tests.create_molecules import create_ethanol
from openff.units import unit

from openff.interchange import Interchange
from openff.interchange.constants import kj_mol
from openff.interchange.drivers.openmm import get_openmm_energies
from openff.interchange.interop.openmm._import import from_openmm
from openff.interchange.tests import _BaseTest


class TestFromOpenMM(_BaseTest):
    def test_simple_roundtrip(self, sage_unconstrained):
        molecule = create_ethanol()
        molecule.generate_conformers(n_conformers=1)

        interchange = Interchange.from_smirnoff(
            sage_unconstrained,
            [molecule],
            box=[4, 4, 4] * unit.nanometer,
        )

        system = interchange.to_openmm(combine_nonbonded_forces=True)

        converted = from_openmm(
            topology=interchange.topology.to_openmm(),
            system=system,
            positions=interchange.positions,
            box_vectors=interchange.box,
        )

        get_openmm_energies(interchange).compare(
            get_openmm_energies(converted),
            tolerances={
                "Bond": 1e-6 * kj_mol,
                "Angle": 1e-6 * kj_mol,
                "Torsion": 1e-6 * kj_mol,
                "Nonbonded": 1e-3 * kj_mol,
            },
        )
