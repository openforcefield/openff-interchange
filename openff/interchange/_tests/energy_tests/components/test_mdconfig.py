from openff.units import Quantity, unit

from openff.interchange._tests import MoleculeWithConformer, needs_gmx, needs_lmp
from openff.interchange.drivers import get_gromacs_energies, get_openmm_energies


class TestLJPME:
    @needs_gmx
    @needs_lmp
    def test_ljpme(self, sage):
        sage["vdW"].periodic_method = "Ewald3D"

        topology = MoleculeWithConformer.from_smiles("C#N").to_topology()
        topology.box_vectors = [2, 2, 2] * unit.nanometer
        interchange = sage.create_interchange(topology)

        get_openmm_energies(interchange, combine_nonbonded_forces=True).compare(
            get_gromacs_energies(interchange),
            tolerances={
                "Nonbonded": Quantity(0.001, "kilojoule / mole"),
            },
        )
