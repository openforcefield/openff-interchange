import pytest
from openff.toolkit import Quantity

from openff.interchange._tests import needs_gmx
from openff.interchange.drivers.gromacs import get_gromacs_energies
from openff.interchange.interop.gromacs._import._import import from_files
from openff.interchange.interop.gromacs._interchange import to_interchange


class TestToInterchange:

    @needs_gmx
    def test_torsion_multiplicities_stored_in_keys(
        self,
        ethanol,
        sage_unconstrained,
        monkeypatch,
    ):
        """Reproduce issue #978."""
        parmed = pytest.importorskip("parmed")

        monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")

        ethanol.generate_conformers(n_conformers=1)
        topology = ethanol.to_topology()
        topology.box_vectors = Quantity([4, 4, 4], "nanometer")

        out = sage_unconstrained.create_interchange(topology)
        out.to_gromacs("ethanol", decimal=8)

        parsed = from_files("ethanol.top", "ethanol.gro")

        num_torsions_in_file = len(parmed.load_file("ethanol.top").dihedrals)

        # Check that dihedrals are not override in the intermediate GROMACSSystem object
        assert num_torsions_in_file == len(parsed.molecule_types["MOL0"].dihedrals)

        # Check that GROMACSSystem -> Interchange doesn't lose information
        assert num_torsions_in_file == len(
            to_interchange(parsed)["ProperTorsions"].key_map,
        )

        get_gromacs_energies(
            out,
        ).compare(
            get_gromacs_energies(to_interchange(parsed)),
            tolerances={"Electrostatics": Quantity(0.01, "kilojoule / mole")},
        )
