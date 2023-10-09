import pytest
from openff.units import Quantity, unit
from openff.utilities.testing import skip_if_missing

from openff.interchange._tests import MoleculeWithConformer

# Some of these tests are basically copied from [1] with an extra step into GROMACS files on disk.  Might be useful to
# collapse these in order to avoid divergence in the future.
# [1] unit_tests/interop/openmm/test_virtual_sites.py


@skip_if_missing("openmm")
class TestBondChargeVirtualSite:
    @pytest.mark.parametrize(
        (
            "distance_",
            "expected_w1",
            "expected_w2",
        ),
        [
            (0.8, 1.5, -0.5),
            (0.0, 1.0, 0.0),
            (-0.8, 0.5, 0.5),
            (-1.6, 0.0, 1.0),
            (-2.4, -0.5, 1.5),
        ],
    )
    def test_bond_charge_weights(
        self,
        sage_with_bond_charge,
        water,
        distance_,
        expected_w1,
        expected_w2,
    ):
        import openmm
        import openmm.app

        sage_with_bond_charge["VirtualSites"].parameters[0].distance = Quantity(
            distance_,
            unit.angstrom,
        )

        sage_with_bond_charge.create_interchange(
            MoleculeWithConformer.from_mapped_smiles(
                "[H:3][C:2]([H:4])([H:5])[Cl:1]",
            ).to_topology(),
        ).to_top(f"_foo{distance_}.top")

        system = openmm.app.GromacsTopFile(f"_foo{distance_}.top").createSystem()

        assert system.getNumParticles() == 6

        for index in range(5):
            assert system.getParticleMass(index)._value > 0.0

            try:
                system.getVirtualSite(index)
            except openmm.OpenMMException as error:
                # This method raises an error, not sure if there's another method
                # that just returns a boolean as to whether or not it's a virtual site
                assert "This particle is not a virtual site" in str(error)

        assert system.getParticleMass(5)._value == 0.0

        virtual_site = system.getVirtualSite(5)

        assert isinstance(virtual_site, openmm.TwoParticleAverageSite)

        assert virtual_site.getParticle(0) == 0
        assert virtual_site.getParticle(1) == 1

        assert virtual_site.getWeight(0) == pytest.approx(expected_w1)
        assert virtual_site.getWeight(1) == pytest.approx(expected_w2)
