"""Test SMIRNOFF-GROMACS conversion."""

import pytest
from openff.toolkit import Molecule, unit

from openff.interchange import Interchange
from openff.interchange.interop.gromacs.models.models import GROMACSMolecule
from openff.interchange.smirnoff._gromacs import (
    _convert,
    _convert_angles,
    _convert_bonds,
    _convert_settles,
)


class TestConvert:
    def test_residue_names(self, sage):
        """Reproduce issue #642."""
        ligand = Molecule.from_smiles("CCO")
        ligand.generate_conformers(n_conformers=1)

        for atom in ligand.atoms:
            atom.metadata["residue_name"] = "LIG"

        system = _convert(
            Interchange.from_smirnoff(
                sage,
                [ligand],
            ),
        )

        for molecule_type in system.molecule_types.values():
            for atom in molecule_type.atoms:
                assert atom.residue_name == "LIG"


class TestSettles:
    @pytest.fixture()
    def tip3p_interchange(self, tip3p, water):
        return tip3p.create_interchange(water.to_topology())

    @pytest.fixture()
    def sage_tip3p_interchange(self, sage, water):
        return sage.create_interchange(water.to_topology())

    def test_catch_other_water_ordering(self, tip3p):
        molecule = Molecule.from_mapped_smiles("[H:1][O:2][H:3]")
        interchange = tip3p.create_interchange(molecule.to_topology())

        with pytest.raises(Exception, match="OHH"):
            _convert_settles(
                GROMACSMolecule(name="foo"),
                interchange.topology.molecule(0),
                interchange,
            )

    @pytest.mark.parametrize("use_bundled_tip3p", [True, False])
    def test_convert_settles(
        self,
        use_bundled_tip3p,
        tip3p_interchange,
        sage_tip3p_interchange,
    ):
        molecule = GROMACSMolecule(name="foo")

        if use_bundled_tip3p:
            _convert_settles(
                molecule,
                sage_tip3p_interchange.topology.molecule(0),
                sage_tip3p_interchange,
            )
        else:
            _convert_settles(
                molecule,
                tip3p_interchange.topology.molecule(0),
                tip3p_interchange,
            )

        assert len(molecule.settles) == 1

        settle = molecule.settles[0]

        assert settle.first_atom == 1
        assert settle.hydrogen_hydrogen_distance.m_as(unit.angstrom) == pytest.approx(
            1.5139006545247014,
        )
        assert settle.oxygen_hydrogen_distance.m_as(unit.angstrom) == pytest.approx(
            0.9572,
        )

        assert molecule.exclusions[0].first_atom == 1
        assert molecule.exclusions[0].other_atoms == [2, 3]
        assert molecule.exclusions[1].first_atom == 2
        assert molecule.exclusions[1].other_atoms == [1, 3]
        assert molecule.exclusions[2].first_atom == 3
        assert molecule.exclusions[2].other_atoms == [1, 2]

    def test_convert_no_settles_unconstrained_water(self, tip3p_interchange):
        tip3p_interchange.collections["Constraints"].key_map = dict()

        molecule = GROMACSMolecule(name="foo")

        _convert_settles(
            molecule,
            tip3p_interchange.topology.molecule(0),
            tip3p_interchange,
        )

        assert len(molecule.settles) == 0

    def test_convert_no_settles_no_constraints(self, tip3p_interchange):
        tip3p_interchange.collections.pop("Constraints")

        molecule = GROMACSMolecule(name="foo")

        _convert_settles(
            molecule,
            tip3p_interchange.topology.molecule(0),
            tip3p_interchange,
        )

        assert len(molecule.settles) == 0

    @pytest.mark.parametrize("use_bundled_tip3p", [True, False])
    def test_error_if_water_partially_constrained(
        self,
        use_bundled_tip3p,
        tip3p_interchange,
        sage_tip3p_interchange,
    ):
        from openff.interchange.models import BondKey

        # Manually remove the H-H constraint parameter from each,
        tip3p_interchange["Constraints"].potentials.pop(
            tip3p_interchange["Constraints"].key_map[BondKey(atom_indices=(0, 1))],
        )

        sage_tip3p_interchange["Constraints"].potentials.pop(
            sage_tip3p_interchange["Constraints"].key_map[BondKey(atom_indices=(0, 1))],
        )

        molecule = GROMACSMolecule(name="foo")

        if not use_bundled_tip3p:
            with pytest.raises(
                RuntimeError,
                match="Could not find a constraint distance .*0.*1",
            ):
                # ... and ensure this is an error when there is no bond parameter
                # to fall back on
                _convert_settles(
                    GROMACSMolecule(name="foo"),
                    tip3p_interchange.topology.molecule(0),
                    tip3p_interchange,
                )
        else:
            # ... but uses a O-H distance if there is one (here, from Sage)
            _convert_settles(
                molecule,
                sage_tip3p_interchange.topology.molecule(0),
                sage_tip3p_interchange,
            )

            # b88 in both 2.0 and 2.1
            assert molecule.settles[0].oxygen_hydrogen_distance.m_as(
                unit.angstrom,
            ) == pytest.approx(
                0.97167633126,
            )

    def test_no_bonds_or_angles_if_settle(self, tip3p_interchange):
        molecule = GROMACSMolecule(name="foo")

        for function in [_convert_settles, _convert_bonds, _convert_angles]:
            function(
                molecule,
                tip3p_interchange.topology.molecule(0),
                tip3p_interchange,
            )

        assert len(molecule.settles) == 1
        assert len(molecule.angles) == 0
        assert len(molecule.bonds) == 0
