import numpy as np
import pytest
from openforcefield.topology import Molecule, Topology
from pydantic import ValidationError

from .. import unit
from ..system import ProtoSystem, System
from ..typing.smirnoff.data import (
    SMIRNOFFTermCollection,
    SMIRNOFFvdWTerm,
    build_slot_smirks_map,
    build_smirks_potential_map,
)
from ..utils import compare_forcefields
from .base_test import BaseTest


class TestPotentialEnergyTerm(BaseTest):
    def test_term_conversion(self, argon_ff, argon_top):
        term = SMIRNOFFvdWTerm.build_from_toolkit_data(
            handler=argon_ff["vdW"], topology=argon_top,
        )

        assert term.potentials["[#18:1]"].parameters["sigma"] == 0.3 * unit.nm


class TestSystem(BaseTest):
    @pytest.fixture
    def slot_smirks_map(self, argon_ff, argon_top):
        return build_slot_smirks_map(forcefield=argon_ff, topology=argon_top)

    @pytest.fixture
    def smirks_potential_map(self, argon_ff, slot_smirks_map):
        return build_smirks_potential_map(
            forcefield=argon_ff, smirks_map=slot_smirks_map
        )

    @pytest.fixture
    def term_collection(self, argon_ff, argon_top):
        return SMIRNOFFTermCollection.from_toolkit_data(
            toolkit_forcefield=argon_ff, toolkit_topology=argon_top,
        )

    def test_constructor(
        self,
        argon_ff,
        argon_top,
        argon_coords,
        argon_box,
        slot_smirks_map,
        smirks_potential_map,
        term_collection,
    ):
        """
        Test the basic constructor behavior from SMIRNOFF

        TODO: Check that the instances are reasonable, not just don't error
        """
        System(
            topology=argon_top,
            forcefield=argon_ff,
            positions=argon_coords,
            box=argon_box,
        )

        System(
            topology=argon_top, forcefield=argon_ff, positions=argon_coords, box=None,
        )

        System(
            topology=argon_top,
            term_collection=term_collection,
            slot_smirks_map=slot_smirks_map,
            smirks_potential_map=smirks_potential_map,
            positions=argon_coords,
            box=argon_box,
        )

        with pytest.raises(ValidationError):
            System(
                term_collection=term_collection,
                potential_map=smirks_potential_map,
                positions=argon_coords,
                box=argon_box,
            )

        # This would raise ValidationError before we consider un-typed `System`s
        # with pytest.raises(ValidationError):
        #    System(
        #        topology=argon_top,
        #        term_collection=term_collection,
        #        positions=argon_coords,
        #        box=argon_box,
        #    )
        #    raise ValidationError

        # This would raise an exception before we consider un-typed `System`s
        # with pytest.raises(ValidationError):
        #    System(
        #        topology=argon_top,
        #        potential_map=smirks_potential_map,
        #        positions=argon_coords,
        #        box=argon_box,
        #    )
        #    raise ValidationError

    def test_construct_single_molecule(self, parsley):
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=1)
        top = Topology.from_molecules(mol)

        test_system = System.from_toolkit(topology=top, forcefield=parsley)

        assert test_system.positions.shape == (9, 3)

    def test_construct_multiple_molecules(self, parsley):
        mol = Molecule.from_smiles("CCO")
        mol.generate_conformers(n_conformers=1)
        top = Topology.from_molecules([mol, mol])

        test_system = System.from_toolkit(topology=top, forcefield=parsley)

        assert test_system.positions.shape == (18, 3)

    def test_automatic_typing(self, argon_ff, argon_top, argon_coords, argon_box):
        """
        Test that, when only forcefield is provided, typing dicts are built automatically.

        # TODO: Make sure the results are reasonble, not just existent.
        """
        test_system = System(
            topology=argon_top,
            forcefield=argon_ff,
            positions=argon_coords,
            box=argon_box,
        )

        assert test_system.slot_smirks_map is not None
        assert test_system.smirks_potential_map is not None

    def test_from_proto_system(
        self, argon_ff, argon_top, argon_coords, argon_box,
    ):

        proto_system = ProtoSystem(
            topology=argon_top, positions=argon_coords, box=argon_box,
        )

        assert proto_system.topology is not None

        ref = System(
            topology=argon_top,
            positions=argon_coords,
            box=argon_box,
            forcefield=argon_ff,
        )

        converted = System.from_proto_system(
            proto_system=proto_system, forcefield=argon_ff,
        )

        assert np.allclose(converted.box, ref.box)
        assert np.allclose(converted.positions, ref.positions)

        # TODO: replace with == if there is ever a safe Topology.__eq__()
        assert converted.topology is ref.topology

    def test_apply_single_handler(self, argon_ff, argon_top, argon_coords, argon_box):

        argon = System(
            topology=argon_top,
            positions=argon_coords,
            box=argon_box,
            smirks_potential_map=dict(),
            slot_smirks_map=dict(),
            term_collection=dict(),
        )

        assert not argon.slot_smirks_map
        assert not argon.smirks_potential_map
        assert not argon.term_collection.terms

        argon.apply_single_handler(argon_ff["vdW"])

        assert "vdW" in argon.slot_smirks_map.keys()
        assert "vdW" in argon.smirks_potential_map.keys()
        assert "vdW" in argon.term_collection.terms.keys()


class TestProtoSystem(BaseTest):
    def test_constructor(self, argon_top, argon_coords, argon_box):
        """
        TODO: Check that the instances are reasonable, not just don't error
        """
        ProtoSystem(topology=argon_top, positions=argon_coords, box=argon_box)

        with pytest.raises(ValidationError):
            ProtoSystem(positions=argon_coords, box=argon_box)

        with pytest.raises(ValidationError):
            ProtoSystem(topology=argon_top, box=argon_box)

        with pytest.raises(ValidationError):
            ProtoSystem(positions=argon_coords, box=argon_box)


class TestValidators(BaseTest):
    @pytest.mark.parametrize(
        "values,units",
        [
            ([4, 4, 4], "nm"),
            ([60, 60, 60], "angstrom"),
            ([[4, 0, 0], [0, 4, 0], [0, 0, 4]], "nm"),
        ],
    )
    def test_valiate_box(self, values, units):
        box = ProtoSystem.validate_box(unit.Quantity(values, units=units))

        assert box.shape == (3, 3)
        assert box.units == unit.Unit(units)

    @pytest.mark.parametrize(
        "values,units",
        [
            (3 * [4], "hour"),
            (3 * [4], "acre"),
            (2 * [4], "nm"),
            (4 * [4], "nm"),
            (5 * [4], "kilojoule"),
        ],
    )
    def test_validate_box_bad(self, values, units):
        with pytest.raises(TypeError):
            ProtoSystem.validate_box(values, units=units)

    def test_validate_forcefield(self, parsley):
        for ff in [parsley, parsley._to_smirnoff_data(), "openff-1.0.0.offxml"]:
            compare_forcefields(parsley, System.validate_forcefield(ff))
