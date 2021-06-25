import mdtraj as md
import numpy as np
import parmed as pmd
import pytest
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit
from parmed.amber import readparm
from pmdtest.utils import get_fn as get_pmd_fn
from simtk import unit as omm_unit

from openff.interchange.components.interchange import Interchange
from openff.interchange.drivers.gromacs import (
    _get_mdp_file,
    _run_gmx_energy,
    get_gromacs_energies,
)
from openff.interchange.tests import BaseTest
from openff.interchange.tests.utils import top_from_smiles
from openff.interchange.utils import get_test_file_path


class TestParmedConversion(BaseTest):
    @pytest.fixture()
    def box(self):
        return np.array([4.0, 4.0, 4.0])

    def test_box(self, argon_ff, argon_top, box):
        off_sys = Interchange.from_smirnoff(
            force_field=argon_ff, topology=argon_top, box=box
        )
        off_sys.positions = (
            np.zeros(shape=(argon_top.n_topology_atoms, 3)) * unit.angstrom
        )
        struct = off_sys._to_parmed()

        assert np.allclose(
            struct.box[:3],
            [40, 40, 40],
        )

    def test_basic_conversion_argon(self, argon_ff, argon_top, box):
        off_sys = Interchange.from_smirnoff(
            force_field=argon_ff, topology=argon_top, box=box
        )
        off_sys.positions = np.zeros(shape=(argon_top.n_topology_atoms, 3))
        struct = off_sys._to_parmed()

        # As partial sanity check, see if it they save without error
        struct.save("x.top", combine="all")
        struct.save("x.gro", combine="all")

        assert np.allclose(struct.box, np.array([40, 40, 40, 90, 90, 90]))

    def test_basic_conversion_params(self, box):
        top = top_from_smiles("C")
        parsley = ForceField("openff_unconstrained-1.0.0.offxml")

        off_sys = Interchange.from_smirnoff(force_field=parsley, topology=top, box=box)
        # UnitArray(...)
        off_sys.positions = np.zeros(shape=(top.n_topology_atoms, 3))
        struct = off_sys._to_parmed()

        sigma0 = struct.atoms[0].atom_type.sigma
        epsilon0 = struct.atoms[0].atom_type.epsilon

        sigma1 = struct.atoms[1].atom_type.sigma
        epsilon1 = struct.atoms[1].atom_type.epsilon

        bond_k = struct.bonds[0].type.k
        req = struct.bonds[0].type.req

        angle_k = struct.angles[0].type.k
        theteq = struct.angles[0].type.theteq

        assert sigma0 == pytest.approx(3.3996695084235347)
        assert epsilon0 == pytest.approx(0.1094)

        assert sigma1 == pytest.approx(2.649532787749369)
        assert epsilon1 == pytest.approx(0.0157)

        assert bond_k == pytest.approx(379.04658864565)
        assert req == pytest.approx(1.092888378383)

        assert angle_k == pytest.approx(37.143507635885)
        assert theteq == pytest.approx(107.5991506326)

        assert np.allclose(struct.box, np.array([40, 40, 40, 90, 90, 90]))

    def test_basic_conversion_ammonia(self, ammonia_ff, ammonia_top, box):
        off_sys = Interchange.from_smirnoff(
            force_field=ammonia_ff, topology=ammonia_top, box=box
        )
        off_sys.positions = np.zeros(shape=(ammonia_top.n_topology_atoms, 3))
        struct = off_sys._to_parmed()

        # As partial sanity check, see if it they save without error
        struct.save("x.top", combine="all")
        struct.save("x.gro", combine="all")

        assert np.allclose(struct.box, np.array([40, 40, 40, 90, 90, 90]))

    @pytest.mark.slow
    @pytest.mark.xfail
    def test_parmed_roundtrip(self):
        original = pmd.load_file(get_test_file_path("ALA_GLY/ALA_GLY.top"))
        gro = pmd.load_file(get_test_file_path("ALA_GLY/ALA_GLY.gro"))
        original.box = gro.box
        original.positions = gro.positions

        openff_sys = Interchange._from_parmed(original)
        openff_sys.topology.mdtop = md.Topology.from_openmm(gro.topology)

        #  Some sanity checks, including that residues are stored ...
        assert openff_sys.topology.mdtop.n_atoms == 29
        # TODO: Assert number of topology molecules after refactor
        assert openff_sys.topology.mdtop.n_residues == 4

        # ... and written out
        openff_sys.to_gro("has_residues.gro", writer="internal")
        assert len(pmd.load_file("has_residues.gro").residues) == 4

        roundtrip = openff_sys._to_parmed()

        roundtrip.save("conv.gro", overwrite=True)
        roundtrip.save("conv.top", overwrite=True)

        original_energy = _run_gmx_energy(
            top_file=get_test_file_path("ALA_GLY/ALA_GLY.top"),
            gro_file=get_test_file_path("ALA_GLY/ALA_GLY.gro"),
            mdp_file=_get_mdp_file("cutoff_hbonds"),
        )
        internal_energy = get_gromacs_energies(openff_sys, mdp="cutoff_hbonds")

        roundtrip_energy = _run_gmx_energy(
            top_file="conv.top",
            gro_file="conv.gro",
            mdp_file=_get_mdp_file("cutoff_hbonds"),
        )

        # Differences in bond energies appear to be related to ParmEd's rounding
        # of the force constant and equilibrium bond length
        original_energy.compare(internal_energy)
        internal_energy.compare(
            roundtrip_energy,
            custom_tolerances={
                "Bond": 0.02 * omm_unit.kilojoule_per_mole,
            },
        )
        original_energy.compare(
            roundtrip_energy,
            custom_tolerances={
                "Bond": 0.02 * omm_unit.kilojoule_per_mole,
            },
        )


class TestParmEdAmber:
    @pytest.mark.slow
    def test_load_prmtop(self):
        struct = readparm.LoadParm(get_pmd_fn("trx.prmtop"))
        other_struct = readparm.AmberParm(get_pmd_fn("trx.prmtop"))
        prmtop = Interchange._from_parmed(struct)
        other_prmtop = Interchange._from_parmed(other_struct)

        for handler_key in prmtop.handlers:
            # TODO: Closer inspection of data
            assert handler_key in other_prmtop.handlers

        assert not prmtop.box

        struct.box = [20, 20, 20, 90, 90, 90]
        prmtop_converted = Interchange._from_parmed(struct)
        np.testing.assert_allclose(
            prmtop_converted.box, np.eye(3) * 2.0 * unit.nanometer
        )

    @pytest.mark.slow
    def test_read_box_parm7(self):
        top = readparm.LoadParm(get_pmd_fn("solv2.parm7"))
        out = Interchange._from_parmed(top)
        # pmd.load_file(get_pmd_fn("solv2.rst7")))
        # top = readparm.LoadParm(get_pmd_fn("solv2.parm7"), xyz=coords.coordinates)
        np.testing.assert_allclose(
            np.diag(out.box.m_as(unit.angstrom)), top.parm_data["BOX_DIMENSIONS"][1:]
        )
