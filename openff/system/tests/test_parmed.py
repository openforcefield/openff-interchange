import mdtraj as md
import parmed as pmd
import pytest
from simtk import unit as omm_unit

from openff.system.components.system import System
from openff.system.tests.base_test import BaseTest
from openff.system.tests.energy_tests.gromacs import (
    _get_mdp_file,
    _run_gmx_energy,
    get_gromacs_energies,
)
from openff.system.utils import get_test_file_path


class TestParmEd(BaseTest):
    @pytest.mark.slow
    def test_parmed_roundtrip(self):
        original = pmd.load_file(get_test_file_path("ALA_GLY/ALA_GLY.top"))
        gro = pmd.load_file(get_test_file_path("ALA_GLY/ALA_GLY.gro"))
        original.box = gro.box
        original.positions = gro.positions

        openff_sys = System._from_parmed(original)
        openff_sys.topology.mdtop = md.Topology.from_openmm(gro.topology)

        #  Some sanity checks, including that residues are stored ...
        assert openff_sys.topology._topology_molecules[0].n_atoms == 29
        assert openff_sys.topology.n_topology_molecules == 1
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
