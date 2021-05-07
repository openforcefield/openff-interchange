import glob

import foyer
import mdtraj as md
import numpy as np
import parmed as pmd
import pytest
from openff.toolkit.topology import Molecule
from openff.units import unit
from simtk import unit as simtk_unit

from openff.system.components.foyer import from_foyer
from openff.system.components.misc import OFFBioTop
from openff.system.tests.base_test import BaseTest
from openff.system.tests.energy_tests.gromacs import (
    _get_mdp_file,
    _run_gmx_energy,
    get_gromacs_energies,
)
from openff.system.tests.energy_tests.openmm import get_openmm_energies
from openff.system.utils import get_test_files_dir_path


class TestFoyer(BaseTest):
    @pytest.fixture(scope="session")
    def oplsaa(self):
        return foyer.forcefields.load_OPLSAA()

    @pytest.fixture(scope="session")
    def oplsaa_system_ethanol(self, oplsaa):

        molecule = Molecule.from_file(
            get_test_files_dir_path("foyer_test_molecules") + "/ethanol.sdf"
        )
        molecule.name = "ETH"

        top = OFFBioTop.from_molecules(molecule)
        top.mdtop = md.Topology.from_openmm(top.to_openmm())
        oplsaa = foyer.Forcefield(name="oplsaa")
        system = from_foyer(topology=top, ff=oplsaa)
        system.positions = molecule.conformers[0].value_in_unit(simtk_unit.nanometer)
        system.box = [4, 4, 4]
        return system

    @pytest.fixture(scope="session")
    def get_systems(self, oplsaa):
        def systems_from_path(molecule_path):
            molecule_or_molecules = Molecule.from_file(molecule_path)
            off_bio_top = OFFBioTop.from_molecules(molecule_or_molecules)
            off_bio_top.mdtop = md.Topology.from_openmm(off_bio_top.to_openmm())
            openff_system = from_foyer(off_bio_top, oplsaa)
            if isinstance(molecule_or_molecules, list):
                openff_system.positions = np.vstack(
                    tuple(
                        molecule.conformers[0].value_in_unit(simtk_unit.nanometer)
                        for molecule in molecule_or_molecules
                    )
                )
            else:
                openff_system.positions = molecule_or_molecules.conformers[
                    0
                ].value_in_unit(simtk_unit.nanometer)
            openff_system.box = [4, 4, 4]

            parmed_struct = pmd.openmm.load_topology(off_bio_top.to_openmm())
            parmed_struct.positions = openff_system.positions.m_as(unit.angstrom)
            parmed_struct.box = [40, 40, 40, 90, 90, 90]

            return openff_system, parmed_struct

        return systems_from_path

    def test_handlers_exist(self, oplsaa_system_ethanol):
        for _, handler in oplsaa_system_ethanol.handlers.items():
            assert handler

        assert oplsaa_system_ethanol["vdW"].scale_14 == 0.5
        assert oplsaa_system_ethanol["Electrostatics"].scale_14 == 0.5

    @pytest.mark.slow
    def test_ethanol_energies(self, oplsaa_system_ethanol):
        gmx_energies = get_gromacs_energies(oplsaa_system_ethanol)
        omm_energies = get_openmm_energies(oplsaa_system_ethanol)

        gmx_energies.compare(
            omm_energies,
            custom_tolerances={
                "vdW": 12.0 * unit.kilojoule / unit.mole,
                "Electrostatics": 12.0 * unit.kilojoule / unit.mole,
            },
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        argnames="molecule_path",
        argvalues=glob.glob(get_test_files_dir_path("foyer_test_molecules") + "/*.sdf"),
    )
    def test_system_energies(self, molecule_path, get_systems, oplsaa):
        openff_system, pmd_structure = get_systems(molecule_path)
        parameterized_pmd_structure = oplsaa.apply(pmd_structure)
        openff_energy = get_gromacs_energies(openff_system)
        print(openff_system.handlers["Bonds"])
        parameterized_pmd_structure.save("from_foyer.gro")
        parameterized_pmd_structure.save("from_foyer.top")

        through_foyer = _run_gmx_energy(
            top_file="from_foyer.top",
            gro_file="from_foyer.gro",
            mdp_file=_get_mdp_file("cutoff_hbonds"),
        )

        openff_energy.compare(
            through_foyer,
            custom_tolerances={
                "Bond": 1.8 * simtk_unit.kilojoule_per_mole,
                "Angle": 0.06 * simtk_unit.kilojoule_per_mole,
                "Nonbonded": 30 * simtk_unit.kilojoule_per_mole,
            },
        )
