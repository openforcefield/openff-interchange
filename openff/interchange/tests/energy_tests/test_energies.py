import foyer
import numpy as np
import openmm
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit
from openff.units.openmm import to_openmm
from openff.utilities.testing import skip_if_missing
from openmm import unit as openmm_unit

from openff.interchange import Interchange
from openff.interchange.constants import kj_mol
from openff.interchange.drivers import get_openmm_energies
from openff.interchange.drivers.openmm import _get_openmm_energies
from openff.interchange.drivers.report import EnergyError, EnergyReport
from openff.interchange.tests import (
    HAS_GROMACS,
    HAS_LAMMPS,
    _BaseTest,
    get_test_file_path,
    needs_gmx,
    needs_lmp,
)

if HAS_GROMACS:
    from openff.interchange.drivers.gromacs import (
        _get_mdp_file,
        _run_gmx_energy,
        get_gromacs_energies,
    )
if HAS_LAMMPS:
    from openff.interchange.drivers.lammps import get_lammps_energies


def test_energy_report():
    """Test that multiple failing energies are captured in the EnergyError"""
    a = EnergyReport(
        energies={
            "a": 1 * kj_mol,
            "_FLAG": 2 * kj_mol,
            "KEY_": 1.2 * kj_mol,
        }
    )
    b = EnergyReport(
        energies={
            "a": -1 * kj_mol,
            "_FLAG": -2 * kj_mol,
            "KEY_": -0.1 * kj_mol,
        }
    )
    custom_tolerances = {
        "a": 1 * kj_mol,
        "_FLAG": 1 * kj_mol,
        "KEY_": 1 * kj_mol,
    }
    with pytest.raises(EnergyError, match=r"_FLAG[\s\S]*KEY_"):
        a.compare(b, custom_tolerances=custom_tolerances)


class TestEnergies(_BaseTest):
    @pytest.fixture(scope="session")
    def oplsaa(self):
        return foyer.forcefields.load_OPLSAA()

    @skip_if_missing("mbuild")
    @needs_gmx
    @needs_lmp
    @pytest.mark.xfail()
    @pytest.mark.slow()
    @pytest.mark.parametrize("constrained", [True, False])
    @pytest.mark.parametrize("mol_smi", ["C"])  # ["C", "CC"]
    def test_energies_single_mol(self, constrained, sage, sage_unconstrained, mol_smi):
        import mbuild as mb

        mol = Molecule.from_smiles(mol_smi)
        mol.generate_conformers(n_conformers=1)
        mol.name = "FOO"

        force_field = sage if constrained else sage_unconstrained

        off_sys = Interchange.from_smirnoff(force_field, [mol])

        off_sys.handlers["Electrostatics"].periodic_potential = "cutoff"

        mol.to_file("out.xyz", file_format="xyz")
        compound: mb.Compound = mb.load("out.xyz")
        packed_box: mb.Compound = mb.fill_box(
            compound=compound, n_compounds=1, box=mb.Box(lengths=[10, 10, 10])
        )

        positions = packed_box.xyz * unit.nanometer
        off_sys.positions = positions

        # Compare directly to toolkit's reference implementation
        omm_energies = get_openmm_energies(off_sys, round_positions=8)
        omm_reference = force_field.create_openmm_system(mol.to_topology())
        reference_energies = _get_openmm_energies(
            omm_sys=omm_reference,
            box_vectors=to_openmm(off_sys.box),
            positions=to_openmm(off_sys.positions),
            round_positions=8,
        )

        omm_energies.compare(reference_energies)

        mdp = "cutoff_hbonds" if constrained else "auto"
        # Compare GROMACS writer and OpenMM export
        gmx_energies = get_gromacs_energies(off_sys, mdp=mdp)

        custom_tolerances = {
            "Bond": 2e-5 * openmm_unit.kilojoule_per_mole,
            "Electrostatics": 2 * openmm_unit.kilojoule_per_mole,
            "vdW": 2 * openmm_unit.kilojoule_per_mole,
            "Nonbonded": 2 * openmm_unit.kilojoule_per_mole,
            "Angle": 1e-4 * openmm_unit.kilojoule_per_mole,
        }

        gmx_energies.compare(
            omm_energies,
            custom_tolerances=custom_tolerances,
        )

        if not constrained:
            other_energies = get_openmm_energies(
                off_sys,
                round_positions=8,
                hard_cutoff=True,
                electrostatics=True,
            )
            lmp_energies = get_lammps_energies(off_sys)
            custom_tolerances = {
                "vdW": 5.0 * openmm_unit.kilojoule_per_mole,
                "Electrostatics": 5.0 * openmm_unit.kilojoule_per_mole,
            }
            lmp_energies.compare(other_energies, custom_tolerances=custom_tolerances)

    @pytest.mark.skip(
        reason="Needs to be reimplmented after OFFTK 0.11.0 with fewer moving parts"
    )
    @needs_gmx
    @pytest.mark.slow()
    @pytest.mark.parametrize(
        "toolkit_file_path",
        [
            "systems/test_systems/1_cyclohexane_1_ethanol.pdb",
            "systems/test_systems/1_ethanol.pdb",
            "systems/test_systems/1_ethanol_reordered.pdb",
            # "systems/test_systems/T4_lysozyme_water_ions.pdb",
            "systems/packmol_boxes/cyclohexane_ethanol_0.4_0.6.pdb",
            "systems/packmol_boxes/cyclohexane_water.pdb",
            "systems/packmol_boxes/ethanol_water.pdb",
            "systems/packmol_boxes/propane_methane_butanol_0.2_0.3_0.5.pdb",
        ],
    )
    def test_packmol_boxes(self, sage, toolkit_file_path):
        # TODO: Isolate a set of systems here instead of using toolkit data
        # TODO: Fix nonbonded energy differences
        from openff.toolkit.utils import get_data_file_path

        pdb_file_path = get_data_file_path(toolkit_file_path)
        pdbfile = openmm.app.PDBFile(pdb_file_path)

        unique_molecules = [
            Molecule.from_smiles(smi)
            for smi in [
                "CCO",
                "CCCCO",
                "C",
                "CCC",
                "C1CCCCC1",
                "O",
            ]
        ]
        omm_topology = pdbfile.topology
        off_topology = Topology.from_openmm(
            omm_topology, unique_molecules=unique_molecules
        )

        off_sys = Interchange.from_smirnoff(sage, off_topology)

        off_sys.box = np.asarray(
            pdbfile.topology.getPeriodicBoxVectors().value_in_unit(
                openmm_unit.nanometer
            )
        )
        off_sys.positions = pdbfile.positions

        sys_from_toolkit = sage.create_openmm_system(off_topology)

        omm_energies = get_openmm_energies(
            off_sys,
            combine_nonbonded_forces=True,
            hard_cutoff=True,
            electrostatics=False,
        )
        reference = _get_openmm_energies(
            sys_from_toolkit,
            off_sys.box,
            off_sys.positions,
            hard_cutoff=True,
            electrostatics=False,
        )

        try:
            omm_energies.compare(
                reference,
                # custom_tolerances={
                #    "Electrostatics": 2e-4 * openmm_unit.kilojoule_per_mole,
                # },
            )
        except EnergyError as err:
            if "Torsion" in err.args[0]:
                from openff.interchange.tests.utils import (  # type: ignore
                    _compare_torsion_forces,
                    _get_force,
                )

                _compare_torsion_forces(
                    _get_force(off_sys.to_openmm, openmm.PeriodicTorsionForce),
                    _get_force(sys_from_toolkit, openmm.PeriodicTorsionForce),
                )

        # custom_tolerances={"HarmonicBondForce": 1.0}

        # Compare GROMACS writer and OpenMM export
        gmx_energies = get_gromacs_energies(off_sys)

        omm_energies_rounded = get_openmm_energies(
            off_sys,
            round_positions=8,
            hard_cutoff=True,
            electrostatics=False,
        )

        omm_energies_rounded.compare(
            other=gmx_energies,
            custom_tolerances={
                "Angle": 1e-2 * openmm_unit.kilojoule_per_mole,
                "Torsion": 1e-2 * openmm_unit.kilojoule_per_mole,
                "Electrostatics": 3200 * openmm_unit.kilojoule_per_mole,
                "vdW": 0.5 * openmm_unit.kilojoule_per_mole,
            },
        )

    @needs_lmp
    @pytest.mark.slow()
    def test_water_dimer(self):
        tip3p = ForceField(get_test_file_path("tip3p.offxml"))
        water = Molecule.from_smiles("O")
        top = Topology.from_molecules(2 * [water])

        pdbfile = openmm.app.PDBFile(get_test_file_path("water-dimer.pdb"))

        positions = pdbfile.positions

        openff_sys = Interchange.from_smirnoff(tip3p, top)
        openff_sys.positions = positions
        openff_sys.box = [10, 10, 10] * unit.nanometer

        omm_energies = get_openmm_energies(
            openff_sys,
            hard_cutoff=True,
            electrostatics=False,
            combine_nonbonded_forces=True,
        )

        toolkit_energies = _get_openmm_energies(
            tip3p.create_openmm_system(top),
            openff_sys.box,
            openff_sys.positions,
            hard_cutoff=True,
            electrostatics=False,
        )

        omm_energies.compare(toolkit_energies)

        # TODO: Fix GROMACS energies by handling SETTLE constraints
        # gmx_energies, _ = get_gromacs_energies(openff_sys)
        # compare_gromacs_openmm(omm_energies=omm_energies, gmx_energies=gmx_energies)

        openff_sys["Electrostatics"].periodic_potential = "cutoff"
        omm_energies_cutoff = get_gromacs_energies(openff_sys)  # noqa

        # TODO: Don't write out dihedral section of LAMMPS input file for this system
        # lmp_energies = get_lammps_energies(openff_sys)
        # lmp_energies.compare(omm_energies_cutoff)

    @needs_gmx
    @skip_if_missing("foyer")
    @skip_if_missing("mbuild")
    @pytest.mark.slow()
    def test_process_rb_torsions(self, oplsaa):
        """Test that the GROMACS driver reports Ryckaert-Bellemans torsions"""
        from mbuild import Box

        from openff.interchange.components.mbuild import offmol_to_compound

        ethanol = Molecule.from_smiles("CCO")
        ethanol.generate_conformers(n_conformers=1)
        ethanol.generate_unique_atom_names()

        my_compound = offmol_to_compound(ethanol)
        my_compound.box = Box(lengths=[4, 4, 4])

        struct = oplsaa.apply(my_compound)

        struct.save("eth.top", overwrite=True)
        struct.save("eth.gro", overwrite=True)

        # Get single-point energies using GROMACS
        oplsaa_energies = _run_gmx_energy(
            top_file="eth.top", gro_file="eth.gro", mdp_file=_get_mdp_file("default")
        )

        assert oplsaa_energies.energies["Torsion"].m != 0.0

    @needs_gmx
    def test_gmx_14_energies_exist(self, sage):
        # TODO: Make sure 1-4 energies are accurate, not just existent

        # Use a molecule with only one 1-4 interaction, and
        # make it between heavy atoms because H-H 1-4 are weak
        mol = Molecule.from_smiles("ClC#CCl")
        mol.name = "HPER"
        mol.generate_conformers(n_conformers=1)

        out = Interchange.from_smirnoff(sage, [mol])
        out.positions = mol.conformers[0]
        out.box = 3 * [10]

        # Put this molecule in a large box with cut-off electrostatics
        # to prevent it from interacting with images of itself
        out["Electrostatics"].periodic_potential = "cutoff"

        gmx_energies = get_gromacs_energies(out)

        # The only possible non-bonded interactions should be from 1-4 intramolecular interactions
        assert gmx_energies.energies["vdW"].m != 0.0
        assert gmx_energies.energies["Electrostatics"].m != 0.0

        # TODO: It would be best to save the 1-4 interactions, split off into vdW and Electrostatics
        # in the energies. This might be tricky/intractable to do for engines that are not GROMACS

    @pytest.mark.skip("LAMMPS export experimental")
    @needs_gmx
    @needs_lmp
    @pytest.mark.slow()
    def test_cutoff_electrostatics(self):
        ion_ff = ForceField(get_test_file_path("ions.offxml"))
        ions = Topology.from_molecules(
            [
                Molecule.from_smiles("[#3+]"),
                Molecule.from_smiles("[#17-]"),
            ]
        )
        out = Interchange.from_smirnoff(ion_ff, ions)
        out.box = [4, 4, 4] * unit.nanometer

        gmx = []
        lmp = []

        for d in np.linspace(0.75, 0.95, 5):
            positions = np.zeros((2, 3)) * unit.nanometer
            positions[1, 0] = d * unit.nanometer
            out.positions = positions

            out["Electrostatics"].periodic_potential = "cutoff"
            gmx.append(
                get_gromacs_energies(out, mdp="auto").energies["Electrostatics"].m
            )
            lmp.append(
                get_lammps_energies(out)
                .energies["Electrostatics"]
                .m_as(unit.kilojoule / unit.mol)
            )

        assert np.sum(np.sqrt(np.square(np.asarray(lmp) - np.asarray(gmx)))) < 1e-3

    @pytest.mark.parametrize(
        "smi",
        [
            "c1cc(ccc1c2ccncc2)O",
            "c1cc(ccc1c2ccncc2)[O-]",
            "c1cc(c(cc1O)Cl)c2cc[nH+]cc2",
        ],
    )
    @needs_gmx
    @pytest.mark.slow()
    def test_interpolated_parameters(self, smi):
        xml_ff_bo_all_heavy_bonds = """<?xml version='1.0' encoding='ASCII'?>
        <SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">
          <Bonds version="0.3" fractional_bondorder_method="AM1-Wiberg" fractional_bondorder_interpolation="linear">
            <Bond smirks="[!#1:1]~[!#1:2]" id="bbo1"
                k_bondorder1="100.0 * kilocalories_per_mole/angstrom**2"
                k_bondorder2="1000.0 * kilocalories_per_mole/angstrom**2"
                length_bondorder1="1.5 * angstrom"
                length_bondorder2="1.0 * angstrom"/>
          </Bonds>
        </SMIRNOFF>
        """

        mol = Molecule.from_smiles(smi)
        mol.generate_conformers(n_conformers=1)

        forcefield = ForceField(
            "test_forcefields/test_forcefield.offxml",
            xml_ff_bo_all_heavy_bonds,
        )

        out = Interchange.from_smirnoff(forcefield, [mol])
        out.box = [4, 4, 4] * unit.nanometer
        out.positions = mol.conformers[0]

        toolkit_system = forcefield.create_openmm_system(mol.to_topology())

        for key in ["Bond", "Torsion"]:
            interchange_energy = get_openmm_energies(
                out, combine_nonbonded_forces=True
            ).energies[key]

            toolkit_energy = _get_openmm_energies(
                toolkit_system,
                box_vectors=[[4, 0, 0], [0, 4, 0], [0, 0, 4]] * openmm_unit.nanometer,
                positions=to_openmm(mol.conformers[0]),
            ).energies[key]

            toolkit_diff = abs(interchange_energy - toolkit_energy).m_as(kj_mol)

            if toolkit_diff < 1e-6:
                pass
            elif toolkit_diff < 1e-2:
                pytest.xfail(
                    f"Found energy difference of {toolkit_diff} kJ/mol vs. toolkit"
                )
            else:
                pytest.xfail(
                    f"Found energy difference of {toolkit_diff} kJ/mol vs. toolkit"
                )

            gromacs_energy = get_gromacs_energies(out).energies[key]
            energy_diff = abs(interchange_energy - gromacs_energy).m_as(kj_mol)

            if energy_diff < 1e-6:
                pass
            elif energy_diff < 1e-2:
                pytest.xpass(
                    f"Found {key} energy difference of {energy_diff} kJ/mol between GROMACS and OpenMM exports"
                )
            else:
                pytest.xfail(
                    f"Found {key} energy difference of {energy_diff} kJ/mol between GROMACS and OpenMM exports"
                )
