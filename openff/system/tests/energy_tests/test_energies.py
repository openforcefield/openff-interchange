from copy import deepcopy

import mdtraj as md
import numpy as np
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.units import unit
from openff.utilities.testing import skip_if_missing
from openff.utilities.utils import has_executable
from simtk import openmm
from simtk import unit as simtk_unit
from simtk.openmm import app

from openff.system.components.misc import OFFBioTop
from openff.system.stubs import ForceField
from openff.system.tests.energy_tests.openmm import (
    _get_openmm_energies,
    get_openmm_energies,
)
from openff.system.utils import get_test_file_path

HAS_GROMACS = any(has_executable(e) for e in ["gmx", "gmx_d"])
HAS_LAMMPS = any(has_executable(e) for e in ["lammps", "lmp_mpi", "lmp_serial"])

if HAS_GROMACS:
    from openff.system.tests.energy_tests.gromacs import (
        _get_mdp_file,
        _run_gmx_energy,
        get_gromacs_energies,
    )
if HAS_LAMMPS:
    from openff.system.tests.energy_tests.lammps import get_lammps_energies

needs_gmx = pytest.mark.skipif(not HAS_GROMACS, reason="Needs GROMACS")
needs_lmp = pytest.mark.skipif(not HAS_LAMMPS, reason="Needs GROMACS")


@skip_if_missing("mbuild")
@needs_gmx
@needs_lmp
@pytest.mark.xfail
@pytest.mark.slow
@pytest.mark.parametrize("constrained", [True, False])
@pytest.mark.parametrize("mol_smi", ["C"])  # ["C", "CC"]
def test_energies_single_mol(constrained, mol_smi):
    import mbuild as mb

    mol = Molecule.from_smiles(mol_smi)
    mol.generate_conformers(n_conformers=1)
    mol.name = "FOO"
    top = mol.to_topology()
    top.box_vectors = None  # [10, 10, 10] * simtk_unit.nanometer

    if constrained:
        parsley = ForceField("openff-1.0.0.offxml")
    else:
        parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    off_sys = parsley.create_openff_system(top)

    off_sys.handlers["Electrostatics"].method = "cutoff"

    mol.to_file("out.xyz", file_format="xyz")
    compound: mb.Compound = mb.load("out.xyz")
    packed_box: mb.Compound = mb.fill_box(
        compound=compound, n_compounds=1, box=mb.Box(lengths=[10, 10, 10])
    )

    positions = packed_box.xyz * unit.nanometer
    off_sys.positions = positions

    # Compare directly to toolkit's reference implementation
    omm_energies = get_openmm_energies(off_sys, round_positions=8)
    omm_reference = parsley.create_openmm_system(top)
    reference_energies = _get_openmm_energies(
        omm_sys=omm_reference,
        box_vectors=off_sys.box,
        positions=off_sys.positions,
        round_positions=8,
    )

    omm_energies.compare(reference_energies)

    mdp = "cutoff_hbonds" if constrained else "auto"
    # Compare GROMACS writer and OpenMM export
    gmx_energies = get_gromacs_energies(off_sys, mdp=mdp)

    custom_tolerances = {
        "Bond": 2e-5 * simtk_unit.kilojoule_per_mole,
        "Electrostatics": 2 * simtk_unit.kilojoule_per_mole,
        "vdW": 2 * simtk_unit.kilojoule_per_mole,
        "Nonbonded": 2 * simtk_unit.kilojoule_per_mole,
        "Angle": 1e-4 * simtk_unit.kilojoule_per_mole,
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
            "vdW": 5.0 * simtk_unit.kilojoule_per_mole,
            "Electrostatics": 5.0 * simtk_unit.kilojoule_per_mole,
        }
        lmp_energies.compare(other_energies, custom_tolerances=custom_tolerances)


@needs_gmx
@needs_lmp
@pytest.mark.slow
def test_liquid_argon():
    argon = Molecule.from_smiles("[#18]")
    pdbfile = app.PDBFile(get_test_file_path("packed-argon.pdb"))

    top = Topology.from_openmm(pdbfile.topology, unique_molecules=[argon])

    argon_ff = ForceField(get_test_file_path("argon.offxml"))

    out = argon_ff.create_openff_system(top)
    out.positions = pdbfile.positions

    omm_energies = get_openmm_energies(out)

    gmx_energies = get_gromacs_energies(
        out,
        mdp="auto",
        writer="internal",
    )

    omm_energies.compare(
        gmx_energies,
        custom_tolerances={
            "vdW": 0.008 * simtk_unit.kilojoule_per_mole,
        },
    )

    argon_ff_no_switch = deepcopy(argon_ff)
    argon_ff_no_switch["vdW"].switch_width *= 0

    out_no_switch = argon_ff_no_switch.create_openff_system(top)
    out_no_switch.positions = pdbfile.positions

    lmp_energies = get_lammps_energies(out_no_switch)

    omm_energies.compare(
        lmp_energies,
        custom_tolerances={
            "vdW": 10.5 * simtk_unit.kilojoule_per_mole,
        },
    )


@needs_gmx
@pytest.mark.skip("Skip until residues are matched between gro and top")
@pytest.mark.parametrize(
    "toolkit_file_path",
    [
        # ("systems/test_systems/1_cyclohexane_1_ethanol.pdb", 18.165),
        "systems/packmol_boxes/cyclohexane_ethanol_0.4_0.6.pdb",
    ],
)
def test_packmol_boxes(toolkit_file_path):
    # TODO: Isolate a set of systems here instead of using toolkit data
    # TODO: Fix nonbonded energy differences
    from openff.toolkit.utils import get_data_file_path

    pdb_file_path = get_data_file_path(toolkit_file_path)
    pdbfile = openmm.app.PDBFile(pdb_file_path)

    ethanol = Molecule.from_smiles("CCO")
    cyclohexane = Molecule.from_smiles("C1CCCCC1")
    omm_topology = pdbfile.topology
    off_topology = OFFBioTop.from_openmm(
        omm_topology, unique_molecules=[ethanol, cyclohexane]
    )
    off_topology.mdtop = md.Topology.from_openmm(omm_topology)

    parsley = ForceField("openff_unconstrained-1.0.0.offxml")

    off_sys = parsley.create_openff_system(off_topology)

    off_sys.box = np.asarray(
        pdbfile.topology.getPeriodicBoxVectors().value_in_unit(simtk_unit.nanometer)
    )
    off_sys.positions = pdbfile.positions

    sys_from_toolkit = parsley.create_openmm_system(off_topology)

    omm_energies = get_openmm_energies(off_sys, hard_cutoff=True, electrostatics=False)
    reference = _get_openmm_energies(
        sys_from_toolkit,
        off_sys.box,
        off_sys.positions,
        hard_cutoff=True,
        electrostatics=False,
    )

    omm_energies.compare(
        reference,
        custom_tolerances={
            "Electrostatics": 2e-2 * simtk_unit.kilojoule_per_mole,
        },
    )

    # custom_tolerances={"HarmonicBondForce": 1.0}

    # Compare GROMACS writer and OpenMM export
    gmx_energies = get_gromacs_energies(off_sys, electrostatics=False)

    omm_energies_rounded = get_openmm_energies(
        off_sys,
        round_positions=8,
        hard_cutoff=True,
        electrostatics=False,
    )

    omm_energies_rounded.compare(
        other=gmx_energies,
        custom_tolerances={
            "Angle": 1e-2 * simtk_unit.kilojoule_per_mole,
            "Torsion": 1e-2 * simtk_unit.kilojoule_per_mole,
            "Electrostatics": 3200 * simtk_unit.kilojoule_per_mole,
        },
    )


@needs_lmp
@pytest.mark.slow
def test_water_dimer():
    tip3p = ForceField(get_test_file_path("tip3p.offxml"))
    water = Molecule.from_smiles("O")
    top = Topology.from_molecules(2 * [water])
    top.mdtop = md.Topology.from_openmm(top.to_openmm())

    pdbfile = openmm.app.PDBFile(get_test_file_path("water-dimer.pdb"))

    positions = pdbfile.positions

    openff_sys = tip3p.create_openff_system(top)
    openff_sys.positions = positions
    openff_sys.box = [10, 10, 10] * unit.nanometer

    omm_energies = get_openmm_energies(
        openff_sys,
        hard_cutoff=True,
        electrostatics=False,
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

    lmp_energies = get_lammps_energies(openff_sys, electrostatics=False)

    lmp_energies.compare(omm_energies)


@needs_gmx
@skip_if_missing("foyer")
@skip_if_missing("mbuild")
@pytest.mark.slow
def test_process_rb_torsions():
    """Test that the GROMACS driver reports Ryckaert-Bellemans torsions"""

    import foyer
    import mbuild as mb

    oplsaa = foyer.Forcefield(name="oplsaa")

    ethanol = Molecule.from_smiles("CCO")
    ethanol.generate_conformers(n_conformers=1)
    ethanol.generate_unique_atom_names()

    # Run this OFFMol through MoSDeF infrastructure and OPLS-AA
    from openff.system.tests.energy_tests.utils import offmol_to_compound

    my_compound = offmol_to_compound(ethanol)
    my_compound.box = mb.Box(lengths=[4, 4, 4])

    oplsaa = foyer.Forcefield(name="oplsaa")
    struct = oplsaa.apply(my_compound)

    struct.save("eth.top", overwrite=True)
    struct.save("eth.gro", overwrite=True)

    # Get single-point energies using GROMACS
    oplsaa_energies = _run_gmx_energy(
        top_file="eth.top", gro_file="eth.gro", mdp_file=_get_mdp_file("default")
    )

    assert oplsaa_energies.energies["Torsion"].m != 0.0


@needs_gmx
def test_gmx_14_energies_exist():
    # TODO: Make sure 1-4 energies are accurate, not just existent

    # Use a molecule with only one 1-4 interaction, and
    # make it between heavy atoms because H-H 1-4 are weak
    mol = Molecule.from_smiles("ClC#CCl")
    mol.name = "HPER"
    mol.generate_conformers(n_conformers=1)

    parsley = ForceField("openff-1.0.0.offxml")

    out = parsley.create_openff_system(topology=mol.to_topology())
    out.positions = mol.conformers[0]

    # Put this molecule in a large box with cut-off electrostatics
    # to prevent it from interacting with images of itself
    out.box = [40, 40, 40]
    out["Electrostatics"].method = "cutoff"

    gmx_energies = get_gromacs_energies(out)

    # The only possible non-bonded interactions should be from 1-4 intramolecular interactions
    assert gmx_energies.energies["vdW"].m != 0.0
    assert gmx_energies.energies["Electrostatics"].m != 0.0

    # TODO: It would be best to save the 1-4 interactions, split off into vdW and Electrostatics
    # in the energies. This might be tricky/intractable to do for engines that are not GROMACS


@needs_gmx
@needs_lmp
@pytest.mark.xfail
@pytest.mark.slow
def test_cutoff_electrostatics():
    ion_ff = ForceField(get_test_file_path("ions.offxml"))
    ions = Topology.from_molecules(
        [
            Molecule.from_smiles("[#3]"),
            Molecule.from_smiles("[#17]"),
        ]
    )
    out = ion_ff.create_openff_system(ions)
    out.box = [4, 4, 4] * unit.nanometer

    gmx = []
    lmp = []

    for d in np.linspace(0.75, 0.95, 5):
        positions = np.zeros((2, 3)) * unit.nanometer
        positions[1, 0] = d * unit.nanometer
        out.positions = positions

        out["Electrostatics"].method = "cutoff"
        gmx.append(get_gromacs_energies(out, mdp="auto").energies["Electrostatics"].m)
        lmp.append(
            get_lammps_energies(out)
            .energies["Electrostatics"]
            .m_as(unit.kilojoule / unit.mol)
        )

    assert np.sum(np.sqrt(np.square(np.asarray(lmp) - np.asarray(gmx)))) < 1e-3
