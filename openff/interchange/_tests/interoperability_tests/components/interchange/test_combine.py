import pytest
from openff.toolkit import Quantity

from openff.interchange import Interchange
from openff.interchange._tests import MoleculeWithConformer
from openff.interchange.drivers import get_openmm_energies


def test_combine_after_from_openmm_with_mainline_openmm_force_field(
    popc,
    sage,
):
    # amber/lipid17.xml is not shipped with OpenMM
    pytest.importorskip("openmmforcefields")

    import openmm.app
    import openmm.unit

    ligand = MoleculeWithConformer.from_smiles("c1ccccc1")
    ligand._conformers[0] += Quantity([3, 3, 3], "angstrom")
    topology = ligand.to_topology()
    topology.box_vectors = Quantity([6, 6, 6], "nanometer")

    popc_topology = popc.to_topology()
    popc_topology.box_vectors = Quantity([6, 6, 6], "nanometer")

    lipid17 = openmm.app.ForceField("amber/lipid17.xml")

    popc_system = lipid17.createSystem(
        topology=popc_topology.to_openmm(),
        nonbondedMethod=openmm.app.PME,
        constraints=openmm.app.HBonds,
    )

    imported = Interchange.from_openmm(
        system=popc_system,
        topology=popc_topology,
        positions=popc_topology.get_positions(),
    )

    ligand_interchange = sage.create_interchange(topology)

    for item in [imported, ligand_interchange]:
        item["vdW"].cutoff = Quantity(9, "angstrom")
        item["vdW"].switch_width = Quantity(1, "angstrom")
        item["Electrostatics"].cutoff = Quantity(9, "angstrom")

    combined = imported.combine(ligand_interchange)

    combined.to_openmm()

    energies = {
        "ligand": get_openmm_energies(ligand_interchange),
        "popc": get_openmm_energies(imported),
        "combined": get_openmm_energies(combined),
    }

    for key in ["Bond", "Angle", "Torsion"]:
        assert (energies["ligand"][key] + energies["popc"][key]).m == pytest.approx(
            energies["combined"][key].m,
        ), key

    # Non-bonded energies are not linear sums of component non-bonded energies, so
    # just make sure it's not a NaN (though a NaN should probably cause a crash ...)
    assert isinstance(energies["combined"]["Nonbonded"].m, float)
