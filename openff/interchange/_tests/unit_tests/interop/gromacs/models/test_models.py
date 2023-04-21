import numpy
import pytest
from openff.toolkit import Molecule, Topology
from openff.units import unit

from openff.interchange import Interchange
from openff.interchange._tests import _BaseTest, needs_gmx
from openff.interchange.components.mdconfig import get_intermol_defaults
from openff.interchange.drivers.gromacs import _process, _run_gmx_energy
from openff.interchange.interop.gromacs.export._export import GROMACSWriter
from openff.interchange.smirnoff._gromacs import _convert


class TestAddRemoveMoleculeType(_BaseTest):
    @pytest.fixture()
    def molecule1(self):
        molecule = Molecule.from_smiles(
            "[H][O][c]1[c]([H])[c]([O][H])[c]([H])[c]([O][H])[c]1[H]",
        )
        molecule.generate_conformers(n_conformers=1)
        molecule.name = "MOL1"

        return molecule

    @pytest.fixture()
    def molecule2(self):
        molecule = Molecule.from_smiles("C1=C(C=C(C=C1C(=O)O)C(=O)O)C(=O)O")
        molecule.generate_conformers(n_conformers=1)
        molecule.name = "MOL2"

        molecule.conformers[0] += numpy.array([5, 0, 0]) * unit.angstrom

        return molecule

    @pytest.fixture()
    def system1(self, molecule1, sage):
        box = 5 * numpy.eye(3) * unit.nanometer

        return _convert(Interchange.from_smirnoff(sage, [molecule1], box=box))

    @pytest.fixture()
    def system2(self, molecule2, sage):
        box = 5 * numpy.eye(3) * unit.nanometer

        return _convert(Interchange.from_smirnoff(sage, [molecule2], box=box))

    @pytest.fixture()
    def combined_system(self, sage, molecule1, molecule2):
        box = 5 * numpy.eye(3) * unit.nanometer

        return _convert(
            Interchange.from_smirnoff(
                sage,
                Topology.from_molecules([molecule1, molecule2]),
                box=box,
            ),
        )

    @needs_gmx
    @pytest.mark.parametrize("molecule_name", ["MOL1", "MOL2"])
    def test_remove_basic(self, combined_system, molecule_name):
        combined_system.remove_molecule_type(molecule_name)

        # Just a sanity check
        writer = GROMACSWriter(
            system=combined_system,
            top_file=f"{molecule_name}.top",
            gro_file=f"{molecule_name}.gro",
        )

        writer.to_top()
        writer.to_gro(decimal=8)

        get_intermol_defaults(periodic=True).write_mdp_file("tmp.mdp")

        _process(
            _run_gmx_energy(f"{molecule_name}.top", f"{molecule_name}.gro", "tmp.mdp"),
        )

    @pytest.mark.parametrize("molecule_name", ["MOL1", "MOL2"])
    def test_add_existing_molecule_type(self, combined_system, molecule_name):
        with pytest.raises(
            ValueError,
            match=f"The molecule type {molecule_name} is already present in this system.",
        ):
            combined_system.add_molecule_type(
                combined_system.molecule_types[molecule_name],
                1,
            )

    @needs_gmx
    def test_molecule_order_independent(self, system1, system2):
        positions1 = numpy.vstack([system1.positions, system2.positions])
        positions2 = numpy.vstack([system2.positions, system1.positions])

        system1.add_molecule_type(system2.molecule_types["MOL2"], 1)
        system1.positions = positions1

        system2.add_molecule_type(system1.molecule_types["MOL1"], 1)
        system2.positions = positions2

        writer1 = GROMACSWriter(
            system=system1,
            top_file="order1.top",
            gro_file="order1.gro",
        )

        writer2 = GROMACSWriter(
            system=system2,
            top_file="order2.top",
            gro_file="order2.gro",
        )

        for writer in [writer1, writer2]:
            writer.to_top()
            writer.to_gro(decimal=8)

        get_intermol_defaults(periodic=True).write_mdp_file("tmp.mdp")

        _process(_run_gmx_energy("order1.top", "order1.gro", "tmp.mdp")).compare(
            _process(_run_gmx_energy("order2.top", "order2.gro", "tmp.mdp")),
        )

    def test_clashing_atom_types(self, combined_system, system1, system2):
        with pytest.raises(
            ValueError,
            match="The molecule type MOL1 is already present in this system.",
        ):
            combined_system.add_molecule_type(system1.molecule_types["MOL1"], 1)

        with pytest.raises(
            ValueError,
            match="The molecule type MOL2 is already present in this system.",
        ):
            combined_system.add_molecule_type(system2.molecule_types["MOL2"], 1)

        with pytest.raises(
            ValueError,
            match="The molecule type MOL1 is already present in this system.",
        ):
            system1.add_molecule_type(system1.molecule_types["MOL1"], 1)

        with pytest.raises(
            ValueError,
            match="The molecule type MOL2 is already present in this system.",
        ):
            system2.add_molecule_type(system2.molecule_types["MOL2"], 1)
