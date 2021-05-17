from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ImproperTorsionHandler
from simtk import unit as simtk_unit

from openff.system.components.smirnoff import SMIRNOFFImproperTorsionHandler
from openff.system.models import TopologyKey


def test_store_improper_torsion_matches():

    formaldehyde: Molecule = Molecule.from_mapped_smiles("[H:3][C:1]([H:4])=[O:2]")

    parameter_handler = ImproperTorsionHandler(version=0.3)
    parameter_handler.add_parameter(
        parameter=ImproperTorsionHandler.ImproperTorsionType(
            smirks="[*:1]~[#6X3:2](~[*:3])~[*:4]",
            periodicity1=2,
            phase1=180.0 * simtk_unit.degree,
            k1=1.1 * simtk_unit.kilocalorie_per_mole,
        )
    )

    potential_handler = SMIRNOFFImproperTorsionHandler()
    potential_handler.store_matches(parameter_handler, formaldehyde.to_topology())

    assert len(potential_handler.slot_map) == 3

    assert TopologyKey(atom_indices=(0, 1, 2, 3), mult=0) in potential_handler.slot_map
    assert TopologyKey(atom_indices=(0, 2, 3, 1), mult=0) in potential_handler.slot_map
    assert TopologyKey(atom_indices=(0, 3, 1, 2), mult=0) in potential_handler.slot_map
