from system import System
from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField


# Load Parsley and populate a dummy topology (ignoring positions for the moment)
openff_forcefield = ForceField('scratch/ar.offxml')

# Create OpenFF Molecule containing only one Argon atom
mol = Molecule.from_smiles('[#18]')
mol.generate_conformers(n_conformers=1)

# Generate an OpenFF Topology from 10 Argon "molecules"
openff_topology = Topology.from_molecules(10 * [mol])
#
# Construct an OpenFF System with the force field and topology
openff_system = System(
    toolkit_topology=openff_topology,
    toolkit_forcefield=openff_forcefield,
    positions=None,
    box=None
)

openff_system.populate_from_toolkit_data()

stuff_to_print = {
    'openff_system.forcefield':                     openff_system.forcefield,
    'openff_system.forcefield["n1"]':               openff_system.forcefield['n1'],
    'openff_system.forcefield["n1"].expression':    openff_system.forcefield['n1'].expression,
    'openff_system.forcefield["n1"].parameters':    openff_system.forcefield['n1'].parameters,
    }
for key, val in stuff_to_print.items():
    print(f'Calling {key}:\n\treturns {val}\n')
