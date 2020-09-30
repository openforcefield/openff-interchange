import numpy as np
from openforcefield.topology import Molecule, Topology
from simtk import unit


def top_from_smiles(
    smiles: str,
    n_molecules: int = 1,
) -> Topology:
    """Create a gas phase OpenFF Topology from a single-molecule SMILES

    Parameters
    ----------
    smiles : str
        The SMILES of the input molecule
    n_molecules : int, optional, default = 1
        The number of copies of the SMILES molecule from which to
        compose a topology

    Returns
    -------
    top : openforcefield.topology.Topology
        A single-molecule, gas phase-like topology

    """
    mol = Molecule.from_smiles(smiles)
    mol.generate_conformers(n_conformers=1)
    top = Topology.from_molecules(n_molecules * [mol])
    # Add dummy box vectors
    # TODO: Revisit if/after Topology.is_periodic
    top.box_vectors = np.eye(3) * 10 * unit.nanometer
    return top
