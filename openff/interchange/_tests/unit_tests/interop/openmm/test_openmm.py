import pandas
from openff.toolkit import ForceField, Molecule
from openff.units import unit
from openff.utilities.testing import skip_if_missing

from openff.interchange.components.mdconfig import get_intermol_defaults
from openff.interchange.drivers import get_summary_data


def run(smiles: str, constrained: bool) -> pandas.DataFrame:
    molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=1)
    topology = molecule.to_topology()
    topology.box_vectors = [4, 4, 4] * unit.nanometer

    if constrained:
        force_field = ForceField("openff-2.0.0.offxml")
    else:
        force_field = ForceField("openff_unconstrained-2.0.0.offxml")

    interchange = force_field.create_interchange(topology)
    get_intermol_defaults(periodic=True).apply(interchange)
    return get_summary_data(interchange)


def compare(smiles: str) -> float:
    return run(smiles, True).std()["Bond"] - run(smiles, False).std()["Bond"]


@skip_if_missing("openmm")
class TestToOpenMM:
    def test_combine_nonbonded_forces_vdw_14(self, sage_unconstrained):
        molecule = Molecule.from_mapped_smiles("[C:2](#[C:3][Br:4])[Br:1]")
        molecule.generate_conformers(n_conformers=1)
        topology = molecule.to_topology()
        topology.box_vectors = [4, 4, 4] * unit.nanometer

        interchange = sage_unconstrained.create_interchange(topology)

        split_system = interchange.to_openmm(combine_nonbonded_forces=False)

        vdw_14_force = [
            force
            for force in split_system.getForces()
            if force.getName() == "vdW 1-4 force"
        ][0]
        vdw_force = [
            force
            for force in split_system.getForces()
            if force.getName() == "vdW force"
        ][0]

        assert vdw_force.getParticleParameters(0) == vdw_force.getParticleParameters(3)

        _, _, (sigma, epsilon) = vdw_14_force.getBondParameters(0)

        # epsilon of 1-4 interaction matches either atom's sigma
        assert sigma == vdw_force.getParticleParameters(0)[0]

        # epsilon of 1-4 interaction matches either atom's sigma
        scale = sage_unconstrained["vdW"].scale14
        assert epsilon == vdw_force.getParticleParameters(0)[1] * scale
