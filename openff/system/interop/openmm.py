from simtk import openmm, unit

from openff.system.exceptions import UnsupportedCutoffMethodError
from openff.system.interop.parmed import _lj_params_from_potential

kcal_mol = unit.kilocalorie_per_mole
kcal_ang = kcal_mol / unit.angstrom ** 2
kcal_rad = kcal_mol / unit.radian ** 2

kj_mol = unit.kilojoule_per_mole
kj_nm = kj_mol / unit.nanometer ** 2
kj_rad = kj_mol / unit.radian ** 2


def to_openmm(openff_sys) -> openmm.System:
    """Convert an OpenFF System to a ParmEd Structure"""

    openmm_sys = openmm.System()

    # OpenFF box stored implicitly as nm, and that happens to be what
    # OpenMM casts box vectors to if provided only an np.ndarray
    if openff_sys.box is not None:
        openmm_sys.setDefaultPeriodicBoxVectors(*openff_sys.box)

    # Add particles (both atoms and virtual sites) with appropriate masses
    for atom in openff_sys.topology.topology_particles:
        openmm_sys.addParticle(atom.atom.mass)

    _process_nonbonded_forces(openff_sys, openmm_sys)
    _process_proper_torsion_forces(openff_sys, openmm_sys)
    if len(openff_sys.handlers["ImproperTorsions"].slot_map) > 0:
        _process_improper_torsion_forces(openff_sys, openmm_sys)
    _process_angle_forces(openff_sys, openmm_sys)
    _process_bond_forces(openff_sys, openmm_sys)

    return openmm_sys


def _process_bond_forces(openff_sys, openmm_sys):
    harmonic_bond_force = openmm.HarmonicBondForce()
    openmm_sys.addForce(harmonic_bond_force)

    bond_handler = openff_sys.handlers["Bonds"]
    for bond, key in bond_handler.slot_map.items():
        indices = eval(bond)
        params = bond_handler.potentials[key].parameters
        k = params["k"] * kcal_ang / kj_nm
        length = params["length"] * unit.angstrom / unit.nanometer

        harmonic_bond_force.addBond(
            particle1=indices[0],
            particle2=indices[1],
            length=length,
            k=k,
        )


def _process_angle_forces(openff_sys, openmm_sys):
    harmonic_angle_force = openmm.HarmonicAngleForce()
    openmm_sys.addForce(harmonic_angle_force)

    angle_handler = openff_sys.handlers["Angles"]
    for angle, key in angle_handler.slot_map.items():
        indices = eval(angle)
        params = angle_handler.potentials[key].parameters
        k = params["k"] * kcal_rad / kj_rad
        angle = params["angle"] * unit.degree

        harmonic_angle_force.addAngle(
            particle1=indices[0],
            particle2=indices[1],
            particle3=indices[2],
            angle=angle,
            k=k,
        )


def _process_proper_torsion_forces(openff_sys, openmm_sys):
    proper_torsion_force = openmm.PeriodicTorsionForce()
    openmm_sys.addForce(proper_torsion_force)

    torsion_handler = openff_sys.handlers["ProperTorsions"]
    idivf = torsion_handler.idivf

    for torsion_key, key in torsion_handler.slot_map.items():
        torsion, idx = torsion_key.split("_")
        indices = eval(torsion)
        params = torsion_handler.potentials[key].parameters

        k = params["k"] * kcal_mol / kj_mol
        periodicity = int(params["periodicity"])
        phase = params["phase"] * unit.degree

        proper_torsion_force.addTorsion(
            indices[0],
            indices[1],
            indices[2],
            indices[3],
            periodicity,
            phase,
            k / idivf,
        )


def _process_improper_torsion_forces(openff_sys, openmm_sys):
    raise NotImplementedError


def _process_nonbonded_forces(openff_sys, openmm_sys):
    # Store the pairings, not just the supported methods for each
    supported_cutoff_methods = [["cutoff", "pme"]]

    vdw_handler = openff_sys.handlers["vdW"]
    if vdw_handler.method not in [val[0] for val in supported_cutoff_methods]:
        raise UnsupportedCutoffMethodError()

    vdw_cutoff = vdw_handler.cutoff * unit.angstrom

    electrostatics_handler = openff_sys.handlers["Electrostatics"]  # Split this out
    if electrostatics_handler.method.lower() not in [
        v[1] for v in supported_cutoff_methods
    ]:
        raise UnsupportedCutoffMethodError()

    non_bonded_force = openmm.NonbondedForce()
    openmm_sys.addForce(non_bonded_force)

    for _ in openff_sys.topology.topology_particles:
        non_bonded_force.addParticle(0.0, 1.0, 0.0)

    if vdw_handler.method == "cutoff":
        if openff_sys.box is None:
            non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        else:
            non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
            non_bonded_force.setUseDispersionCorrection(True)
            non_bonded_force.setCutoffDistance(vdw_cutoff)

    for vdw_atom, vdw_smirks in vdw_handler.slot_map.items():
        atom_idx = eval(vdw_atom)[0]

        partial_charge = electrostatics_handler.charge_map[vdw_atom]
        vdw_potential = vdw_handler.potentials[vdw_smirks]
        sigma, epsilon = _lj_params_from_potential(vdw_potential)
        sigma = sigma * unit.angstrom
        epsilon = epsilon * kcal_mol

        non_bonded_force.setParticleParameters(atom_idx, partial_charge, sigma, epsilon)

    # from vdWHandler.postprocess_system
    bond_particle_indices = []

    for topology_molecule in openff_sys.topology.topology_molecules:

        top_mol_particle_start_index = topology_molecule.atom_start_topology_index

        for topology_bond in topology_molecule.bonds:

            top_index_1 = topology_molecule._ref_to_top_index[
                topology_bond.bond.atom1_index
            ]
            top_index_2 = topology_molecule._ref_to_top_index[
                topology_bond.bond.atom2_index
            ]

            top_index_1 += top_mol_particle_start_index
            top_index_2 += top_mol_particle_start_index

            bond_particle_indices.append((top_index_1, top_index_2))

    # OpenMM thinks these exceptions were already added
    non_bonded_force.createExceptionsFromBonds(
        bond_particle_indices,
        electrostatics_handler.scale_14,
        vdw_handler.scale_14,
    )
