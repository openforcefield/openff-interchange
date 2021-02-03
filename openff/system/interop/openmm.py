from simtk import openmm, unit

from openff.system import unit as off_unit
from openff.system.exceptions import UnsupportedCutoffMethodError
from openff.system.interop.parmed import _lj_params_from_potential

kcal_mol = unit.kilocalorie_per_mole
kcal_ang = kcal_mol / unit.angstrom ** 2
kcal_rad = kcal_mol / unit.radian ** 2

kj_mol = unit.kilojoule_per_mole
kj_nm = kj_mol / unit.nanometer ** 2
kj_rad = kj_mol / unit.radian ** 2


def to_openmm(openff_sys) -> openmm.System:
    """Convert an OpenFF System to a ParmEd Structure

    Parameters
    ----------
    openff_sys : openff.system.System
        An OpenFF System object

    Returns
    -------
    openmm_sys : openmm.System
        The corresponding OpenMM System object

    """

    openmm_sys = openmm.System()

    # OpenFF box stored implicitly as nm, and that happens to be what
    # OpenMM casts box vectors to if provided only an np.ndarray
    if openff_sys.box is not None:
        box = openff_sys.box.to(off_unit.nanometer).magnitude
        openmm_sys.setDefaultPeriodicBoxVectors(*box)

    # Add particles (both atoms and virtual sites) with appropriate masses
    for atom in openff_sys.topology.topology_particles:
        openmm_sys.addParticle(atom.atom.mass)

    _process_nonbonded_forces(openff_sys, openmm_sys)
    _process_proper_torsion_forces(openff_sys, openmm_sys)
    _process_improper_torsion_forces(openff_sys, openmm_sys)
    _process_angle_forces(openff_sys, openmm_sys)
    _process_bond_forces(openff_sys, openmm_sys)

    return openmm_sys


def _process_bond_forces(openff_sys, openmm_sys):
    """Process the Bonds section of an OpenFF System into a corresponding openmm.HarmonicBondForce"""
    harmonic_bond_force = openmm.HarmonicBondForce()
    openmm_sys.addForce(harmonic_bond_force)

    bond_handler = openff_sys.handlers["Bonds"]
    for bond, key in bond_handler.slot_map.items():
        indices = eval(bond)
        params = bond_handler.potentials[key].parameters
        k = params["k"].to(off_unit.Unit(str(kcal_ang))).magnitude * kcal_ang / kj_nm
        length = params["length"].to(off_unit.nanometer).magnitude

        harmonic_bond_force.addBond(
            particle1=indices[0],
            particle2=indices[1],
            length=length,
            k=k,
        )


def _process_angle_forces(openff_sys, openmm_sys):
    """Process the Angles section of an OpenFF System into a corresponding openmm.HarmonicAngleForce"""
    harmonic_angle_force = openmm.HarmonicAngleForce()
    openmm_sys.addForce(harmonic_angle_force)

    angle_handler = openff_sys.handlers["Angles"]
    for angle, key in angle_handler.slot_map.items():
        indices = eval(angle)
        params = angle_handler.potentials[key].parameters
        k = params["k"].to(off_unit.Unit(str(kcal_rad))).magnitude
        k = k * kcal_rad / kj_rad
        angle = params["angle"].to(off_unit.degree).magnitude
        angle = angle * unit.degree / unit.radian

        harmonic_angle_force.addAngle(
            particle1=indices[0],
            particle2=indices[1],
            particle3=indices[2],
            angle=angle,
            k=k,
        )


def _process_proper_torsion_forces(openff_sys, openmm_sys):
    """Process the Propers section of an OpenFF System into corresponding
    forces within an openmm.PeriodicTorsionForce"""
    torsion_force = openmm.PeriodicTorsionForce()
    openmm_sys.addForce(torsion_force)

    proper_torsion_handler = openff_sys.handlers["ProperTorsions"]

    for torsion_key, key in proper_torsion_handler.slot_map.items():
        torsion, idx = torsion_key.split("_")
        indices = eval(torsion)
        params = proper_torsion_handler.potentials[key].parameters

        k = params["k"].to(off_unit.Unit(str(kcal_mol))).magnitude * kcal_mol / kj_mol
        periodicity = int(params["periodicity"])
        phase = params["phase"].to(off_unit.degree).magnitude
        phase = phase * unit.degree / unit.radian
        idivf = int(params["idivf"])
        torsion_force.addTorsion(
            indices[0],
            indices[1],
            indices[2],
            indices[3],
            periodicity,
            phase,
            k / idivf,
        )


def _process_improper_torsion_forces(openff_sys, openmm_sys):
    """Process the Impropers section of an OpenFF System into corresponding
    forces within an openmm.PeriodicTorsionForce"""
    if "ImproperTorsions" not in openff_sys.handlers.keys():
        raise Exception

    for force in openmm_sys.getForces():
        if type(force) == openmm.PeriodicTorsionForce:
            torsion_force = force
            break
    else:
        # TODO: Support case of no propers but some impropers?
        raise Exception

    improper_torsion_handler = openff_sys.handlers["ImproperTorsions"]

    for torsion_key, key in improper_torsion_handler.slot_map.items():
        torsion, idx = torsion_key.split("_")
        indices = eval(torsion)
        params = improper_torsion_handler.potentials[key].parameters

        k = params["k"].to(off_unit.Unit(str(kcal_mol))).magnitude * kcal_mol / kj_mol
        periodicity = int(params["periodicity"])
        phase = params["phase"].to(off_unit.degree).magnitude
        phase = phase * unit.degree / unit.radian
        idivf = int(params["idivf"])

        other_atoms = [indices[0], indices[2], indices[3]]
        for p in [
            (other_atoms[i], other_atoms[j], other_atoms[k])
            for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        ]:
            torsion_force.addTorsion(
                indices[1],
                p[0],
                p[1],
                p[2],
                periodicity,
                phase,
                k / idivf,
            )


def _process_nonbonded_forces(openff_sys, openmm_sys):
    """Process the vdW and Electrostatics sections of an OpenFF System into a corresponding openmm.NonbondedForce"""
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

    if openff_sys.box is None:
        non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    else:
        non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
        non_bonded_force.setUseDispersionCorrection(True)
        non_bonded_force.setCutoffDistance(vdw_cutoff)

    for vdw_atom, vdw_smirks in vdw_handler.slot_map.items():
        atom_idx = eval(vdw_atom)[0]

        partial_charge = electrostatics_handler.charge_map[vdw_atom]
        partial_charge = (partial_charge / off_unit.elementary_charge).magnitude
        vdw_potential = vdw_handler.potentials[vdw_smirks]
        # these are floats, implicitly angstrom and kcal/mol
        sigma, epsilon = _lj_params_from_potential(vdw_potential)
        sigma = sigma * unit.angstrom / unit.nanometer
        epsilon = epsilon * unit.kilocalorie_per_mole / unit.kilojoule_per_mole

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

    non_bonded_force.createExceptionsFromBonds(
        bond_particle_indices,
        electrostatics_handler.scale_14,
        vdw_handler.scale_14,
    )

    non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
    non_bonded_force.setCutoffDistance(9.0 * unit.angstrom)
    non_bonded_force.setEwaldErrorTolerance(1.0e-4)

    # It's not clear why this needs to happen here, but it cannot be set above
    # and satisfy vdW/Electrostatics methods Cutoff and PME; see create_force
    # and postprocess_system methods in toolkit
    if openff_sys.box is None:
        non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
