from openff.units import unit as off_unit
from simtk import openmm, unit

from openff.system.exceptions import UnsupportedCutoffMethodError
from openff.system.interop.parmed import _lj_params_from_potential
from openff.system.utils import pint_to_simtk

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
        box = openff_sys.box.m_as(off_unit.nanometer)
        openmm_sys.setDefaultPeriodicBoxVectors(*box)

    # Add particles (both atoms and virtual sites) with appropriate masses
    for atom in openff_sys.topology.topology_particles:
        openmm_sys.addParticle(atom.atom.mass)

    _process_nonbonded_forces(openff_sys, openmm_sys)
    _process_torsion_forces(openff_sys, openmm_sys)
    _process_improper_torsion_forces(openff_sys, openmm_sys)
    _process_angle_forces(openff_sys, openmm_sys)
    _process_bond_forces(openff_sys, openmm_sys)
    _process_constraints(openff_sys, openmm_sys)
    return openmm_sys


def _process_constraints(openff_sys, openmm_sys):
    """Process the Constraints section of an OpenFF System into a corresponding constraints in the OpenMM System"""
    try:
        constraint_handler = openff_sys.handlers["Constraints"]
    except KeyError:
        return

    for top_key, pot_key in constraint_handler.slot_map.items():
        indices = top_key.atom_indices
        params = constraint_handler.constraints[pot_key].parameters
        distance = params["distance"]
        distance_omm = distance.m_as(off_unit.nanometer)

        openmm_sys.addConstraint(indices[0], indices[1], distance_omm)


def _process_bond_forces(openff_sys, openmm_sys):
    """Process the Bonds section of an OpenFF System into a corresponding openmm.HarmonicBondForce"""
    harmonic_bond_force = openmm.HarmonicBondForce()
    openmm_sys.addForce(harmonic_bond_force)

    try:
        bond_handler = openff_sys.handlers["Bonds"]
    except KeyError:
        return

    try:
        constraint_handler = openff_sys.handlers["Constraints"]
        has_constraint_handler = True
    except KeyError:
        has_constraint_handler = False

    for top_key, pot_key in bond_handler.slot_map.items():
        if has_constraint_handler:
            # If this bond show up in the constraints ...
            if top_key in constraint_handler.slot_map:
                # ... don't add it as an interacting bond
                continue
        indices = top_key.atom_indices
        params = bond_handler.potentials[pot_key].parameters
        k = params["k"].m_as(
            off_unit.kilojoule / off_unit.nanometer ** 2 / off_unit.mol
        )
        length = params["length"].m_as(off_unit.nanometer)

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

    try:
        angle_handler = openff_sys.handlers["Angles"]
    except KeyError:
        return

    for top_key, pot_key in angle_handler.slot_map.items():
        indices = top_key.atom_indices
        params = angle_handler.potentials[pot_key].parameters
        k = params["k"].m_as(off_unit.kilojoule / off_unit.rad / off_unit.mol)
        angle = params["angle"].m_as(off_unit.radian)

        harmonic_angle_force.addAngle(
            particle1=indices[0],
            particle2=indices[1],
            particle3=indices[2],
            angle=angle,
            k=k,
        )


def _process_torsion_forces(openff_sys, openmm_sys):
    if "ProperTorsions" in openff_sys.handlers:
        _process_proper_torsion_forces(openff_sys, openmm_sys)
    if "RBTorsions" in openff_sys.handlers:
        _process_rb_torsion_forces(openff_sys, openmm_sys)


def _process_proper_torsion_forces(openff_sys, openmm_sys):
    """Process the Propers section of an OpenFF System into corresponding
    forces within an openmm.PeriodicTorsionForce"""
    torsion_force = openmm.PeriodicTorsionForce()
    openmm_sys.addForce(torsion_force)

    proper_torsion_handler = openff_sys.handlers["ProperTorsions"]

    for top_key, pot_key in proper_torsion_handler.slot_map.items():
        indices = top_key.atom_indices
        params = proper_torsion_handler.potentials[pot_key].parameters

        k = params["k"].m_as(off_unit.kilojoule / off_unit.mol)
        periodicity = int(params["periodicity"])
        phase = params["phase"].m_as(off_unit.radian)
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


def _process_rb_torsion_forces(openff_sys, openmm_sys):
    """Process Ryckaert-Bellemans torsions"""
    rb_force = openmm.RBTorsionForce()
    openmm_sys.addForce(rb_force)

    rb_torsion_handler = openff_sys.handlers["RBTorsions"]

    for top_key, pot_key in rb_torsion_handler.slot_map.items():
        indices = top_key.atom_indices
        params = rb_torsion_handler.potentials[pot_key].parameters

        c0 = params["C0"].m_as(off_unit.kilojoule / off_unit.mol)
        c1 = params["C1"].m_as(off_unit.kilojoule / off_unit.mol)
        c2 = params["C2"].m_as(off_unit.kilojoule / off_unit.mol)
        c3 = params["C3"].m_as(off_unit.kilojoule / off_unit.mol)
        c4 = params["C4"].m_as(off_unit.kilojoule / off_unit.mol)
        c5 = params["C5"].m_as(off_unit.kilojoule / off_unit.mol)

        rb_force.addTorsion(
            indices[0],
            indices[1],
            indices[2],
            indices[3],
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
        )


def _process_improper_torsion_forces(openff_sys, openmm_sys):
    """Process the Impropers section of an OpenFF System into corresponding
    forces within an openmm.PeriodicTorsionForce"""
    if "ImproperTorsions" not in openff_sys.handlers.keys():
        return

    for force in openmm_sys.getForces():
        if type(force) == openmm.PeriodicTorsionForce:
            torsion_force = force
            break
    else:
        torsion_force = openmm.PeriodicTorsionForce()

    improper_torsion_handler = openff_sys.handlers["ImproperTorsions"]

    for top_key, pot_key in improper_torsion_handler.slot_map.items():
        indices = top_key.atom_indices
        params = improper_torsion_handler.potentials[pot_key].parameters

        k = params["k"].m_as(off_unit.kilojoule / off_unit.mol)
        periodicity = int(params["periodicity"])
        phase = params["phase"].m_as(off_unit.radian)
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

    if "vdW" in openff_sys.handlers:
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

        for top_key, pot_key in vdw_handler.slot_map.items():
            atom_idx = top_key.atom_indices[0]

            partial_charge = electrostatics_handler.charges[top_key]
            partial_charge = partial_charge.m_as(off_unit.elementary_charge)
            vdw_potential = vdw_handler.potentials[pot_key]
            # these are floats, implicitly angstrom and kcal/mol
            sigma, epsilon = _lj_params_from_potential(vdw_potential)
            sigma = sigma.m_as(off_unit.nanometer)
            epsilon = epsilon.m_as(off_unit.kilojoule / off_unit.mol)

            non_bonded_force.setParticleParameters(
                atom_idx, partial_charge, sigma, epsilon
            )

    elif "Buckingham-6" in openff_sys.handlers:
        buck_handler = openff_sys.handlers["Buckingham-6"]

        non_bonded_force = openmm.CustomNonbondedForce(
            "A * exp(-B * r) - C * r ^ -6; A = sqrt(A1 * A2); B = 2 / (1 / B1 + 1 / B2); C = sqrt(C1 * C2)"
        )
        non_bonded_force.addPerParticleParameter("A")
        non_bonded_force.addPerParticleParameter("B")
        non_bonded_force.addPerParticleParameter("C")
        openmm_sys.addForce(non_bonded_force)

        for _ in openff_sys.topology.topology_particles:
            non_bonded_force.addParticle([0.0, 0.0, 0.0])

        if openff_sys.box is None:
            non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        else:
            non_bonded_force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
            non_bonded_force.setCutoffDistance(buck_handler.cutoff * unit.angstrom)

        for top_key, pot_key in buck_handler.slot_map.items():
            atom_idx = top_key.atom_indices[0]

            # TODO: Add electrostatics
            params = buck_handler.potentials[pot_key].parameters
            a = pint_to_simtk(params["A"])
            b = pint_to_simtk(params["B"])
            c = pint_to_simtk(params["C"])
            non_bonded_force.setParticleParameters(atom_idx, [a, b, c])

        return

    # TODO: Figure out all of this post-processing with CustomNonbondedForce

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
