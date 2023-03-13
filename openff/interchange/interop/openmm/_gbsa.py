from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import openmm

    from openff.interchange import Interchange


def _process_gbsa(
    interchange: "Interchange",
    system: "openmm.System",
):
    import openmm.app
    import openmm.unit
    from openff.units import unit

    existing_forces = [
        force
        for force in system.getForces()
        if isinstance(force, (openmm.NonbondedForce, openmm.CustomNonbondedForce))
    ]

    assert (
        len(existing_forces) == 1
    ), "GBSA implementation assumes only one NonbondedForce is present."

    non_bonded_force = existing_forces[0]

    if non_bonded_force.getNonbondedMethod() != openmm.NonbondedForce.NoCutoff:
        amber_cutoff = None
    else:
        amber_cutoff = non_bonded_force.getCutoffDistance()

    try:
        collection = interchange.collections["GBSA"]
    except KeyError:
        return

    if collection.gb_model == "OBC2":
        force = openmm.GBSAOBCForce()
    elif collection.gb_model in ["OBC1", "HCT"]:
        if collection.gb_model == "HCT":
            force_type = openmm.app.internal.customgbforces.GBSAHCTForce
        elif collection.gb_model == "OBC1":
            force_type = openmm.app.internal.customgbforces.GBSAOBC1Force

        force = force_type(
            solventDielectric=collection.solvent_dielectric,
            soluteDielectric=collection.solute_dielectric,
            SA=collection.sa_model,
            cutoff=amber_cutoff,
            kappa=0,
        )

    system.addForce(force)

    if amber_cutoff is not None:
        force.setCutoffDistance(amber_cutoff)

    if non_bonded_force.usesPeriodicBoundaryConditions():
        force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    else:
        force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

    if collection.gb_model == "OBC2":
        force.setSolventDielectric(collection.solvent_dielectric)
        force.setSoluteDielectric(collection.solute_dielectric)
        force.setSurfaceAreaEnergy(
            collection.surface_area_penalty if collection.sa_model is not None else 0,
        )

    for topology_key, potential_key in collection.key_map.items():
        charge, *_ = non_bonded_force.getParticleParameters(
            topology_key.atom_indices[0],
        )
        _parameters = collection.potentials[potential_key].parameters

        parameters = [
            charge,
            _parameters["radius"].m_as(unit.nanometer),
            _parameters["scale"],
        ]

        if collection.gb_model == "OBC2":
            force.addParticle(*parameters)
        else:
            force.addParticle(parameters)

    if collection.gb_model != "OBC2":
        force.finalize()
