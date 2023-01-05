from typing import List, Optional, Union

from openff.toolkit import ForceField, Molecule, Topology
from openff.units import Quantity
from packaging.version import Version

from openff.interchange import Interchange
from openff.interchange.components.smirnoff import (
    SMIRNOFFAngleHandler,
    SMIRNOFFBondHandler,
    SMIRNOFFConstraintHandler,
    SMIRNOFFElectrostaticsHandler,
    SMIRNOFFImproperTorsionHandler,
    SMIRNOFFProperTorsionHandler,
    SMIRNOFFvdWHandler,
    SMIRNOFFVirtualSiteHandler,
)
from openff.interchange.smirnoff._positions import _infer_positions


def _create_interchange(
    force_field: ForceField,
    topology: Union[Topology, List[Molecule]],
    box: Optional[Quantity] = None,
    positions: Optional[Quantity] = None,
    charge_from_molecules: Optional[List[Molecule]] = None,
    partial_bond_orders_from_molecules: Optional[List[Molecule]] = None,
    allow_nonintegral_charges: bool = False,
) -> Interchange:

    # Create empty Interchange
    interchange = Interchange()

    # Copy topology
    _topology = Interchange.validate_topology(topology)

    # Set positions
    interchange.positions = _infer_positions(positions, _topology)

    # Set box
    interchange.box = box

    # Create ParameterBlobs, one by one

    # Bonds
    if force_field["Bonds"].version == Version("0.3"):
        from openff.interchange.smirnoff._valence import _upconvert_bondhandler

        _upconvert_bondhandler(force_field["Bonds"])

    interchange.blobs.update(
        {
            "Bonds": SMIRNOFFBondHandler._from_toolkit(
                parameter_handler=force_field["Bonds"],
                topology=_topology,
                partial_bond_orders_from_molecules=partial_bond_orders_from_molecules,
            ),
        },
    )

    # Constraints
    interchange.blobs.update(
        {
            "Constraints": SMIRNOFFConstraintHandler._from_toolkit(
                parameter_handler=[
                    handler
                    for handler in [
                        force_field._parameter_handlers.get("Bonds", None),
                        force_field._parameter_handlers.get("Constraints", None),
                    ]
                    if handler is not None
                ],
                topology=_topology,
            ),
        },
    )

    # Angles
    interchange.blobs.update(
        {
            "Angles": SMIRNOFFAngleHandler._from_toolkit(
                parameter_handler=force_field["Angles"],
                topology=_topology,
            ),
        },
    )

    # Proper torsions
    interchange.blobs.update(
        {
            "ProperTorsions": SMIRNOFFProperTorsionHandler._from_toolkit(
                parameter_handler=force_field["ProperTorsions"],
                topology=_topology,
                partial_bond_orders_from_molecules=partial_bond_orders_from_molecules,
            ),
        },
    )

    # Improper torsions
    interchange.blobs.update(
        {
            "ImproperTorsions": SMIRNOFFImproperTorsionHandler._from_toolkit(
                parameter_handler=force_field["ImproperTorsions"],
                topology=_topology,
            ),
        },
    )

    # vdW
    interchange.blobs.update(
        {
            "vdW": SMIRNOFFvdWHandler._from_toolkit(
                parameter_handler=force_field["vdW"],
                topology=_topology,
            ),
        },
    )

    # Electrostatics
    interchange.blobs.update(
        {
            "Electrostatics": SMIRNOFFElectrostaticsHandler._from_toolkit(
                parameter_handler=[
                    handler
                    for handler in [
                        force_field._parameter_handlers.get(name, None)
                        for name in [
                            "Electrostatics",
                            "ChargeIncrementModel",
                            "ToolkitAM1BCC",
                            "LibraryCharges",
                        ]
                    ]
                    if handler is not None
                ],
                topology=_topology,
                charge_from_molecules=charge_from_molecules,
                allow_nonintegral_charges=allow_nonintegral_charges,
            ),
        },
    )

    # Virtual sites
    virtual_site_handler = SMIRNOFFVirtualSiteHandler()
    virtual_site_handler.exclusion_policy = force_field["VirtualSites"].exclusion_policy
    virtual_site_handler.store_matches(
        parameter_handler=force_field["VirtualSites"],
        topology=_topology,
    )
    virtual_site_handler.store_potentials(
        parameter_handler=force_field["VirtualSites"],
        vdw_handler=interchange["vdW"],
        electrostatics_handler=interchange["Electrostatics"],
    )
    interchange.blobs.update({"VirtualSites": virtual_site_handler})

    # Set topology
    interchange.topology = _topology

    return interchange
