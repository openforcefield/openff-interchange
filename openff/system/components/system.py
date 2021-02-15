from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from openff.toolkit.topology.topology import Topology
from pydantic import validator

from openff.system.components.potentials import PotentialHandler
from openff.system.interop.openmm import to_openmm
from openff.system.interop.parmed import to_parmed
from openff.system.types import ArrayQuantity, DefaultModel


class System(DefaultModel):
    """
    A molecular system object.

    .. warning :: This object is in an early and experimental state and unsuitable for production.**
    .. warning :: This API is experimental and subject to change.
    """

    handlers: Dict[str, PotentialHandler] = dict()
    topology: Optional[Topology] = None
    box: ArrayQuantity["nanometer"] = None  # type: ignore
    positions: ArrayQuantity["nanometer"] = None  # type: ignore

    @validator("box")
    def validate_box(cls, val):
        if val is None:
            return val
        if val.shape == (3, 3):
            return val
        elif val.shape == (3,):
            val = val * np.eye(3)
            return val
        else:
            raise ValueError  # InvalidBoxError

    def to_gro(self, file_path: Union[Path, str], writer="parmed"):
        """Export this system to a .gro file using ParmEd"""

        if self.positions is None:
            raise Exception

        # TODO: Enum-style class for handling writer arg?
        if writer == "parmed":
            from openff.system.interop.external import ParmEdWrapper

            ParmEdWrapper().to_file(self, file_path)

        elif writer == "internal":
            from openff.system.interop import internal

            internal.to_gro(self, file_path)

    def to_top(self, file_path: Union[Path, str], writer="parmed"):
        """Export this system to a .top file using ParmEd"""
        if writer == "parmed":
            from openff.system.interop.external import ParmEdWrapper

            ParmEdWrapper().to_file(self, file_path)

        elif writer == "internal":
            from openff.system.interop import internal

            internal.to_top(self, file_path)

    def to_openmm(self):
        """Export this sytem to an OpenMM System"""
        assert self._check_nonbonded_compatibility()
        return to_openmm(self)

    def to_parmed(self):
        """Export this sytem to a ParmEd Structure"""
        return to_parmed(self)

    def _get_nonbonded_methods(self):
        nonbonded_ = {
            "electrostatics_method": self.handlers["Electrostatics"].method,
            "vdw_method": self.handlers["vdW"].method,
            "periodic_topology": self.box is not None,
        }

        return nonbonded_

    def _check_nonbonded_compatibility(self):
        from openff.system.interop.compatibility.nonbonded import (
            check_nonbonded_compatibility,
        )

        nonbonded_ = self._get_nonbonded_methods()
        return check_nonbonded_compatibility(nonbonded_)
