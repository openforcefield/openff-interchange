"""Runtime settings for MD simulations."""
from typing import TYPE_CHECKING, Literal, Optional

from openff.units import unit
from pydantic import Field

from openff.interchange.exceptions import (
    UnsupportedCutoffMethodError,
    UnsupportedExportError,
)
from openff.interchange.models import DefaultModel
from openff.interchange.types import FloatQuantity

if TYPE_CHECKING:
    from openff.interchange import Interchange

MDP_HEADER = """
nsteps                   = 0
nstenergy                = 1000
continuation             = yes
cutoff-scheme            = verlet

DispCorr                 = Ener
"""


class MDConfig(DefaultModel):
    """A partial superset of runtime configurations for MD engines."""

    periodic: bool = Field(
        True,
        description="Whether or not the system is periodic.",
    )
    constraints: Literal["none", "h-bonds", "all-bonds", "all-angles"] = Field(
        "none", description="The type of constraints to be used in the simulation."
    )
    vdw_method: Optional[Literal["cutoff", "pme", "no-cutoff"]] = Field(
        None, description="The method used to calculate the vdW interactions."
    )
    vdw_cutoff: Optional[FloatQuantity["angstrom"]] = Field(  # type: ignore
        None,
        description="The distance at which pairwise interactions are truncated",
    )
    mixing_rule: Optional[Literal["lorentz-berthelot", "geometric"]] = Field(
        None,
        description="The mixing rule (combination rule, combining rule) used in computing pairwise vdW interactions",
    )

    switching_function: Optional[bool] = Field(
        None,
        description="Whether or not to use a switching function for the vdw interactions",
    )
    switching_distance: Optional[FloatQuantity["angstrom"]] = Field(  # type: ignore
        None,
        description="The distance at which the switching function is applied",
    )
    coul_method: Optional[Literal["cutoff", "pme", "reaction-field"]] = Field(
        None,
        description="The method used to compute pairwise electrostatic interactions",
    )
    coul_cutoff: Optional[FloatQuantity["angstrom"]] = Field(  # type: ignore
        None,
        description="The distance at which electrostatic interactions are truncated or transformed.",
    )

    @classmethod
    def from_interchange(cls, interchange: "Interchange") -> "MDConfig":
        """Generate a MDConfig object from an Interchange object."""
        mdconfig = cls(
            periodic=interchange.box is not None,
            constraints=_infer_constraints(interchange),
        )
        if "vdW" in interchange.handlers:
            mdconfig.vdw_cutoff = interchange.handlers["vdW"].cutoff
            mdconfig.vdw_method = interchange.handlers["vdW"].method
            mdconfig.mixing_rule = interchange.handlers["vdW"].mixing_rule
            mdconfig.switching_function = (
                interchange.handlers["vdW"].switch_width is not None
            )
            mdconfig.switching_distance = (
                mdconfig.vdw_cutoff - interchange.handlers["vdW"].switch_width
            )

        if "Electrostatics" in interchange.handlers:
            mdconfig.coul_method = interchange.handlers["Electrostatics"].method
            mdconfig.coul_cutoff = interchange.handlers["Electrostatics"].cutoff

        return mdconfig

    def write_mdp_file(self, mdp_file: str = "auto_generated.mdp") -> None:
        with open(mdp_file, "w") as mdp:
            mdp.write(MDP_HEADER)

            if self.periodic:
                mdp.write("pbc = xyz\n")
            else:
                mdp.write("pbc = no\n")

            mdp.write(f"constraints = {self.constraints}\n")

            coul_cutoff = round(self.coul_cutoff.m_as(unit.nanometer), 4)
            if self.coul_method == "cutoff":
                mdp.write("coulombtype = Cut-off\n")
                mdp.write("coulomb-modifier = None\n")
                mdp.write(f"rcoulomb = {coul_cutoff}\n")
            elif self.coul_method == "pme":
                if not self.periodic:
                    raise UnsupportedCutoffMethodError(
                        "PME is not valid with a non-periodic system."
                    )
                mdp.write("coulombtype = PME\n")
                mdp.write(f"rcoulomb = {coul_cutoff}\n")
            elif self.coul_method == "reactionfield":
                mdp.write("coulombtype = Reaction-field\n")
                mdp.write(f"rcoulomb = {coul_cutoff}\n")
            else:
                raise UnsupportedExportError(
                    f"Electrostatics method {coul_method} not supported"
                )

            if self.vdw_method == "cutoff":
                mdp.write("vdwtype = cutoff\n")
            elif self.vdw_method == "pme":
                mdp.write("vdwtype = PME\n")
            else:
                raise UnsupportedExportError(f"vdW method {vdw_method} not supported")

            vdw_cutoff = round(self.vdw_cutoff.m_as(unit.nanometer), 4)
            mdp.write(f"rvdw = {vdw_cutoff}\n")

            if self.switching_function:
                mdp.write("vdw-modifier = Potential-switch\n")
                mdp.write(
                    f"rvdwswitch = {round(self.switching_distance.m_as(unit.nanometer), 4)}\n"
                )


class AmberMDConfig(MDConfig):
    """Runtime settings for AMBER simulations."""


class OpenMMMDConfig(MDConfig):
    """Runtime settings for OpenMM simulations."""


class LAMMPSMDConfig(MDConfig):
    """Runtime settings for LAMMPS simulations."""


def _infer_constraints(interchange: "Interchange") -> str:
    if "Constraints" not in interchange.handlers:
        return "none"
    elif "Bonds" not in interchange.handlers:
        return "none"
    else:
        num_constraints = len(interchange["Constraints"].slot_map)
        if num_constraints == 0:
            return "none"
        else:
            from openff.interchange.components.toolkit import _get_num_h_bonds

            num_h_bonds = _get_num_h_bonds(interchange.topology)

            num_bonds = len(interchange["Bonds"].slot_map)
            num_angles = len(interchange["Angles"].slot_map)

            if num_constraints == num_h_bonds:
                return "h-bonds"
            elif num_constraints == len(interchange["Bonds"].slot_map):
                return "all-bonds"
            elif num_constraints == (num_bonds + num_angles):
                return "all-angles"

            else:
                raise Exception("Generic failure while inferring constraints")
