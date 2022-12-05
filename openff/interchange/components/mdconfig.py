"""Runtime settings for MD simulations."""
from typing import TYPE_CHECKING, Literal

from openff.models.models import DefaultModel
from openff.models.types import FloatQuantity
from openff.units import unit
from pydantic import Field

from openff.interchange.constants import _PME
from openff.interchange.exceptions import (
    UnsupportedCutoffMethodError,
    UnsupportedExportError,
)

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
    vdw_method: str = Field(
        None, description="The method used to calculate the vdW interactions."
    )
    vdw_cutoff: FloatQuantity["angstrom"] = Field(
        None,
        description="The distance at which pairwise interactions are truncated",
    )
    mixing_rule: Literal["lorentz-berthelot", "geometric"] = Field(
        None,
        description="The mixing rule (combination rule, combining rule) used in computing pairwise vdW interactions",
    )

    switching_function: bool = Field(
        None,
        description="Whether or not to use a switching function for the vdw interactions",
    )
    switching_distance: FloatQuantity["angstrom"] = Field(
        None,
        description="The distance at which the switching function is applied",
    )
    coul_method: str = Field(
        None,
        description="The method used to compute pairwise electrostatic interactions",
    )
    coul_cutoff: FloatQuantity["angstrom"] = Field(
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
            vdw_handler = interchange["vdW"]
            mdconfig.vdw_cutoff = vdw_handler.cutoff
            mdconfig.vdw_method = vdw_handler.method
            mdconfig.mixing_rule = vdw_handler.mixing_rule

            if vdw_handler.switch_width is not None:
                if vdw_handler.switch_width.m == 0:
                    mdconfig.switching_function = False
                else:
                    mdconfig.switching_function = True
                    mdconfig.switching_distance = (
                        mdconfig.vdw_cutoff - vdw_handler.switch_width
                    )
            else:
                mdconfig.switching_function = False

        if "Electrostatics" in interchange.handlers:
            mdconfig.coul_method = interchange["Electrostatics"].periodic_potential
            mdconfig.coul_cutoff = interchange["Electrostatics"].cutoff

        return mdconfig

    def write_mdp_file(self, mdp_file: str = "auto_generated.mdp") -> None:
        """Write a GROMACS `.mdp` file for running single-point energies."""
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
            elif self.coul_method == _PME:
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
                    f"Electrostatics method {self.coul_method} not supported"
                )

            if self.vdw_method == "cutoff":
                mdp.write("vdwtype = cutoff\n")
            elif self.vdw_method == "pme":
                mdp.write("vdwtype = PME\n")
            else:
                raise UnsupportedExportError(
                    f"vdW method {self.vdw_method} not supported"
                )

            vdw_cutoff = round(self.vdw_cutoff.m_as(unit.nanometer), 4)
            mdp.write(f"rvdw = {vdw_cutoff}\n")

            if self.switching_function:
                mdp.write("vdw-modifier = Potential-switch\n")
                distance = round(self.switching_distance.m_as(unit.nanometer), 4)
                mdp.write(f"rvdwswitch = {distance}\n")

    def write_lammps_input(self, input_file: str = "run.in") -> None:
        """Write a LAMMPS input file for running single-point energies."""
        with open(input_file, "w") as lmp:
            lmp.write(
                "units real\n"
                "atom_style full\n"
                "\n"
                "dimension 3\nboundary p p p\n\n"
            )

            lmp.write("bond_style hybrid harmonic\n")
            lmp.write("angle_style hybrid harmonic\n")
            lmp.write("dihedral_style hybrid fourier\n")
            lmp.write("improper_style cvff\n")

            # TODO: LAMMPS puts this information in the "run" file. Should it live in MDConfig or not?
            scale_factors = {
                "vdW": {
                    "1-2": 0.0,
                    "1-3": 0.0,
                    "1-4": 0.5,
                    "1-5": 1,
                },
                "Electrostatics": {
                    "1-2": 0.0,
                    "1-3": 0.0,
                    "1-4": 0.8333333333,
                    "1-5": 1,
                },
            }
            lmp.write(
                "special_bonds lj "
                f"{scale_factors['vdW']['1-2']} "
                f"{scale_factors['vdW']['1-3']} "
                f"{scale_factors['vdW']['1-4']} "
                "coul "
                f"{scale_factors['Electrostatics']['1-2']} "
                f"{scale_factors['Electrostatics']['1-3']} "
                f"{scale_factors['Electrostatics']['1-4']} "
                "\n"
            )

            vdw_cutoff = round(self.vdw_cutoff.m_as(unit.angstrom), 4)
            coul_cutoff = round(self.coul_cutoff.m_as(unit.angstrom), 4)
            if self.coul_method == _PME:
                lmp.write(f"pair_style lj/cut/coul/long {vdw_cutoff} {coul_cutoff}\n")
            elif self.coul_method == "cutoff":
                lmp.write(f"pair_style lj/cut/coul/cut {vdw_cutoff} {coul_cutoff}\n")
            else:
                raise UnsupportedExportError(
                    f"Unsupported electrostatics method {self.coul_method}"
                )

            if self.mixing_rule == "lorentz-berthelot":
                lmp.write("pair_modify mix arithmetic tail yes\n\n")
            elif self.mixing_rule == "geometric":
                lmp.write("pair_modify mix geometric tail yes\n\n")
            else:
                raise UnsupportedExportError(
                    f"Mixing rule {self.mixing_rule} not supported"
                )
            lmp.write("read_data out.lmp\n\n")
            lmp.write(
                "thermo_style custom ebond eangle edihed eimp epair evdwl ecoul elong etail pe\n\n"
            )

            if self.coul_method == "pme":
                # Note: LAMMPS will error out if using kspace on something with all zero charges,
                # so this may not work if all partial charges are zero
                lmp.write("kspace_style pppm 1e-6\n")

            lmp.write("run 0\n")

    def write_sander_input_file(self, input_file: str = "run.in") -> None:
        """Write a Sander input file for running single-point energies."""
        with open(input_file, "w") as sander:
            sander.write("single-point energy\n&cntrl\nimin=1,\nmaxcyc=0,\nntb=1,\n")

            if self.switching_function is not None:
                distance = round(self.switching_distance.m_as(unit.angstrom), 4)
                sander.write(f"fswitch={distance},\n")

            if self.constraints in ["none", None]:
                sander.write("ntc=1,\nntf=1,\n")
            elif self.constraints == "h-bonds":
                sander.write("ntc=2,\nntf=2,\n")
            # TODO: Is there a clear analog to GROMACS's all-bonds?
            elif self.constraints == "angles":
                raise UnsupportedExportError(
                    "Unclear how to constrain angles with sander"
                )

            if self.vdw_method == "cutoff":
                vdw_cutoff = round(self.vdw_cutoff.m_as(unit.angstrom), 4)
                sander.write(f"cut={vdw_cutoff},\n")
            else:
                raise UnsupportedExportError(
                    f"vdW method {self.vdw_method} not supported"
                )

            if self.coul_method == "pme":
                sander.write("/\n&ewald\norder=4\nskinnb=1.0\n/")

            sander.write("/\n")


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
                import warnings

                warnings.warn(
                    "Ambiguous failure while processing constraints. Constraining h-bonds as a stopgap."
                )

                return "h-bonds"
