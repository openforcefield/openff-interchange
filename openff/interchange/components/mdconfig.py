"""Runtime settings for MD simulations."""

import warnings
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from openff.toolkit import Quantity
from pydantic import Field

from openff.interchange._annotations import _DistanceQuantity
from openff.interchange.components.toolkit import _get_num_h_bonds
from openff.interchange.constants import _PME
from openff.interchange.exceptions import (
    NonperiodicNoCutoffNotSupportedError,
    UnsupportedCutoffMethodError,
    UnsupportedExportError,
)
from openff.interchange.pydantic import _BaseModel
from openff.interchange.warnings import SwitchingFunctionNotImplementedWarning

if TYPE_CHECKING:
    from openff.interchange import Interchange


MDP_HEADER = """\
; Single point energy
nsteps                   = 0      ; Perform no MD steps
nstenergy                = 1000   ; Write energies to .EDR file
continuation             = yes    ; Do not apply constraints at start
cutoff-scheme            = verlet ; Use a Verlet pair list for cutoffs

; Force field configuration
disp-corr                = Ener
"""


class MDConfig(_BaseModel):
    """A partial superset of runtime configurations for MD engines."""

    periodic: bool = Field(
        True,
        description="Whether or not the system is periodic.",
    )
    constraints: str = Field(
        "none",
        description="The type of constraints to be used in the simulation.",
    )
    vdw_method: Literal["cutoff", "pme", "no-cutoff"] = Field(
        "cutoff",
        description="The method used to calculate the vdW interactions.",
    )
    vdw_cutoff: _DistanceQuantity = Field(
        Quantity(9.0, "angstrom"),
        description="The distance at which pairwise interactions are truncated",
    )
    mixing_rule: str = Field(
        "lorentz-berthelot",
        description="The mixing rule (combination rule, combining rule) used in computing pairwise vdW interactions",
    )

    switching_function: bool = Field(
        False,
        description="Whether or not to use a switching function for the vdw interactions",
    )
    switching_distance: _DistanceQuantity = Field(
        Quantity(0.0, "angstrom"),
        description="The distance at which the switching function is applied",
    )
    coul_method: str = Field(
        "",
        description="The method used to compute pairwise electrostatic interactions",
    )
    coul_cutoff: _DistanceQuantity = Field(
        Quantity(9.0, "angstrom"),
        description=(
            "The distance at which electrostatic interactions are truncated or transition from short- to long-range."
        ),
    )

    @classmethod
    def from_interchange(cls, interchange: "Interchange") -> "MDConfig":
        """Generate a MDConfig object from an Interchange object."""
        mdconfig = cls(
            periodic=interchange.box is not None,
            constraints=_infer_constraints(interchange),
        )
        if "vdW" in interchange.collections:
            vdw_collection = interchange["vdW"]

            if interchange.box is None:
                mdconfig.vdw_method = vdw_collection.nonperiodic_method
            else:
                mdconfig.vdw_method = vdw_collection.periodic_method
                mdconfig.vdw_cutoff = vdw_collection.cutoff

            mdconfig.mixing_rule = vdw_collection.mixing_rule

            if vdw_collection.switch_width is not None:
                if vdw_collection.switch_width.m == 0:
                    mdconfig.switching_function = False
                else:
                    mdconfig.switching_function = True
                    mdconfig.switching_distance = mdconfig.vdw_cutoff - vdw_collection.switch_width
            else:
                mdconfig.switching_function = False

        if "Electrostatics" in interchange.collections:
            mdconfig.coul_method = getattr(
                interchange["Electrostatics"],
                "periodic_potential" if mdconfig.periodic else "nonperiodic_potential",
            )
            mdconfig.coul_cutoff = interchange["Electrostatics"].cutoff

        return mdconfig

    def apply(self, interchange: "Interchange"):
        """Attempt to apply these settings to an Interchange object."""
        if self.periodic:
            if interchange.box is None:
                interchange.box = Quantity([10, 10, 10], "nanometer")
        else:
            interchange.box = None

        if "vdW" in interchange.collections:
            vdw_collection = interchange["vdW"]

            if interchange.box is None:
                vdw_collection.nonperiodic_method = self.vdw_method
            else:
                vdw_collection.periodic_method = self.vdw_method

            vdw_collection.cutoff = self.vdw_cutoff
            vdw_collection.mixing_rule = self.mixing_rule

            if self.switching_function:
                vdw_collection.switch_width = self.vdw_cutoff - self.switching_distance
            else:
                vdw_collection.switch_width = Quantity(0.0, "angstrom")

        if "Electrostatics" in interchange.collections:
            electrostatics = interchange["Electrostatics"]
            if self.coul_method.lower() == "pme":
                electrostatics.periodic_potential = _PME  # type: ignore[assignment]
            else:
                electrostatics.periodic_potential = self.coul_method  # type: ignore[assignment]
            electrostatics.cutoff = self.coul_cutoff

    def write_mdp_file(self, mdp_file: str = "auto_generated.mdp") -> None:
        """
        Write a GROMACS `.mdp` file for running single-point energies.

        Parameters
        ----------
        mdp_file
            The name of the file to write the input to, defaults to `auto_generated.mdp`.
        """
        # Construct the MDP file in memory so nothing is written to disk if an
        # error is encountered.
        with StringIO() as mdp:
            mdp.write(MDP_HEADER)

            if self.periodic:
                mdp.write("pbc                      = xyz\n")
            else:
                mdp.write("pbc                      = no\n")

            mdp.write(f"constraints              = {self.constraints}\n")

            coul_cutoff = round(self.coul_cutoff.m_as("nanometer"), 4)

            if self.coul_method == "cutoff":
                mdp.write("coulombtype              = Cut-off\n")
                mdp.write("coulomb-modifier         = None\n")
                mdp.write(f"rcoulomb                 = {coul_cutoff}\n")
            elif self.coul_method in (_PME, "PME", "pme"):
                if not self.periodic:
                    raise UnsupportedCutoffMethodError(
                        "PME is not valid with a non-periodic system.",
                    )
                mdp.write("coulombtype              = PME\n")
                mdp.write(f"rcoulomb                 = {coul_cutoff}\n")
                mdp.write("coulomb-modifier         = None\n")
                mdp.write("fourier-spacing          = 0.12\n")
                # TODO: Wire this through like `ewald_tolerance` in `to_openmm`
                mdp.write("ewald-rtol               = 1e-4\n")
            elif self.coul_method == "reactionfield":
                mdp.write("coulombtype              = Reaction-field\n")
                mdp.write(f"rcoulomb                 = {coul_cutoff}\n")
            else:
                raise UnsupportedExportError(
                    f"Electrostatics method {self.coul_method} not supported",
                )

            if self.vdw_method == "cutoff":
                mdp.write("vdwtype                  = cutoff\n")
            elif self.vdw_method in ("Ewald3D", "pme", "PME", _PME):
                mdp.write("vdwtype                  = PME\n")
                # TODO: Wire this through like `ewald_tolerance` in `to_openmm`
                # TODO: Should this match electrostatics PME tolerance?
                mdp.write("ewald-rtol-lj            = 1e-4\n")
                mdp.write("lj-pme-comb-rule         = geometric\n")
            else:
                raise UnsupportedExportError(
                    f"vdW method {self.vdw_method} not supported",
                )

            vdw_cutoff = round(self.vdw_cutoff.m_as("nanometer"), 4)
            mdp.write(f"rvdw                     = {vdw_cutoff}\n")

            if self.switching_function and self.vdw_method == "cutoff":
                mdp.write("vdw-modifier             = Potential-switch\n")
                distance = round(self.switching_distance.m_as("nanometer"), 4)
                mdp.write(f"rvdw-switch              = {distance}\n")
            else:
                mdp.write("vdw-modifier             = None\n")
                mdp.write("rvdwswitch               = 0\n")

            # No errors, we can now write to disk!
            Path(mdp_file).write_text(mdp.getvalue())

    def write_lammps_input(
        self,
        interchange: "Interchange",
        input_file: str = "run.in",
        data_file: str = "out.lmp",
    ) -> None:
        """
        Write a LAMMPS input file for running single-point energies.

        Parameters
        ----------
        interchange
            The `Interchange` used when preparing this LAMMPS input.
        input_file
            The name of the file to write the input to, defaults to `run.in`.
        data_file
            The name of the file to write the data to, defaults to `out.lmp`.
        """
        # TODO: Get constrained angles
        # TODO: Process rigid water

        def _get_coeffs_of_constrained_bonds_and_angles(
            interchange: "Interchange",
        ) -> tuple[set[int], set[int]]:
            """
            Get coefficients of bonds and angles that appear to be constrained.

            Refactor this when LAMMPS export uses a dedicated class.

            * Coefficients are matched by stored SMIRKS
            * Coefficients are ints associated with Bond Coeffs / Angle Coeffs section
            * Coefficients are zero-indexed
            """
            constraint_styles = {key.associated_handler for key in interchange["Constraints"].potentials}

            if len(constraint_styles.difference({"Bonds", "Angles"})) > 0:
                raise NotImplementedError(
                    "Found unsupported constraints case in LAMMPS input writer.",
                )

            constrained_bond_smirks = {
                key.id for key in interchange["Constraints"].potentials if key.associated_handler == "Bonds"
            }

            constrained_angle_smirks = {
                key.id for key in interchange["Constraints"].potentials if key.associated_handler == "Angles"
            }

            return (
                {
                    key
                    for key, val in dict(
                        enumerate(interchange["Bonds"].potentials),
                    ).items()
                    if val.id in constrained_bond_smirks
                },
                {
                    key
                    for key, val in dict(
                        enumerate(interchange["Angles"].potentials),
                    ).items()
                    if val.id in constrained_angle_smirks
                },
            )

        # zero-indexed here
        (
            constrained_bond_coeffs,
            constrained_angle_coeffs,
        ) = _get_coeffs_of_constrained_bonds_and_angles(interchange)

        # Construct the input file in memory so nothing is written to disk if an
        # error is encountered.
        with StringIO() as lmp:
            if self.switching_function is not None:
                if self.switching_distance.m > 0.0:
                    warnings.warn(
                        f"A switching distance {self.switching_distance} was specified by the "
                        "force field, but LAMMPS may not implement a switching function as "
                        "specified by SMIRNOFF. Using a hard cut-off instead. Non-bonded "
                        "interactions will be affected.",
                        SwitchingFunctionNotImplementedWarning,
                    )

            lmp.write(
                "units real\natom_style full\n\ndimension 3\nboundary p p p\n\n",
            )

            if len(interchange["Bonds"].key_map) > 0:
                lmp.write("bond_style harmonic\n")

            if len(interchange["Angles"].key_map) > 0:
                lmp.write("angle_style harmonic\n")

            try:
                if len(interchange["ProperTorsions"].key_map) > 0:
                    lmp.write("dihedral_style fourier\n")
            except LookupError:
                # no torsions here
                pass

            try:
                if len(interchange["ImproperTorsions"].key_map) > 0:
                    lmp.write("improper_style cvff\n")
            except LookupError:
                # no impropers here
                pass

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
                "\n",
            )

            vdw_cutoff = round(self.vdw_cutoff.m_as("angstrom"), 4)
            coul_cutoff = round(self.coul_cutoff.m_as("angstrom"), 4)

            if self.coul_method == _PME:
                lmp.write(f"pair_style lj/cut/coul/long {vdw_cutoff} {coul_cutoff}\n")
            elif self.coul_method == "cutoff":
                lmp.write(f"pair_style lj/cut/coul/cut {vdw_cutoff} {coul_cutoff}\n")
            else:
                raise UnsupportedExportError(
                    f"Unsupported electrostatics method {self.coul_method}",
                )

            if self.mixing_rule == "lorentz-berthelot":
                lmp.write("pair_modify mix arithmetic tail yes\n\n")
            elif self.mixing_rule == "geometric":
                lmp.write("pair_modify mix geometric tail yes\n\n")
            else:
                raise UnsupportedExportError(
                    f"Mixing rule {self.mixing_rule} not supported",
                )

            lmp.write(f"read_data {data_file}\n\n")
            lmp.write(
                "thermo_style custom ebond eangle edihed eimp epair evdwl ecoul elong etail pe\n\n",
            )

            if len(constrained_bond_coeffs.union(constrained_angle_coeffs)) > 0:
                # https://docs.lammps.org/fix_shake.html
                # TODO: Apply fix to just a group (sub-group)?
                lmp.write(
                    "fix 100 all shake 0.0001 20 10 ",
                )

                if constrained_bond_coeffs:
                    lmp.write(
                        f"b {' '.join([str(val + 1) for val in constrained_bond_coeffs])}",
                    )

                if constrained_angle_coeffs:
                    lmp.write(
                        f"a {' '.join([str(val + 1) for val in constrained_angle_coeffs])}",
                    )

                lmp.write("\n")

            if self.coul_method == _PME:
                # Note: LAMMPS will error out if using kspace on something with all zero charges,
                # so this may not work if all partial charges are zero
                lmp.write("kspace_style pppm 1e-4\n")

            lmp.write("run 0\n")

            # No errors, safe to write to disk!
            Path(input_file).write_text(lmp.getvalue())

    def write_sander_input_file(
        self,
        input_file: str = "run.in",
    ) -> None:
        """
        Write a Sander input file for running single-point energies.

        Parameters
        ----------
        input_file
            The name of the file to write the input to, defaults to `run.in`.

        """
        # Construct the file in memory so nothing is written to disk if an
        # error is encountered.
        with StringIO() as sander:
            sander.write("single-point energy\n&cntrl\nimin=1,\nmaxcyc=0,\nntb=1,\n")

            if self.switching_function is not None:
                if self.switching_distance.m > 0.0:
                    warnings.warn(
                        f"A switching distance {self.switching_distance} was specified by the "
                        "force field, but Amber does not implement a switching function. Using a "
                        "hard cut-off instead. Non-bonded interactions will be affected.",
                        SwitchingFunctionNotImplementedWarning,
                    )

                # Whether this is stored as zero or positive distance, pass a
                # negative value to ensure it's turned off.
                sander.write(f"fswitch={-1.0},\n")

            if self.constraints in ["none", None]:
                sander.write("ntc=1,\nntf=1,\n")
            # TODO: This is an approximation, but most of the time these will be set to 2
            #       Amber cannot ignore H-O-H angle energy without ignoring all H-X-X angles,
            #       See 21.7.1. in Amber22 manual
            elif self.constraints in ("h-bonds", "all-bonds", "all-angles"):
                sander.write(
                    "ntc=1,\n"  # do NOT perform shake, since it will modify positions, but ...
                    "ntf=2,\n",  # ... ignore interactions of bonds including hydrogen atoms
                )
            # TODO: Cover other cases, though hard to reach with mainline OpenFF force fields
            else:
                raise UnsupportedExportError(
                    f"Unclear how to apply {self.constraints} with sander",
                )

            if self.vdw_method == "cutoff":
                vdw_cutoff = round(self.vdw_cutoff.m_as("angstrom"), 4)
                sander.write(f"cut={vdw_cutoff},\n")
            else:
                raise NonperiodicNoCutoffNotSupportedError(
                    f"vdW method {self.vdw_method} not supported",
                )

            if self.coul_method == _PME:
                sander.write("/\n&ewald\norder=4\nskinnb=1.0\n/")

            sander.write("/\n")

            # No errors, safe to write to disk!
            Path(input_file).write_text(sander.getvalue())


def _infer_constraints(interchange: "Interchange") -> str:
    if "Constraints" not in interchange.collections:
        return "none"
    elif "Bonds" not in interchange.collections:
        return "none"
    else:
        num_constraints = len(interchange["Constraints"].key_map)
        if num_constraints == 0:
            return "none"
        else:
            num_h_bonds = _get_num_h_bonds(interchange.topology)

            num_bonds = len(interchange["Bonds"].key_map)
            num_angles = len(interchange["Angles"].key_map)

            if num_constraints == num_h_bonds:
                return "h-bonds"
            elif num_constraints == num_bonds:
                return "all-bonds"
            elif num_constraints == (num_bonds + num_angles):
                return "all-angles"

            else:
                # TODO: Rigid waters may not have bond and angle parameters, but still have 3 constraints
                #       per molecule. There should be a better way to process these, but it's non-trivial
                #       to detect water molecules in a performance, scalable way without false positives.
                warnings.warn(
                    "Ambiguous failure while processing constraints. Constraining h-bonds as a stopgap.",
                )

                return "h-bonds"


def get_smirnoff_defaults(periodic: bool = False) -> MDConfig:
    """
    Return an `MDConfig` object that matches settings used in SMIRNOFF force fields (through Sage).

    Parameters
    ----------
    periodic
        Whether to use periodic boundary conditions.

    Returns
    -------
    config
        An `MDConfig` object with settings that match those used in SMIRNOFF force fields.

    """
    return MDConfig(
        periodic=periodic,
        constraints="h-bonds",
        vdw_method="cutoff",
        vdw_cutoff=Quantity(0.9, "nanometer"),
        mixing_rule="lorentz-berthelot",
        switching_function=True,
        switching_distance=Quantity(0.8, "nanometer"),
        coul_method="PME" if periodic else "Coulomb",
    )


def get_intermol_defaults(periodic: bool = False) -> MDConfig:
    """
    Return an `MDConfig` object that attempts to match settings used in InterMol tests.

    These settings are poor choices for production but can be useful for testing. See also
        - 10.1007/s10822-016-9977-1
        - https://github.com/shirtsgroup/InterMol/blob/master/intermol/tests/
            /gromacs/grompp_vacuum.mdp
            /lammps/unit_tests/atom_style-full_vacuum/atom_style-full-data_vacuum.input
            /amber/min_vacuum.in

    Parameters
    ----------
    periodic
        Whether to use periodic boundary conditions.

    Returns
    -------
    config
        An `MDConfig` object with settings that match those used in InterMol tests.

    """
    return MDConfig(
        periodic=periodic,
        constraints="none",
        vdw_method="cutoff",
        vdw_cutoff=Quantity(0.9, "nanometer"),
        mixing_rule="lorentz-berthelot",
        switching_function=False,
        switching_distance=Quantity(0.0, "angstrom"),
        coul_method="PME" if periodic else "cutoff",
        coul_cutoff=Quantity(0.9, "nanometer") if periodic else Quantity(2.0, "nanometer"),
    )
