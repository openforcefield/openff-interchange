from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
from openff.units import unit

if TYPE_CHECKING:
    from openff.interchange.components.interchange import Interchange


def to_prmtop(interchange: "Interchange", file_path: Union[Path, str]):
    """
    Write a .prmtop file. See http://ambermd.org/prmtop.pdf for details.

    """
    if isinstance(file_path, str):
        path = Path(file_path)
    if isinstance(file_path, Path):
        path = file_path

    with open(path, "w") as prmtop:
        import datetime

        now = datetime.datetime.now()
        prmtop.write(
            "%%VERSION  VERSION_STAMP = V0001.000  DATE = "
            f"{now.month:02d}/{now.day:02d}/{(now.year % 100):02}  "
            f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}"
        )

        from openff.interchange.interop.internal.gromacs import _build_typemap

        typemap = _build_typemap(interchange)  # noqa


def to_inpcrd(interchange: "Interchange", file_path: Union[Path, str]):
    """
    Write a .prmtop file. See https://ambermd.org/FileFormats.php#restart for details.

    """
    if isinstance(file_path, str):
        path = Path(file_path)
    if isinstance(file_path, Path):
        path = file_path

    n_atoms = interchange.topology.mdtop.n_atoms  # type: ignore
    time = 0.0

    with open(path, "w") as inpcrd:
        inpcrd.write(f"\n{n_atoms:5d}{time:15.7e}\n")

        fmt = "%12.7f%12.7f%12.7f" "%12.7f%12.7f%12.7f\n"
        coords = interchange.positions.m_as(unit.angstrom)
        reshaped = coords.reshape((-1, 6))
        for row in reshaped:
            inpcrd.write(fmt % (row[0], row[1], row[2], row[3], row[4], row[5]))

        box = interchange.box.to(unit.angstrom).magnitude
        if (box == np.diag(np.diagonal(box))).all():
            for i in range(3):
                inpcrd.write(f"{box[i, i]:12.7f}")
            for _ in range(3):
                inpcrd.write("  90.0000000")
        else:
            # TODO: Handle non-rectangular
            raise NotImplementedError

        inpcrd.write("\n")
