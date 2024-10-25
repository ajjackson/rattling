"""Selective phonon-mode rattling"""

from importlib.resources import files
from os import remove
from pathlib import Path
from typing import Annotated, Optional

import ase.io
import ase.units
import numpy as np
import typer

from rattling.api import random_rattle_iter


def main(
    structure: Path = files("rattling.data") / "sample.extxyz",
    fc_file: Path = files("rattling.data") / "sample_fc.npy",
    temperature: float = 300.0,
    quantum: bool = True,
    seed: int = 1,
    num_configs: int = 10,
    indices: Annotated[
        str,
        typer.Option(
            help=(
                "0-based indices of selected phonon modes, "
                "separated by commas (e.g. '0,1,2')"
            )
        ),
    ] = "",
    index_range: Annotated[
        Optional[tuple[int, int]],
        typer.Option(
            help="0-based index range of selected phonon modes (inclusive)"
        ),
    ] = None,
    frequency_range: Annotated[
        Optional[tuple[float, float]],
        typer.Option(help="Energy range in cm⁻¹ of selected phonon modes"),
    ] = None,
    output_file: Path = "rattled.extxyz",
) -> None:
    atoms = ase.io.read(structure)
    force_constants = np.load(fc_file)

    if indices:
        indices = np.array(indices.split(","), dtype=int)
    else:  # empty string -> None
        indices = None

    if index_range is not None:
        index_range = slice(index_range[0], index_range[1] + 1)

    rattled_iter = random_rattle_iter(
        atoms=atoms,
        force_constants=force_constants,
        temperature=temperature,
        quantum=quantum,
        seed=seed,
        num_configs=num_configs,
        indices=indices,
        index_range=index_range,
        frequency_range=frequency_range,
    )

    if output_file.exists():
        remove(output_file)

    for out_atoms in rattled_iter:
        out_atoms.write(output_file, append=True)


def app() -> None:
    typer.run(main)


if __name__ == "__main__":
    app()
