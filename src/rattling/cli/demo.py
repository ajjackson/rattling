"""Selective phonon-mode rattling"""

from importlib.resources import files
from os import remove
from pathlib import Path
from typing import Annotated, Optional

from ase import Atoms
import ase.io
from ase.phonons import Phonons
import ase.units
import numpy as np
import typer

from rattling import (
    calculate_random_displacements,
    get_phonon_modes,
    get_rattled_atoms,
    PhononModes,
)


def _get_selection(
    indices: str,
    mode_index_range: tuple[int, int] | None,
    frequency_range: tuple[float, float] | None,
    modes: PhononModes,
) -> slice | np.ndarray:
    """Handle user input for mode selection"""
    n_inputs = sum(
        (
            bool(indices),
            mode_index_range is not None,
            frequency_range is not None,
        )
    )
    if n_inputs == 0:
        return None
    if n_inputs > 1:
        raise ValueError(
            "Only one of --indices, --index-range or --frequency-range may be provided"
        )
    if indices:
        return np.array(indices.split(","), dtype=int)
    if mode_index_range is not None:
        return slice(mode_index_range[0], mode_index_range[1] + 1)

    # The remaining option is frequency range
    frequencies = modes.frequencies / ase.units.invcm
    mask = np.logical_and(
        frequencies >= min(frequency_range),
        frequencies < max(frequency_range),
    )
    return np.where(mask)


def _print_selection_info(
    modes: PhononModes, selection: np.ndarray | slice | None
) -> None:
    if selection is None:
        return

    selected_frequencies = modes.frequencies[selection]

    print("Selected modes with frequencies (cm⁻¹):")
    print(selected_frequencies / ase.units.invcm)


def main(
    structure: Path = files("rattling.data") / "sample.extxyz",
    fc_file: Path = files("rattling.data") / "sample_fc.npy",
    temperature: float = 300.0,
    quantum: bool = True,
    seed: int = 1,
    frames: int = 10,
    indices: Annotated[
        str,
        typer.Option(
            help="0-based indices of selected phonon modes, separated by commas (e.g. '0,1,2')"
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
    rng = np.random.default_rng(seed=seed)

    modes = get_phonon_modes(
        force_constants,
        atoms.get_masses(),
        temperature_K=temperature,
        quantum=quantum,
        failfast=True,
    )

    selection = _get_selection(indices, index_range, frequency_range, modes)
    _print_selection_info(modes, selection)

    if output_file.exists():
        remove(output_file)

    for _ in range(frames):
        phonon_rattle = calculate_random_displacements(
            atoms.get_masses(), modes, rng=rng.random, indices=selection
        )

        out_atoms = get_rattled_atoms(atoms, rattle=phonon_rattle)
        out_atoms.write(output_file, append=True)


def app() -> None:
    typer.run(main)


if __name__ == "__main__":
    app()
