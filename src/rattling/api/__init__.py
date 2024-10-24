"""Selective phonon-mode rattling"""

from typing import Iterator

from ase import Atoms
import ase.io
import ase.units
import numpy as np

from rattling import (
    calculate_random_displacements,
    get_phonon_modes,
    get_rattled_atoms,
    PhononModes,
)


def _get_selection(
    indices: np.ndarray | None,
    mode_index_range: slice | None,
    frequency_range: tuple[float, float] | None,
    modes: PhononModes,
) -> slice | np.ndarray:
    """Handle user input for mode selection"""
    n_inputs = sum(
        (
            indices is not None,
            mode_index_range is not None,
            frequency_range is not None,
        )
    )
    if n_inputs == 0:
        return None
    if n_inputs > 1:
        raise ValueError(
            "Only one of --indices, --index-range or "
            "--frequency-range may be provided"
        )
    if indices is not None:
        return indices
    if mode_index_range is not None:
        return mode_index_range

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


def random_rattle(
    atoms: Atoms,
    force_constants: np.ndarray,
    temperature: float = 300.0,
    quantum: bool = True,
    seed: int = 1,
    num_configs: int = 10,
    indices: np.ndarray | None = None,
    index_range: slice | None = None,
    frequency_range: tuple[int, int] | None = None,
    verbose: bool = True,
) -> Iterator[Atoms]:
    """High-level Python interface for mode-selective rattling

    Users are encouraged to examine the source code and use the same core
    functions for alternate workflows.

    Yields:
        A new phonon-rattled Atoms

    """
    rng = np.random.default_rng(seed=seed)

    modes = get_phonon_modes(
        force_constants,
        atoms.get_masses(),
        temperature_K=temperature,
        quantum=quantum,
        failfast=True,
    )

    selection = _get_selection(indices, index_range, frequency_range, modes)
    if verbose:
        _print_selection_info(modes, selection)

    for _ in range(num_configs):
        phonon_rattle = calculate_random_displacements(
            atoms.get_masses(), modes, rng=rng.random, indices=selection
        )
        yield get_rattled_atoms(atoms, rattle=phonon_rattle)
