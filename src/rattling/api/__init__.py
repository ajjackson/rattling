"""Selective phonon-mode rattling"""


from itertools import chain
from math import ceil
from typing import Iterator

from ase import Atoms
import ase.io
import ase.units
from joblib import Parallel, delayed
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
    wavenumbers = modes.energies / ase.units.invcm
    mask = np.logical_and(
        wavenumbers >= min(frequency_range),
        wavenumbers < max(frequency_range),
    )
    return np.where(mask)


def _print_selection_info(
    modes: PhononModes, selection: np.ndarray | slice | None
) -> None:
    if selection is None:
        return

    selected_frequencies = modes.energies[selection]

    print("Selected modes with frequencies:")
    print(" (cm⁻¹)    (meV)")
    for energy in selected_frequencies:
        print(
            "{cm:8.3f} {mev:8.4f}".format(
                cm=(energy / ase.units.invcm),
                mev=(energy * 1000),
            ),
        )

def random_rattle_iter(
    atoms: Atoms,
    force_constants: np.ndarray,
    temperature: float = 300.0,
    quantum: bool = True,
    seed: int = 1,
    rng: np.random.Generator | None = None,
    num_configs: int = 10,
    indices: np.ndarray | None = None,
    index_range: slice | None = None,
    frequency_range: tuple[float, float] | None = None,
    include_velocities: bool = True,
    verbose: bool = True,
) -> Iterator[Atoms]:
    """High-level Python interface for mode-selective rattling

    Users are encouraged to examine the source code and use the same core
    functions for alternate workflows.

    Args:
        atoms: Input structure
        force_constants: 3Nx3N force constants in eV Å⁻² a.m.u.⁻¹
        temperature: Rattling temperature in Kelvin
        quantum: If true, use Bose-Einstein distibution (including zero-point
            motions). Otherwise, use classical occupation.
        seed: Seed for random number generator. This can be used to ensure or
            avoid repeated results, as appropriate.
        rng: If provided, use this Generator instead of creating a new one with seed
        num_configs: Number of output structures.
        indices: integer array of indices for included vibrational modes.
            Only one of "indices", "index_range" and "frequency_range" may be
            used. If none are used, all modes will be included.
        index_range: A Python slice object selecting an index range
        frequency_range: Two values in wavenumber (1/cm), selecting range of
            included vibrational modes.
        include_velocities: calculate velocities (otherwise set to 0)
        verbose:
            Print additional information where relevant.

    Yields:
        A new phonon-rattled Atoms

    """
    if rng is None:
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
            atoms.get_masses(),
            modes,
            rng=rng.random,
            indices=selection,
            include_velocities=include_velocities,
        )
        yield get_rattled_atoms(atoms, rattle=phonon_rattle)

def _random_rattle_parallel_iter(
    atoms: Atoms,
    force_constants: np.ndarray,
    temperature: float = 300.0,
    quantum: bool = True,
    seed: int = 1,
    rng: np.random.Generator | None = None,
    num_configs: int = 10,
    indices: np.ndarray | None = None,
    index_range: slice | None = None,
    frequency_range: tuple[float, float] | None = None,
    include_velocities: bool = True,
    verbose: bool = True) -> list[Atoms]:
    """High-level Python interface for mode-selective rattling

    Users are encouraged to examine the source code and use the same core
    functions for alternate workflows.

    Args:
        atoms: Input structure
        force_constants: 3Nx3N force constants in eV Å⁻² a.m.u.⁻¹
        temperature: Rattling temperature in Kelvin
        quantum: If true, use Bose-Einstein distibution (including zero-point
            motions). Otherwise, use classical occupation.
        seed: Seed for random number generator. This can be used to ensure or
            avoid repeated results, as appropriate.
        rng: If provided, use this Generator instead of creating a new one with
            seed
        num_configs: Number of output structures.
        indices: integer array of indices for included vibrational modes.
            Only one of "indices", "index_range" and "frequency_range" may be
            used. If none are used, all modes will be included.
        index_range: A Python slice object selecting an index range
        frequency_range: Two values in wavenumber (1/cm), selecting range of
            included vibrational modes.
        include_velocities: calculate velocities (otherwise set to 0)
        verbose:
            Print additional information where relevant.

    Returns:
        Series of new Atoms with displacements and velocities.

    """

    if rng is None:
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

    def f(rng: int) -> Atoms:
        phonon_rattle = calculate_random_displacements(
            atoms.get_masses(),
            modes,
            rng=rng.random,
            indices=selection,
            include_velocities=include_velocities,
        )
        return get_rattled_atoms(atoms, rattle=phonon_rattle)

    delayed_inner = (delayed(f)(rng=rng.spawn(1)[0]) for _ in range(num_configs))
    batched_rattle = Parallel(n_jobs=-1,
                              return_as="generator_unordered")(delayed_inner)
    yield from batched_rattle


def random_rattle_binned_gaussian(
        atoms: Atoms,
        force_constants: np.ndarray,
        energy_bins: np.ndarray,
        fwhm: float,
        temperature: float = 300.0,
        quantum: bool = True,
        seed: int = 1,
        rng: np.random.Generator | None = None,
        num_configs: int = 10,
        include_velocities: bool = True,
        verbose: bool = True,
) -> Iterator[list[Atoms]]:
    """Rattle atoms over a series of energy bins

    Args:
        energy_bins: bin edges in eV
        fwhm: full-width half-maximum of Gaussian broadening profile

    """
    from functools import partial
    from rattling import calculate_binned_random_displacements
    from scipy.stats import norm

    if rng is None:
        rng = np.random.default_rng(seed=seed)

    modes = get_phonon_modes(
        force_constants,
        atoms.get_masses(),
        temperature_K=temperature,
        quantum=quantum,
        failfast=True,
    )

    def _fwhm_to_std(width: float) -> float:
        return width / np.sqrt(8 * np.log(2))

    # TODO more robust energy bin handling
    bin_centres = (energy_bins[1:] + energy_bins[:-1]) * 0.5
    bin_width = energy_bins[1] - energy_bins[0]


    def energy_distribution(energy, bin_centres) -> np.ndarray:
        return norm.pdf(bin_centres, loc=energy, scale=_fwhm_to_std(fwhm)) * bin_width

    disp_generator = calculate_binned_random_displacements(
        masses=atoms.get_masses(),
        modes=modes,
        rng=rng.random,
        bin_centres=bin_centres,
        energy_distribution=energy_distribution,
        num_configs=num_configs,
        include_velocities=include_velocities
    )

    # Yield a list of atoms for each energy bin
    from functools import partial
    _get_rattled_atoms = partial(get_rattled_atoms, atoms)

    for phonon_rattles in disp_generator:
        yield list(map(_get_rattled_atoms, phonon_rattles))

def random_rattle(
    atoms: Atoms,
    force_constants: np.ndarray,
    temperature: float = 300.0,
    quantum: bool = True,
    seed: int = 1,
    rng: np.random.Generator | None = None,
    num_configs: int = 10,
    indices: np.ndarray | None = None,
    index_range: slice | None = None,
    frequency_range: tuple[float, float] | None = None,
    include_velocities: bool = True,
    verbose: bool = True,
) -> list[Atoms]:
    """High-level Python interface for mode-selective rattling

    Users are encouraged to examine the source code and use the same core
    functions for alternate workflows.

    Args:
        atoms: Input structure
        force_constants: 3Nx3N force constants in eV Å⁻² a.m.u.⁻¹
        temperature: Rattling temperature in Kelvin
        quantum: If true, use Bose-Einstein distibution (including zero-point
            motions). Otherwise, use classical occupation.
        seed: Seed for random number generator. This can be used to ensure or
            avoid repeated results, as appropriate.
        rng: If provided, use this Generator instead of creating a new one with
            seed
        num_configs: Number of output structures.
        indices: integer array of indices for included vibrational modes.
            Only one of "indices", "index_range" and "frequency_range" may be
            used. If none are used, all modes will be included.
        index_range: A Python slice object selecting an index range
        frequency_range: Two values in wavenumber (1/cm), selecting range of
            included vibrational modes.
        include_velocities: calculate velocities (otherwise set to 0)
        verbose:
            Print additional information where relevant.

    Returns:
        Series of new Atoms with displacements and velocities.

    """
    return list(random_rattle_iter(
        atoms=atoms,
        force_constants=force_constants,
        temperature=temperature,
        quantum=quantum,
        seed=seed,
        rng=rng,
        num_configs=num_configs,
        indices=indices,
        index_range=index_range,
        frequency_range=frequency_range,
        include_velocities=include_velocities,
        verbose=verbose
    ))

def random_rattle_parallel(
    atoms: Atoms,
    force_constants: np.ndarray,
    temperature: float = 300.0,
    quantum: bool = True,
    seed: int = 1,
    rng: np.random.Generator | None = None,
    num_configs: int = 10,
    indices: np.ndarray | None = None,
    index_range: slice | None = None,
    frequency_range: tuple[float, float] | None = None,
    include_velocities: bool = True,
    verbose: bool = True) -> list[Atoms]:
    """High-level Python interface for mode-selective rattling

    Users are encouraged to examine the source code and use the same core
    functions for alternate workflows.

    This function uses joblib for "embarassingly parallel" distribution of
    displacements over processors. There is some overhead, so for calculations
    taking around 2 seconds or less it is recommended to use the serial
    version.

    Args:
        atoms: Input structure
        force_constants: 3Nx3N force constants in eV Å⁻² a.m.u.⁻¹
        temperature: Rattling temperature in Kelvin
        quantum: If true, use Bose-Einstein distibution (including zero-point
            motions). Otherwise, use classical occupation.
        seed: Seed for random number generator. This can be used to ensure or
            avoid repeated results, as appropriate.
        rng: If provided, use this Generator instead of creating a new one with
            seed
        num_configs: Number of output structures.
        indices: integer array of indices for included vibrational modes.
            Only one of "indices", "index_range" and "frequency_range" may be
            used. If none are used, all modes will be included.
        index_range: A Python slice object selecting an index range
        frequency_range: Two values in wavenumber (1/cm), selecting range of
            included vibrational modes.
        include_velocities: calculate velocities (otherwise set to 0)
        verbose:
            Print additional information where relevant.

    Returns:
        Series of new Atoms with displacements and velocities.

    """
    return(list(_random_rattle_parallel_iter(
        atoms=atoms,
        force_constants=force_constants,
        temperature=temperature,
        quantum=quantum,
        seed=seed,
        rng=rng,
        num_configs=num_configs,
        indices=indices,
        index_range=index_range,
        frequency_range=frequency_range,
        include_velocities=include_velocities,
        verbose=verbose        
    )))
