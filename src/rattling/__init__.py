"""Library for phonon-rattling of atomistic structures"""

__version__ = "0.1"


from dataclasses import dataclass
from typing import Callable, NamedTuple

from ase import Atoms
import ase.units
import numpy as np


@dataclass
class PhononModes:
    """Collection of intermediate phonon data

    At this stage the force constants have been solved to a fixed set of
    frequencies and eigenvectors.

    Thermally-appropriate amplitudes are assigned; these have not yet been
    Gaussian-distributed or mass-weighted

    Args:
        frequencies:
            Angular frequency in ASE-friendly units (Å √(amu/eV))
        eigenvectors:
            Normalised eigenvectors
        amplitudes:
            Precalculated mode amplitudes at some temperature / quantum
            distribution

    Properties:
        energies:
            (read-only) energies in eV

    """

    frequencies: np.ndarray
    eigenvectors: np.ndarray
    amplitudes: np.ndarray

    @property
    def energies(self) -> np.ndarray:
        return self.frequencies * ase.units.s * ase.units._hbar * ase.units.J


class PhononRattle(NamedTuple):
    """Collection of atomic displacements and velocities"""

    displacements: np.ndarray
    velocities: np.ndarray


def get_phonon_modes(
    force_constants: np.ndarray,
    masses: np.ndarray,
    temperature_K: float = 300.0,
    *,
    quantum: bool = False,
    failfast: bool = True,
) -> PhononModes:
    r"""Return phonon mode data at a given temperature

    Args:

        force_constants: array of size 3N x 3N
            force constants (Hessian) of the system in eV/Å²

        masses: array of length N
            masses of the structure in amu

        temperature_K: float
            Temperature in Kelvin.

        quantum: bool
            True for Bose-Einstein distribution, False for Maxwell-Boltzmann
            (classical limit)

        failfast: bool
            True for sanity checking the phonon spectrum for negative
            frequencies at Gamma

    Returns:

        Eigenvectors, frequencies and thermal amplitudes

    Purpose:

        Calculate relevant data for thermal excitation of phonon modes.

    The end goal is to excite all phonon modes randomly so that each
    contributes, on average, equally to the given temperature.  Both potential
    energy and kinetic energy will be consistent with the phononic vibrations
    characteristic of the specified temperature.

    In other words the system will be equilibrated for an MD run at
    that temperature.

    force_constants should be the matrix as force constants, e.g.,
    as computed by the ase.phonons module.

    Let X_ai be the phonon modes indexed by atom and mode, w_i the
    phonon frequencies, and let 0 < Q_i <= 1 and 0 <= R_i < 1 be
    uniformly random numbers.  Then

    .. code-block:: none


                    1/2
       _     / k T \     ---  1  _             1/2
       R  += | --- |      >  --- X   (-2 ln Q )    cos (2 pi R )
        a    \  m  /     ---  w   ai         i                i
                 a        i    i


                    1/2
       _     / k T \     --- _            1/2
       v   = | --- |      >  X  (-2 ln Q )    sin (2 pi R )
        a    \  m  /     ---  ai        i                i
                 a        i

    Reference: [West, Estreicher; PRL 96, 22 (2006)]

    The amplitudes produced by this function are sqrt(kT) / w (or the quantum
    equivalent). These may be combined with the masses, eigenvectors and
    appropriate (quasi)random distributions to implement the equation above.

    """

    # Handle the temperature units
    temperature_eV = ase.units.kB * temperature_K

    # Build dynamical matrix
    rminv = (masses**-0.5).repeat(3)
    dynamical_matrix = force_constants * rminv[:, None] * rminv[None, :]

    # Solve eigenvalue problem to compute phonon spectrum and eigenvectors
    w2_s, X_is = np.linalg.eigh(dynamical_matrix)

    # Check for soft modes
    if failfast:
        zeros = w2_s[:3]
        worst_zero = np.abs(zeros).max()
        if worst_zero > 1e-3:
            msg = "Translational deviate from 0 significantly: "
            raise ValueError(msg + f"{w2_s[:3]}")

        w2min = w2_s[3:].min()
        if w2min < 0:
            msg = "Dynamical matrix has negative eigenvalues such as "
            raise ValueError(msg + f"{w2min}")

    # First three modes are translational so ignore:
    nw = len(w2_s) - 3
    n_atoms = len(masses)
    w_s = np.sqrt(w2_s[3:])
    X_acs = X_is[:, 3:].reshape(n_atoms, 3, nw)

    # Assign the amplitudes according to Bose-Einstein distribution
    # or high temperature (== classical) limit
    if quantum:
        hbar = ase.units._hbar * ase.units.J * ase.units.s
        A_s = np.sqrt(
            hbar * (2 * n_BE(temperature_eV, hbar * w_s) + 1) / (2 * w_s)
        )
    else:
        A_s = np.sqrt(temperature_eV) / w_s

    return PhononModes(w_s, X_acs, A_s)


def calculate_random_displacements(
    masses: np.ndarray,
    modes: PhononModes,
    rng: Callable[int, np.ndarray],
    indices: int | slice | np.ndarray | None = None,
) -> PhononRattle:
    """Use PhononModes to compute a random set of displacements and velocities

    For the mathematical formalism, see docstring of :fun:`get_phonon_modes`

    Args:
        masses: atomic masses corresponding to phonon modes
        modes: frequency, eigenvector and nominal amplitude data
        rng: random number generator accepting a target number N, and
            generating a uniform distribution of N values in half-open interval
            [0.0, 1.0)
        indices: If provided, limit the set of included phonon modes.

    """

    w_s, X_acs, A_s = modes.frequencies, modes.eigenvectors, modes.amplitudes

    # compute the gaussian distribution for the amplitudes
    # We need 0 < P <= 1.0 and not 0 0 <= P < 1.0 for the logarithm
    # to avoid (highly improbable) NaN.

    # Box Muller [en.wikipedia.org/wiki/Box–Muller_transform]:
    spread = np.sqrt(-2.0 * np.log(1.0 - rng(len(w_s))))

    # assign amplitudes and phases
    A_s = A_s * spread
    phi_s = 2.0 * np.pi * rng(len(w_s))

    # Zero out amplitude of ignored modes
    if indices is not None:
        mask = np.zeros_like(A_s, dtype=bool)
        mask[indices] = True
        A_s[np.logical_not(mask)] = 0.0

    # Assign velocities and displacements
    v_ac = (w_s * A_s * np.cos(phi_s) * X_acs).sum(axis=2)
    v_ac /= np.sqrt(masses)[:, None]

    d_ac = (A_s * np.sin(phi_s) * X_acs).sum(axis=2)
    d_ac /= np.sqrt(masses)[:, None]

    return PhononRattle(d_ac, v_ac)


def get_rattled_atoms(atoms: Atoms, rattle: PhononRattle) -> Atoms:
    """Get a new Atoms with displacements and velocities from PhononRattle"""

    new_atoms = atoms.copy()
    new_atoms.positions = atoms.positions + rattle.displacements
    new_atoms.set_velocities(rattle.velocities)

    return new_atoms


def n_BE(temp, omega):
    """Bose-Einstein distribution function.

    Args:
        temp: temperature converted to eV (*ase.units.kB)
        omega: sequence of frequencies converted to eV

    Returns:
        Value of Bose-Einstein distribution function for each energy

    """
    # define a ``zero'' temperature to avoid divisions by zero
    eps_temp = 1e-12

    omega = np.asarray(omega)

    # 0K limit
    if temp < eps_temp:
        n = np.zeros_like(omega)
    else:
        n = 1 / (np.exp(omega / (temp)) - 1)
    return n
