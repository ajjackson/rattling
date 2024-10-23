"""Selective phonon-mode rattling"""

# Take 1, based on ASE phonon rattling implementation

from pathlib import Path
from typing import NamedTuple

from asap3 import EMT
from ase import Atoms
import ase.io
from ase.phonons import Phonons
import ase.units
import numpy as np
import typer

def phonon_harmonics(
    force_constants,
    masses,
    temperature_K=None,
    *,
    rng=np.random.rand,
    quantum=False,
    failfast=True,
):
    r"""Return displacements and velocities that produce a given temperature.

    Parameters:

    force_constants: array of size 3N x 3N
        force constants (Hessian) of the system in eV/Å²

    masses: array of length N
        masses of the structure in amu

    temperature_K: float
        Temperature in Kelvin.

    rng: function
        Random number generator function, e.g., np.random.rand

    quantum: bool
        True for Bose-Einstein distribution, False for Maxwell-Boltzmann
        (classical limit)

    failfast: bool
        True for sanity checking the phonon spectrum for negative
        frequencies at Gamma

    Returns:

    Displacements, velocities generated from the eigenmodes,
    (optional: eigenvalues, eigenvectors of dynamical matrix)

    Purpose:

    Excite phonon modes to specified temperature.

    This excites all phonon modes randomly so that each contributes,
    on average, equally to the given temperature.  Both potential
    energy and kinetic energy will be consistent with the phononic
    vibrations characteristic of the specified temperature.

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
    """

    # Handle the temperature units
    temperature_eV = ase.units.kB * temperature_K

    # Build dynamical matrix
    rminv = (masses ** -0.5).repeat(3)
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
        A_s = np.sqrt(hbar * (2 * n_BE(temperature_eV, hbar * w_s) + 1) / (2 * w_s))
    else:
        A_s = np.sqrt(temperature_eV) / w_s

    # compute the gaussian distribution for the amplitudes
    # We need 0 < P <= 1.0 and not 0 0 <= P < 1.0 for the logarithm
    # to avoid (highly improbable) NaN.

    # Box Muller [en.wikipedia.org/wiki/Box–Muller_transform]:
    spread = np.sqrt(-2.0 * np.log(1.0 - rng(nw)))

    # assign amplitudes and phases
    A_s *= spread
    phi_s = 2.0 * np.pi * rng(nw)

    # Assign velocities and displacements
    v_ac = (w_s * A_s * np.cos(phi_s) * X_acs).sum(axis=2)
    v_ac /= np.sqrt(masses)[:, None]

    d_ac = (A_s * np.sin(phi_s) * X_acs).sum(axis=2)
    d_ac /= np.sqrt(masses)[:, None]

    return d_ac, v_ac


def PhononHarmonics(
    atoms,
    force_constants,
    temperature_K=None,
    *,
    rng=np.random,
    quantum=False,
    plus_minus=False,
    return_eigensolution=False,
    failfast=True,
):
    r"""Excite phonon modes to specified temperature.

    This will displace atomic positions and set the velocities so as
    to produce a random, phononically correct state with the requested
    temperature.

    Parameters:

    atoms: ase.atoms.Atoms() object
        Positions and momenta of this object are perturbed.

    force_constants: ndarray of size 3N x 3N
        Force constants for the the structure represented by atoms in eV/Å²

    temperature_K: float
        Temperature in Kelvin.

    rng: Random number generator
        RandomState or other random number generator, e.g., np.random.rand

    quantum: bool
        True for Bose-Einstein distribution, False for Maxwell-Boltzmann
        (classical limit)

    failfast: bool
        True for sanity checking the phonon spectrum for negative frequencies
        at Gamma.

    """

    # Receive displacements and velocities from phonon_harmonics()
    d_ac, v_ac = phonon_harmonics(
        force_constants=force_constants,
        masses=atoms.get_masses(),
        temperature_K=temperature_K,
        rng=rng.rand,
        plus_minus=plus_minus,
        quantum=quantum,
        failfast=failfast,
        return_eigensolution=False,
    )

    # Assign new positions (with displacements) and velocities
    atoms.positions += d_ac
    atoms.set_velocities(v_ac)


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

def _generate_sample_atoms(filename: Path) -> Atoms:
    import ase.build
    atoms = ase.build.bulk("Ag", cubic=True) * 2
    atoms.write(filename)
    return atoms

def _generate_sample_fc(atoms: Atoms,
                        filename: Path = Path("sample_fc.npy")):
    phonons = Phonons(atoms, calc=EMT(), supercell=(1, 1, 1))
    phonons.clean()  # Remove pre-existing
    phonons.run()
    phonons.read()
    force_constants = phonons.get_force_constant()[0]
    np.save(filename, force_constants, allow_pickle=False)
    return force_constants

def _get_atoms(structure: Path) -> Atoms:
    if structure.exists():
        return ase.io.read(structure)

    return _generate_sample_atoms(filename=structure)

def _get_force_constants(atoms: Atoms, fc_file: Path) -> np.ndarray:
    if fc_file.exists():
        return np.load(fc_file)

    return _generate_sample_fc(atoms, filename=fc_file)


def main(structure: Path = "sample.extxyz",
         fc_file: Path = "sample_fc.npy",
         temperature: float = 300.,
         quantum: bool = True,
         seed: int = 1) -> None:
    atoms = _get_atoms(structure)
    force_constants = _get_force_constants(atoms, fc_file)

    rng = np.random.default_rng(seed=seed)

    print(
        phonon_harmonics(
            force_constants,
            atoms.get_masses(),
            temperature_K=temperature,
            rng=rng.random,
            quantum=quantum,
            failfast=True,
        )
    )

if __name__ == '__main__':
    typer.run(main)
