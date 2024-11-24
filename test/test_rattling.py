import json
from pathlib import Path

import ase
import numpy
from numpy.random import Generator, PCG64
from numpy.testing import assert_allclose
import pytest

from rattling import calculate_random_displacements, get_phonon_modes, PhononModes


TEST_DATA = Path(__file__).parent / "data"


@pytest.fixture
def atoms() -> ase.Atoms:
    return ase.io.read(TEST_DATA / "graphene_441/supercell.extxyz")

def test_get_phonon_modes(atoms):
    force_constants = numpy.load(
        TEST_DATA / "graphene_441/force_constants.npy")
    phonon_modes = get_phonon_modes(force_constants, atoms.get_masses())

    with open(TEST_DATA / "graphene_441/ref_phonon_modes.json", "r") as fd:
        ref_data = json.load(fd)

    for key, value in ref_data.items():
        assert_allclose(getattr(phonon_modes, key), value)


def test_random_displacements(atoms):
    rng = Generator(PCG64(seed=1))

    with open(TEST_DATA / "graphene_441/ref_phonon_modes.json", "r") as fd:
        phonon_mode_dict = json.load(fd)
    phonon_modes = PhononModes(**{key: numpy.array(value)
                                  for key, value in phonon_mode_dict.items()})

    phonon_rattle = calculate_random_displacements(
        masses=atoms.get_masses(),
        modes=phonon_modes,
        rng=rng.random,
        indices=None)

    with open(TEST_DATA / "graphene_441/ref_phonon_rattle.json", "r") as fd:
        ref_data = json.load(fd)

    assert_allclose(phonon_rattle.displacements, ref_data["displacements"])
    assert_allclose(phonon_rattle.velocities, ref_data["velocities"])    
