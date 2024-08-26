#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
import numpy as np
from pathlib import Path

import scine_utilities as utils
from scine_chemoton import _read_file_to_molecule, _get_arguments, _create_reactive_complexes

# Local application tests imports
from .resources import resources_root_path


def test_read_file_to_molecule():
    rr = resources_root_path()
    water_path = os.path.join(rr, "water.xyz")
    water = _read_file_to_molecule(Path(water_path))
    assert isinstance(water, utils.AtomCollection)
    assert water.size() == 3

    ethene_path = os.path.join(rr, "ethene.mol")
    ethene = _read_file_to_molecule(Path(ethene_path))
    assert isinstance(ethene, utils.AtomCollection)
    assert ethene.size() == 6


def test_parsing_arguments():
    rr = resources_root_path()
    water_path = os.path.join(rr, "water.xyz")
    arguments = _get_arguments(["-m1", water_path, "-m2", water_path,
                                "-l", "0", "1",
                                "-r", "0", "1",
                                "-ma"])
    assert arguments.molecule1 == Path(water_path)
    assert arguments.molecule2 == Path(water_path)
    assert arguments.lhs == [0, 1]
    assert arguments.rhs == [0, 1]
    assert arguments.n_rotamers == 2
    assert arguments.multiple_attack_points
    assert not arguments.verbose


def test_creating_reactive_complexes():
    rr = resources_root_path()
    water_path = os.path.join(rr, "water.xyz")
    water = _read_file_to_molecule(Path(water_path))
    _create_reactive_complexes(water, water, [0, 1], [0, 1], n_rotamers=1,
                               multiple_attack_points=False, verbose=False)
    rc_path = os.path.join(os.getcwd(), "rc.0.xyz")
    assert os.path.exists("rc.0.xyz")

    rr = resources_root_path()
    ref_water_rc_path = os.path.join(rr, "water_rc.xyz")
    ref_water_rc = utils.io.read(ref_water_rc_path)[0]
    water_rc = utils.io.read(rc_path)[0]

    assert ref_water_rc.elements == water_rc.elements
    assert np.allclose(ref_water_rc.positions, water_rc.positions, rtol=0, atol=1e-10)
    # clean up
    os.remove(rc_path)
