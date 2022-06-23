#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import numpy as np

# Local application imports
from ....utilities.reactive_complexes.lebedev_sphere import LebedevSphere


def test_distance():
    lebedev = LebedevSphere()
    for point in lebedev.points:
        assert abs(np.linalg.norm(point) - 1.0) < 1e-6


def test_neighbors():
    lebedev = LebedevSphere()
    for i, neighbors in enumerate(lebedev.nearest_neighbors):
        for j in neighbors:
            assert i in lebedev.nearest_neighbors[j]
