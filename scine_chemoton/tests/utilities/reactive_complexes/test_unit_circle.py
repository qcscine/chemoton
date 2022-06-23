#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import numpy as np

# Local application imports
from ....utilities.reactive_complexes.unit_circle import UnitCircle


def test_distance():
    unit_circle = UnitCircle()
    for point in unit_circle.points:
        assert abs(np.linalg.norm(point) - 1.0) < 1e-6


def test_neighbors():
    unit_circle = UnitCircle()
    for i, neighbors in enumerate(unit_circle.nearest_neighbors):
        for j in neighbors:
            assert i in unit_circle.nearest_neighbors[j]
