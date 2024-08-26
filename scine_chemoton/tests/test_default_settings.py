#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from scine_utilities import ValueCollection

from scine_chemoton import default_settings


def test_execution_without_argument():
    for f in default_settings.__dict__.values():
        if callable(f):
            vc = f()
            assert isinstance(vc, ValueCollection)
