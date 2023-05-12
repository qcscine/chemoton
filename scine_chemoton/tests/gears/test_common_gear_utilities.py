#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from scine_database import Model

from scine_chemoton.gears import Gear
# required to be in the namespace
from scine_chemoton.gears.compound import BasicAggregateHousekeeping  # pylint: disable=unused-import  # noqa: F401
from scine_chemoton.gears.kinetics import (  # pylint: disable=unused-import  # noqa: F401
    MinimalConnectivityKinetics,  # pylint: disable=unused-import  # noqa: F401
    BasicBarrierHeightKinetics,  # pylint: disable=unused-import  # noqa: F401
    MaximumFluxKinetics,  # pylint: disable=unused-import  # noqa: F401
    PathfinderKinetics  # pylint: disable=unused-import  # noqa: F401
)
from scine_chemoton.gears.reaction import BasicReactionHousekeeping  # pylint: disable=unused-import  # noqa: F401
from scine_chemoton.gears.refinement import NetworkRefinement  # pylint: disable=unused-import  # noqa: F401
from scine_chemoton.gears.scheduler import Scheduler  # pylint: disable=unused-import  # noqa: F401
from scine_chemoton.gears.thermo import BasicThermoDataCompletion  # pylint: disable=unused-import  # noqa: F401
from scine_chemoton.gears.elementary_steps.minimal import \
    MinimalElementarySteps  # pylint: disable=unused-import  # noqa: F401
from scine_chemoton.gears.elementary_steps.brute_force import \
    BruteForceElementarySteps  # pylint: disable=unused-import  # noqa: F401
from scine_chemoton.gears.elementary_steps.minimum_energy_conformer import \
    MinimumEnergyConformerElementarySteps  # pylint: disable=unused-import  # noqa: F401


def test_comparisons():
    for name, cls in globals().items():
        if name.startswith('__'):
            continue
        try:
            if issubclass(cls, Gear):
                a = cls()
                b = cls()
                assert a == b
                a.options.model = Model("foo", "bar", "")
                assert a != b
        except TypeError:
            pass
