#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from scine_database import Model

from scine_chemoton.gears import Gear
# required to be in the namespace
from scine_chemoton.gears.compound import (  # pylint: disable=unused-import  # noqa: F401
    BasicAggregateHousekeeping,  # pylint: disable=unused-import  # noqa: F401
    ThermoAggregateHousekeeping  # pylint: disable=unused-import  # noqa: F401
)
from scine_chemoton.gears.kinetics import (  # pylint: disable=unused-import  # noqa: F401
    MinimalConnectivityKinetics,  # pylint: disable=unused-import  # noqa: F401
    BasicBarrierHeightKinetics,  # pylint: disable=unused-import  # noqa: F401
    MaximumFluxKinetics,  # pylint: disable=unused-import  # noqa: F401
    PathfinderKinetics  # pylint: disable=unused-import  # noqa: F401
)
from scine_chemoton.gears.reaction import BasicReactionHousekeeping  # pylint: disable=unused-import  # noqa: F401
from scine_chemoton.gears.refinement import NetworkRefinement  # pylint: disable=unused-import  # noqa: F401
from scine_chemoton.gears.rerun_calculations import RerunCalculations  # pylint: disable=unused-import  # noqa: F401
from scine_chemoton.gears.scheduler import Scheduler  # pylint: disable=unused-import  # noqa: F401
from scine_chemoton.gears.thermo import BasicThermoDataCompletion  # pylint: disable=unused-import  # noqa: F401
from scine_chemoton.gears.elementary_steps.minimal import \
    MinimalElementarySteps  # pylint: disable=unused-import  # noqa: F401
from scine_chemoton.gears.elementary_steps.brute_force import \
    BruteForceElementarySteps  # pylint: disable=unused-import  # noqa: F401
from scine_chemoton.gears.elementary_steps.minimum_energy_conformer import \
    MinimumEnergyConformerElementarySteps  # pylint: disable=unused-import  # noqa: F401
from scine_chemoton.gears.elementary_steps.selected_structures import \
    SelectedStructuresElementarySteps  # pylint: disable=unused-import  # noqa: F401


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


def test_slots():
    for name, cls in globals().items():
        if name.startswith('__'):
            continue
        try:
            if issubclass(cls, Gear):
                assert hasattr(cls, "Options")
                assert hasattr(cls.Options, '__slots__')
                try:
                    if isinstance(cls.Options.__slots__, str):
                        attr = {cls.__slots__: getattr(cls, cls.__slots__, None)}
                    else:
                        attr = {k: getattr(cls, k, None) for k in cls.__slots__}
                    for v in attr.values():
                        if issubclass(v, Gear.Options):
                            assert hasattr(v, '__slots__')
                            assert v.__slots__
                except AttributeError:
                    pass
        except TypeError:
            pass
