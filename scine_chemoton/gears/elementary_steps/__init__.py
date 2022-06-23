#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""


# Local application imports
from .compound_filters import CompoundFilter
from .. import Gear


class ElementaryStepGear(Gear):
    """
    Base class for elementary step reaction generators
    """

    class Options:
        """
        The options for an ElementarySteps Gear.
        """

        __slots__ = ("cycle_time", "enable_unimolecular_trials", "enable_bimolecular_trials")

        def __init__(self):
            self.cycle_time = 10
            """
            int
                Sleep time between cycles, in seconds.
            """
            self.enable_unimolecular_trials = True
            """
            bool
                If `True`, enables the exploration of unimolecular reactions.
            """
            self.enable_bimolecular_trials = True
            """
            bool
                If `True`, enables the exploration of bimolecular reactions.
            """

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self.compound_filter: CompoundFilter = CompoundFilter()
        self.trial_generator = None

    def _sanity_check_configuration(self):
        if not isinstance(self.compound_filter, CompoundFilter):
            raise TypeError("Expected a CompoundFilter (or a class derived "
                            "from it) in ElementaryStepGear.compound_filter.")
