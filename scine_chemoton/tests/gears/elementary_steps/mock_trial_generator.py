#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import List

# Third party imports
import scine_database as db

# Local application imports
from ....gears.elementary_steps.trial_generator import TrialGenerator


class MockGenerator(TrialGenerator):
    """
    A mock trial generator that counts how often the `unimolecular_reactions`
    and `bimolecular_reactions` methods are called.

    Attributes
    ----------
    unimol_counter : int
        How often  `unimolecular_reactions` was called.
    bimol_counter : ReactiveSiteFilter
        How often  `bimolecular_reactions` was called.
    """

    def __init__(self):
        super().__init__()
        self.unimol_counter = 0
        self.bimol_counter = 0

    def unimolecular_reactions(self, structure: db.Structure) -> None:
        self.unimol_counter += 1

    def bimolecular_reactions(self, structure_list: List[db.Structure]) -> None:
        self.bimol_counter += 1
