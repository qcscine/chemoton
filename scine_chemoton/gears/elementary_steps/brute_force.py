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
from . import ElementaryStepGear


class BruteForceElementarySteps(ElementaryStepGear):
    """
    This Gear probes Reactions by trying to react all Structures of each
    Compound with all Structures of each other Compound.
    For each Structure--Structure combination multiple arrangements (possible
    Elementary Steps) will be tested.

    This Gear does not consider Flasks/Complexes as reactive, they are not probed
    for elementary steps.

    Attributes
    ----------
    options :: BruteForceElementarySteps.Options
        The options for the BruteForceElementarySteps Gear.
    filter :: scine_chemoton.gears.elementary_steps.compound_filter.CompoundFilter
        A filter for allowed reaction combinations, per default everything
        is permitted, no filter is applied.

    Notes
    -----
    This function assumes maximum spin when adding two Structures into one
    reactive complex.
    The need for elementary step guesses is tested by:

    a. for bimolecular reactions: checking whether there is already a
        calculation to search for a bimolecular reaction of the same
        structures with the same job order
    b. for unimolecular reactions: checking whether there is already a
        calculation to search for an intramolecular reaction of the same
        structure  with the same job order
    """

    def _get_eligible_structures(self, compound: db.Compound) -> List[db.ID]:
        eligible = []
        for sid in compound.get_structures():
            structure = db.Structure(sid, self._structures)
            # Only consider optimized structures, no guess structures or duplicates
            if not structure.explore() or structure.get_label() not in [
                db.Label.MINIMUM_OPTIMIZED,
                db.Label.USER_OPTIMIZED,
            ]:
                continue
            eligible.append(sid)
        return eligible
