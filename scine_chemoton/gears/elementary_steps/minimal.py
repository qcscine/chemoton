#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import List
from warnings import warn

# Third party imports
import scine_database as db

# Local application imports
from . import ElementaryStepGear


class MinimalElementarySteps(ElementaryStepGear):
    """
    This Gear probes Reactions by trying to react

    1. one Structure of each compound with one Structure of each other compound
       (intermolecular reactions)
    2. one Structure with itself intramoleculary for each compound.

    For each combination multiple arrangements (possible Elementary Steps) will
    be tested.

    This Gear does not consider Flasks/Complexes as reactive, they are not probed
    for elementary steps.

    Attributes
    ----------
    options :: MinimalElementarySteps.Options
        The options for the MinimalElementarySteps Gear.
    aggregate_filter :: scine_chemoton.gears.elementary_steps.aggregate_filters.AggregateFilter
        A filter for allowed reaction combinations, per default everything
        is permitted, no filter is applied.
    trial_generator :: TrialGenerator
        The generator to set up elementary step trial calculations by enumerating
        reactive complexes and trial reaction coordinates

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
        centroid = db.Structure(compound.get_centroid(), self._structures)
        if not centroid.explore() or centroid.get_label() not in [
            db.Label.MINIMUM_OPTIMIZED,
            db.Label.USER_OPTIMIZED,
        ]:
            warn(f"{self.name} picked centroid {centroid.id()} for compound {compound.id()}, but this is actually not "
                 f"a valid structure.")
        return [centroid.id()]
