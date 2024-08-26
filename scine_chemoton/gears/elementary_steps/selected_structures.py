#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import List, Union

# Third party imports
import scine_database as db

# Local application imports
from . import ElementaryStepGear


class SelectedStructuresElementarySteps(ElementaryStepGear):
    """
    This Gear probes Reactions by trying to react all predefined Structures
    with each other.
    For each Structure-Structure combination multiple arrangements (possible
    Elementary Steps) will be tested.

    Attributes
    ----------
    options : ElementaryStepGear.Options
        The options for the BruteForceElementarySteps Gear.
    aggregate_filter : scine_chemoton.gears.elementary_steps.aggregate_filters.AggregateFilter
        A filter for allowed reaction combinations, per default everything
        is permitted, no filter is applied.
    trial_generator : TrialGenerator
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
        structure with the same job order
    """
    class Options(ElementaryStepGear.Options):
        """
        The options of the SelectedStructuresElementaryStepGear.
        """
        __slots__ = "selected_structures"

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.selected_structures: List[db.ID] = []
            """
            List[db.ID]
                The IDs of structure that are considered reactive
            """

    options: Options

    def _get_eligible_structures(self, aggregate: Union[db.Compound, db.Flask]) -> List[db.ID]:
        return [sid for sid in aggregate.get_structures() if sid in self.options.selected_structures
                and self._check_structure_model(db.Structure(sid, self._structures))]
