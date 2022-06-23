#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
from typing import List

# Third party imports
import scine_database as db

# Local application imports
from . import Gear
from ..utilities.queries import identical_reaction, stop_on_timeout


class BasicReactionHousekeeping(Gear):
    """
    This Gear updates all Elementary Steps stored in the database such that they
    are added to an existing Reaction or that a new Reaction is created if the
    Step does not fit an existing one.

    Attributes
    ----------
    options :: BasicReactionHousekeeping.Options
        The options for the BasicReactionHousekeeping Gear.

    Notes
    -----
    Checks for all Elementary Steps without a 'reaction'.
    """

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self._elementary_steps = "required"
        self._structures = "required"
        self._reactions = "required"
        self._compounds = "required"

    class Options:
        """
        The options for the BasicReactionHousekeeping Gear.
        """

        __slots__ = "cycle_time"

        def __init__(self):
            self.cycle_time = 10
            """
            int
                The minimum number of seconds between two cycles of the Gear.
                Cycles are finished independent of this option, thus if a cycle
                takes longer than the cycle_time will effectively lead to longer
                cycle times and not cause multiple cycles of the same Gear.
            """

    def _loop_impl(self):
        # Setup query for elementary steps without reactions
        selection = {"$and": [{"reaction": {"$eq": ""}}, {"analysis_disabled": {"$ne": True}}]}
        # Loop over all results
        for step in stop_on_timeout(self._elementary_steps.iterate_elementary_steps(dumps(selection))):
            step.link(self._elementary_steps)
            reactants = step.get_reactants(db.Side.BOTH)
            lhs = reactants[0]
            rhs = reactants[1]

            # Check if all structures have compounds assigned
            lhs_compounds = []
            rhs_compounds = []
            all_structures_have_compounds = True
            for sid in lhs:
                structure = db.Structure(sid)
                structure.link(self._structures)
                if not structure.has_compound():
                    all_structures_have_compounds = False
                    break
                else:
                    lhs_compounds.append(structure.get_compound())
            if not all_structures_have_compounds:
                continue
            for sid in rhs:
                structure = db.Structure(sid)
                structure.link(self._structures)
                if not structure.has_compound():
                    all_structures_have_compounds = False
                    break
                else:
                    rhs_compounds.append(structure.get_compound())
            if not all_structures_have_compounds:
                continue

            # Check for a reactions with the same structures/compounds
            true_hit = identical_reaction(lhs_compounds, rhs_compounds, self._reactions)
            if true_hit is not None:
                # Add elementary step to reaction
                # TODO Deduplicate
                reaction = true_hit
                reaction.add_elementary_step(step.id())
                step.set_reaction(reaction.id())
                continue
            # Generate new reaction
            reaction = db.Reaction()
            reaction.link(self._reactions)
            reaction.create(lhs_compounds, rhs_compounds)
            reaction.add_elementary_step(step.id())
            step.set_reaction(reaction.id())
            self._add_reaction_to_compounds(lhs_compounds + rhs_compounds, reaction.get_id())

    def _add_reaction_to_compounds(self, compounds_to_change: List[db.ID], reaction_id: db.ID):
        for compound_id in compounds_to_change:
            compound = db.Compound(compound_id)
            compound.link(self._compounds)
            compound.add_reaction(reaction_id)
