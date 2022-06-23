#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps

# Third party imports
import scine_database as db

# Local application imports
from . import ElementaryStepGear
from ...utilities.queries import stop_on_timeout
from .trial_generator.bond_based import BondBased


class BruteForceElementarySteps(ElementaryStepGear):
    """
    This Gear probes Reactions by trying to react all Structures of each
    Compound with all Structures of each other Compound.
    For each Structure--Structure combination multiple arrangements (possible
    Elementary Steps) will be tested.

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
        calculation to search for an bimolecular reaction of the same
        structures with the same job order
    b. for unimolecular reactions: checking whether there is already a
        calculation to search for an intramolecular reaction of the same
        structure  with the same job order
    """

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self._calculations = "required"
        self._structures = "required"
        self._compounds = "required"
        self._reactions = "required"
        self._properties = "required"
        self.trial_generator = BondBased()

    def _propagate_db_manager(self, manager: db.Manager):
        self.trial_generator.initialize_collections(manager)

    def _loop_impl(self):

        self._sanity_check_configuration()

        # Loop over all compounds
        selection = {"exploration_disabled": {"$ne": True}}
        for compound_one in stop_on_timeout(self._compounds.iterate_compounds(dumps(selection))):
            compound_one.link(self._compounds)
            if self.compound_filter.filter(compound_one):
                # Do intramolecular reaction
                for sid_one in compound_one.get_structures():
                    structure_one = db.Structure(sid_one)
                    structure_one.link(self._structures)
                    # Only consider optimized structures, no guess structures or duplicates
                    if structure_one.get_label() not in [
                        db.Label.MINIMUM_OPTIMIZED,
                        db.Label.USER_OPTIMIZED,
                    ]:
                        continue
                    if self.options.enable_unimolecular_trials:
                        self.trial_generator.unimolecular_reactions(structure_one)

            # Get intermolecular reaction partners
            if not self.options.enable_bimolecular_trials:
                continue
            selection = {"exploration_disabled": {"$ne": True}}
            for compound_two in stop_on_timeout(self._compounds.iterate_compounds(dumps(selection))):
                compound_two.link(self._compounds)
                # Make this loop run lower triangular + diagonal only
                c_id_one = compound_one.id().string()
                c_id_two = compound_two.id().string()
                sorted_ids = sorted([c_id_one, c_id_two])
                # Second criterion needed to not exclude diagonal
                if sorted_ids[0] == c_id_two and c_id_one != c_id_two:
                    continue
                # Filter
                if not self.compound_filter.filter(compound_one, compound_two):
                    continue

                # TODO this should be a reduced query really.
                for sid_one in compound_one.get_structures():
                    structure_one = db.Structure(sid_one)
                    structure_one.link(self._structures)
                    # Only consider optimized structures, no guess structures or duplicates
                    if structure_one.get_label() not in [
                        db.Label.MINIMUM_OPTIMIZED,
                        db.Label.USER_OPTIMIZED,
                    ]:
                        continue
                    for sid_two in compound_two.get_structures():
                        structure_two = db.Structure(sid_two)
                        structure_two.link(self._structures)
                        # Only consider optimized structures, no guess structures or duplicates
                        if structure_two.get_label() not in [
                            db.Label.MINIMUM_OPTIMIZED,
                            db.Label.USER_OPTIMIZED,
                        ]:
                            continue
                        # Lower triangular only if same compound
                        if c_id_one == c_id_two:
                            s_id_one = structure_one.id().string()
                            s_id_two = structure_two.id().string()
                            sorted_ids = sorted([s_id_one, s_id_two])
                            # Second criterion needed to not exclude diagonal
                            if sorted_ids[0] == s_id_two and s_id_one != s_id_two:
                                continue

                        # Do intermolecular reactions
                        self.trial_generator.bimolecular_reactions([structure_one, structure_two])
