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


class MinimalElementarySteps(ElementaryStepGear):
    """
    This Gear probes Reactions by trying to react

    1. one Structure of each compound with one Structure of each other compound
       (intermolecular reactions)
    2. one Structure with itself intramoleculary for each compound.

    For each combination multiple arrangements (possible Elementary Steps) will
    be tested.

    Attributes
    ----------
    options :: MinimalElementarySteps.Options
        The options for the MinimalElementarySteps Gear.
    compound_filter :: scine_chemoton.gears.elementary_steps.compound_filters.CompoundFilter
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
        calculation to search for an bimolecular reaction of the same
        structures with the same job order
    b. for unimolecular reactions: checking whether there is already a
        calculation to search for an intramolecular reaction of the same
        structure  with the same job order
    """

    def __init__(self):
        super().__init__()
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
                centroid_one = db.Structure(compound_one.get_centroid())
                centroid_one.link(self._structures)
                # Do intramolecular reaction
                if self.options.enable_unimolecular_trials:
                    self.trial_generator.unimolecular_reactions(centroid_one)

            # Get intermolecular reaction partners
            if not self.options.enable_bimolecular_trials:
                continue
            selection = {"exploration_disabled": {"$ne": True}}
            for compound_two in stop_on_timeout(self._compounds.iterate_compounds(dumps(selection))):
                compound_two.link(self._compounds)
                centroid_one = db.Structure(compound_one.get_centroid())
                centroid_one.link(self._structures)
                centroid_two = db.Structure(compound_two.get_centroid())
                centroid_two.link(self._structures)
                # Make this loop run lower triangular + diagonal only
                id_one = compound_one.id().string()
                id_two = compound_two.id().string()
                sorted_ids = sorted([id_one, id_two])
                # Second criterion needed to not exclude diagonal
                if sorted_ids[0] == id_two and id_one != id_two:
                    continue
                # Filter
                if not self.compound_filter.filter(compound_one, compound_two):
                    continue
                # Do intermolecular reactions
                self.trial_generator.bimolecular_reactions([centroid_one, centroid_two])
