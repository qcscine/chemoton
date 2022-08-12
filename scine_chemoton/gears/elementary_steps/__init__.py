#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from abc import abstractmethod, ABC
from json import dumps
from typing import List

import scine_database as db

# Local application imports
from .compound_filters import CompoundFilter
from .trial_generator.bond_based import BondBased
from .. import Gear
from scine_chemoton.utilities.queries import stop_on_timeout


class ElementaryStepGear(Gear, ABC):
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
        self._required_collections = ["calculations", "compounds", "properties", "reactions", "structures"]
        self.options = self.Options()
        self.compound_filter: CompoundFilter = CompoundFilter()
        self.trial_generator = BondBased()

    def _sanity_check_configuration(self):
        if not isinstance(self.compound_filter, CompoundFilter):
            raise TypeError("Expected a CompoundFilter (or a class derived "
                            "from it) in ElementaryStepGear.compound_filter.")

    def _propagate_db_manager(self, manager: db.Manager):
        self.trial_generator.initialize_collections(manager)
        if hasattr(self.trial_generator, 'reactive_site_filter'):
            self.trial_generator.reactive_site_filter.initialize_collections(manager)
        if hasattr(self, 'compound_filter'):
            self.compound_filter.initialize_collections(manager)

    def _loop_impl(self):

        self._sanity_check_configuration()

        # Loop over all compounds
        selection = {"exploration_disabled": {"$ne": True}}
        for compound_one in stop_on_timeout(self._compounds.iterate_compounds(dumps(selection))):
            compound_one.link(self._compounds)
            eligible_sid_one = None
            if self.options.enable_unimolecular_trials and self.compound_filter.filter(compound_one):
                eligible_sid_one = sorted(self._get_eligible_structures(compound_one))
                for sid_one in eligible_sid_one:
                    structure_one = db.Structure(sid_one, self._structures)
                    self.trial_generator.unimolecular_reactions(structure_one)
            # Get intermolecular reaction partners
            if not self.options.enable_bimolecular_trials:
                continue
            if eligible_sid_one is None:
                eligible_sid_one = sorted(self._get_eligible_structures(compound_one))
            if not eligible_sid_one:
                break
            c_id_one = compound_one.id().string()
            selection = {"exploration_disabled": {"$ne": True}}
            for compound_two in stop_on_timeout(self._compounds.iterate_compounds(dumps(selection))):
                compound_two.link(self._compounds)
                # Make this loop run lower triangular + diagonal only
                c_id_two = compound_two.id().string()
                sorted_ids = sorted([c_id_one, c_id_two])
                # Second criterion needed to not exclude diagonal
                if sorted_ids[0] == c_id_two and c_id_one != c_id_two:
                    continue
                # Filter
                if not self.compound_filter.filter(compound_one, compound_two):
                    continue
                eligible_sid_two = sorted(self._get_eligible_structures(compound_two))
                if not eligible_sid_two:
                    continue
                same_compounds = c_id_one == c_id_two
                for i, sid_one in enumerate(eligible_sid_one):
                    for j, sid_two in enumerate(eligible_sid_two):
                        if same_compounds and j > i:
                            break
                        structure_one = db.Structure(sid_one, self._structures)
                        structure_two = db.Structure(sid_two, self._structures)
                        self.trial_generator.bimolecular_reactions([structure_one, structure_two])

    @abstractmethod
    def _get_eligible_structures(self, compound: db.Compound) -> List[db.ID]:
        pass
