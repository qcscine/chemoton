#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import unittest

import scine_database as db

from scine_database import test_database_setup as db_setup

from ....gears import HoldsCollections
from ....utilities.db_object_wrappers.wrapper_caches import MultiModelCacheFactory
from ....utilities.model_combinations import ModelCombination
from ....utilities.db_object_wrappers.aggregate_cache import AggregateCache
from ....utilities.db_object_wrappers.reaction_cache import ReactionCache
from ....utilities.db_object_wrappers.aggregate_wrapper import Aggregate
from ....utilities.db_object_wrappers.reaction_wrapper import Reaction


class TestWrapperFactorySingleton(unittest.TestCase, HoldsCollections):
    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "reactions", "compounds", "flasks"]
        self.initialize_collections(manager)

    def tearDown(self) -> None:
        self._manager.wipe()

    def set_up_database(self):
        n_compounds = 10
        n_flasks = 6
        n_reactions = 16
        max_r_per_c = 10
        max_n_products_per_r = 3
        max_n_educts_per_r = 2
        max_s_per_c = 3
        max_steps_per_r = 1
        barrier_limits = (0.1, 100.0)
        n_inserts = 3
        manager = db_setup.get_random_db(
            n_compounds,
            n_flasks,
            n_reactions,
            max_r_per_c,
            "chemoton_test_wrapper_factory_singleton",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)
        MultiModelCacheFactory().clear()

    def test_wrapper_singleton(self):
        self.set_up_database()
        combi = ModelCombination(db_setup.get_fake_model())
        # Check if the factory is able to provide functioning caches.
        for only_electronic in [True, False]:
            aggregate_cache = MultiModelCacheFactory().get_aggregates_cache(only_electronic, combi, self._manager)
            assert isinstance(aggregate_cache, AggregateCache)
            for compound in self._compounds.iterate_all_compounds():
                aggregate = aggregate_cache.get_or_produce(compound.id())
                assert isinstance(aggregate, Aggregate)
                assert aggregate.get_db_id() == compound.id()

            reaction_cache = MultiModelCacheFactory().get_reaction_cache(only_electronic, combi, self._manager)
            assert isinstance(reaction_cache, ReactionCache)
            for db_reaction in self._reactions.iterate_all_reactions():
                reaction = reaction_cache.get_or_produce(db_reaction.id())
                assert isinstance(reaction, Reaction)
                assert reaction.get_db_id() == db_reaction.id()

        # Check if there is only one instance of the factory and this instance returns the same objects if the
        # same settings are requested.
        assert MultiModelCacheFactory() == MultiModelCacheFactory()
        instance_1 = MultiModelCacheFactory().get_aggregates_cache(True, combi, self._manager)
        instance_2 = MultiModelCacheFactory().get_aggregates_cache(True, combi, self._manager)
        assert instance_1 == instance_2
        # Check for different settings.
        combi_2 = ModelCombination(db.Model("some", "other", "model"))
        instance_2 = MultiModelCacheFactory().get_aggregates_cache(True, combi_2, self._manager)
        assert instance_1 != instance_2

        instance_2 = MultiModelCacheFactory().get_aggregates_cache(False, combi, self._manager)
        assert instance_1 != instance_2
