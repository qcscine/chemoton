#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import unittest
from json import dumps
import pytest

# Third party imports
import scine_database as db

# Local application tests imports
from scine_database import test_database_setup as db_setup
from scine_chemoton.gears import HoldsCollections
from scine_chemoton.engine import Engine
from scine_chemoton.gears.network_refinement.aggregate_based_refinement import AggregateBasedRefinement
from scine_chemoton.filters.aggregate_filters import IdFilter, AtomNumberFilter
from scine_chemoton.utilities.db_object_wrappers.wrapper_caches import MultiModelCacheFactory
from scine_chemoton.gears.network_refinement.enabling import ApplyToAllStructuresInAggregate, EnableStructureByModel
from scine_chemoton.filters.aggregate_filters import HasStructureWithModel


class TestAggregateBasedRefinement(unittest.TestCase, HoldsCollections):
    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "reactions", "compounds", "flasks",
                                      "properties", "calculations"]
        self.initialize_collections(manager)
        MultiModelCacheFactory().clear()

    def tearDown(self) -> None:
        self._manager.wipe()

    def set_up_database_and_gear(self):
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
            "chemoton_test_aggregate_based_refinement",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)
        n_calculations = self._calculations.count(dumps({}))
        assert n_calculations == 0
        return self.set_up_engine()

    def set_up_engine(self, pre_refine_model=None, post_refine_model=None):
        gear = AggregateBasedRefinement()
        engine = Engine(self._manager.get_credentials(), fork=False)
        engine.set_gear(gear)
        if pre_refine_model is None:
            pre_refine_model = db_setup.get_fake_model()
        if post_refine_model is None:
            post_refine_model = db.Model("post", "refine", "model")
        gear.options.model = pre_refine_model
        gear.options.post_refine_model = post_refine_model
        gear.options.only_electronic_energies = True
        gear.options.n_lowest = 100
        gear.options.energy_window = 200.0
        return gear, engine

    def test_sp_refinement(self):
        gear, engine = self.set_up_database_and_gear()
        gear.options.refinement["refine_single_points"] = True
        for _ in range(5):
            engine.run(single=True)
        n_structures = self._structures.count(dumps({"label": {"$ne": "ts_optimized"}}))
        n_calculations = self._calculations.count(dumps({}))
        assert n_calculations > 0
        assert n_calculations == n_structures
        for calculation in self._calculations.iterate_all_calculations():
            calculation.link(self._calculations)
            assert calculation.get_job().order == gear.options.sp_job.order

    def test_optimization(self):
        gear, engine = self.set_up_database_and_gear()
        gear.options.refinement["refine_optimizations"] = True
        for _ in range(5):
            engine.run(single=True)
        n_structures = self._structures.count(dumps({"label": {"$ne": "ts_optimized"}}))
        print("not TS", n_structures, "n total", self._structures.count(dumps({})))
        n_calculations = self._calculations.count(dumps({}))
        assert n_calculations > 0
        assert n_calculations == n_structures
        for calculation in self._calculations.iterate_all_calculations():
            calculation.link(self._calculations)
            assert calculation.get_job().order == gear.options.opt_job.order

    def test_window_restriction_n(self):
        gear, engine = self.set_up_database_and_gear()
        gear.options.refinement["refine_optimizations"] = True
        gear.options.n_lowest = 1
        for _ in range(5):
            engine.run(single=True)
        n_aggregates = self._compounds.count(dumps({})) + self._flasks.count(dumps({}))
        n_calculations = self._calculations.count(dumps({}))
        assert n_calculations == n_aggregates

    def test_window_restriction(self):
        gear, engine = self.set_up_database_and_gear()
        gear.options.refinement["refine_optimizations"] = True
        gear.options.energy_window = 0.0  # kJ/mol
        for _ in range(5):
            engine.run(single=True)
        n_aggregates = self._compounds.count(dumps({})) + self._flasks.count(dumps({}))
        n_calculations = self._calculations.count(dumps({}))
        assert n_calculations == n_aggregates

    def test_filter(self):
        gear, engine = self.set_up_database_and_gear()
        gear.options.refinement["refine_optimizations"] = True
        gear.options.energy_window = 0.0  # kJ/mol
        random_compound = self._compounds.get_one_compound(dumps({}))
        gear.aggregate_filter = IdFilter(ids=[random_compound.id().string()])
        for _ in range(5):
            engine.run(single=True)
        n_calculations = self._calculations.count(dumps({}))
        assert n_calculations == 1

    def test_filter_with_collection(self):
        gear, engine = self.set_up_database_and_gear()
        gear.options.refinement["refine_optimizations"] = True
        gear.options.energy_window = 0.0  # kJ/mol
        gear.aggregate_filter = AtomNumberFilter(2)
        for _ in range(5):
            engine.run(single=True)
        n_calculations = self._calculations.count(dumps({}))
        assert n_calculations == 0
        gear.aggregate_filter = AtomNumberFilter(3)
        for _ in range(5):
            engine.run(single=True)
        n_calculations = self._calculations.count(dumps({}))
        assert n_calculations == self._compounds.count(dumps({})) + self._flasks.count(dumps({}))

    def test_identical_models(self):
        gear, engine = self.set_up_database_and_gear()
        gear.options.post_refine_model = gear.options.model
        with pytest.raises(RuntimeError):
            engine.run(single=True)

    def test_wrong_pre_refinement_model(self):
        gear, engine = self.set_up_database_and_gear()
        gear.options.refinement["refine_optimizations"] = True
        gear.options.model = db.Model("some", "other", "model")
        for _ in range(5):
            engine.run(single=True)
        n_calculations = self._calculations.count(dumps({}))
        assert n_calculations == 0

    def test_deactivated(self):
        _, engine = self.set_up_database_and_gear()
        for _ in range(5):
            engine.run(single=True)
        n_calculations = self._calculations.count(dumps({}))
        assert n_calculations == 0

    def test_unknown_refinement_option(self):
        gear, engine = self.set_up_database_and_gear()
        gear.options.refinement["scine_step_refinement"] = True
        with pytest.raises(RuntimeError):
            engine.run(single=True)

    def test_enabling_instead_of_refinement(self):
        gear, engine = self.set_up_database_and_gear()
        gear.options.refinement["refine_single_points"] = True
        random_compounds = self._compounds.random_select_compounds(3)
        for c in random_compounds:
            c.link(self._compounds)
        random_flasks = self._flasks.random_select_flasks(2)
        for c in random_flasks:
            c.link(self._flasks)

        n_structures = self._structures.count(dumps({"label": {"$ne": "ts_optimized"}}))
        test_model = db.Model("My", "test", "model")
        gear.aggregate_enabling = ApplyToAllStructuresInAggregate(EnableStructureByModel(test_model))
        gear.aggregate_validation = HasStructureWithModel(test_model)

        n_structures_skiped_by_enabling = 0
        for aggregate in random_flasks + random_compounds:
            n_structures_skiped_by_enabling += len(aggregate.get_structures())
            aggregate.add_structure(db_setup._fake_structure(aggregate, self._structures, self._properties, False,
                                                             model=test_model))
        engine.run(single=True)

        n_calculations = self._calculations.count(dumps({}))
        assert n_calculations > 0
        assert n_calculations == n_structures - n_structures_skiped_by_enabling

        gear.aggregate_validation = HasStructureWithModel(db.Model("A", "Third", "Model"))
        engine.run(single=True)
        n_calculations = self._calculations.count(dumps({}))
        assert n_calculations == n_structures
