#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import unittest

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application tests imports
from scine_database import test_database_setup as db_setup
from ....gears import HoldsCollections
from ....utilities.model_combinations import ModelCombination
from scine_chemoton.filters.elementary_step_filters import (
    ElementaryStepBarrierFilter,
    BarrierlessElementaryStepFilter,
    StopDuringExploration,
    ConsistentEnergyModelFilter
)
from ....utilities.db_object_wrappers.wrapper_caches import MultiModelCacheFactory


class TestElementaryStepFilter(unittest.TestCase, HoldsCollections):
    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "reactions", "compounds", "flasks",
                                      "properties", "calculations"]
        self.initialize_collections(manager)
        MultiModelCacheFactory().clear()

    def tearDown(self) -> None:
        self._manager.wipe()

    def test_elementary_step_barrier_filter(self):
        n_compounds = 10
        n_flasks = 0  # no barrier-less reactions.
        n_reactions = 10
        max_r_per_c = 10
        max_n_products_per_r = 2
        max_n_educts_per_r = 2
        max_s_per_c = 1
        max_steps_per_r = 1
        barrier_limits = (100, 120)  # + a 10 kJ/mol freedom on top of this range because of the aggregate energies
        n_inserts = 3
        manager = db_setup.get_random_db(
            n_compounds,
            n_flasks,
            n_reactions,
            max_r_per_c,
            "chemoton_test_elementary_step_barrier_filter",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)
        model = db_setup.get_fake_model()
        combi = ModelCombination(model, model)
        for side in [db.Side.LHS, db.Side.RHS, db.Side.BOTH]:
            f = ElementaryStepBarrierFilter(160.0, combi, only_electronic_energies=True, accessible_side=side)
            f.initialize_collections(manager)
            for step in self._elementary_steps.iterate_all_elementary_steps():
                step.link(self._elementary_steps)
                assert f.filter(step)
            f2 = ElementaryStepBarrierFilter(80.0, combi, only_electronic_energies=True, accessible_side=side)
            f2.initialize_collections(manager)
            for step in self._elementary_steps.iterate_all_elementary_steps():
                step.link(self._elementary_steps)
                assert not f2.filter(step)

    def add_elementary_step(self, manager: db.Manager):
        _, s_id_1 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.COMPLEX_OPTIMIZED)
        _, s_id_2 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.COMPLEX_OPTIMIZED)
        s_1 = db.Structure(s_id_1, self._structures)
        s_2 = db.Structure(s_id_2, self._structures)
        e1 = -634.6730820353
        e2 = -634.6568134574
        db_setup.add_random_energy(s_1, (e1 / utils.HARTREE_PER_KJPERMOL, e1 / utils.HARTREE_PER_KJPERMOL),
                                   self._properties)
        db_setup.add_random_energy(s_2, (e2 / utils.HARTREE_PER_KJPERMOL, e2 / utils.HARTREE_PER_KJPERMOL),
                                   self._properties)
        new_step = db.ElementaryStep()
        new_step.link(self._elementary_steps)
        new_step.create([s_id_1], [s_id_2])
        new_step.set_type(db.ElementaryStepType.BARRIERLESS)
        return new_step, s_1, s_2

    def test_barrierless_elementary_step_filter(self):
        manager = db_setup.get_clean_db("chemoton_test_barrierless_elementary_step_filter")
        self.custom_setup(manager)

        new_step, _, _ = self.add_elementary_step(manager)

        f = BarrierlessElementaryStepFilter()
        f.initialize_collections(manager)
        assert not f.filter(new_step)
        f2 = BarrierlessElementaryStepFilter(exclude_barrierless=False)
        f2.initialize_collections(manager)
        assert f2.filter(new_step)

    def test_stop_during_exploration(self):
        n_compounds = 10
        n_flasks = 0
        n_reactions = 10
        max_r_per_c = 10
        max_n_products_per_r = 2
        max_n_educts_per_r = 2
        max_s_per_c = 1
        max_steps_per_r = 1
        barrier_limits = (10, 20)
        n_inserts = 3
        manager = db_setup.get_random_db(
            n_compounds,
            n_flasks,
            n_reactions,
            max_r_per_c,
            "chemoton_test_stop_during_exploration_filter",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)
        wait_for = ["please-wait-for-me"]
        ignore = ["ignore-me"]

        f = StopDuringExploration(orders_to_wait_for=wait_for)
        f.initialize_collections(manager)
        for reaction in self._elementary_steps.iterate_all_elementary_steps():
            reaction.link(self._elementary_steps)
            assert f.filter(reaction)
        model = db_setup.get_fake_model()
        new_calculation = db.Calculation.make(model, db.Job(wait_for[0]), [db.ID()], self._calculations)
        for status in [db.Status.NEW, db.Status.HOLD, db.Status.PENDING]:
            new_calculation.set_status(status)
            for step in self._elementary_steps.iterate_all_elementary_steps():
                step.link(self._elementary_steps)
                assert not f.filter(step)
            new_calculation.set_status(db.Status.COMPLETE)
        for step in self._elementary_steps.iterate_all_elementary_steps():
            step.link(self._elementary_steps)
            assert f.filter(step)

        new_calculation = db.Calculation.make(model, db.Job(ignore[0]), [db.ID()], self._calculations)
        new_calculation.set_status(db.Status.PENDING)
        for step in self._elementary_steps.iterate_all_elementary_steps():
            step.link(self._elementary_steps)
            assert f.filter(step)

    def test_consistent_energy_model_filter(self):
        manager = db_setup.get_clean_db("chemoton_test_consistent_energy_model_filter")
        self.custom_setup(manager)

        new_step, s_1, s_2 = self.add_elementary_step(manager)
        test_model = db.Model("My", "Test", "Model")

        f = ConsistentEnergyModelFilter(test_model)
        f.initialize_collections(manager)

        assert not f.filter(new_step)

        db_setup.add_random_energy(s_1, (-100.0, -101.0), self._properties, test_model)
        assert not f.filter(new_step)

        db_setup.add_random_energy(s_2, (-100.0, -101.0), self._properties, test_model)
        assert f.filter(new_step)
