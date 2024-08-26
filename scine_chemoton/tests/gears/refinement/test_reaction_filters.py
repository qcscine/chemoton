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
from ....utilities.db_object_wrappers.wrapper_caches import MultiModelCacheFactory
from scine_database.insert_concentration import insert_concentration_for_structure
from scine_chemoton.filters.reaction_filters import (
    ReactionBarrierFilter,
    ReactionNumberPropertyFilter,
    BarrierlessReactionFilter,
    StopDuringExploration,
    ReactionHasStepWithModel,
    MaximumTransitionStateEnergyFilter
)


class TestReactionFilter(unittest.TestCase, HoldsCollections):
    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "reactions", "compounds", "flasks",
                                      "properties", "calculations"]
        self.initialize_collections(manager)
        MultiModelCacheFactory().clear()

    def tearDown(self) -> None:
        self._manager.wipe()

    def insert_concentration_for_reaction(self, label, reaction, value, model):
        aggregate = db.Compound(reaction.get_reactants(db.Side.BOTH)[0][0], self._compounds)
        property_label = reaction.id().string() + label
        insert_concentration_for_structure(self._manager, value, model, aggregate.get_centroid(),
                                           label=property_label)

    def test_barrier_filter(self):
        n_compounds = 10
        n_flasks = 0  # no barrierless reactions please!
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
            "chemoton_test_barrier_filter",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)
        for compound in self._compounds.iterate_all_compounds():
            compound.link(self._compounds)
            compound.enable_exploration()
            compound.enable_analysis()
        for flask in self._flasks.iterate_all_flasks():
            flask.link(self._flasks)
            flask.enable_exploration()
        model = db_setup.get_fake_model()
        combi = ModelCombination(model, model)
        f = ReactionBarrierFilter(30.0, combi, only_electronic_energies=True)
        f.initialize_collections(manager)
        for reaction in self._reactions.iterate_all_reactions():
            reaction.link(self._reactions)
            assert f.filter(reaction)

        f2 = ReactionBarrierFilter(8.0, combi, only_electronic_energies=True)
        f2.initialize_collections(manager)
        for reaction in self._reactions.iterate_all_reactions():
            reaction.link(self._reactions)
            assert not f2.filter(reaction)

        other_model = db.Model("other", "model", "maybe")
        f3 = ReactionBarrierFilter(100.0, ModelCombination(other_model), only_electronic_energies=True)
        f3.initialize_collections(manager)
        for reaction in self._reactions.iterate_all_reactions():
            reaction.link(self._reactions)
            assert not f3.filter(reaction)

        f4 = ReactionBarrierFilter(100.0, combi, only_electronic_energies=False)
        f4.initialize_collections(manager)
        for reaction in self._reactions.iterate_all_reactions():
            reaction.link(self._reactions)
            assert not f4.filter(reaction)

    def add_reaction(self, manager: db.Manager):
        f_id_1, s_id_1 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.COMPLEX_OPTIMIZED)
        f_id_2, s_id_2 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.COMPLEX_OPTIMIZED)
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
        new_reaction = db.Reaction()
        new_reaction.link(self._reactions)
        new_reaction.create([f_id_1], [f_id_2], [db.CompoundOrFlask.FLASK], [db.CompoundOrFlask.FLASK])
        new_reaction.add_elementary_step(new_step.id())

        return new_reaction, new_step, s_1, s_2

    def test_barrierless_reaction_filter(self):
        manager = db_setup.get_clean_db("chemoton_test_barrierless_reaction_filter")
        self.custom_setup(manager)

        new_reaction, _, _, _ = self.add_reaction(manager)

        model = db_setup.get_fake_model()
        combi = ModelCombination(model, model)
        f = BarrierlessReactionFilter(combi, only_electronic_energies=True)
        f.initialize_collections(manager)
        assert not f.filter(new_reaction)
        f2 = BarrierlessReactionFilter(combi, only_electronic_energies=True, exclude_barrierless=False)
        f2.initialize_collections(manager)
        assert f2.filter(new_reaction)

    def test_reaction_number_property_filter(self):
        n_compounds = 12
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
            "chemoton_test_reaction_number_property_filter",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)
        label = "_concentration_flux"
        model = db_setup.get_fake_model()
        f = ReactionNumberPropertyFilter(label, 1e-1, model, threshold_must_be_exceeded=True)
        f2 = ReactionNumberPropertyFilter(label, 1e-1, model, threshold_must_be_exceeded=False)
        f.initialize_collections(manager)
        f2.initialize_collections(manager)
        n_insert = 3
        for reaction in self._reactions.iterate_all_reactions():
            reaction.link(self._reactions)
            assert not f.filter(reaction)
            assert f2.filter(reaction)
            if n_insert:
                self.insert_concentration_for_reaction(label, reaction, 1.0, model)
                n_insert -= 1
        n_insert = 3
        for reaction in self._reactions.iterate_all_reactions():
            reaction.link(self._reactions)
            if n_insert:
                assert f.filter(reaction)
                assert not f2.filter(reaction)
                n_insert -= 1
            else:
                assert not f.filter(reaction)
                assert f2.filter(reaction)

    def test_reaction_filter_and_or_array(self):
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
            "chemoton_test_barrier_filter",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)
        model = db_setup.get_fake_model()
        label1 = "_concentration_flux_1"
        label2 = "_concentration_flux_2"

        f = ReactionNumberPropertyFilter(label1, 1e-1, model) and ReactionNumberPropertyFilter(label2, 1.0, model)
        f2 = ReactionNumberPropertyFilter(label1, 1e-1, model) or ReactionNumberPropertyFilter(label2, 1.0, model)
        f.initialize_collections(manager)
        f2.initialize_collections(manager)
        n_insert_one = 5
        n_insert_two = 3
        for reaction in self._reactions.iterate_all_reactions():
            reaction.link(self._reactions)
            assert not f.filter(reaction)
            assert not f2.filter(reaction)
            if n_insert_one:
                self.insert_concentration_for_reaction(label1, reaction, 0.5, model)
                n_insert_one -= 1
            if n_insert_two:
                self.insert_concentration_for_reaction(label2, reaction, 2.0, model)
                n_insert_two -= 1
            else:
                self.insert_concentration_for_reaction(label2, reaction, 0.5, model)
        n_insert_one = 5
        n_insert_two = 3
        for reaction in self._reactions.iterate_all_reactions():
            reaction.link(self._reactions)
            if n_insert_two and n_insert_one:
                assert f.filter(reaction)
                assert f2.filter(reaction)
                n_insert_one -= 1
                n_insert_two -= 1
            else:
                assert not f.filter(reaction)
                if n_insert_one:
                    assert f2.filter(reaction)
                    n_insert_one -= 1
                else:
                    assert not f2.filter(reaction)

    def test_stop_during_exploration_filter(self):
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
        for reaction in self._reactions.iterate_all_reactions():
            reaction.link(self._reactions)
            assert f.filter(reaction)
        model = db_setup.get_fake_model()
        new_calculation = db.Calculation.make(model, db.Job(wait_for[0]), [db.ID()], self._calculations)
        for status in [db.Status.NEW, db.Status.HOLD, db.Status.PENDING]:
            new_calculation.set_status(status)
            for reaction in self._reactions.iterate_all_reactions():
                reaction.link(self._reactions)
                assert not f.filter(reaction)
            new_calculation.set_status(db.Status.COMPLETE)
        for reaction in self._reactions.iterate_all_reactions():
            reaction.link(self._reactions)
            assert f.filter(reaction)

        new_calculation = db.Calculation.make(model, db.Job(ignore[0]), [db.ID()], self._calculations)
        new_calculation.set_status(db.Status.PENDING)
        for reaction in self._reactions.iterate_all_reactions():
            reaction.link(self._reactions)
            assert f.filter(reaction)

    def test_has_step_with_model_filter_energy(self):
        manager = db_setup.get_clean_db("chemoton_test_has_step_with_model_filter_energy")
        self.custom_setup(manager)

        new_reaction, _, s_1, s_2 = self.add_reaction(manager)
        _, step_2, _, _ = self.add_reaction(manager)
        new_reaction.add_elementary_step(step_2.id())
        step_2.set_reaction(new_reaction.id())

        test_model = db.Model("My", "test", "model")
        f = ReactionHasStepWithModel(test_model, check_only_energies=True)
        f.initialize_collections(manager)

        assert not f.filter(new_reaction)

        db_setup.add_random_energy(s_1, (-100.0, -101.0), self._properties, test_model)
        assert not f.filter(new_reaction)

        db_setup.add_random_energy(s_2, (-100.0, -101.0), self._properties, test_model)
        assert f.filter(new_reaction)

    def test_has_step_with_model_filter_structure(self):
        manager = db_setup.get_clean_db("chemoton_test_has_step_with_model_filter_structure")
        self.custom_setup(manager)

        new_reaction, _, s_1, s_2 = self.add_reaction(manager)
        _, step_2, _, _ = self.add_reaction(manager)
        new_reaction.add_elementary_step(step_2.id())
        step_2.set_reaction(new_reaction.id())

        test_model = db.Model("My", "test", "model")
        f = ReactionHasStepWithModel(test_model, check_only_energies=False)
        f.initialize_collections(manager)

        assert not f.filter(new_reaction)

        s_1.set_model(test_model)
        assert not f.filter(new_reaction)

        s_2.set_model(test_model)
        assert f.filter(new_reaction)

    def test_maximum_energy_transition_state_filter(self):
        manager = db_setup.get_clean_db("chemoton_test_maximum_energy_transition_state_filter")
        self.custom_setup(manager)

        new_reaction, step_1, s_1, s_2 = self.add_reaction(manager)
        _, _, s_3, _ = self.add_reaction(manager)
        e_ts = -634.6568134574  # Energy of the TS.

        model = s_2.get_model()
        other_model = db.Model("Other", "model", "other_model")
        s_1.set_model(other_model)

        f = MaximumTransitionStateEnergyFilter(ModelCombination(model), e_ts + 0.1, True)
        f.initialize_collections(manager)
        # Wrong model for s_1
        assert not f.filter(new_reaction)
        s_1.set_model(model)
        # Everything matching.
        assert f.filter(new_reaction)

        # Assert that it also works for regualr elementary steps.
        step_1.set_transition_state(s_3.id())
        step_1.set_type(db.ElementaryStepType.REGULAR)
        assert f.filter(new_reaction)

        # Assert that the model matters.
        f = MaximumTransitionStateEnergyFilter(ModelCombination(other_model), e_ts + 0.1, True)
        f.initialize_collections(manager)
        assert not f.filter(new_reaction)

        # Max energy-cut off.
        f = MaximumTransitionStateEnergyFilter(ModelCombination(model), e_ts - 0.1, True)
        f.initialize_collections(manager)
        assert not f.filter(new_reaction)
