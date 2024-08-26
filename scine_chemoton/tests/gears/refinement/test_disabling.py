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

# Local application tests imports
from scine_database import test_database_setup as db_setup
from ....gears import HoldsCollections
from ....gears.network_refinement.disabling import (
    ReactionDisabling,
    DisableReactionByJob,
    DisableAllReactions,
    StepDisabling,
    DisableStepByJob,
    DisableAllSteps,
    AggregateDisabling,
    DisableAllAggregates,
    DisableAllStepsByModel
)


class TestDisabling(unittest.TestCase, HoldsCollections):
    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "reactions", "structures", "compounds", "flasks"]
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
            "chemoton_test_disabling",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)

    def run_reaction_policy(self, policy: ReactionDisabling, job_name: str = ""):
        for reaction in self._reactions.iterate_all_reactions():
            reaction.link(self._reactions)
            policy.process(reaction, job_name)

    def run_step_policy(self, policy: StepDisabling, job_name: str = ""):
        for step in self._elementary_steps.iterate_all_elementary_steps():
            step.link(self._elementary_steps)
            policy.process(step, job_name)

    def run_aggregate_policy(self, policy: AggregateDisabling):
        for compound in self._compounds.iterate_all_compounds():
            compound.link(self._compounds)
            policy.process(compound)
        for flask in self._flasks.iterate_all_flasks():
            flask.link(self._flasks)
            policy.process(flask)

    def check_all_reactions(self, enabled: bool):
        for reaction in self._reactions.iterate_all_reactions():
            reaction.link(self._reactions)
            assert reaction.explore() is enabled
            assert reaction.analyze() is enabled
        self.check_all_steps(enabled)

    def check_all_steps(self, enabled: bool):
        for step in self._elementary_steps.iterate_all_elementary_steps():
            step.link(self._elementary_steps)
            assert step.explore() is enabled
            assert step.analyze() is enabled

    def check_all_aggregates(self, enabled: bool):
        for compound in self._compounds.iterate_all_compounds():
            compound.link(self._compounds)
            assert compound.explore() is enabled
            assert compound.analyze() is enabled
        for flask in self._flasks.iterate_all_flasks():
            flask.link(self._flasks)
            assert flask.explore() is enabled
            assert flask.analyze() is enabled
        self.check_all_structures(enabled)

    def check_all_structures(self, enabled: bool):
        for structure in self._structures.iterate_all_structures():
            structure.link(self._structures)
            if structure.get_label() == db.Label.TS_OPTIMIZED:
                continue
            assert structure.explore() is enabled
            assert structure.analyze() is enabled

    def test_disable_none(self):
        self.set_up_database()
        reaction_policy = ReactionDisabling()
        reaction_policy.initialize_collections(self._manager)
        self.run_reaction_policy(reaction_policy)
        self.check_all_reactions(enabled=True)

        step_policy = StepDisabling()
        self.run_step_policy(step_policy)
        self.check_all_steps(enabled=True)

    def test_disable_all_reactions(self):
        self.set_up_database()
        reaction_policy = DisableAllReactions()
        reaction_policy.initialize_collections(self._manager)
        self.run_reaction_policy(reaction_policy)
        self.check_all_reactions(enabled=False)

    def test_disable_all_steps(self):
        self.set_up_database()
        step_policy = DisableAllSteps()
        self.run_step_policy(step_policy)
        self.check_all_steps(enabled=False)

    def test_disable_reactions_by_job(self):
        self.set_up_database()
        black_list = ["disable-me", "really-disable-me"]
        reaction_policy = DisableReactionByJob(black_list=black_list)
        reaction_policy.initialize_collections(self._manager)
        self.run_reaction_policy(reaction_policy, "still-enabled")
        self.check_all_reactions(enabled=True)
        self.run_reaction_policy(reaction_policy, black_list[0])
        self.check_all_reactions(enabled=False)

    def test_disable_steps_by_job(self):
        self.set_up_database()
        black_list = ["disable-me", "really-disable-me"]
        step_policy = DisableStepByJob(black_list=black_list)
        self.run_step_policy(step_policy, "still-enabled")
        self.check_all_steps(enabled=True)
        self.run_step_policy(step_policy, black_list[0])
        self.check_all_steps(enabled=False)

    def test_disable_all_aggregates(self):
        self.set_up_database()
        policy = DisableAllAggregates()
        policy.initialize_collections(self._manager)
        self.run_aggregate_policy(policy)
        self.check_all_aggregates(False)

    def test_disable_all_steps_by_model(self):
        self.set_up_database()
        model = db_setup.get_fake_model()
        other_model = db.Model("some", "other", "model")
        step_policy = DisableAllStepsByModel(other_model)
        step_policy.initialize_collections(self._manager)
        self.run_step_policy(step_policy)
        self.check_all_steps(enabled=True)

        step_policy = DisableAllStepsByModel(model)
        step_policy.initialize_collections(self._manager)
        self.run_step_policy(step_policy)
        self.check_all_steps(enabled=False)
