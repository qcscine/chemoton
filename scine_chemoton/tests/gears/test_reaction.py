#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
from typing import List, Tuple
import os
import inspect
import unittest

# Local application tests imports
from scine_chemoton.gears import HoldsCollections
from .. import test_database_setup as db_setup

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application imports
from ...engine import Engine
from ...gears.reaction import BasicReactionHousekeeping
from ..resources import resources_root_path


class ReactionTests(unittest.TestCase, HoldsCollections):

    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "reactions", "compounds",
                                      "properties"]
        self.initialize_collections(manager)

    def tearDown(self) -> None:
        self._manager.wipe()

    def test_elementary_step_deduplication(self):
        manager = db_setup.get_clean_db(inspect.currentframe().f_code.co_name)
        self.custom_setup(manager)
        manager = self._manager
        steps = self._elementary_steps

        # set up 2 compounds
        c1_id, s1_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.USER_OPTIMIZED)
        c2_id, s2_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.MINIMUM_OPTIMIZED)

        # set up step between compounds
        step_one = self._add_new_regular_step([s1_id, s2_id], (70.0, 71.0))

        reaction_gear = BasicReactionHousekeeping()
        reaction_engine = Engine(manager.get_credentials(), fork=False)
        reaction_engine.set_gear(reaction_gear)
        reaction_engine.run(single=True)

        n_reactions = self._reactions.count(dumps({}))
        assert n_reactions == 1
        compound_one = db.Compound(c1_id, self._compounds)
        compound_two = db.Compound(c2_id, self._compounds)
        reaction_from_one = compound_one.get_reactions()
        reaction_from_two = compound_two.get_reactions()
        assert reaction_from_one[0] == reaction_from_two[0]

        # insert duplicate
        step_two = db.ElementaryStep(db.ID(), steps)
        step_two.create([s1_id], [s2_id])
        step_two.set_transition_state(step_one.get_transition_state())

        reaction_engine.run(single=True)
        n_reactions = self._reactions.count(dumps({}))
        assert n_reactions == 1
        reaction = db.Reaction(reaction_from_one[0], self._reactions)
        assert len(reaction.get_elementary_steps()) == 1
        assert not step_two.explore() and not step_two.analyze()

        # insert step with same structures and TS for the same reaction with different energy
        step_three = self._add_new_regular_step([s1_id, s2_id], (68.0, 69.0))

        # deactivate energy criterion
        reaction_gear.options.use_energy_deduplication = False
        reaction_engine.set_gear(reaction_gear)
        reaction_engine.run(single=True)

        # duplicated without energy criterion
        n_reactions = self._reactions.count(dumps({}))
        assert n_reactions == 1
        reaction = db.Reaction(compound_one.get_reactions()[0], self._reactions)
        assert len(reaction.get_elementary_steps()) == 1
        assert not step_three.explore() and not step_three.analyze()

        # no deduplication with energy criterion
        step_three.enable_exploration()
        step_three.enable_analysis()

        reaction_gear.options.use_energy_deduplication = True
        reaction_engine.set_gear(reaction_gear)
        reaction_engine.run(single=True)
        n_reactions = self._reactions.count(dumps({}))
        assert n_reactions == 1
        reaction = db.Reaction(compound_one.get_reactions()[0], self._reactions)
        assert len(reaction.get_elementary_steps()) == 2
        assert step_three.explore() and step_three.analyze()

    def test_barrierless_disable(self):
        manager = db_setup.get_clean_db(inspect.currentframe().f_code.co_name)
        self.custom_setup(manager)
        manager = self._manager

        # set up 4 compounds
        _, s1_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.USER_OPTIMIZED)
        _, s2_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.MINIMUM_OPTIMIZED)
        _, s3_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.USER_OPTIMIZED)
        _, s4_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.MINIMUM_OPTIMIZED)

        # set up one step each between two pair of structures
        step0_regular_0 = self._add_new_regular_step([s1_id, s2_id])
        step1_free_0 = self._add_new_barrierless_step([s3_id, s4_id])

        reaction_gear = BasicReactionHousekeeping()
        reaction_engine = Engine(manager.get_credentials(), fork=False)
        reaction_engine.set_gear(reaction_gear)
        reaction_engine.run(single=True)

        assert self._reactions.count(dumps({})) == 2
        assert step0_regular_0.explore() and step0_regular_0.analyze()
        assert step1_free_0.explore() and step1_free_0.analyze()

        # insert different step for the same reactions, but other types than before
        step0_free_0 = self._add_new_barrierless_step([s1_id, s2_id])
        step1_regular_0 = self._add_new_regular_step([s3_id, s4_id])

        # we are warned about conflicting types
        with self.assertWarns(Warning):
            reaction_engine.run(single=True)
        assert self._reactions.count(dumps({})) == 2
        for reaction in self._reactions.query_reactions("{}"):
            assert len(reaction.get_elementary_steps()) == 2
        # reaction 0
        assert step0_regular_0.explore() and step0_regular_0.analyze()
        assert not step0_free_0.explore() and not step0_free_0.analyze()
        # reaction 1
        assert step1_regular_0.explore() and step1_regular_0.analyze()
        assert not step1_free_0.explore() and not step1_free_0.analyze()

        # insert again alternating step type
        step0_regular_1 = self._add_new_regular_step([s1_id, s2_id])
        step1_free_1 = self._add_new_barrierless_step([s3_id, s4_id])
        # we are warned about conflicting types
        with self.assertWarns(Warning):
            reaction_engine.run(single=True)
        assert self._reactions.count(dumps({})) == 2
        for reaction in self._reactions.query_reactions("{}"):
            assert len(reaction.get_elementary_steps()) == 3
        # reaction 0
        assert step0_regular_0.explore() and step0_regular_0.analyze()
        assert step0_regular_1.explore() and step0_regular_1.analyze()
        assert not step0_free_0.explore() and not step0_free_0.analyze()
        # reaction 1
        assert step1_regular_0.explore() and step1_regular_0.analyze()
        assert not step1_free_0.explore() and not step1_free_0.analyze()
        assert not step1_free_1.explore() and not step1_free_1.analyze()

    def _add_new_regular_step(self, structures: List[db.ID], ts_energy_range: Tuple[float, float] = (70.0, 71.0)) \
            -> db.ElementaryStep:
        step = db.ElementaryStep(db.ID(), self._elementary_steps)
        step.create(*[[s] for s in structures])
        ts = db.Structure(db.ID(), self._structures)
        ts.create(os.path.join(resources_root_path(), "water.xyz"), 0, 1)
        ts.set_label(db.Label.TS_GUESS)
        ts.set_model(db.Model("FAKE", "FAKE", "F-AKE"))
        db_setup.add_random_energy(ts, ts_energy_range, self._properties)
        step.set_transition_state(ts.get_id())
        return step

    def _add_new_barrierless_step(self, structures: List[db.ID]) -> db.ElementaryStep:
        step = db.ElementaryStep(db.ID(), self._elementary_steps)
        step.create(*[[s] for s in structures])
        step.set_type(db.ElementaryStepType.BARRIERLESS)
        return step

    def test_elementary_step_inversion(self):
        import numpy as np
        manager = db_setup.get_clean_db(inspect.currentframe().f_code.co_name)
        self.custom_setup(manager)
        manager = self._manager
        structures = manager.get_collection("structures")

        # set up 2 compounds
        _, s1_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.USER_OPTIMIZED)
        _, s2_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.MINIMUM_OPTIMIZED)

        # set up step between compounds
        step_one = self._add_new_regular_step([s1_id, s2_id], (70.0, 71.0))
        step_two = self._add_new_regular_step([s2_id, s1_id], (70.0, 71.0))
        structure_two = db.Structure(s1_id, structures)

        knots = np.asarray([0.0, 0.7, 1.0])
        data = np.asarray([[0.1, 0.0, 0.0, 0.0], [0.2, 0.0, 0.1, 0.0], [0.1, 0.0, 0.3, 0.0]])
        ts_position = 0.7

        spline = utils.bsplines.TrajectorySpline(structure_two.get_atoms().elements, knots, data, ts_position)
        step_two.set_spline(spline)

        reaction_gear = BasicReactionHousekeeping()
        reaction_engine = Engine(manager.get_credentials(), fork=False)
        reaction_engine.set_gear(reaction_gear)
        reaction_engine.run(single=True)

        step_one_reactants = step_one.get_reactants(db.Side.BOTH)
        step_two_reactants = step_two.get_reactants(db.Side.BOTH)

        assert step_one_reactants[0] == step_two_reactants[0]
        assert step_one_reactants[1] == step_two_reactants[1]

        assert step_two.has_spline()
        new_spline = step_two.get_spline()
        assert new_spline.knots[0] == 0.0
        assert abs(new_spline.knots[1] - 0.3) < 1e-9
        assert abs(new_spline.knots[2] - 1.0) < 1e-9

        ref_data = np.flipud(np.asarray([[0.1, 0.0, 0.0, 0.0], [0.2, 0.0, 0.1, 0.0], [0.1, 0.0, 0.3, 0.0]]))
        assert np.sum(np.abs(ref_data - new_spline.data)) < 1e-9
        assert abs(new_spline.ts_position - 0.3) < 1e-9

        assert step_one.has_reaction()
        assert step_two.has_reaction()
        assert step_one.get_reaction() == step_two.get_reaction()

        manager.wipe()

    def test_elementary_step_structure_deduplication(self):
        manager = db_setup.get_clean_db(inspect.currentframe().f_code.co_name)
        self.custom_setup(manager)
        manager = self._manager
        structures = manager.get_collection("structures")

        # set up 2 compounds
        _, s1_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.USER_OPTIMIZED)
        a2_id, s2_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.MINIMUM_OPTIMIZED)
        _, duplicate_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.DUPLICATE)
        duplicate = db.Structure(duplicate_id, structures)
        duplicate.set_as_duplicate_of(s2_id)
        duplicate.set_aggregate(a2_id)
        step = self._add_new_regular_step([s1_id, duplicate_id], (70.0, 71.0))

        reaction_gear = BasicReactionHousekeeping()
        reaction_engine = Engine(manager.get_credentials(), fork=False)
        reaction_engine.set_gear(reaction_gear)
        reaction_engine.run(single=True)

        reactants = step.get_reactants(db.Side.BOTH)
        assert reactants[0][0] == s1_id
        assert reactants[1][0] == s2_id
