#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import List, Tuple
import unittest
import os
import pytest
from json import dumps

# Third party imports
import scine_database as db

# Local application tests imports
from .. import test_database_setup as db_setup
from ...gears import HoldsCollections

# Local application imports
from ..resources import resources_root_path
from ...engine import Engine
from ...gears.kinetics import MinimalConnectivityKinetics, BasicBarrierHeightKinetics, MaximumFluxKinetics, \
    PathfinderKinetics


class KineticsTests(unittest.TestCase, HoldsCollections):

    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "reactions", "compounds", "flasks",
                                      "properties"]
        self.initialize_collections(manager)

    def tearDown(self) -> None:
        self._manager.wipe()

    def test_activation_all_compounds(self):
        n_compounds = 10
        n_reactions = 6
        max_r_per_c = 10
        max_n_products_per_r = 2
        max_n_educts_per_r = 2
        max_s_per_c = 1
        max_steps_per_r = 1
        barrier_limits = (0.1, 2000.0)
        n_inserts = 3
        n_flasks = 0
        manager = db_setup.get_random_db(
            n_compounds,
            n_flasks,
            n_reactions,
            max_r_per_c,
            "chemoton_test_activation_all_compounds",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)

        kinetics_gear = MinimalConnectivityKinetics()  # activate all compounds
        kinetics_gear.options.restart = True
        kinetics_engine = Engine(manager.get_credentials(), fork=False)
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(n_reactions):  # should be faster, but ensures that iterative procedure goes through
            kinetics_engine.run(single=True)

        selection = {"exploration_disabled": {"$ne": True}}
        assert self._compounds.count(dumps(selection)) == n_compounds

        kinetics_gear._disable_all_aggregates()
        selection = {"exploration_disabled": {"$ne": False}}
        assert self._compounds.count(dumps(selection)) == n_compounds

    def test_activation_all_compounds_and_flasks(self):
        n_compounds = 10
        n_reactions = 8
        max_r_per_c = 10
        max_n_products_per_r = 3
        max_n_educts_per_r = 2
        max_s_per_c = 1
        max_steps_per_r = 1
        barrier_limits = (0.1, 2000.0)
        n_inserts = 3
        n_flasks = 3
        manager = db_setup.get_random_db(
            n_compounds,
            n_flasks,
            n_reactions,
            max_r_per_c,
            "chemoton_test_activation_all_compounds",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)

        kinetics_gear = MinimalConnectivityKinetics()  # activate all compounds
        kinetics_gear.options.restart = True
        kinetics_engine = Engine(manager.get_credentials(), fork=False)
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(n_reactions):  # should be faster, but ensures that iterative procedure goes through
            kinetics_engine.run(single=True)

        selection = {"exploration_disabled": {"$ne": True}}
        assert self._compounds.count(dumps(selection)) == n_compounds
        assert self._flasks.count(dumps(selection)) == n_flasks

        kinetics_gear._disable_all_aggregates()
        selection = {"exploration_disabled": {"$ne": False}}
        assert self._compounds.count(dumps(selection)) == n_compounds
        assert self._flasks.count(dumps(selection)) == n_flasks

    def test_user_insertion_verification(self):
        manager = db_setup.get_clean_db("chemoton_test_user_insertion_verification")
        self.custom_setup(manager)
        compound = db.Compound(db_setup.insert_single_empty_structure_aggregate(manager, db.Label.USER_GUESS)[0])
        compound.link(self._compounds)

        kinetics_gear = MinimalConnectivityKinetics()  # activate all compounds
        kinetics_engine = Engine(manager.get_credentials(), fork=False)
        kinetics_engine.set_gear(kinetics_gear)
        kinetics_engine.run(single=True)

        assert compound.explore()

    def test_barrier_limit(self):
        manager = db_setup.get_clean_db("chemoton_test_barrier_limit")
        self.custom_setup(manager)
        # set up 2 compounds
        c1_id, s1_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.USER_GUESS)
        c2_id, s2_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_GUESS)

        # set up step between compounds
        step = db.ElementaryStep()
        step.link(self._elementary_steps)
        step.create([s1_id], [s2_id])

        # set up TS and energies
        db_setup.add_random_energy(db.Structure(s1_id, self._structures), (0.0, 1.0), self._properties)
        db_setup.add_random_energy(db.Structure(s2_id, self._structures), (50.0, 51.0), self._properties)
        ts = db.Structure(db_setup.insert_single_empty_structure_aggregate(manager, db.Label.TS_GUESS)[1])
        ts.link(self._structures)
        ts_prop_id = db_setup.add_random_energy(ts, (70.0, 71.0), self._properties)
        step.set_transition_state(ts.get_id())

        # set up reaction
        reaction = db.Reaction()
        reaction.link(self._reactions)
        reaction.create([c1_id], [c2_id])
        reaction.set_elementary_steps([step.get_id()])
        compound_1 = db.Compound(c1_id)
        compound_2 = db.Compound(c2_id)
        compound_1.link(self._compounds)
        compound_2.link(self._compounds)
        compound_1.set_reactions([reaction.get_id()])
        compound_2.set_reactions([reaction.get_id()])

        # run barrier filter gear
        kinetics_gear = BasicBarrierHeightKinetics()
        kinetics_gear.options.model = db.Model("FAKE", "FAKE", "F-AKE")
        kinetics_gear.options.restart = True
        kinetics_gear.options.max_allowed_barrier = 100.0
        kinetics_engine = Engine(manager.get_credentials(), fork=False)
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(2):
            kinetics_engine.run(single=True)

        assert compound_1.explore() and compound_2.explore()

        # make barrier too high
        ts_prop = db.NumberProperty(ts_prop_id)
        ts_prop.link(self._properties)
        ts_prop.set_data(110.0)

        # No change expected, will be run from cache
        kinetics_gear.options.restart = True
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(2):
            kinetics_engine.run(single=True)
        assert compound_1.explore() and compound_2.explore()

        # Reset cache -> change expected
        kinetics_gear.options.restart = True
        kinetics_gear.clear_cache()
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(2):
            kinetics_engine.run(single=True)
        assert compound_1.explore() and not compound_2.explore()

    def test_barrier_limit_with_barrierless(self):
        manager = db_setup.get_clean_db("chemoton_test_barrier_limit_barrierless")
        self.custom_setup(manager)
        # set up 2 compounds
        c1_id, s1_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.USER_GUESS)
        c2_id, s2_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_GUESS)
        c3_id, s3_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_GUESS)

        # set up steps between compounds
        step_down = db.ElementaryStep(db.ID(), self._elementary_steps)
        step_down.create([s1_id], [s2_id])
        step_down.set_type(db.ElementaryStepType.BARRIERLESS)

        step_up = db.ElementaryStep(db.ID(), self._elementary_steps)
        step_up.create([s2_id], [s3_id])
        step_up.set_type(db.ElementaryStepType.BARRIERLESS)

        db_setup.add_random_energy(db.Structure(s1_id, self._structures), (100.0, 101.0), self._properties)
        db_setup.add_random_energy(db.Structure(s2_id, self._structures), (10.0, 11.0), self._properties)
        e3_id = db_setup.add_random_energy(db.Structure(s3_id, self._structures), (20.0, 21.0), self._properties)

        # set up reaction
        reaction = db.Reaction(db.ID(), self._reactions)
        reaction.create([c1_id], [c2_id])
        reaction.set_elementary_steps([step_down.get_id()])
        compound_1 = db.Compound(c1_id, self._compounds)
        compound_2 = db.Compound(c2_id, self._compounds)
        compound_1.set_reactions([reaction.get_id()])
        compound_2.set_reactions([reaction.get_id()])

        reaction = db.Reaction(db.ID(), self._reactions)
        reaction.create([c2_id], [c3_id])
        reaction.set_elementary_steps([step_up.get_id()])
        compound_3 = db.Compound(c3_id, self._compounds)
        compound_2.add_reaction(reaction.get_id())
        compound_3.add_reaction(reaction.get_id())

        # run barrier filter gear
        kinetics_gear = BasicBarrierHeightKinetics()
        kinetics_gear.options.model = db.Model("FAKE", "FAKE", "F-AKE")
        kinetics_gear.options.restart = True
        kinetics_gear.options.max_allowed_barrier = 50.0
        kinetics_engine = Engine(manager.get_credentials(), fork=False)
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(3):
            kinetics_engine.run(single=True)

        assert compound_1.explore() and compound_2.explore() and compound_3.explore()

        # make barrier too high
        e3 = db.NumberProperty(e3_id, self._properties)
        e3.set_data(70.0)

        # No change expected, will be run from cache
        kinetics_gear.options.restart = True
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(3):
            kinetics_engine.run(single=True)
        assert compound_1.explore() and compound_2.explore() and compound_3.explore()

        # Reset cache -> change expected
        kinetics_gear.clear_cache()
        kinetics_gear.options.restart = True
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(3):
            kinetics_engine.run(single=True)
        assert compound_1.explore() and compound_2.explore() and not compound_3.explore()

    def test_barrier_with_random_network(self):
        # these numbers can be lowered if we want faster unit tests
        n_compounds = 50
        n_reactions = 20
        max_r_per_c = 10
        max_n_products_per_r = 4
        max_n_educts_per_r = 2
        max_s_per_c = 2
        max_steps_per_r = 1
        barrier_limits = (50.1, 9000.1)
        n_inserts = 5
        n_flasks = 0
        manager = db_setup.get_random_db(
            n_compounds,
            n_flasks,
            n_reactions,
            max_r_per_c,
            "chemoton_test_barrier_with_random_network",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)

        kinetics_gear = BasicBarrierHeightKinetics()
        kinetics_gear.options.model = db.Model("wrong", "model", "")
        kinetics_gear.options.restart = True
        kinetics_gear.options.max_allowed_barrier = 1e10
        kinetics_engine = Engine(manager.get_credentials(), fork=False)
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(n_reactions):
            kinetics_engine.run(single=True)

        # check if all compounds but the user inputs are still deactivated because of wrong model
        selection = {"exploration_disabled": {"$ne": True}}
        assert self._compounds.count(dumps(selection)) == n_inserts

        kinetics_gear.options.model = db.Model("FAKE", "FAKE", "F-AKE")
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(n_reactions):  # should be faster, but ensures that iterative procedure goes through
            kinetics_engine.run(single=True)

        # check if all compounds are activated
        assert self._compounds.count(dumps(selection)) == n_compounds

        # set barrier limit so low that all are too high
        kinetics_gear.options.max_allowed_barrier = 10
        kinetics_gear.options.restart = True
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(n_reactions):  # should be faster, but ensures that iterative procedure goes through
            kinetics_engine.run(single=True)

        # all barriers are now too high, so only user inserted compounds should be enabled
        selection = {"exploration_disabled": {"$ne": True}}
        assert self._compounds.count(dumps(selection)) == n_inserts

    def test_concentration_based_selection(self):
        manager = db_setup.get_clean_db("chemoton_test_concentration_based_selection")
        self.custom_setup(manager)
        # set up 2 compounds
        c1_id, s1_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.USER_GUESS)
        c2_id, s2_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_GUESS)

        # set up step between compounds
        step = db.ElementaryStep()
        step.link(self._elementary_steps)
        step.create([s1_id], [s2_id])

        # set up TS and energies
        s1 = db.Structure(s1_id, self._structures)
        s2 = db.Structure(s2_id, self._structures)
        db_setup.add_random_energy(s1, (0.0, 1.0), self._properties)
        db_setup.add_random_energy(s2, (50.0, 51.0), self._properties)
        ts = db.Structure(db_setup.insert_single_empty_structure_aggregate(manager, db.Label.TS_GUESS)[1],
                          self._structures)
        db_setup.add_random_energy(ts, (70.0, 71.0), self._properties)
        step.set_transition_state(ts.get_id())

        # set up reactions
        reaction = db.Reaction(db.ID(), self._reactions)
        reaction.create([c1_id], [c2_id])
        reaction.set_elementary_steps([step.get_id()])
        compound_1 = db.Compound(c1_id, self._compounds)
        compound_2 = db.Compound(c2_id, self._compounds)
        compound_1.set_reactions([reaction.get_id()])
        compound_2.set_reactions([reaction.get_id()])

        # set up concentration properties
        kinetics_gear = MaximumFluxKinetics()
        concentration_label = kinetics_gear.options.property_label
        flux_label = kinetics_gear.options.flux_property_label
        model = db.Model("FAKE", "FAKE", "F-AKE")
        conc_prop1 = db.NumberProperty.make(concentration_label, model, 100, self._properties)
        conc_prop2 = db.NumberProperty.make(concentration_label, model, 100, self._properties)
        flux_prop1 = db.NumberProperty.make(flux_label, model, 100, self._properties)
        s1.add_property(concentration_label, conc_prop1.id())
        s1.add_property(flux_label, flux_prop1.id())
        flux_prop1.set_structure(s1_id)

        # run barrier filter gear
        kinetics_gear.options.model = model
        kinetics_gear.options.restart = True
        kinetics_gear.options.min_allowed_concentration = 1.0
        kinetics_engine = Engine(manager.get_credentials(), fork=False)
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(2):
            kinetics_engine.run(single=True)

        assert compound_1.explore() and not compound_2.explore()

        # add concentration for the compound_2
        conc_prop2.set_structure(s2_id)
        s2.add_property(concentration_label, conc_prop2.id())

        kinetics_gear.options.restart = True
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(2):
            kinetics_engine.run(single=True)
        assert compound_1.explore() and compound_2.explore()

    def test_barrier_with_flasks_network(self):
        # these numbers can be lowered if we want faster unit tests
        n_compounds = 50
        n_reactions = 20
        max_r_per_c = 10
        max_n_products_per_r = 4
        max_n_educts_per_r = 2
        max_s_per_c = 2
        max_steps_per_r = 1
        barrier_limits = (50.1, 9000.1)
        n_inserts = 5
        n_flasks = 5
        manager = db_setup.get_random_db(
            n_compounds,
            n_flasks,
            n_reactions,
            max_r_per_c,
            "chemoton_test_barrier_with_random_network",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)

        kinetics_gear = BasicBarrierHeightKinetics()
        kinetics_gear.options.model = db.Model("wrong", "model", "")
        kinetics_gear.options.restart = True
        kinetics_gear.options.max_allowed_barrier = 1e10
        kinetics_engine = Engine(manager.get_credentials(), fork=False)
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(n_reactions):
            kinetics_engine.run(single=True)

        # check if all compounds but the user inputs are still deactivated because of wrong model
        selection = {"exploration_disabled": {"$ne": True}}
        assert self._compounds.count(dumps(selection)) == n_inserts

        kinetics_gear.options.model = db.Model("FAKE", "FAKE", "F-AKE")
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(n_reactions):  # should be faster, but ensures that iterative procedure goes through
            kinetics_engine.run(single=True)

        # check if all compounds are activated
        assert self._compounds.count(dumps(selection)) == n_compounds

        # set barrier limit so low that all are too high
        kinetics_gear.options.max_allowed_barrier = -1
        kinetics_gear.options.restart = True
        kinetics_engine.set_gear(kinetics_gear)
        for _ in range(n_reactions):  # should be faster, but ensures that iterative procedure goes through
            kinetics_engine.run(single=True)

        # all barriers are now too high, so only user inserted compounds should be enabled
        selection = {"exploration_disabled": {"$ne": True}}
        assert self._compounds.count(dumps(selection)) == n_inserts

    def _insert_single_elementary_step_reaction(self,
                                                s_ids: Tuple[List[db.ID],
                                                             List[db.ID]],
                                                s_energies: Tuple[List[float],
                                                                  List[float]],
                                                ts_energy,
                                                c_ids: Tuple[List[db.ID],
                                                             List[db.ID]],
                                                c_types: Tuple[List[db.CompoundOrFlask],
                                                               List[db.CompoundOrFlask]] = ([],
                                                                                            [])):
        step = db.ElementaryStep()
        step.link(self._elementary_steps)
        step.create(s_ids[0], s_ids[1])

        # add energies to structures
        for side in range(0, len(s_ids)):
            for s_id, s_energy in zip(s_ids[side], s_energies[side]):
                s_tmp = db.Structure(s_id, self._structures)
                if not s_tmp.has_property("electronic_energy"):
                    db_setup.add_random_energy(s_tmp, (s_energy, s_energy), self._properties)

        # set TS
        ts = db.Structure()
        ts.link(self._structures)
        ts.create(os.path.join(resources_root_path(), "water.xyz"), 0, 1)
        db_setup.add_random_energy(ts, (ts_energy, ts_energy), self._properties)
        ts.set_model(db.Model("FAKE", "FAKE", "F-AKE"))
        ts.set_label(db.Label.TS_OPTIMIZED)
        step.set_transition_state(ts.get_id())
        # set up reaction
        reaction = db.Reaction(db.ID(), self._reactions)
        reaction.create(c_ids[0], c_ids[1], c_types[0], c_types[1])
        reaction.set_elementary_steps([step.get_id()])

    @pytest.mark.filterwarnings("ignore:.+Not all start compounds in graph:UserWarning")
    def test_pathfinder_selection(self):
        # set up three reactions, one isolated; check if stuff is activated
        # for allow unsolved on false
        manager = db_setup.get_clean_db("chemoton_test_pathfinder_based_selection")
        self.custom_setup(manager)
        # Set up 1 compounds
        c1_id, s1_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.USER_OPTIMIZED)
        cmp_struct_dict = {1: {"c": c1_id, "s": s1_id}}
        # Create 4 compounds with 4 structures
        for i in range(2, 6):
            c_tmp_id, s_tmp_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
            cmp_struct_dict[i] = {"c": c_tmp_id, "s": s_tmp_id}
        f1_id, s1_flask_id = db_setup.insert_single_empty_structure_flask(manager, db.Label.COMPLEX_OPTIMIZED)
        # Insert Reaction A = B
        self._insert_single_elementary_step_reaction(([cmp_struct_dict[1]["s"]], [cmp_struct_dict[2]["s"]]),
                                                     ([-100.0], [-110.0]),
                                                     -70.0,
                                                     ([cmp_struct_dict[1]["c"]], [cmp_struct_dict[2]["c"]]))
        # Insert Reaction A = C
        self._insert_single_elementary_step_reaction(([cmp_struct_dict[1]["s"]], [cmp_struct_dict[3]["s"]]),
                                                     ([-100.0], [-90.0]),
                                                     -90.0,
                                                     ([cmp_struct_dict[1]["c"]], [cmp_struct_dict[3]["c"]]))
        # Insert Reaction Y = Z
        self._insert_single_elementary_step_reaction(([cmp_struct_dict[4]["s"]], [cmp_struct_dict[5]["s"]]),
                                                     ([-200.0], [-190.0]),
                                                     -185.0,
                                                     ([cmp_struct_dict[4]["c"]], [cmp_struct_dict[5]["c"]]))
        # Insert Reaction B + C = F
        self._insert_single_elementary_step_reaction(([cmp_struct_dict[2]["s"], cmp_struct_dict[3]["s"]],
                                                     [s1_flask_id]), ([-110.0, -90.0], [-220.0]), -270.0,
                                                     ([cmp_struct_dict[2]["c"], cmp_struct_dict[3]["c"]], [f1_id]),
                                                     ([], [db.CompoundOrFlask.FLASK]))

        # Set up gear
        ref_start_value = 0.5
        kinetics_gear = PathfinderKinetics()
        kinetics_gear.options.max_compound_cost = 10
        kinetics_gear.options.model = db.Model("FAKE", "FAKE", "F-AKE")
        kinetics_gear.options.structure_model = db.Model("FAKE", "FAKE", "F-AKE")
        # Insert compound with wrong structure model (not included in graph)
        c_blunt_id, s_blunt_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        s_blunt = db.Structure(s_blunt_id, self._structures)
        s_blunt.set_model(db.Model("fake", "fake", "f-ake"))
        kinetics_gear.options.start_conditions = {
            c_blunt_id.string(): ref_start_value,
        }
        # Setup engine
        kinetics_engine = Engine(manager.get_credentials(), fork=False)
        kinetics_engine.set_gear(kinetics_gear)
        # Run engine
        kinetics_engine.run(single=True)

        assert kinetics_gear.finder.graph_handler is None

        # Correct starting conditions
        kinetics_gear.options.start_conditions = {
            cmp_struct_dict[1]["c"].string(): ref_start_value,
        }
        kinetics_gear.options.allow_unsolved_compound_costs = False
        kinetics_gear.options.filter_negative_barriers = True
        # Run engine
        kinetics_engine.run(single=True)

        # Check, that all compounds are not active for exploration
        for key, value in cmp_struct_dict.items():
            c_tmp = db.Compound(value["c"], self._compounds)
            assert not c_tmp.explore()
            # Check, that start condition is set
            if key == 1:
                s_start = db.Structure(value["s"], self._structures)
                assert s_start.has_property("compound_cost")
                prop_id = s_start.get_properties("compound_cost")[-1]
                prop = db.NumberProperty(prop_id, self._properties)
                assert prop.get_data() == ref_start_value

        # Allow unsolved compound costs, enables compound 1 and 3
        kinetics_gear.options.allow_unsolved_compound_costs = True
        kinetics_engine.run(single=True)
        for key, value in cmp_struct_dict.items():
            if key == 1 or key == 3:
                c_tmp = db.Compound(value["c"], self._compounds)
                assert c_tmp.explore()
            else:
                c_tmp = db.Compound(value["c"], self._compounds)
                assert not c_tmp.explore()
        # Check that the flask is deactivated
        f1 = db.Flask(f1_id, self._flasks)
        assert not f1.explore()
        assert abs(kinetics_gear._old_ratio - 10 / 6) < 1e-12

        # Insert two new structures to activate trigger
        db_setup._fake_structure(c_tmp, self._structures, self._properties, False, (-180, -188))
        db_setup._fake_structure(c_tmp, self._structures, self._properties, False, (-180, -188))
        # Reset finder
        kinetics_gear.options.max_compound_cost = 0.0
        kinetics_gear.options.restart = True
        # Restart and allow only compound 1 to be active
        kinetics_engine.run(single=True)
        assert not kinetics_gear.options.restart
        for key, value in cmp_struct_dict.items():
            if key == 1:
                c_tmp = db.Compound(value["c"], self._compounds)
                assert c_tmp.explore()
            else:
                c_tmp = db.Compound(value["c"], self._compounds)
                assert not c_tmp.explore()
        # Check that the flask is deactivated
        f1 = db.Flask(f1_id, self._flasks)
        assert not f1.explore()
        # Check ratio to be overwritten
        assert (kinetics_gear._old_ratio - 12 / 6) < 1e-12


def test_concentration_based_selection_chained_flasks():
    manager = db_setup.get_clean_db("chemoton_test_concentration_based_selection_chained_flask")
    # Reaction setup
    # lhs_c_id -> lhs_f_id -> TS -> rhs_f_id -> rhs_c_id
    lhs_c_id, lhs_s_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.USER_OPTIMIZED)
    rhs_c_id, rhs_s_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
    lhs_f_id, lhs_comp_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.COMPLEX_GUESS)
    rhs_f_id, rhs_comp_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.COMPLEX_GUESS)

    steps = manager.get_collection("elementary_steps")
    structures = manager.get_collection("structures")
    properties = manager.get_collection("properties")
    reactions = manager.get_collection("reactions")
    compounds = manager.get_collection("compounds")
    flasks = manager.get_collection("flasks")

    # set up steps between aggregates
    step_barrierless_lhs = db.ElementaryStep()
    step_barrierless_lhs.link(steps)
    step_barrierless_lhs.create([lhs_s_id], [lhs_comp_id])
    step_barrierless_lhs.set_type(db.ElementaryStepType.BARRIERLESS)

    step_barrierless_rhs = db.ElementaryStep()
    step_barrierless_rhs.link(steps)
    step_barrierless_rhs.create([rhs_comp_id], [rhs_s_id])
    step_barrierless_rhs.set_type(db.ElementaryStepType.BARRIERLESS)

    step_central = db.ElementaryStep()
    step_central.link(steps)
    step_central.create([lhs_comp_id], [rhs_comp_id])

    # set up TS and energies
    lhs_comp_structure = db.Structure(lhs_comp_id, structures)
    rhs_comp_structure = db.Structure(rhs_comp_id, structures)
    lhs_s_structure = db.Structure(lhs_s_id, structures)
    rhs_s_structure = db.Structure(rhs_s_id, structures)
    db_setup.add_random_energy(lhs_comp_structure, (0.0, 1.0), properties)
    db_setup.add_random_energy(rhs_comp_structure, (0.0, 1.0), properties)
    db_setup.add_random_energy(lhs_s_structure, (0.0, 1.0), properties)
    db_setup.add_random_energy(rhs_s_structure, (0.0, 1.0), properties)
    ts = db.Structure(db_setup.insert_single_empty_structure_aggregate(manager, db.Label.TS_GUESS)[1], structures)
    db_setup.add_random_energy(ts, (70.0, 71.0), properties)
    step_central.set_transition_state(ts.get_id())

    # set up reactions
    reaction_barrierless_lhs = db.Reaction()
    reaction_barrierless_lhs.link(reactions)
    reaction_barrierless_lhs.create([lhs_c_id], [lhs_f_id], [db.CompoundOrFlask.COMPOUND], [db.CompoundOrFlask.FLASK])
    reaction_barrierless_lhs.set_elementary_steps([step_barrierless_lhs.get_id()])

    reaction_barrierless_rhs = db.Reaction()
    reaction_barrierless_rhs.link(reactions)
    reaction_barrierless_rhs.create([rhs_f_id], [rhs_c_id], [db.CompoundOrFlask.FLASK], [db.CompoundOrFlask.COMPOUND])
    reaction_barrierless_rhs.set_elementary_steps([step_barrierless_rhs.get_id()])

    reaction_central = db.Reaction()
    reaction_central.link(reactions)
    reaction_central.create([lhs_f_id], [rhs_f_id], [db.CompoundOrFlask.FLASK], [db.CompoundOrFlask.FLASK])
    reaction_central.set_elementary_steps([step_central.get_id()])

    lhs_flask = db.Flask(lhs_f_id, flasks)
    rhs_flask = db.Flask(rhs_f_id, flasks)
    lhs_compound = db.Compound(lhs_c_id, compounds)
    rhs_compound = db.Compound(rhs_c_id, compounds)
    lhs_compound.set_reactions([reaction_barrierless_lhs.id()])
    lhs_flask.set_reactions([reaction_barrierless_lhs.id(), reaction_central.id()])
    rhs_flask.set_reactions([reaction_central.id(), reaction_barrierless_rhs.id()])
    rhs_compound.set_reactions([reaction_barrierless_rhs.id()])

    # set up concentration properties
    model = db.Model("FAKE", "FAKE", "F-AKE")
    kinetics_gear = MaximumFluxKinetics()
    kinetics_gear.options.model = model
    concentration_label = kinetics_gear.options.property_label
    flux_label = kinetics_gear.options.flux_property_label
    conc_prop1 = db.NumberProperty.make(concentration_label, model, 100, properties)
    conc_prop2 = db.NumberProperty.make(concentration_label, model, 100, properties)
    flux_prop1 = db.NumberProperty.make(flux_label, model, 100, properties)
    flux_prop2 = db.NumberProperty.make(flux_label, model, 100, properties)
    lhs_s_structure.add_property(concentration_label, conc_prop1.id())
    lhs_s_structure.add_property(flux_label, flux_prop1.id())
    flux_prop1.set_structure(lhs_s_id)
    conc_prop1.set_structure(lhs_s_id)

    # run barrier filter gear
    kinetics_gear.options.model = model
    kinetics_gear.options.restart = True
    kinetics_gear.options.min_allowed_concentration = 1.0
    kinetics_engine = Engine(manager.get_credentials(), fork=False)
    kinetics_engine.set_gear(kinetics_gear)
    for _ in range(2):
        kinetics_engine.run(single=True)

    assert lhs_compound.explore() and not rhs_compound.explore() and not lhs_flask.explore() and not rhs_flask.explore()

    # add concentration for the compound_2
    conc_prop2.set_structure(rhs_s_id)
    rhs_s_structure.add_property(concentration_label, conc_prop2.id())
    flux_prop2.set_structure(rhs_s_id)
    rhs_s_structure.add_property(flux_label, flux_prop2.id())

    kinetics_gear.options.restart = True
    kinetics_engine.set_gear(kinetics_gear)
    for _ in range(2):
        kinetics_engine.run(single=True)
    print(lhs_compound.explore())
    print(rhs_compound.explore())
    print(lhs_flask.explore())
    print(rhs_flask.explore())
    assert lhs_compound.explore() and rhs_compound.explore() and not lhs_flask.explore() and not rhs_flask.explore()
