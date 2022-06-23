#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
import pytest
import random

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application tests imports
from .. import test_database_setup as db_setup

# Local application imports
from ...utilities.queries import identical_reaction, model_query
from ...engine import Engine
from ...gears.refinement import NetworkRefinement


def create_new_calculations(elementary_steps, calculations, pre_model):
    n_new_calcs = 0
    for step in elementary_steps.iterate_all_elementary_steps():
        n_new_calcs += 1
        step.link(elementary_steps)
        calculation = db.Calculation()
        calculation.link(calculations)
        calculation.create(pre_model, db.Job("some_react_job"), step.get_reactants(db.Side.LHS)[0])
        calculation.set_status(db.Status.COMPLETE)
        rc_keys = [
            "rc_x_alignment_0",
            "rc_x_alignment_1",
            "rc_x_rotation",
            "rc_x_spread",
            "rc_displacement",
        ]
        rc_settings = utils.ValueCollection({k: random.uniform(0.1, 2.0) for k in rc_keys})
        rc_settings.update(
            {
                "rc_spin_multiplicity": 1,
                "rc_molecular_charge": 0,
                "some_nt_setting": "something",
            }
        )
        calculation.set_settings(rc_settings)
        results = calculation.get_results()
        results.set_structures(step.get_reactants(db.Side.RHS)[1])
        results.add_elementary_step(step.id())
        calculation.set_results(results)
    return n_new_calcs


def test_incorrect_refinement_input():
    manager = db_setup.get_clean_db("chemoton_test_incorrect_refinement_input")
    refinement_gear = NetworkRefinement()
    refinement_gear.options.pre_refine_model = db.Model("something", "wrong", "because", "identical")
    refinement_gear.options.post_refine_model = db.Model("something", "wrong", "because", "identical")
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    with pytest.raises(RuntimeError):
        refinement_engine.run(single=True)

    refinement_gear.options.refinements["double_ended_new_connections"] = True
    refinement_engine.set_gear(refinement_gear)
    refinement_engine.run(single=True)  # should not fail

    refinement_gear.options.refinements["redo_single_points"] = True
    refinement_engine.set_gear(refinement_gear)
    with pytest.raises(RuntimeError):
        refinement_engine.run(single=True)


def test_if_refinement_runs_deactivated():
    n_compounds = 9
    n_reactions = 6
    max_r_per_c = 10
    max_n_products_per_r = 2
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_reactions,
        max_r_per_c,
        "chemoton_test_if_refinement_runs_deactivated",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")
    structures = manager.get_collection("structures")
    n_structures_before = structures.count(dumps({}))

    # enable all compounds
    for c in compounds.iterate_all_compounds():
        c.link(compounds)
        c.enable_exploration()

    # does nothing per default
    refinement_gear = NetworkRefinement()
    refinement_gear.options.pre_refine_model = db.Model("FAKE", "", "")
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(5):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    assert calculations.count(dumps({})) == 0  # random db does not have calculations


def test_refinement_with_wrong_model():
    n_compounds = 9
    n_reactions = 6
    max_r_per_c = 10
    max_n_products_per_r = 2
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_reactions,
        max_r_per_c,
        "chemoton_test_refinement_wrong_model",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")
    structures = manager.get_collection("structures")
    elementary_steps = manager.get_collection("elementary_steps")
    n_structures_before = structures.count(dumps({}))

    # enable all compounds
    for c in compounds.iterate_all_compounds():
        c.link(compounds)
        c.enable_exploration()

    added_calculations = create_new_calculations(elementary_steps, calculations, db.Model("FAKE", "", ""))
    pre_model = db.Model("NON_EXISTENT", "", "")
    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = NetworkRefinement()
    refinement_gear.options.pre_refine_model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.refinements = {
        "refine_single_points": True,
        "refine_optimizations": True,
        "double_ended_refinement": True,
        "double_ended_new_connections": True,
        "refine_single_ended_search": True,
        "refine_structures_and_irc": True,
    }
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(2):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    assert calculations.count(dumps({})) == added_calculations  # random db does not have calculations


def test_sp_refinement():
    n_compounds = 9
    n_reactions = 6
    max_r_per_c = 10
    max_n_products_per_r = 2
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_reactions,
        max_r_per_c,
        "chemoton_test_sp_refinement",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")
    structures = manager.get_collection("structures")
    elementary_steps = manager.get_collection("elementary_steps")
    n_structures_before = structures.count(dumps({}))

    # enable all compounds
    for c in compounds.iterate_all_compounds():
        c.link(compounds)
        c.enable_exploration()

    structures_to_refine = []
    for e in elementary_steps.iterate_all_elementary_steps():
        e.link(elementary_steps)
        reactants_products = e.get_reactants(db.Side.BOTH)
        transition_state_id = e.get_transition_state()
        structure_ids = reactants_products[0] + [transition_state_id] + reactants_products[1]
        for s_id in structure_ids:
            if s_id not in structures_to_refine:
                structures_to_refine.append(s_id)

    pre_model = db.Model("FAKE", "", "")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model)
    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = NetworkRefinement()
    refinement_gear.options.pre_refine_model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.refinements = {
        "refine_single_points": True,
        "refine_optimizations": False,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": False,
    }
    refinement_gear.options.max_barrier = 2001.0
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(2):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    # random db does not have calculations, now calculation for every structure
    assert calculations.count(dumps({})) == len(structures_to_refine) + added_calculations
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == len(structures_to_refine)
    calc = calculations.find(dumps({"$and": model_query(refine_model)}))
    calc.link(calculations)
    assert calc.get_job().order == refinement_gear.options.sp_job.order
    assert calc.get_settings() == refinement_gear.options.sp_job_settings


def test_barrier_screening():
    n_compounds = 9
    n_reactions = 6
    max_r_per_c = 10
    max_n_products_per_r = 2
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (101, 1000.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_reactions,
        max_r_per_c,
        "chemoton_test_barrier",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")
    structures = manager.get_collection("structures")
    elementary_steps = manager.get_collection("elementary_steps")
    n_structures_before = structures.count(dumps({}))

    # enable all compounds
    for c in compounds.iterate_all_compounds():
        c.link(compounds)
        c.enable_exploration()

    pre_model = db.Model("FAKE", "", "")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model)
    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = NetworkRefinement()
    refinement_gear.options.pre_refine_model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.refinements = {
        "refine_single_points": True,
        "refine_optimizations": False,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": False,
    }
    refinement_gear.options.max_barrier = 100.0
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(5):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    # random db does not have calculations, now calculation for every structure
    assert calculations.count(dumps({})) == added_calculations
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == 0


def test_opt_refinement():
    n_compounds = 9
    n_reactions = 6
    max_r_per_c = 10
    max_n_products_per_r = 2
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_reactions,
        max_r_per_c,
        "chemoton_test_opt_refinement",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")
    structures = manager.get_collection("structures")
    elementary_steps = manager.get_collection("elementary_steps")
    n_structures_before = structures.count(dumps({}))

    # enable all compounds
    for c in compounds.iterate_all_compounds():
        c.link(compounds)
        c.enable_exploration()

    structures_to_refine = []
    for e in elementary_steps.iterate_all_elementary_steps():
        e.link(elementary_steps)
        reactants_products = e.get_reactants(db.Side.BOTH)
        transition_state_id = e.get_transition_state()
        structure_ids = reactants_products[0] + [transition_state_id] + reactants_products[1]
        for s_id in structure_ids:
            if s_id not in structures_to_refine:
                structures_to_refine.append(s_id)

    pre_model = db.Model("FAKE", "", "")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model)
    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = NetworkRefinement()
    refinement_gear.options.pre_refine_model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.refinements = {
        "refine_single_points": False,
        "refine_optimizations": True,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": False,
    }
    refinement_gear.options.max_barrier = 2001.0
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(2):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    # random db does not have calculations, now calculation for every structure
    assert calculations.count(dumps({})) == len(structures_to_refine) + added_calculations
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == len(structures_to_refine)
    calc = calculations.find(dumps({"$and": model_query(refine_model)}))
    calc.link(calculations)
    structure = db.Structure(calc.get_structures()[0])
    structure.link(structures)
    target_job = (
        refinement_gear.options.tsopt_job
        if structure.get_label() == db.Label.TS_OPTIMIZED
        else refinement_gear.options.opt_job
    )
    target_settings = (
        refinement_gear.options.tsopt_job_settings
        if structure.get_label() == db.Label.TS_OPTIMIZED
        else refinement_gear.options.opt_job_settings
    )

    assert calc.get_job().order == target_job.order
    assert calc.get_settings() == target_settings


def test_sp_and_opt_refinement():
    n_compounds = 9
    n_reactions = 6
    max_r_per_c = 10
    max_n_products_per_r = 2
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_reactions,
        max_r_per_c,
        "chemoton_test_sp_and_opt_refinement",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")
    structures = manager.get_collection("structures")
    elementary_steps = manager.get_collection("elementary_steps")
    n_structures_before = structures.count(dumps({}))

    # enable all compounds
    for c in compounds.iterate_all_compounds():
        c.link(compounds)
        c.enable_exploration()

    structures_to_refine = []
    for e in elementary_steps.iterate_all_elementary_steps():
        e.link(elementary_steps)
        reactants_products = e.get_reactants(db.Side.BOTH)
        transition_state_id = e.get_transition_state()
        structure_ids = reactants_products[0] + [transition_state_id] + reactants_products[1]
        for s_id in structure_ids:
            if s_id not in structures_to_refine:
                structures_to_refine.append(s_id)

    pre_model = db.Model("FAKE", "", "")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model)
    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = NetworkRefinement()
    refinement_gear.options.pre_refine_model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.refinements = {
        "refine_single_points": True,
        "refine_optimizations": True,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": False,
    }
    refinement_gear.options.max_barrier = 2001.0
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(5):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    # random db does not have calculations, now calculation for every structure
    assert calculations.count(dumps({})) == 2 * len(structures_to_refine) + added_calculations
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == 2 * len(structures_to_refine)
    calc = calculations.find(dumps({"$and": model_query(refine_model)}))
    calc.link(calculations)
    assert calc.get_job().order in [
        refinement_gear.options.sp_job.order,
        refinement_gear.options.opt_job.order,
        refinement_gear.options.tsopt_job.order,
    ]


def test_double_ended_refinement():
    n_compounds = 9
    n_reactions = 6
    max_r_per_c = 10
    max_n_products_per_r = 2
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1  # must be kept so we know the number of expected calculations
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_reactions,
        max_r_per_c,
        "chemoton_test_double_ended_refinement",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")
    structures = manager.get_collection("structures")
    elementary_steps = manager.get_collection("elementary_steps")
    n_structures_before = structures.count(dumps({}))

    # enable all compounds
    for c in compounds.iterate_all_compounds():
        c.link(compounds)
        c.enable_exploration()

    # create calculation for each step
    pre_model = db.Model("FAKE", "", "")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model)
    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = NetworkRefinement()
    refinement_gear.options.pre_refine_model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.refinements = {
        "refine_single_points": False,
        "refine_optimizations": False,
        "double_ended_refinement": True,
        "double_ended_new_connections": False,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": False,
    }
    refinement_gear.options.max_barrier = 2001.0
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(5):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    # random db does not have calculations, made calculation for every step, now should have double the calculations
    assert max_steps_per_r == 1
    assert calculations.count(dumps({})) == n_reactions + added_calculations
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == n_reactions
    calc = calculations.random_select_calculations(1)[0]
    calc.link(calculations)
    while calc.get_model() != refine_model:
        calc = calculations.random_select_calculations(1)[0]
        calc.link(calculations)
    assert calc.get_job().order == refinement_gear.options.double_ended_job.order


def test_double_ended_new():
    n_compounds = 9
    n_reactions = 6
    max_r_per_c = 10
    max_n_products_per_r = 2
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_reactions,
        max_r_per_c,
        "chemoton_test_double_ended_new",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")
    structures = manager.get_collection("structures")
    reactions = manager.get_collection("reactions")
    n_structures_before = structures.count(dumps({}))

    # enable all compounds
    for c in compounds.iterate_all_compounds():
        c.link(compounds)
        c.enable_exploration()

    pre_model = db.Model("FAKE", "", "")
    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = NetworkRefinement()
    refinement_gear.options.pre_refine_model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.refinements = {
        "refine_single_points": False,
        "refine_optimizations": False,
        "double_ended_refinement": False,
        "double_ended_new_connections": True,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": False,
    }
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(5):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    # in random db all structures are water --> all same PES --> every pair that is not already connected,
    # should have calculation now
    expected_calculations = 0
    for c_i in compounds.iterate_all_compounds():
        c_i.link(compounds)
        for c_j in compounds.iterate_all_compounds():
            c_j.link(compounds)
            if c_i.id() >= c_j.id():  # avoid self pair and double counting
                continue
            if not identical_reaction([c_i.id()], [c_j.id()], reactions):
                expected_calculations += len(c_i.get_structures()) * len(c_j.get_structures())

    assert calculations.count(dumps({})) == expected_calculations
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == expected_calculations
    calc = calculations.random_select_calculations(1)[0]
    calc.link(calculations)
    assert calc.get_job().order == refinement_gear.options.double_ended_job.order
    assert calc.get_settings() == refinement_gear.options.double_ended_job_settings


def test_single_ended_refinement():
    n_compounds = 9
    n_reactions = 6
    max_r_per_c = 10
    max_n_products_per_r = 2
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1  # must be kept so we know the number of expected calculations
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_reactions,
        max_r_per_c,
        "chemoton_test_single_ended_refinement",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")
    structures = manager.get_collection("structures")
    elementary_steps = manager.get_collection("elementary_steps")
    n_structures_before = structures.count(dumps({}))

    # enable all compounds
    for c in compounds.iterate_all_compounds():
        c.link(compounds)
        c.enable_exploration()

    # create calculation for each step
    pre_model = db.Model("FAKE", "", "")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model)
    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = NetworkRefinement()
    refinement_gear.options.pre_refine_model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.refinements = {
        "refine_single_points": False,
        "refine_optimizations": False,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": True,
        "refine_structures_and_irc": False,
    }
    refinement_gear.options.max_barrier = 2001.0
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(5):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    # random db does not have calculations, made calculation for every step, now should have double the calculations
    assert max_steps_per_r == 1
    assert calculations.count(dumps({})) == n_reactions + added_calculations
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == n_reactions
    calc = calculations.random_select_calculations(1)[0]
    calc.link(calculations)
    while calc.get_model() != refine_model:
        calc = calculations.random_select_calculations(1)[0]
        calc.link(calculations)
    assert calc.get_job().order == refinement_gear.options.single_ended_job.order
    assert "some_nt_setting" not in calc.get_settings().keys()
    assert "rc_molecular_charge" in calc.get_settings().keys()


def test_refine_structures_and_irc():
    n_compounds = 9
    n_reactions = 6
    max_r_per_c = 10
    max_n_products_per_r = 2
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (0.1, 200)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_reactions,
        max_r_per_c,
        "chemoton_test_refine_structures_and_irc",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")
    structures = manager.get_collection("structures")
    n_structures_before = structures.count(dumps({}))

    # enable all compounds
    for c in compounds.iterate_all_compounds():
        c.link(compounds)
        c.enable_exploration()

    # create calculation for each step
    pre_model = db.Model("FAKE", "", "")
    elementary_steps = manager.get_collection("elementary_steps")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model)

    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = NetworkRefinement()
    refinement_gear.options.pre_refine_model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.refinements = {
        "refine_single_points": False,
        "refine_optimizations": False,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": True,
    }
    refinement_gear.options.max_barrier = 2001
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(5):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    # random db does not have calculations, now calculation for every structure
    assert calculations.count(dumps({})) == added_calculations + n_reactions
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == n_reactions
    calc = calculations.random_select_calculations(1)[0]
    calc.link(calculations)
    while calc.get_model() != refine_model:
        calc = calculations.random_select_calculations(1)[0]
        calc.link(calculations)
    assert calc.get_job().order == refinement_gear.options.single_ended_step_refinement_job.order
