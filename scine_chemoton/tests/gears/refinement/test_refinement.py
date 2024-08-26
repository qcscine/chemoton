#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
import pytest
import random
from typing import Union

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application tests imports
from scine_database.queries import identical_reaction, model_query
from scine_database.energy_query_functions import get_energy_for_structure
from scine_database import test_database_setup as db_setup

# Local application imports
from scine_chemoton.engine import Engine
from scine_chemoton.gears.network_refinement.reaction_based_refinement import ReactionBasedRefinement
from scine_chemoton.gears.network_refinement.calculation_based_refinement import CalculationBasedRefinement
from scine_chemoton.filters.elementary_step_filters import ElementaryStepBarrierFilter, StopDuringExploration
from scine_chemoton.filters.reaction_filters import ReactionIDFilter
from scine_chemoton.filters.reaction_filters import StopDuringExploration as StopReactionFilter
from scine_chemoton.utilities.model_combinations import ModelCombination
from scine_chemoton.gears.network_refinement.disabling import DisableAllSteps, DisableAllReactions
from scine_chemoton.gears.network_refinement.enabling import ApplyToAllStepsInReaction, FilteredStepEnabling
from scine_chemoton.filters.elementary_step_filters import ConsistentEnergyModelFilter
from scine_chemoton.filters.reaction_filters import ReactionHasStepWithModel
from scine_chemoton.utilities.db_object_wrappers.wrapper_caches import MultiModelCacheFactory
from scine_chemoton.gears.network_refinement.enabling import (
    EnableJobSpecificCalculations,
    EnableAllStructures,
    AggregateEnabling
)


def create_new_calculations(elementary_steps, calculations, pre_model, structures):
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
        reactants = step.get_reactants(db.Side.BOTH)
        calculation.set_structures(reactants[0])
        results = calculation.get_results()
        results.set_structures(reactants[1] + [step.get_transition_state()])
        results.add_elementary_step(step.id())
        calculation.set_results(results)
        lhs_struc_ids = step.get_reactants(db.Side.LHS)[0]
        for s_id in lhs_struc_ids:
            structure = db.Structure(s_id, structures)
            structure.add_calculation(calculation.job.order, calculation.id())
    return n_new_calcs


def create_fake_optimization_calculations(elementary_steps, post_model, sides, job, settings, calculations, structures):
    s_id_list = list()
    for step in elementary_steps.iterate_all_elementary_steps():
        step.link(elementary_steps)
        reactants = step.get_reactants(sides)
        all_structure_ids = reactants[0] + reactants[1]
        for s_id in all_structure_ids:
            if s_id not in s_id_list:
                calculation = db.Calculation()
                calculation.link(calculations)
                calculation.create(post_model, job, [s_id])
                calculation.set_settings(settings)
                calculation.set_status(db.Status.COMPLETE)
                results = calculation.get_results()
                results.add_structure(s_id)
                calculation.set_results(results)
                structure = db.Structure(s_id, structures)
                structure.add_calculation(calculation.job.order, calculation.id())
                structure.set_graph("masm_cbor_graph", "asdfghjkl")
                structure.set_graph("masm_decision_list", "")
                s_id_list.append(s_id)
    return len(s_id_list)


@pytest.mark.filterwarnings("ignore:.+is not implemented by default:UserWarning")
def test_incorrect_refinement_input():
    manager = db_setup.get_clean_db("chemoton_test_incorrect_refinement_input")
    refinement_gear = CalculationBasedRefinement()
    refinement_gear.options.model = db.Model("something", "wrong", "because", "identical")
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
    # Cleaning
    manager.wipe()


def test_if_refinement_runs_deactivated():
    n_compounds = 10
    n_flasks = 0
    n_reactions = 5
    max_r_per_c = 10
    max_n_products_per_r = 3
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_flasks,
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

    # does nothing per default
    refinement_gear = CalculationBasedRefinement()
    refinement_gear.options.model = db.Model("FAKE", "FAKE", "F-AKE")
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(5):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    assert calculations.count(dumps({})) == 0  # random db does not have calculations
    # Cleaning
    manager.wipe()


@pytest.mark.filterwarnings("ignore:.+is not implemented by default:UserWarning")
def test_refinement_with_wrong_model():
    n_compounds = 10
    n_flasks = 0
    n_reactions = 5
    max_r_per_c = 10
    max_n_products_per_r = 3
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_flasks,
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

    added_calculations = create_new_calculations(elementary_steps, calculations, db.Model("FAKE", "FAKE", "F-AKE"),
                                                 structures)
    pre_model = db.Model("NON_EXISTENT", "FAKE", "F-AKE")
    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = CalculationBasedRefinement()
    refinement_gear.options.model = pre_model
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
    # Cleaning
    manager.wipe()


def test_sp_refinement():
    n_compounds = 10
    n_flasks = 0
    n_reactions = 5
    max_r_per_c = 10
    max_n_products_per_r = 3
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_flasks,
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

    structures_to_refine = []
    for e in elementary_steps.iterate_all_elementary_steps():
        e.link(elementary_steps)
        reactants_products = e.get_reactants(db.Side.BOTH)
        transition_state_id = e.get_transition_state()
        structure_ids = reactants_products[0] + [transition_state_id] + reactants_products[1]
        for s_id in structure_ids:
            if s_id not in structures_to_refine:
                structures_to_refine.append(s_id)
    pre_model = db_setup.get_fake_model()
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model, structures)
    assert added_calculations == calculations.count(dumps({}))

    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = CalculationBasedRefinement()
    refinement_gear.options.model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.only_electronic_energies = True
    refinement_gear.options.refinements = {
        "refine_single_points": True,
        "refine_optimizations": False,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": False,
    }
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
    # Cleaning
    manager.wipe()


def test_barrier_screening():
    n_compounds = 10
    n_flasks = 0
    n_reactions = 5
    max_r_per_c = 10
    max_n_products_per_r = 3
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (101, 1000.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_flasks,
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

    pre_model = db.Model("FAKE", "FAKE", "F-AKE")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model, structures)
    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = CalculationBasedRefinement()
    refinement_gear.options.model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.refinements = {
        "refine_single_points": True,
        "refine_optimizations": False,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": False,
    }
    refinement_gear.elementary_step_filter = ElementaryStepBarrierFilter(100.0, ModelCombination(pre_model),
                                                                         only_electronic_energies=True)
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(5):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    # random db does not have calculations, now calculation for every structure
    assert calculations.count(dumps({})) == added_calculations
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == 0
    # Cleaning
    manager.wipe()


@pytest.mark.filterwarnings("ignore:.+is not implemented by default:UserWarning")
def test_opt_refinement():
    n_compounds = 10
    n_flasks = 0
    n_reactions = 5
    max_r_per_c = 10
    max_n_products_per_r = 3
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_flasks,
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

    structures_to_refine = []
    for e in elementary_steps.iterate_all_elementary_steps():
        e.link(elementary_steps)
        reactants_products = e.get_reactants(db.Side.BOTH)
        transition_state_id = e.get_transition_state()
        structure_ids = reactants_products[0] + [transition_state_id] + reactants_products[1]
        for s_id in structure_ids:
            if s_id not in structures_to_refine:
                structures_to_refine.append(s_id)

    pre_model = db.Model("FAKE", "FAKE", "F-AKE")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model, structures)
    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = CalculationBasedRefinement()
    refinement_gear.options.model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.only_electronic_energies = True
    refinement_gear.options.refinements = {
        "refine_single_points": False,
        "refine_optimizations": True,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": False,
    }
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
    # Cleaning
    manager.wipe()


@pytest.mark.filterwarnings("ignore:.+is not implemented by default:UserWarning")
def test_sp_and_opt_refinement():
    n_compounds = 10
    n_flasks = 0
    n_reactions = 5
    max_r_per_c = 10
    max_n_products_per_r = 3
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_flasks,
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

    structures_to_refine = []
    for e in elementary_steps.iterate_all_elementary_steps():
        e.link(elementary_steps)
        reactants_products = e.get_reactants(db.Side.BOTH)
        transition_state_id = e.get_transition_state()
        structure_ids = reactants_products[0] + [transition_state_id] + reactants_products[1]
        for s_id in structure_ids:
            if s_id not in structures_to_refine:
                structures_to_refine.append(s_id)

    pre_model = db.Model("FAKE", "FAKE", "F-AKE")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model, structures)
    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = CalculationBasedRefinement()
    refinement_gear.options.model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.only_electronic_energies = True
    refinement_gear.options.refinements = {
        "refine_single_points": True,
        "refine_optimizations": True,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": False,
    }
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
    # Cleaning
    manager.wipe()


@pytest.mark.filterwarnings("ignore:.+is not implemented by default:UserWarning")
def test_double_ended_new():
    n_compounds = 10
    n_flasks = 0
    n_reactions = 5
    max_r_per_c = 10
    max_n_products_per_r = 3
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_flasks,
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

    pre_model = db.Model("FAKE", "FAKE", "F-AKE")
    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = CalculationBasedRefinement()
    refinement_gear.options.model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.only_electronic_energies = True
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
            if not identical_reaction([c_i.id()], [c_j.id()], [db.CompoundOrFlask.COMPOUND],
                                      [db.CompoundOrFlask.COMPOUND], reactions):
                expected_calculations += len(c_i.get_structures()) * len(c_j.get_structures())

    assert calculations.count(dumps({})) == expected_calculations
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == expected_calculations
    calc = calculations.random_select_calculations(1)[0]
    calc.link(calculations)
    assert calc.get_job().order == refinement_gear.options.double_ended_job.order
    assert calc.get_settings() == refinement_gear.options.double_ended_job_settings
    # Cleaning
    manager.wipe()


def test_single_ended_refinement():
    n_compounds = 10
    n_flasks = 0
    n_reactions = 5
    max_r_per_c = 10
    max_n_products_per_r = 3
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1  # must be kept so we know the number of expected calculations
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_flasks,
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

    # create calculation for each step
    pre_model = db.Model("FAKE", "FAKE", "F-AKE")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model, structures)
    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = CalculationBasedRefinement()
    n_opt_calculations = create_fake_optimization_calculations(elementary_steps, refine_model, db.Side.LHS,
                                                               refinement_gear.options.opt_job,
                                                               refinement_gear.options.opt_job_settings,
                                                               calculations, structures)
    refinement_gear.options.model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.only_electronic_energies = True
    refinement_gear.step_disabling = DisableAllSteps()
    refinement_gear.options.refinements = {
        "refine_single_points": False,
        "refine_optimizations": False,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": True,
        "refine_structures_and_irc": False,
    }
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(5):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    # random db does not have calculations, made calculation for every step, now should have double the calculations
    assert max_steps_per_r == 1
    assert calculations.count(dumps({})) == n_reactions + added_calculations + n_opt_calculations
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == n_reactions + n_opt_calculations
    calc = calculations.random_select_calculations(1)[0]
    calc.link(calculations)
    while calc.get_model() != refine_model or calc.get_job().order == refinement_gear.options.opt_job.order:
        calc = calculations.random_select_calculations(1)[0]
        calc.link(calculations)
    assert calc.get_job().order == refinement_gear.options.single_ended_job.order
    assert "some_nt_setting" not in calc.get_settings().keys()
    assert "rc_molecular_charge" in calc.get_settings().keys()

    for step in elementary_steps.iterate_all_elementary_steps():
        step.link(elementary_steps)
        assert not step.analyze()
        assert not step.explore()

    # Cleaning
    manager.wipe()


def test_refine_structures_and_irc():
    n_compounds = 10
    n_flasks = 0
    n_reactions = 5
    max_r_per_c = 10
    max_n_products_per_r = 3
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (0.1, 200)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_flasks,
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
    MultiModelCacheFactory().clear()
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")
    structures = manager.get_collection("structures")
    n_structures_before = structures.count(dumps({}))

    # create calculation for each step
    pre_model = db.Model("FAKE", "FAKE", "F-AKE")
    elementary_steps = manager.get_collection("elementary_steps")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model, structures)

    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = CalculationBasedRefinement()
    n_opt_calculations = create_fake_optimization_calculations(elementary_steps, refine_model, db.Side.LHS,
                                                               refinement_gear.options.opt_job,
                                                               refinement_gear.options.opt_job_settings,
                                                               calculations, structures)

    refinement_gear.options.model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.only_electronic_energies = True
    refinement_gear.options.refinements = {
        "refine_single_points": False,
        "refine_optimizations": False,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": True,
    }
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(5):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    # random db does not have calculations, now calculation for every structure
    assert calculations.count(dumps({})) == added_calculations + n_reactions + n_opt_calculations
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == n_reactions + n_opt_calculations
    calc = calculations.random_select_calculations(1)[0]
    calc.link(calculations)
    while calc.get_model() != refine_model or calc.get_job().order == refinement_gear.options.opt_job.order:
        calc = calculations.random_select_calculations(1)[0]
        calc.link(calculations)
    assert calc.get_job().order == refinement_gear.options.single_ended_step_refinement_job.order
    # Cleaning
    manager.wipe()


def prepare_database():
    n_compounds = 10
    n_flasks = 0
    n_reactions = 5
    max_r_per_c = 10
    max_n_products_per_r = 3
    max_n_educts_per_r = 2
    max_s_per_c = 3
    max_steps_per_r = 3
    barrier_limits = (0.1, 200)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_flasks,
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

    # enable all compounds
    for c in compounds.iterate_all_compounds():
        c.link(compounds)
        c.enable_exploration()
    MultiModelCacheFactory().clear()
    return manager, n_reactions, n_compounds


def test_refine_structures_and_irc_reaction_based_loop():
    manager, n_reactions, n_compounds = prepare_database()
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")
    structures = manager.get_collection("structures")
    reactions = manager.get_collection("reactions")
    n_structures_before = structures.count(dumps({}))

    # create calculation for each step
    pre_model = db.Model("FAKE", "FAKE", "F-AKE")
    elementary_steps = manager.get_collection("elementary_steps")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model, structures)

    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = ReactionBasedRefinement()
    refinement_gear.reaction_disabling = DisableAllReactions()
    n_opt_calculations = create_fake_optimization_calculations(elementary_steps, refine_model, db.Side.LHS,
                                                               refinement_gear.options.opt_job,
                                                               refinement_gear.options.opt_job_settings,
                                                               calculations, structures)
    refinement_gear.options.model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.only_electronic_energies = True
    refinement_gear.options.refinements = {
        "refine_single_points": False,
        "refine_optimizations": False,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": True,
    }
    refinement_gear.options.transition_state_energy_window = 0.0
    refinement_gear.reaction_filter = StopReactionFilter(["some_react_job"])
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)

    calculation = db.Calculation()
    calculation.link(calculations)
    calculation.create(pre_model, db.Job("some_react_job"), [])
    calculation.set_status(db.Status.PENDING)

    for _ in range(5):
        refinement_engine.run(single=True)

    assert calculations.count(dumps({})) == added_calculations + n_opt_calculations + 1

    calculation.set_status(db.Status.COMPLETE)
    for _ in range(5):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before

    assert calculations.count(dumps({})) == added_calculations + n_reactions + n_opt_calculations + 1
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == n_reactions + n_opt_calculations
    calc = calculations.random_select_calculations(1)[0]
    calc.link(calculations)
    while calc.get_model() != refine_model or calc.get_job().order == refinement_gear.options.opt_job.order:
        calc = calculations.random_select_calculations(1)[0]
        calc.link(calculations)
    assert calc.get_job().order == refinement_gear.options.single_ended_step_refinement_job.order

    for reaction in reactions.iterate_all_reactions():
        reaction.link(reactions)
        assert not reaction.analyze()
        assert not reaction.explore()

    # Cleaning
    manager.wipe()


def test_refine_manual_reaction_selection():
    manager, _, _ = prepare_database()
    calculations = manager.get_collection("calculations")
    structures = manager.get_collection("structures")
    reactions = manager.get_collection("reactions")

    # create calculation for each step
    pre_model = db.Model("FAKE", "FAKE", "F-AKE")
    elementary_steps = manager.get_collection("elementary_steps")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model, structures)

    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = ReactionBasedRefinement()
    n_opt_calculations = create_fake_optimization_calculations(elementary_steps, refine_model, db.Side.LHS,
                                                               refinement_gear.options.opt_job,
                                                               refinement_gear.options.opt_job_settings,
                                                               calculations, structures)
    refinement_gear.options.model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.only_electronic_energies = True
    refinement_gear.options.refinements = {
        "refine_single_points": False,
        "refine_optimizations": False,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": True,
    }
    random_reaction = reactions.get_one_reaction(dumps({}))
    refinement_gear.reaction_filter = (StopReactionFilter(["some_react_job"])
                                       and ReactionIDFilter([random_reaction.id()]))

    refinement_gear.options.transition_state_energy_window = 0.0
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)

    refinement_engine.run(single=True)
    assert calculations.count(dumps({})) == added_calculations + n_opt_calculations + 1
    assert calculations.count(dumps({"job.order": "scine_step_refinement"})) == 1

    # test old result enabling
    old_calc = calculations.find(dumps({"job.order": "scine_step_refinement"}))
    old_calc.link(calculations)
    _, results_s_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
    results_structure = db.Structure(results_s_id, structures)
    results_structure.disable_analysis()
    old_calc.disable_exploration()
    old_calc.disable_analysis()
    results = old_calc.get_results()
    results.set_structures([results_s_id])
    old_calc.set_results(results)
    old_calc.set_status(db.Status.COMPLETE)
    refinement_gear_2 = ReactionBasedRefinement()
    refinement_gear_2.options = refinement_gear.options
    refinement_gear_2.reaction_filter = refinement_gear.reaction_filter
    refinement_gear_2.result_enabling = EnableJobSpecificCalculations(
        refine_model, "scine_step_refinement",
        structure_enabling_policy=EnableAllStructures(aggregate_enabling_policy=AggregateEnabling()))
    refinement_engine_2 = Engine(manager.get_credentials(), fork=False)
    refinement_engine_2.set_gear(refinement_gear_2)

    refinement_engine_2.run(single=True)
    assert calculations.count(dumps({"job.order": "scine_step_refinement"})) == 1
    print("Check calc id", old_calc.id())
    assert old_calc.analyze()
    assert old_calc.explore()
    assert results_structure.analyze()
    assert results_structure.explore()
    # Cleaning
    manager.wipe()


def test_reaction_based_single_points():
    manager, _, _ = prepare_database()
    calculations = manager.get_collection("calculations")
    structures = manager.get_collection("structures")
    reactions = manager.get_collection("reactions")
    compounds = manager.get_collection("compounds")
    flasks = manager.get_collection("flasks")
    properties = manager.get_collection("properties")

    # create calculation for each step
    pre_model = db_setup.get_fake_model()

    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = ReactionBasedRefinement()
    refinement_gear.options.model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.only_electronic_energies = True
    refinement_gear.options.refinements = {
        "refine_single_points": True,
        "refine_optimizations": False,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": False,
    }
    random_reactions = reactions.random_select_reactions(3)
    refinement_gear.reaction_filter = StopReactionFilter(["some_react_job"]) and ReactionIDFilter(
        [r.id() for r in random_reactions])

    refinement_gear.options.transition_state_energy_window = 0.0
    refinement_gear.options.aggregate_energy_window = 0.0
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    unique_agg = set()
    for r in random_reactions:
        r.link(reactions)
        reactants = r.get_reactants(db.Side.BOTH)
        unique_agg.update([a_id.string() for a_id in reactants[0] + reactants[1]])
    refinement_engine.run(single=True)

    assert calculations.count(dumps({})) == len(unique_agg) + len(random_reactions)  # sp for each agg. and ts
    assert calculations.count(dumps({"job.order": "scine_single_point"})) == len(unique_agg) + len(random_reactions)
    for calculation in calculations.iterate_calculations(dumps({"job.order": "scine_single_point"})):
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        structure = db.Structure(calculation.get_structures()[0], structures)
        assert structure.exists()
        e_base = get_energy_for_structure(structure, "electronic_energy", pre_model, structures, properties)
        if structure.label != db.Label.TS_OPTIMIZED:
            aggregate: Union[db.Compound, db.Flask] = db.Compound(structure.get_aggregate(), compounds)
            if not aggregate.exists():
                aggregate = db.Flask(structure.get_aggregate(), flasks)
            for s_id in aggregate.get_structures():
                s = db.Structure(s_id, structures)
                assert e_base <= get_energy_for_structure(s, "electronic_energy", pre_model, structures, properties)

    # Cleaning
    manager.wipe()


def test_single_ended_refinement_set_up_optimizations():
    n_compounds = 10
    n_flasks = 0
    n_reactions = 5
    max_r_per_c = 10
    max_n_products_per_r = 3
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1  # must be kept to know the number of expected calculations
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_flasks,
        n_reactions,
        max_r_per_c,
        "chemoton_test_single_ended_refinement_set_up_optimizations",
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
    pre_model = db.Model("FAKE", "FAKE", "F-AKE")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model, structures)
    s_id_set = set()
    for calculation in calculations.iterate_all_calculations():
        calculation.link(calculations)
        s_id_set = s_id_set.union([s_id.string() for s_id in calculation.get_structures()])
        calculation.set_status(db.Status.PENDING)
    n_unique_lhs = len(s_id_set)

    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = CalculationBasedRefinement()
    refinement_gear.options.model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.only_electronic_energies = True
    refinement_gear.options.refinements = {
        "refine_single_points": False,
        "refine_optimizations": False,
        "double_ended_refinement": False,
        "double_ended_new_connections": False,
        "refine_single_ended_search": True,
        "refine_structures_and_irc": False,
    }
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(5):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    assert max_steps_per_r == 1
    assert calculations.count(dumps({})) == added_calculations

    for calculation in calculations.iterate_all_calculations():
        calculation.link(calculations)
        calculation.set_status(db.Status.COMPLETE)

    for _ in range(5):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert structures.count(dumps({})) == n_structures_before
    assert max_steps_per_r == 1
    assert n_unique_lhs > 0
    assert calculations.count(dumps({})) == added_calculations + n_unique_lhs
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == n_unique_lhs
    # Cleaning
    manager.wipe()


def test_double_ended_refinement():
    n_compounds = 10
    n_flasks = 0
    n_reactions = 5
    max_r_per_c = 10
    max_n_products_per_r = 3
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1  # must be kept to know the number of expected calculations
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_flasks,
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

    # create calculation for each step
    pre_model = db.Model("FAKE", "FAKE", "F-AKE")
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model, structures)
    for calculation in calculations.iterate_all_calculations():
        calculation.link(calculations)
        calculation.set_status(db.Status.PENDING)

    for step in elementary_steps.iterate_all_elementary_steps():
        step.link(elementary_steps)
        if step.get_type() == db.ElementaryStepType.REGULAR:
            reactant_id = step.get_reactants(db.Side.LHS)[0][0]
            s = db.Structure(reactant_id, structures)
            atoms = s.get_atoms()
            rpi = utils.bsplines.ReactionProfileInterpolation()
            rpi.append_structure(atoms, 1.0)
            rpi.append_structure(atoms, 1.0)
            step.set_spline(rpi.spline(2, 1))

    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = CalculationBasedRefinement()
    refinement_gear.options.model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.only_electronic_energies = True
    refinement_gear.elementary_step_filter = StopDuringExploration(orders_to_wait_for=["some_react_job"])
    refinement_gear.options.refinements = {
        "refine_single_points": False,
        "refine_optimizations": False,
        "double_ended_refinement": True,
        "double_ended_new_connections": False,
        "refine_single_ended_search": False,
        "refine_structures_and_irc": False,
    }
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)
    for _ in range(5):
        refinement_engine.run(single=True)

    assert compounds.count(dumps({})) == n_compounds
    assert compounds.count(dumps({})) == n_compounds
    assert max_steps_per_r == 1
    assert structures.count(dumps({})) == n_structures_before
    assert calculations.count(dumps({})) == added_calculations

    for calculation in calculations.iterate_all_calculations():
        calculation.link(calculations)
        calculation.set_status(db.Status.COMPLETE)

    for _ in range(5):
        refinement_engine.run(single=True)

    assert structures.count(dumps({})) == n_structures_before + n_reactions * 2
    assert compounds.count(dumps({})) == n_compounds
    assert calculations.count(dumps({})) == added_calculations + n_reactions
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == n_reactions
    # Cleaning
    manager.wipe()


def test_enable_instead_of_refinement():
    manager, n_reactions, _ = prepare_database()
    calculations = manager.get_collection("calculations")
    structures = manager.get_collection("structures")
    reactions = manager.get_collection("reactions")
    n_structures_before = structures.count(dumps({}))
    elementary_steps = manager.get_collection("elementary_steps")
    properties = manager.get_collection("properties")

    # create calculation for each step
    pre_model = db_setup.get_fake_model()
    added_calculations = create_new_calculations(elementary_steps, calculations, pre_model, structures)

    refine_model = db.Model("a", "b", "c", "d")
    refine_model.solvation = "something"
    refinement_gear = ReactionBasedRefinement()
    n_opt_calculations = create_fake_optimization_calculations(elementary_steps, refine_model, db.Side.LHS,
                                                               refinement_gear.options.opt_job,
                                                               refinement_gear.options.opt_job_settings,
                                                               calculations, structures)
    refinement_gear.options.model = pre_model
    refinement_gear.options.post_refine_model = refine_model
    refinement_gear.options.only_electronic_energies = True
    refinement_gear.options.refinements["refine_structures_and_irc"] = True
    refinement_gear.options.transition_state_energy_window = 0.0
    refinement_gear.reaction_enabling = ApplyToAllStepsInReaction(
        FilteredStepEnabling(ConsistentEnergyModelFilter(refine_model)))
    refinement_gear.reaction_validation = ReactionHasStepWithModel(refine_model, check_only_energies=True)
    refinement_engine = Engine(manager.get_credentials(), fork=False)
    refinement_engine.set_gear(refinement_gear)

    random_reactions = reactions.random_select_reactions(2)
    n_skipped = len(random_reactions)
    for r in random_reactions:
        r.link(reactions)
        step_ids = r.get_elementary_steps()
        step = db.ElementaryStep(step_ids[-1], elementary_steps)
        reactants = step.get_reactants(db.Side.BOTH)
        s_ids = reactants[0] + reactants[1]
        if step.has_transition_state():
            s_ids.append(step.get_transition_state())
        for s_id in s_ids:
            db_setup.add_random_energy(db.Structure(s_id, structures), (-11.0, -10.0), properties, refine_model)

    for _ in range(5):
        refinement_engine.run(single=True)

    assert structures.count(dumps({})) == n_structures_before

    n_new_calculations = n_reactions - n_skipped
    assert calculations.count(dumps({})) == added_calculations + n_opt_calculations + n_new_calculations
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == n_opt_calculations + n_new_calculations

    refinement_gear.reaction_validation = ReactionHasStepWithModel(db.Model("Some", "third", "model"))

    for _ in range(5):
        refinement_engine.run(single=True)

    n_new_calculations = n_reactions
    assert calculations.count(dumps({})) == added_calculations + n_opt_calculations + n_new_calculations
    assert calculations.count(dumps({"$and": model_query(refine_model)})) == n_opt_calculations + n_new_calculations

    # Cleaning
    manager.wipe()
