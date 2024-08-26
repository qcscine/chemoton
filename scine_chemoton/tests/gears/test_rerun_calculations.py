#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
import random

# Third party imports
import scine_database as db
from scine_database.queries import model_query
from scine_database import test_database_setup as db_setup
import scine_utilities as utils

# Local application tests imports
from ...gears.rerun_calculations import RerunCalculations
from ...engine import Engine


def create_new_calculations(elementary_steps, calculations, pre_model, job, old_settings):
    n_new_calcs = 0
    new_calculation_ids = list()
    for i, step in enumerate(elementary_steps.iterate_all_elementary_steps()):
        n_new_calcs += 1
        step.link(elementary_steps)
        calculation = db.Calculation()
        calculation.link(calculations)
        calculation.create(pre_model, job, step.get_reactants(db.Side.LHS)[0])
        calculation.set_status(db.Status.FAILED)
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
                "spin_multiplicity": 1,
                "molecular_charge": 0,
                "some_other_setting": "something",
                "random_settings": i
            }
        )
        rc_settings.update(old_settings)
        calculation.set_settings(rc_settings)
        calculation.set_comment("FAKE comment")
        new_calculation_ids.append(calculation.id())
    return new_calculation_ids


def test_calculation_creation():
    n_compounds = 10
    n_flasks = 0
    n_reactions = 8
    max_r_per_c = 10
    max_n_products_per_r = 2
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
        "chemoton_test_calculation_creation_rerun",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    calculations = manager.get_collection("calculations")
    elementary_steps = manager.get_collection("elementary_steps")

    old_model = db.Model("FAKE", "FAKE", "FAKE")
    new_model = db.Model("Fu", "Bar", "")
    old_job = db.Job("some_job")
    new_job = db.Job("some_other_job")
    old_settings = {
        "some_settings": "some_value"
    }
    new_settings = {
        "some_settings": "some_other_value",
        "a_new_setting": 5,
    }
    added_calculations = len(create_new_calculations(elementary_steps, calculations, old_model, old_job, old_settings))
    rerun_gear = RerunCalculations()
    rerun_gear.options.legacy_existence_check = True
    rerun_gear.options.old_job = old_job
    rerun_gear.options.new_job = new_job
    rerun_gear.options.old_job_settings = old_settings
    rerun_gear.options.new_job_settings = new_settings
    rerun_gear.options.model = old_model
    rerun_gear.options.new_model = new_model
    rerun_gear.options.comment_filter = ["FAKE comment", "some other comment"]
    rerun_gear.options.old_status = "failed"
    rerun_gear.options.change_model = True

    rerun_engine = Engine(manager.get_credentials(), fork=False)
    rerun_engine.set_gear(rerun_gear)
    rerun_engine.run(single=True)
    rerun_engine.run(single=True)

    assert calculations.count(dumps({})) == added_calculations * 2
    assert calculations.count(dumps({"$and": model_query(old_model)})) == added_calculations
    assert calculations.count(dumps({"$and": model_query(new_model)})) == added_calculations
    assert calculations.count(dumps({"job.order": old_job.order})) == added_calculations
    assert calculations.count(dumps({"job.order": new_job.order})) == added_calculations
    assert calculations.count(dumps({"settings.some_settings": "some_other_value"})) == added_calculations
    assert calculations.count(dumps({"settings.a_new_setting": 5})) == added_calculations

    rerun_gear2 = RerunCalculations()
    rerun_gear2.options.legacy_existence_check = True
    rerun_gear2.options.old_job = old_job
    rerun_gear2.options.new_job = new_job
    rerun_gear2.options.old_job_settings = old_settings
    rerun_gear2.options.new_job_settings = new_settings
    rerun_gear2.options.model = old_model
    rerun_gear2.options.new_model = new_model
    rerun_gear2.options.comment_filter = ["FAKE comment", "some other comment"]
    rerun_gear2.options.old_status = "failed"
    rerun_gear2.options.change_model = True

    rerun_engine2 = Engine(manager.get_credentials(), fork=False)
    rerun_engine2.set_gear(rerun_gear2)
    rerun_engine2.run(single=True)
    rerun_engine2.run(single=True)

    assert calculations.count(dumps({})) == added_calculations * 2
    assert calculations.count(dumps({"$and": model_query(old_model)})) == added_calculations
    assert calculations.count(dumps({"$and": model_query(new_model)})) == added_calculations
    assert calculations.count(dumps({"job.order": old_job.order})) == added_calculations
    assert calculations.count(dumps({"job.order": new_job.order})) == added_calculations
    assert calculations.count(dumps({"settings.some_settings": "some_other_value"})) == added_calculations
    assert calculations.count(dumps({"settings.a_new_setting": 5})) == added_calculations

    # Check if the cache is reset correctly.
    third_job = db.Job("some_third_job")
    rerun_gear2.options.new_job = third_job
    rerun_engine2.run(single=True)
    assert calculations.count(dumps({})) == added_calculations * 3
    assert calculations.count(dumps({"job.order": third_job.order})) == added_calculations

    manager.wipe()


def test_settings_specification():
    n_compounds = 10
    n_flasks = 0
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
        n_flasks,
        n_reactions,
        max_r_per_c,
        "chemoton_test_settings_specification_rerun",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    calculations = manager.get_collection("calculations")
    elementary_steps = manager.get_collection("elementary_steps")

    old_model = db.Model("FAKE", "FAKE", "FAKE")
    new_model = db.Model("Fu", "Bar", "")
    old_job = db.Job("some_job")
    new_job = db.Job("some_other_job")
    old_settings = {
        "some_settings": "some_value"
    }
    new_settings = {
        "some_settings": "some_other_value",
        "a_new_setting": 5,
    }
    added_calculations = len(create_new_calculations(elementary_steps, calculations, old_model, old_job, old_settings))
    old_settings["Setting_not_in_calculations"] = "value"
    rerun_gear = RerunCalculations()
    rerun_gear.options.old_job = old_job
    rerun_gear.options.new_job = new_job
    rerun_gear.options.old_job_settings = old_settings
    rerun_gear.options.new_job_settings = new_settings
    rerun_gear.options.model = old_model
    rerun_gear.options.new_model = new_model
    rerun_gear.options.comment_filter = ["FAKE comment", "some other comment"]
    rerun_gear.options.old_status = "failed"
    rerun_gear.options.change_model = True

    rerun_engine = Engine(manager.get_credentials(), fork=False)
    rerun_engine.set_gear(rerun_gear)
    rerun_engine.run(single=True)

    assert calculations.count(dumps({})) == added_calculations
    manager.wipe()


def test_duplication():
    n_compounds = 10
    n_flasks = 0
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
        n_flasks,
        n_reactions,
        max_r_per_c,
        "chemoton_test_duplication_rerun",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    calculations = manager.get_collection("calculations")
    elementary_steps = manager.get_collection("elementary_steps")

    old_model = db.Model("FAKE", "FAKE", "FAKE")
    new_model = old_model
    old_job = db.Job("some_job")
    new_job = old_job
    old_settings = {
        "some_settings": "some_value"
    }
    new_settings = old_settings
    added_calculations = len(create_new_calculations(elementary_steps, calculations, old_model, old_job, old_settings))
    rerun_gear = RerunCalculations()
    rerun_gear.options.old_job = old_job
    rerun_gear.options.new_job = new_job
    rerun_gear.options.old_job_settings = old_settings
    rerun_gear.options.new_job_settings = new_settings
    rerun_gear.options.model = old_model
    rerun_gear.options.new_model = new_model
    rerun_gear.options.comment_filter = ["FAKE comment", "some other comment"]
    rerun_gear.options.old_status = "failed"
    rerun_gear.options.change_model = True

    rerun_engine = Engine(manager.get_credentials(), fork=False)
    rerun_engine.set_gear(rerun_gear)
    rerun_engine.run(single=True)

    assert calculations.count(dumps({})) == added_calculations
    manager.wipe()


def test_comment():
    n_compounds = 10
    n_flasks = 0
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
        n_flasks,
        n_reactions,
        max_r_per_c,
        "chemoton_test_comment_rerun",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    calculations = manager.get_collection("calculations")
    elementary_steps = manager.get_collection("elementary_steps")

    old_model = db.Model("FAKE", "FAKE", "FAKE")
    new_model = db.Model("Fu", "Bar", "")
    old_job = db.Job("some_job")
    new_job = db.Job("some_other_job")
    old_settings = {
        "some_settings": "some_value"
    }
    new_settings = {
        "some_settings": "some_other_value",
        "a_new_setting": 5,
    }
    added_calculations = len(create_new_calculations(elementary_steps, calculations, old_model, old_job, old_settings))
    rerun_gear = RerunCalculations()
    rerun_gear.options.old_job = old_job
    rerun_gear.options.new_job = new_job
    rerun_gear.options.old_job_settings = old_settings
    rerun_gear.options.new_job_settings = new_settings
    rerun_gear.options.model = old_model
    rerun_gear.options.new_model = new_model
    rerun_gear.options.comment_filter = ["FUBAR", "some other comment"]
    rerun_gear.options.old_status = "failed"
    rerun_gear.options.change_model = True

    rerun_engine = Engine(manager.get_credentials(), fork=False)
    rerun_engine.set_gear(rerun_gear)
    rerun_engine.run(single=True)

    assert calculations.count(dumps({})) == added_calculations
    manager.wipe()


def test_status():
    n_compounds = 10
    n_flasks = 0
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
        n_flasks,
        n_reactions,
        max_r_per_c,
        "chemoton_test_status_rerun",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    calculations = manager.get_collection("calculations")
    elementary_steps = manager.get_collection("elementary_steps")

    old_model = db.Model("FAKE", "FAKE", "FAKE")
    new_model = db.Model("Fu", "Bar", "")
    old_job = db.Job("some_job")
    new_job = db.Job("some_other_job")
    old_settings = {
        "some_settings": "some_value"
    }
    new_settings = {
        "some_settings": "some_other_value",
        "a_new_setting": 5,
    }
    added_calculations = len(create_new_calculations(elementary_steps, calculations, old_model, old_job, old_settings))
    rerun_gear = RerunCalculations()
    rerun_gear.options.old_job = old_job
    rerun_gear.options.new_job = new_job
    rerun_gear.options.old_job_settings = old_settings
    rerun_gear.options.new_job_settings = new_settings
    rerun_gear.options.model = old_model
    rerun_gear.options.new_model = new_model
    rerun_gear.options.comment_filter = ["FAKE comment", "some other comment"]
    rerun_gear.options.old_status = "completed"
    rerun_gear.options.change_model = True

    rerun_engine = Engine(manager.get_credentials(), fork=False)
    rerun_engine.set_gear(rerun_gear)
    rerun_engine.run(single=True)

    assert calculations.count(dumps({})) == added_calculations
    manager.wipe()


def test_calculation_creation_id_list():
    n_compounds = 10
    n_flasks = 0
    n_reactions = 8
    max_r_per_c = 10
    max_n_products_per_r = 2
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
        "chemoton_test_calculation_creation_rerun",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    calculations = manager.get_collection("calculations")
    elementary_steps = manager.get_collection("elementary_steps")

    old_model = db.Model("FAKE", "FAKE", "FAKE")
    new_model = db.Model("Fu", "Bar", "")
    old_job = db.Job("some_job")
    new_job = db.Job("some_other_job")
    old_settings = {
        "some_settings": "some_value"
    }
    new_settings = {
        "some_settings": "some_other_value",
        "a_new_setting": 5,
    }
    added_calculation_ids = create_new_calculations(elementary_steps, calculations, old_model, old_job, old_settings)
    rerun_gear = RerunCalculations()
    rerun_gear.options.old_job = old_job
    rerun_gear.options.new_job = new_job
    rerun_gear.options.old_job_settings = old_settings
    rerun_gear.options.new_job_settings = new_settings
    rerun_gear.options.model = old_model
    rerun_gear.options.new_model = new_model
    rerun_gear.options.comment_filter = ["FAKE comment", "some other comment"]
    rerun_gear.options.old_status = "failed"
    rerun_gear.options.change_model = True
    assert len(added_calculation_ids) > 0
    random_calculation = added_calculation_ids[0]
    rerun_gear.options.calculation_id_list = [random_calculation]

    rerun_engine = Engine(manager.get_credentials(), fork=False)
    rerun_engine.set_gear(rerun_gear)
    rerun_engine.run(single=True)
    rerun_engine.run(single=True)

    n_added_calculations = len(added_calculation_ids)
    assert calculations.count(dumps({})) == n_added_calculations + 1
    assert calculations.count(dumps({"$and": model_query(old_model)})) == n_added_calculations
    assert calculations.count(dumps({"$and": model_query(new_model)})) == 1
    assert calculations.count(dumps({"job.order": old_job.order})) == n_added_calculations
    assert calculations.count(dumps({"job.order": new_job.order})) == 1
    assert calculations.count(dumps({"settings.some_settings": "some_other_value"})) == 1
    assert calculations.count(dumps({"settings.a_new_setting": 5})) == 1

    rerun_gear2 = RerunCalculations()
    rerun_gear2.options.old_job = old_job
    rerun_gear2.options.new_job = new_job
    rerun_gear2.options.old_job_settings = old_settings
    rerun_gear2.options.new_job_settings = new_settings
    rerun_gear2.options.model = old_model
    rerun_gear2.options.new_model = new_model
    rerun_gear2.options.comment_filter = ["FAKE comment", "some other comment"]
    rerun_gear2.options.old_status = "failed"
    rerun_gear2.options.change_model = True
    rerun_gear2.options.calculation_id_list = [random_calculation]

    rerun_engine2 = Engine(manager.get_credentials(), fork=False)
    rerun_engine2.set_gear(rerun_gear2)
    rerun_engine2.run(single=True)
    rerun_engine2.run(single=True)

    assert calculations.count(dumps({})) == n_added_calculations + 1
    assert calculations.count(dumps({"$and": model_query(old_model)})) == n_added_calculations
    assert calculations.count(dumps({"$and": model_query(new_model)})) == 1
    assert calculations.count(dumps({"job.order": old_job.order})) == n_added_calculations
    assert calculations.count(dumps({"job.order": new_job.order})) == 1
    assert calculations.count(dumps({"settings.some_settings": "some_other_value"})) == 1
    assert calculations.count(dumps({"settings.a_new_setting": 5})) == 1

    # Check if the cache is reset correctly.
    third_job = db.Job("some_third_job")
    rerun_gear2.options.new_job = third_job
    rerun_gear2.options.calculation_id_list = None
    rerun_engine2.run(single=True)
    assert calculations.count(dumps({})) == n_added_calculations * 2 + 1
    assert calculations.count(dumps({"job.order": third_job.order})) == n_added_calculations

    manager.wipe()


def test_remove_settings():
    n_compounds = 10
    n_flasks = 0
    n_reactions = 8
    max_r_per_c = 10
    max_n_products_per_r = 2
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
        "chemoton_test_calculation_creation_rerun_removed_settings",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    calculations = manager.get_collection("calculations")
    elementary_steps = manager.get_collection("elementary_steps")

    old_model = db.Model("FAKE", "FAKE", "FAKE")
    new_model = db.Model("Fu", "Bar", "")
    old_job = db.Job("some_job")
    new_job = db.Job("some_other_job")
    old_settings = {
        "some_settings": "some_value",
        "setting_to_be_removed": 10,
        "another_setting": 42.0,
    }
    new_settings = {
        "some_settings": "some_other_value",
        "a_new_setting": 5,
    }
    added_calculation_ids = create_new_calculations(elementary_steps, calculations, old_model, old_job, old_settings)
    rerun_gear = RerunCalculations()
    rerun_gear.options.old_settings_to_remove = ["setting_to_be_removed", "another_setting"]
    rerun_gear.options.old_job = old_job
    rerun_gear.options.new_job = new_job
    rerun_gear.options.old_job_settings = old_settings
    rerun_gear.options.new_job_settings = new_settings
    rerun_gear.options.model = old_model
    rerun_gear.options.new_model = new_model
    rerun_gear.options.comment_filter = ["FAKE comment", "some other comment"]
    rerun_gear.options.old_status = "failed"
    rerun_gear.options.change_model = True
    assert len(added_calculation_ids) > 0
    random_calculation = added_calculation_ids[0]
    rerun_gear.options.calculation_id_list = [random_calculation]

    rerun_engine = Engine(manager.get_credentials(), fork=False)
    rerun_engine.set_gear(rerun_gear)
    rerun_engine.run(single=True)
    rerun_engine.run(single=True)

    n_added_calculations = len(added_calculation_ids)
    assert calculations.count(dumps({})) == n_added_calculations + 1
    assert calculations.count(dumps({"$and": model_query(old_model)})) == n_added_calculations
    assert calculations.count(dumps({"$and": model_query(new_model)})) == 1
    assert calculations.count(dumps({"job.order": old_job.order})) == n_added_calculations
    assert calculations.count(dumps({"job.order": new_job.order})) == 1
    assert calculations.count(dumps({"settings.some_settings": "some_other_value"})) == 1
    assert calculations.count(dumps({"settings.a_new_setting": 5})) == 1
    assert calculations.count(dumps({"settings.setting_to_be_removed": 10})) == n_added_calculations
    assert calculations.count(dumps({"settings.another_setting": 42.0})) == n_added_calculations

    manager.wipe()
