#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps

# Third party imports
import scine_database as db

# Local application tests imports
from ... import test_database_setup as db_setup
from ....engine import Engine
from ....gears.kinetic_modeling.kinetic_modeling import KineticModeling


def test_random_kinetic_model():
    n_compounds = 10
    n_flasks = 3
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
        "chemoton_test_random_kinetic_model",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    model = db.Model("FAKE", "FAKE", "F-AKE")
    structures = manager.get_collection("structures")
    properties = manager.get_collection("properties")
    calculations = manager.get_collection("calculations")
    # Set the starting concentrations for all compounds.
    for structure in structures.iterate_all_structures():
        structure.link(structures)
        con_prop = db.NumberProperty.make("start_concentration", model, 0.1, properties)
        con_prop.set_structure(structure.id())
        structure.add_property("start_concentration", con_prop.id())

    # Check the job-set up.
    kinetic_modeling_gear = KineticModeling()
    kinetic_modeling_gear.options.model = model
    kinetic_modeling_gear.options.elementary_step_interval = 4
    kinetic_modeling_gear.options.time_step = 1e-8
    kinetic_modeling_gear.options.solver = "cash_karp_5"
    kinetic_modeling_gear.options.batch_interval = 1000

    kinetic_modeling_engine = Engine(manager.get_credentials(), fork=False)
    kinetic_modeling_engine.set_gear(kinetic_modeling_gear)
    kinetic_modeling_engine.run(single=True)
    assert calculations.count(dumps({})) == 1
    kinetic_modeling_engine.run(single=True)
    assert calculations.count(dumps({})) == 1

    # Test whether the gear notices that the same job is already in the database.
    kinetic_modeling_gear2 = KineticModeling()
    kinetic_modeling_gear2.options.model = model
    kinetic_modeling_gear2.options.elementary_step_interval = 4
    kinetic_modeling_gear2.options.time_step = 1e-8
    kinetic_modeling_gear2.options.solver = "cash_karp_5"
    kinetic_modeling_gear2.options.batch_interval = 1000

    kinetic_modeling_engine2 = Engine(manager.get_credentials(), fork=False)
    kinetic_modeling_engine2.set_gear(kinetic_modeling_gear2)
    kinetic_modeling_engine2.run(single=True)
    assert calculations.count(dumps({})) == 1

    # Change the settings and test again.
    old_calc = calculations.find(dumps({}))
    old_calc.set_status(db.Status.COMPLETE)
    kinetic_modeling_gear3 = KineticModeling()
    kinetic_modeling_gear3.options.model = model
    kinetic_modeling_gear3.options.elementary_step_interval = 4
    kinetic_modeling_gear3.options.time_step = 1e-8
    kinetic_modeling_gear3.options.solver = "explicit_euler"
    kinetic_modeling_gear3.options.batch_interval = 1000

    # Add some concentration flux
    for structure in structures.iterate_all_structures():
        structure.link(structures)
        con_prop = db.NumberProperty.make("concentration_flux", model, 0.1, properties)
        con_prop.set_structure(structure.id())
        structure.add_property("concentration_flux", con_prop.id())

    kinetic_modeling_engine3 = Engine(manager.get_credentials(), fork=False)
    kinetic_modeling_engine3.set_gear(kinetic_modeling_gear3)
    kinetic_modeling_engine3.run(single=True)
    assert calculations.count(dumps({})) == 2

    # Cleaning
    manager.wipe()


def test_random_kinetic_model_sleepy():
    n_compounds = 10
    n_flasks = 2
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
        "chemoton_test_random_kinetic_model_sleepy",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    model = db.Model("FAKE", "FAKE", "F-AKE")
    structures = manager.get_collection("structures")
    properties = manager.get_collection("properties")
    calculations = manager.get_collection("calculations")
    # Set the starting concentrations for all compounds.
    for structure in structures.iterate_all_structures():
        structure.link(structures)
        con_prop = db.NumberProperty.make("start_concentration", model, 0.1, properties)
        con_prop.set_structure(structure.id())
        structure.add_property("start_concentration", con_prop.id())

    calculation = db.Calculation()
    calculation.link(calculations)
    calculation.create(model, db.Job("some_job"), [])
    calculation.set_status(db.Status.PENDING)

    # Check the job-set up.
    kinetic_modeling_gear = KineticModeling()
    kinetic_modeling_gear.options.model = model
    kinetic_modeling_gear.options.elementary_step_interval = 4
    kinetic_modeling_gear.options.time_step = 1e-8
    kinetic_modeling_gear.options.solver = "cash_karp_5"
    kinetic_modeling_gear.options.batch_interval = 1000
    kinetic_modeling_gear.options.sleeper_mode = True

    kinetic_modeling_engine = Engine(manager.get_credentials(), fork=False)
    kinetic_modeling_engine.set_gear(kinetic_modeling_gear)
    kinetic_modeling_engine.run(single=True)
    assert calculations.count(dumps({})) == 1
    kinetic_modeling_engine.run(single=True)
    assert calculations.count(dumps({})) == 1

    calculation.set_status(db.Status.COMPLETE)
    kinetic_modeling_engine.run(single=True)
    assert calculations.count(dumps({})) == 2

    # Cleaning
    manager.wipe()
