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
import scine_utilities as utils

# Local application tests imports
from ... import test_database_setup as db_setup
from ...test_database_setup import insert_single_empty_structure_aggregate
from ....engine import Engine
from ....gears.kinetic_modeling.kinetic_modeling import KineticModeling, KineticModelingJobFactory


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
    compounds = manager.get_collection("compounds")
    steps = manager.get_collection("elementary_steps")
    reactions = manager.get_collection("reactions")
    flasks = manager.get_collection("flasks")

    # Add three more reactions. Two barrierless and one regular one.
    # lhs_c_id -> lhs_f_id -> TS -> rhs_f_id -> rhs_c_id
    random_compounds = compounds.random_select_compounds(4)
    lhs_c_id_one = random_compounds[0].id()
    lhs_c_id_two = random_compounds[1].id()
    rhs_c_id_one = random_compounds[2].id()
    rhs_c_id_two = random_compounds[3].id()
    lhs_compound_one = db.Compound(lhs_c_id_one, compounds)
    lhs_compound_two = db.Compound(lhs_c_id_two, compounds)
    rhs_compound_one = db.Compound(rhs_c_id_one, compounds)
    rhs_compound_two = db.Compound(rhs_c_id_two, compounds)

    lhs_f_id, lhs_comp_id = insert_single_empty_structure_aggregate(manager, db.Label.COMPLEX_OPTIMIZED)
    rhs_f_id, rhs_comp_id = insert_single_empty_structure_aggregate(manager, db.Label.COMPLEX_OPTIMIZED)
    lhs_s_id_one = lhs_compound_one.get_centroid()
    lhs_s_id_two = lhs_compound_two.get_centroid()
    rhs_s_id_one = rhs_compound_one.get_centroid()
    rhs_s_id_two = rhs_compound_two.get_centroid()
    for compound in compounds.iterate_all_compounds():
        compound.link(compounds)
        compound.enable_exploration()
    for flask in flasks.iterate_all_flasks():
        flask.link(flasks)
        flask.enable_exploration()

    # set up steps between aggregates
    step_barrierless_lhs = db.ElementaryStep()
    step_barrierless_lhs.link(steps)
    step_barrierless_lhs.create([lhs_s_id_one, lhs_s_id_two], [lhs_comp_id])
    step_barrierless_lhs.set_type(db.ElementaryStepType.BARRIERLESS)

    step_barrierless_rhs = db.ElementaryStep()
    step_barrierless_rhs.link(steps)
    step_barrierless_rhs.create([rhs_comp_id], [rhs_s_id_one, rhs_s_id_two])
    step_barrierless_rhs.set_type(db.ElementaryStepType.BARRIERLESS)

    step_central = db.ElementaryStep()
    step_central.link(steps)
    step_central.create([lhs_comp_id], [rhs_comp_id])

    # set up TS and energies
    lhs_comp_structure = db.Structure(lhs_comp_id, structures)
    db_setup.add_random_energy(lhs_comp_structure, (0.0, 1.0), properties)
    ts = db.Structure(db_setup.insert_single_empty_structure_aggregate(manager, db.Label.TS_GUESS)[1], structures)
    db_setup.add_random_energy(ts, (70.0, 71.0), properties)
    step_central.set_transition_state(ts.get_id())

    # set up reactions
    reaction_barrierless_lhs = db.Reaction()
    reaction_barrierless_lhs.link(reactions)
    reaction_barrierless_lhs.create([lhs_c_id_one, lhs_c_id_two], [lhs_f_id],
                                    [db.CompoundOrFlask.COMPOUND, db.CompoundOrFlask.COMPOUND],
                                    [db.CompoundOrFlask.FLASK])
    reaction_barrierless_lhs.set_elementary_steps([step_barrierless_lhs.get_id()])

    reaction_barrierless_rhs = db.Reaction()
    reaction_barrierless_rhs.link(reactions)
    reaction_barrierless_rhs.create([rhs_f_id], [rhs_c_id_one, rhs_c_id_two], [db.CompoundOrFlask.FLASK],
                                    [db.CompoundOrFlask.COMPOUND, db.CompoundOrFlask.COMPOUND])
    reaction_barrierless_rhs.set_elementary_steps([step_barrierless_rhs.get_id()])

    reaction_central = db.Reaction()
    reaction_central.link(reactions)
    reaction_central.create([lhs_f_id], [rhs_f_id], [db.CompoundOrFlask.FLASK], [db.CompoundOrFlask.FLASK])
    reaction_central.set_elementary_steps([step_central.get_id()])

    lhs_flask = db.Flask(lhs_f_id, flasks)
    rhs_flask = db.Flask(rhs_f_id, flasks)
    lhs_compound_one.set_reactions([reaction_barrierless_lhs.id()])
    lhs_compound_two.set_reactions([reaction_barrierless_lhs.id()])
    lhs_flask.set_reactions([reaction_barrierless_lhs.id(), reaction_central.id()])
    rhs_flask.set_reactions([reaction_central.id(), reaction_barrierless_rhs.id()])
    rhs_compound_one.set_reactions([reaction_barrierless_rhs.id()])
    rhs_compound_two.set_reactions([reaction_barrierless_rhs.id()])

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
    kinetic_modeling_gear.options.diffusion_controlled_barrierless = False

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
    kinetic_modeling_gear2.options.diffusion_controlled_barrierless = False

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

    kinetic_modeling_engine3 = Engine(manager.get_credentials(), fork=False)
    kinetic_modeling_engine3.set_gear(kinetic_modeling_gear3)
    kinetic_modeling_engine3.run(single=True)
    assert calculations.count(dumps({})) == 2

    kinetic_modeling_engine3.run(single=True)
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
    compounds = manager.get_collection("compounds")
    # Set the starting concentrations for all compounds.
    for structure in structures.iterate_all_structures():
        structure.link(structures)
        con_prop = db.NumberProperty.make("start_concentration", model, 0.1, properties)
        con_prop.set_structure(structure.id())
        structure.add_property("start_concentration", con_prop.id())
    for compound in compounds.iterate_all_compounds():
        compound.link(compounds)
        compound.enable_exploration()

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


def test_random_kinetic_model_local_barrier_analysis():
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
        "chemoton_test_random_kinetic_model_local_barrier_analysis",
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
    compounds = manager.get_collection("compounds")
    # Set the starting concentrations for all compounds.
    for structure in structures.iterate_all_structures():
        structure.link(structures)
        con_prop = db.NumberProperty.make("start_concentration", model, 0.1, properties)
        con_prop.set_structure(structure.id())
        structure.add_property("start_concentration", con_prop.id())
    for compound in compounds.iterate_all_compounds():
        compound.link(compounds)
        compound.enable_exploration()

    # Check the job-set up.
    job_factory = KineticModelingJobFactory(model, manager, "electronic_energy", db.Job('fake_kinetic_modeling'),
                                            1e-5, True, False, 200.0, 0.0, 0.0, 0.0, "_test", True, False)
    settings = utils.ValueCollection({})
    settings["time_step"] = 1e-14
    settings["solver"] = "cash_carp_5"
    settings["batch_interval"] = 10000
    settings["n_batches"] = 10000
    settings["energy_label"] = "electronic_energy"
    settings["convergence"] = 1e-8

    job_factory.create_local_barrier_analysis_jobs(settings)

    assert calculations.count(dumps({})) == n_reactions

    # Cleaning
    manager.wipe()
