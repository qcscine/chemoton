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
from .. import test_database_setup as db_setup

# Local application imports
from ...engine import Engine
from ...gears.kinetics import MinimalConnectivityKinetics, BasicBarrierHeightKinetics, MaximumFluxKinetics


def test_activation_all_compounds():
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
    kinetics_gear = MinimalConnectivityKinetics()  # activate all compounds
    kinetics_gear.options.restart = True
    kinetics_engine = Engine(manager.get_credentials(), fork=False)
    kinetics_engine.set_gear(kinetics_gear)
    for _ in range(n_reactions):  # should be faster, but ensures that iterative procedure goes through
        kinetics_engine.run(single=True)

    compounds = manager.get_collection("compounds")
    selection = {"exploration_disabled": {"$ne": True}}
    assert compounds.count(dumps(selection)) == n_compounds

    kinetics_gear._disable_all_aggregates()
    selection = {"exploration_disabled": {"$ne": False}}
    assert compounds.count(dumps(selection)) == n_compounds
    # Cleaning
    manager.wipe()


def test_activation_all_compounds_and_flasks():
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
    kinetics_gear = MinimalConnectivityKinetics()  # activate all compounds
    kinetics_gear.options.restart = True
    kinetics_engine = Engine(manager.get_credentials(), fork=False)
    kinetics_engine.set_gear(kinetics_gear)
    for _ in range(n_reactions):  # should be faster, but ensures that iterative procedure goes through
        kinetics_engine.run(single=True)

    compounds = manager.get_collection("compounds")
    flasks = manager.get_collection("flasks")
    selection = {"exploration_disabled": {"$ne": True}}
    assert compounds.count(dumps(selection)) == n_compounds
    assert flasks.count(dumps(selection)) == n_flasks

    kinetics_gear._disable_all_aggregates()
    selection = {"exploration_disabled": {"$ne": False}}
    assert compounds.count(dumps(selection)) == n_compounds
    assert flasks.count(dumps(selection)) == n_flasks
    # Cleaning
    manager.wipe()


def test_user_insertion_verification():
    manager = db_setup.get_clean_db("chemoton_test_user_insertion_verification")
    compound = db.Compound(db_setup.insert_single_empty_structure_compound(manager, db.Label.USER_GUESS)[0])
    compounds = manager.get_collection("compounds")
    compound.link(compounds)

    kinetics_gear = MinimalConnectivityKinetics()  # activate all compounds
    kinetics_engine = Engine(manager.get_credentials(), fork=False)
    kinetics_engine.set_gear(kinetics_gear)
    kinetics_engine.run(single=True)

    assert compound.explore()
    # Cleaning
    manager.wipe()


def test_barrier_limit():
    manager = db_setup.get_clean_db("chemoton_test_barrier_limit")
    # set up 2 compounds
    c1_id, s1_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.USER_GUESS)
    c2_id, s2_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.MINIMUM_GUESS)

    steps = manager.get_collection("elementary_steps")
    structures = manager.get_collection("structures")
    properties = manager.get_collection("properties")
    reactions = manager.get_collection("reactions")
    compounds = manager.get_collection("compounds")

    # set up step between compounds
    step = db.ElementaryStep()
    step.link(steps)
    step.create([s1_id], [s2_id])

    # set up TS and energies
    db_setup.add_random_energy(db.Structure(s1_id, structures), (0.0, 1.0), properties)
    db_setup.add_random_energy(db.Structure(s2_id, structures), (50.0, 51.0), properties)
    ts = db.Structure(db_setup.insert_single_empty_structure_compound(manager, db.Label.TS_GUESS)[1])
    ts.link(structures)
    ts_prop_id = db_setup.add_random_energy(ts, (70.0, 71.0), properties)
    step.set_transition_state(ts.get_id())

    # set up reaction
    reaction = db.Reaction()
    reaction.link(reactions)
    reaction.create([c1_id], [c2_id])
    reaction.set_elementary_steps([step.get_id()])
    compound_1 = db.Compound(c1_id)
    compound_2 = db.Compound(c2_id)
    compound_1.link(compounds)
    compound_2.link(compounds)
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
    ts_prop.link(properties)
    ts_prop.set_data(110.0)

    kinetics_gear.options.restart = True
    kinetics_engine.set_gear(kinetics_gear)
    for _ in range(2):
        kinetics_engine.run(single=True)
    assert compound_1.explore() and not compound_2.explore()
    # Cleaning
    manager.wipe()


def test_barrier_limit_with_barrierless():
    manager = db_setup.get_clean_db("chemoton_test_barrier_limit_barrierless")
    # set up 2 compounds
    c1_id, s1_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.USER_GUESS)
    c2_id, s2_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.MINIMUM_GUESS)
    c3_id, s3_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.MINIMUM_GUESS)

    steps = manager.get_collection("elementary_steps")
    structures = manager.get_collection("structures")
    properties = manager.get_collection("properties")
    reactions = manager.get_collection("reactions")
    compounds = manager.get_collection("compounds")

    # set up steps between compounds
    step_down = db.ElementaryStep(db.ID(), steps)
    step_down.create([s1_id], [s2_id])
    step_down.set_type(db.ElementaryStepType.BARRIERLESS)

    step_up = db.ElementaryStep(db.ID(), steps)
    step_up.create([s2_id], [s3_id])
    step_up.set_type(db.ElementaryStepType.BARRIERLESS)

    db_setup.add_random_energy(db.Structure(s1_id, structures), (100.0, 101.0), properties)
    db_setup.add_random_energy(db.Structure(s2_id, structures), (10.0, 11.0), properties)
    e3_id = db_setup.add_random_energy(db.Structure(s3_id, structures), (20.0, 21.0), properties)

    # set up reaction
    reaction = db.Reaction(db.ID(), reactions)
    reaction.create([c1_id], [c2_id])
    reaction.set_elementary_steps([step_down.get_id()])
    compound_1 = db.Compound(c1_id, compounds)
    compound_2 = db.Compound(c2_id, compounds)
    compound_1.set_reactions([reaction.get_id()])
    compound_2.set_reactions([reaction.get_id()])

    reaction = db.Reaction(db.ID(), reactions)
    reaction.create([c2_id], [c3_id])
    reaction.set_elementary_steps([step_up.get_id()])
    compound_3 = db.Compound(c3_id, compounds)
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
    e3 = db.NumberProperty(e3_id, properties)
    e3.set_data(70.0)

    kinetics_gear.options.restart = True
    kinetics_engine.set_gear(kinetics_gear)
    for _ in range(3):
        kinetics_engine.run(single=True)
    assert compound_1.explore() and compound_2.explore() and not compound_3.explore()
    # Cleaning
    manager.wipe()


def test_barrier_with_random_network():
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

    compounds = manager.get_collection("compounds")
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
    assert compounds.count(dumps(selection)) == n_inserts

    kinetics_gear.options.model = db.Model("FAKE", "FAKE", "F-AKE")
    kinetics_engine.set_gear(kinetics_gear)
    for _ in range(n_reactions):  # should be faster, but ensures that iterative procedure goes through
        kinetics_engine.run(single=True)

    # check if all compounds are activated
    assert compounds.count(dumps(selection)) == n_compounds

    # set barrier limit so low that all are too high
    kinetics_gear.options.max_allowed_barrier = 10
    kinetics_gear.options.restart = True
    kinetics_engine.set_gear(kinetics_gear)
    for _ in range(n_reactions):  # should be faster, but ensures that iterative procedure goes through
        kinetics_engine.run(single=True)

    # all barriers are now too high, so only user inserted compounds should be enabled
    selection = {"exploration_disabled": {"$ne": True}}
    assert compounds.count(dumps(selection)) == n_inserts
    # Cleaning
    manager.wipe()


def test_concentration_based_selection():
    manager = db_setup.get_clean_db("chemoton_test_concentration_based_selection")
    # set up 2 compounds
    c1_id, s1_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.USER_GUESS)
    c2_id, s2_id = db_setup.insert_single_empty_structure_compound(manager, db.Label.MINIMUM_GUESS)

    steps = manager.get_collection("elementary_steps")
    structures = manager.get_collection("structures")
    properties = manager.get_collection("properties")
    reactions = manager.get_collection("reactions")
    compounds = manager.get_collection("compounds")

    # set up step between compounds
    step = db.ElementaryStep()
    step.link(steps)
    step.create([s1_id], [s2_id])

    # set up TS and energies
    s1 = db.Structure(s1_id, structures)
    s2 = db.Structure(s2_id, structures)
    db_setup.add_random_energy(s1, (0.0, 1.0), properties)
    db_setup.add_random_energy(s2, (50.0, 51.0), properties)
    ts = db.Structure(db_setup.insert_single_empty_structure_compound(manager, db.Label.TS_GUESS)[1], structures)
    db_setup.add_random_energy(ts, (70.0, 71.0), properties)
    step.set_transition_state(ts.get_id())

    # set up reactions
    reaction = db.Reaction(db.ID(), reactions)
    reaction.create([c1_id], [c2_id])
    reaction.set_elementary_steps([step.get_id()])
    compound_1 = db.Compound(c1_id, compounds)
    compound_2 = db.Compound(c2_id, compounds)
    compound_1.set_reactions([reaction.get_id()])
    compound_2.set_reactions([reaction.get_id()])

    # set up concentration properties
    kinetics_gear = MaximumFluxKinetics()
    concentration_label = kinetics_gear.options.property_label
    flux_label = kinetics_gear.options.flux_property_label
    model = db.Model("FAKE", "FAKE", "F-AKE")
    conc_prop1 = db.NumberProperty.make(concentration_label, model, 100, properties)
    conc_prop2 = db.NumberProperty.make(concentration_label, model, 100, properties)
    flux_prop1 = db.NumberProperty.make(flux_label, model, 100, properties)
    s1.add_property(concentration_label, conc_prop1.id())
    s1.add_property(flux_label, flux_prop1.id())
    flux_prop1.set_structure(s1_id)

    # run barrier filter gear
    kinetics_gear.options.model = model
    kinetics_gear.options.restart = True
    kinetics_gear.options.max_allowed_barrier = 100.0
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
    # Cleaning
    manager.wipe()


def test_barrier_with_flasks_network():
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

    compounds = manager.get_collection("compounds")
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
    assert compounds.count(dumps(selection)) == n_inserts

    kinetics_gear.options.model = db.Model("FAKE", "FAKE", "F-AKE")
    kinetics_engine.set_gear(kinetics_gear)
    for _ in range(n_reactions):  # should be faster, but ensures that iterative procedure goes through
        kinetics_engine.run(single=True)

    # check if all compounds are activated
    assert compounds.count(dumps(selection)) == n_compounds

    # set barrier limit so low that all are too high
    kinetics_gear.options.max_allowed_barrier = -1
    kinetics_gear.options.restart = True
    kinetics_engine.set_gear(kinetics_gear)
    for _ in range(n_reactions):  # should be faster, but ensures that iterative procedure goes through
        kinetics_engine.run(single=True)

    # all barriers are now too high, so only user inserted compounds should be enabled
    selection = {"exploration_disabled": {"$ne": True}}
    assert compounds.count(dumps(selection)) == n_inserts
    # Cleaning
    manager.wipe()
