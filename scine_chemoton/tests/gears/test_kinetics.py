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
from ...gears.kinetics import MinimalConnectivityKinetics, BasicBarrierHeightKinetics


def test_activation_all_compounds():
    n_compounds = 9
    n_reactions = 6
    max_r_per_c = 10
    max_n_products_per_r = 2
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (0.1, 2000.0)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
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

    kinetics_gear._disable_all_compounds()
    selection = {"exploration_disabled": {"$ne": False}}
    assert compounds.count(dumps(selection)) == n_compounds


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
    s1 = db.Structure(s1_id)
    s1.link(structures)
    db_setup.add_random_energy(s1, (0.0, 1.0), properties)
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
    kinetics_gear.options.model = db.Model("FAKE", "", "")
    kinetics_gear.options.restart = True
    kinetics_gear.options.max_allowed_barrier = 100.0
    kinetics_engine = Engine(manager.get_credentials(), fork=False)
    kinetics_engine.set_gear(kinetics_gear)
    for _ in range(2):
        kinetics_engine.run(single=True)

    assert compound_1.explore() and compound_2.explore()

    # make barrier to high
    ts_prop = db.NumberProperty(ts_prop_id)
    ts_prop.link(properties)
    ts_prop.set_data(110.0)

    kinetics_gear.options.restart = True
    kinetics_engine.set_gear(kinetics_gear)
    for _ in range(2):
        kinetics_engine.run(single=True)
    assert compound_1.explore() and not compound_2.explore()


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
    manager = db_setup.get_random_db(
        n_compounds,
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

    kinetics_gear.options.model = db.Model("FAKE", "", "")
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
