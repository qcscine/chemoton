#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
import json
from typing import Dict, List, Tuple

# Third party imports
import scine_database as db
from numpy import ndarray

# Local application tests imports
from ... import test_database_setup as db_setup
from ...resources import resources_root_path

# Local application imports
from ....engine import Engine
from ....gears.elementary_steps.minimal import MinimalElementarySteps
from .mock_trial_generator import MockGenerator


def _sum_unimolecular(coordinates: Dict[str, Dict[str, List[Tuple[List[List[Tuple[int, int]]], int]]]]) -> int:
    s = 0
    for compound in coordinates:
        for structure in coordinates[compound]:
            for coords, _ in coordinates[compound][structure]:
                s += len(coords)
    return s


def _sum_bimolecular(coordinates: Dict[str, Dict[str, Dict[Tuple[List[Tuple[int, int]], int],
                                                           List[Tuple[ndarray, ndarray, float, float]]]]]) -> int:
    s = 0
    for compound in coordinates:
        for structure in coordinates[compound]:
            for trial in coordinates[compound][structure]:
                s += len(coordinates[compound][structure][trial])
    return s


def test_bimol():
    """
    Test whether the correct number of bimolecular combinations is probed
    """
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_minimal_bimol")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")

    # Add fake data
    rr = resources_root_path()
    for mol in ["hydrogenperoxide", "water"]:
        compound = db.Compound()
        compound.link(compounds)
        compound.create([])
        # Adding more structures per compound should not have an effect
        for i in range(2):
            graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
            structure.set_label(db.Label.USER_OPTIMIZED)
            structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
            structure.set_graph("masm_idx_map", graph["masm_idx_map"])
            structure.set_graph("masm_decision_list", str(i))
            compound.add_structure(structure.id())
            structure.set_aggregate(compound.id())

    # Setup gear
    es_gear = MinimalElementarySteps()
    es_gear.trial_generator = MockGenerator()
    es_gear.options.enable_unimolecular_trials = False
    es_gear.options.enable_bimolecular_trials = True
    es_engine = Engine(manager.get_credentials(), fork=False)
    es_engine.set_gear(es_gear)

    # Run a single loop
    es_engine.run(single=True)

    # Expected numbers:
    # Unimolecular: 0
    # Bimolecular: 2 choose 2 with repetition: 3 (H2O + H2O, H2O + H2O2, H2O2)
    assert es_gear.trial_generator.unimol_counter == 0
    assert es_gear.trial_generator.bimol_counter == 3

    # Cleaning
    manager.wipe()


def test_unimol():
    """
    Test whether the correct number of unimolecular combinations is probed
    """
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_minimal_bimol")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")

    # Add fake data
    rr = resources_root_path()
    for mol in ["hydrogenperoxide", "water", "cyclohexene"]:
        compound = db.Compound()
        compound.link(compounds)
        compound.create([])
        # Adding more structures per compound should not have an effect
        for i in range(4):
            graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
            structure.set_label(db.Label.USER_OPTIMIZED)
            structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
            structure.set_graph("masm_idx_map", graph["masm_idx_map"])
            structure.set_graph("masm_decision_list", str(i))
            compound.add_structure(structure.id())
            structure.set_aggregate(compound.id())

    # Setup gear
    es_gear = MinimalElementarySteps()
    es_gear.trial_generator = MockGenerator()
    es_gear.options.enable_unimolecular_trials = True
    es_gear.options.enable_bimolecular_trials = False
    es_engine = Engine(manager.get_credentials(), fork=False)
    es_engine.set_gear(es_gear)

    expected_uni = _sum_unimolecular(es_gear.unimolecular_coordinates(manager.get_credentials()))
    expected_bi = _sum_bimolecular(es_gear.bimolecular_coordinates(manager.get_credentials()))
    es_gear.clear_cache()
    assert expected_uni == 3
    assert expected_bi == 0

    # Run a single loop
    es_engine.run(single=True)

    # Expected numbers:
    # Unimolecular: 3 (once for every compound)
    # Bimolecular: 0
    assert es_gear.trial_generator.unimol_counter == 3
    assert es_gear.trial_generator.bimol_counter == 0

    # Cleaning
    manager.wipe()


def test_unimol_bimol():
    """
    Test whether the correct number of unimolecular and bimolecular combinations
    is probed
    """
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_minimal_unimol_bimol")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")

    # Add fake data
    rr = resources_root_path()
    for mol in ["hydrogenperoxide", "cyclohexene", "water"]:
        compound = db.Compound()
        compound.link(compounds)
        compound.create([])
        # Adding more structures per compound should not have an effect
        for i in range(3):
            graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
            structure.set_label(db.Label.USER_OPTIMIZED)
            structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
            structure.set_graph("masm_idx_map", graph["masm_idx_map"])
            structure.set_graph("masm_decision_list", str(i))
            compound.add_structure(structure.id())
            structure.set_aggregate(compound.id())

    # Setup gear
    es_gear = MinimalElementarySteps()
    es_gear.trial_generator = MockGenerator()
    es_gear.options.enable_unimolecular_trials = True
    es_gear.options.enable_bimolecular_trials = True
    es_engine = Engine(manager.get_credentials(), fork=False)
    es_engine.set_gear(es_gear)

    expected_uni = _sum_unimolecular(es_gear.unimolecular_coordinates(manager.get_credentials()))
    expected_bi = _sum_bimolecular(es_gear.bimolecular_coordinates(manager.get_credentials()))
    es_gear.clear_cache()
    assert expected_uni == 3
    assert expected_bi == 6

    # Run a single loop
    es_engine.run(single=True)

    # Expected numbers:
    # Unimolecular: 3 (once for every compound)
    # Bimolecular: 3 choose 2 with repetition = 6
    assert es_gear.trial_generator.unimol_counter == 3
    assert es_gear.trial_generator.bimol_counter == 6

    # Rerun and build cache
    es_gear2 = MinimalElementarySteps()
    es_gear2.trial_generator = MockGenerator()
    es_gear2.options.enable_unimolecular_trials = True
    es_gear2.options.enable_bimolecular_trials = True
    es_engine2 = Engine(manager.get_credentials(), fork=False)
    es_engine2.set_gear(es_gear2)
    # Run a single loop building cache
    es_engine2.run(single=True)
    assert es_gear2.trial_generator.unimol_counter == 0
    assert es_gear2.trial_generator.bimol_counter == 0
    # Run a single loop using cache
    es_engine2.run(single=True)
    assert es_gear2.trial_generator.unimol_counter == 0
    assert es_gear2.trial_generator.bimol_counter == 0

    # Cleaning
    manager.wipe()


def test_unimol_bimol_enhancement():
    """
    Test whether the correct number of unimolecular and bimolecular combinations
    is probed
    """
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_minimal_unimol_bimol_enhancement")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")

    # Add fake data
    rr = resources_root_path()
    for mol in ["hydrogenperoxide", "cyclohexene", "water"]:
        compound = db.Compound()
        compound.link(compounds)
        compound.create([])
        # Adding more structures per compound should not have an effect
        for i in range(3):
            graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
            structure.set_label(db.Label.USER_OPTIMIZED)
            structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
            structure.set_graph("masm_idx_map", graph["masm_idx_map"])
            structure.set_graph("masm_decision_list", str(i))
            compound.add_structure(structure.id())
            structure.set_aggregate(compound.id())

    # Setup gear
    es_gear = MinimalElementarySteps()
    es_gear.trial_generator = MockGenerator()
    es_gear.options.enable_unimolecular_trials = True
    es_gear.options.enable_bimolecular_trials = True
    es_engine = Engine(manager.get_credentials(), fork=False)
    es_engine.set_gear(es_gear)

    expected_uni = _sum_unimolecular(es_gear.unimolecular_coordinates(manager.get_credentials()))
    expected_bi = _sum_bimolecular(es_gear.bimolecular_coordinates(manager.get_credentials()))
    es_gear.clear_cache()
    assert expected_uni == 3
    assert expected_bi == 6

    # Run a single loop
    es_engine.run(single=True)

    # Expected numbers:
    # Unimolecular: 3 (once for every compound)
    # Bimolecular: 3 choose 2 with repetition = 6
    assert es_gear.trial_generator.unimol_counter == 3
    assert es_gear.trial_generator.bimol_counter == 6

    # Rerun and build cache
    es_gear2 = MinimalElementarySteps()
    es_gear2.trial_generator = MockGenerator()
    es_gear2.options.enable_unimolecular_trials = True
    es_gear2.options.enable_bimolecular_trials = True
    es_engine2 = Engine(manager.get_credentials(), fork=False)
    es_engine2.set_gear(es_gear2)
    expected_uni = _sum_unimolecular(es_gear2.unimolecular_coordinates(manager.get_credentials()))
    expected_bi = _sum_bimolecular(es_gear2.bimolecular_coordinates(manager.get_credentials()))
    es_gear2.clear_cache()
    assert expected_uni == 0
    assert expected_bi == 0
    # Run a single loop building cache
    es_engine2.run(single=True)
    assert es_gear2.trial_generator.unimol_counter == 0
    assert es_gear2.trial_generator.bimol_counter == 0
    # Run a single loop using cache
    es_engine2.run(single=True)
    assert es_gear2.trial_generator.unimol_counter == 0
    assert es_gear2.trial_generator.bimol_counter == 0

    """ See if nothing changes if we run with settings check but nothing changed """
    es_gear.trial_generator.unimol_counter = 0
    es_gear.trial_generator.bimol_counter = 0
    es_gear.options.run_one_cycle_with_settings_enhancement = True
    es_engine.set_gear(es_gear)
    expected_uni = _sum_unimolecular(es_gear.unimolecular_coordinates(manager.get_credentials()))
    expected_bi = _sum_bimolecular(es_gear.bimolecular_coordinates(manager.get_credentials()))
    es_gear.clear_cache()
    assert expected_uni == 0
    assert expected_bi == 0
    es_engine.run(single=True)
    assert es_gear.trial_generator.unimol_counter == 0
    assert es_gear.trial_generator.bimol_counter == 0

    """ See if we get new calculations if we change the calculation settings """
    es_gear.options.run_one_cycle_with_settings_enhancement = True
    es_gear.trial_generator.options.settings = {"something": "changed"}
    es_engine.set_gear(es_gear)
    expected_uni = _sum_unimolecular(es_gear.unimolecular_coordinates(manager.get_credentials()))
    expected_bi = _sum_bimolecular(es_gear.bimolecular_coordinates(manager.get_credentials()))
    es_gear.clear_cache()
    assert expected_uni == 3
    assert expected_bi == 6
    es_engine.run(single=True)
    assert es_gear.trial_generator.unimol_counter == 3
    assert es_gear.trial_generator.bimol_counter == 6

    # Cleaning
    manager.wipe()
