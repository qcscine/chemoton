#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
from json import dumps

# Third party imports
import scine_database as db

# Local application tests imports
from ... import test_database_setup as db_setup
from ...resources import resources_root_path

# Local application imports
from ....engine import Engine
from ....gears.conformers.brute_force import BruteForceConformers


def test_conformer_job_setup():
    # Connect to test DB
    manager = db_setup.get_clean_db("test_conformers_brute_force_conf")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")

    # Add a compound with a single optimized structure
    model = db.Model("FAKE", "FAKE", "F-AKE")
    rr = resources_root_path()
    centroid = db.Structure()
    centroid.link(structures)
    centroid.create(os.path.join(rr, "water.xyz"), 0, 1)
    centroid.set_label(db.Label.MINIMUM_OPTIMIZED)
    centroid.set_graph("masm_cbor_graph", "asdfghjkl")
    centroid.set_graph("masm_decision_list", "(1,2,3,4)")
    compound = db.Compound()
    compound.link(compounds)
    compound.create([centroid.id()])
    centroid.set_aggregate(compound.id())

    # Setup gear
    conformer_gear = BruteForceConformers()
    conformer_gear.options.model = model
    conformer_gear.options.conformer_job = db.Job("fake_conformer_generation")
    conformer_engine = Engine(manager.get_credentials(), fork=False)
    conformer_engine.set_gear(conformer_gear)

    # Run a single loop
    conformer_engine.run(single=True)

    # Checks
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 1

    calculation = db.Calculation(hits[0].id())
    calculation.link(calculations)

    assert len(calculation.get_structures()) == 1
    assert calculation.get_structures()[0].string() == centroid.id().string()
    assert calculation.get_status() == db.Status.HOLD
    assert calculation.get_job().order == "fake_conformer_generation"

    # Run a second loop
    conformer_engine.run(single=True)

    # Check again
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 1

    # Rerun with a different model
    model2 = db.Model("FAKE2", "", "")
    conformer_gear.options.model = model2
    conformer_engine.run(single=True)
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 2

    # Cleaning
    manager.wipe()


def test_optimization_job_setup():
    # Connect to test DB
    manager = db_setup.get_clean_db("test_conformers_brute_force_opt")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")

    # Add a compound
    compound = db.Compound()
    compound.link(compounds)
    compound.create([])
    # Add calculation
    model = db.Model("FAKE", "FAKE", "F-AKE")
    conf_gen = db.Calculation()
    conf_gen.link(calculations)
    conf_gen.create(model, db.Job("fake_conformer_generation"), [])
    conf_gen.set_auxiliary("compound", compound.id())
    conf_gen.set_status(db.Status.COMPLETE)
    # Add conformer guesses and a calculation
    rr = resources_root_path()
    results = conf_gen.get_results()
    for f in os.listdir(os.path.join(rr, "propionic_acid_conf_guesses")):
        s = db.Structure()
        s.link(structures)
        s.create(os.path.join(rr, "propionic_acid_conf_guesses", f), 0, 1)
        s.set_label(db.Label.MINIMUM_GUESS)
        s.set_aggregate(compound.id())
        compound.add_structure(s.id())
        results.add_structure(s.id())
    conf_gen.set_results(results)

    # Setup gear
    conformer_gear = BruteForceConformers()
    conformer_gear.options.model = model
    conformer_gear.options.conformer_job = db.Job("fake_conformer_generation")
    conformer_gear.options.minimization_job = db.Job("fake_opt")
    conformer_engine = Engine(manager.get_credentials(), fork=False)
    conformer_engine.set_gear(conformer_gear)

    # Run a single loop
    conformer_engine.run(single=True)

    # Checks
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 13

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        if calculation.get_job().order == "fake_conformer_generation":
            continue
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "fake_opt"

    # Run a second loop
    conformer_engine.run(single=True)

    # Check again
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 13

    # Cleaning
    manager.wipe()


def test_compound_completion_setup():
    # Connect to test DB
    manager = db_setup.get_clean_db("test_conformers_brute_force_opt")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")
    properties = manager.get_collection("properties")

    # Add a compound
    compound = db.Compound()
    compound.link(compounds)
    compound.create([])
    # Add calculation
    model = db.Model("FAKE", "FAKE", "F-AKE")
    conf_gen = db.Calculation()
    conf_gen.link(calculations)
    conf_gen.create(model, db.Job("fake_conformer_generation"), [])
    conf_gen.set_auxiliary("compound", compound.id())
    conf_gen.set_status(db.Status.COMPLETE)
    # Add conformer guesses, a calculation for the confomer generation and optimization calculations
    rr = resources_root_path()
    results = conf_gen.get_results()
    opt_job = db.Job("fake_opt")
    for f in os.listdir(os.path.join(rr, "propionic_acid_conf_guesses")):
        s = db.Structure()
        s.link(structures)
        s.create(os.path.join(rr, "propionic_acid_conf_guesses", f), 0, 1)
        s.set_label(db.Label.MINIMUM_OPTIMIZED)
        s.set_compound(compound.id())
        compound.add_structure(s.id())
        results.add_structure(s.id())

        energy = db.NumberProperty.make("electronic_energy", model, -2000, properties)
        s.add_property("electronic_energy", energy.id())

        minimization = db.Calculation()
        minimization.link(calculations)
        minimization.create(model, opt_job, [s.id()])
        minimization.set_status(db.Status.COMPLETE)
        min_result = minimization.get_results()
        min_result.add_structure(s.id())
        minimization.set_results(min_result)
    conf_gen.set_results(results)

    # Setup gear
    conformer_gear = BruteForceConformers()
    conformer_gear.options.model = model
    conformer_gear.options.conformer_job = db.Job("fake_conformer_generation")
    conformer_gear.options.minimization_job = opt_job
    conformer_engine = Engine(manager.get_credentials(), fork=False)
    conformer_engine.set_gear(conformer_gear)

    # Run a single loop
    conformer_engine.run(single=True)

    # Checks
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 13

    for sid in compound.get_structures():
        s = db.Structure(sid, structures)
        assert s.has_property("boltzmann_weight")

    s = db.Structure(compound.get_structures()[0], structures)
    prop = db.NumberProperty(s.get_property("boltzmann_weight"), properties)
    boltzmann_weight = prop.get_derived().get_data()
    comment = prop.get_comment()
    # Check comment for the energy zero point
    assert comment == "Energy 0-point: -2000.0"
    assert boltzmann_weight == 1.0

    # Run a second loop
    conformer_engine.run(single=True)

    # Check again
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 13

    # Cleaning
    manager.wipe()
