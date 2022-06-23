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
from .. import test_database_setup as db_setup
from ..resources import resources_root_path

# Local application imports
from ...engine import Engine
from ...gears.compound import BasicCompoundHousekeeping


def test_compound_creation():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_compound_creation")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")

    # Add structure data
    model = db.Model("FAKE", "", "")
    rr = resources_root_path()
    for label in [db.Label.MINIMUM_OPTIMIZED, db.Label.USER_OPTIMIZED]:
        # Setup clean db
        manager.init()

        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(label)
        structure.set_graph("masm_cbor_graph", "asdfghjkl")
        structure.set_graph("masm_decision_list", "")

        # Setup gear
        compound_gear = BasicCompoundHousekeeping()
        compound_gear.options.model = model
        compound_engine = Engine(manager.get_credentials(), fork=False)
        compound_engine.set_gear(compound_gear)

        # Run a single loop
        compound_engine.run(single=True)

        # Checks
        hits = compounds.query_compounds(dumps({}))
        assert len(hits) == 1

        compound = db.Compound(hits[0].id())
        compound.link(compounds)

        assert len(compound.get_structures()) == 1
        assert compound.get_structures()[0].string() == structure.id().string()
        assert compound.get_centroid().string() == structure.id().string()
        assert structure.has_compound()
        assert structure.get_compound().string() == compound.id().string()

        # Cleaning
        manager.wipe()


def test_compound_extension():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_compound_extension")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")

    # Add fake data
    model = db.Model("FAKE", "", "")
    rr = resources_root_path()
    centroid = db.Structure()
    centroid.link(structures)
    centroid.create(os.path.join(rr, "water.xyz"), 0, 1)
    centroid.set_label(db.Label.USER_OPTIMIZED)
    centroid.set_graph("masm_cbor_graph",
                       "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAW"
                       "FzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                       )
    centroid.set_graph("masm_decision_list", "(1,2,3,1)")
    structure = db.Structure()
    structure.link(structures)
    structure.create(os.path.join(rr, "water.xyz"), 0, 1)
    structure.set_label(db.Label.MINIMUM_OPTIMIZED)
    structure.set_graph("masm_cbor_graph",
                        "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAW"
                        "FzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                        )
    structure.set_graph("masm_decision_list", "(181,182,183,1)")
    compound = db.Compound()
    compound.link(compounds)
    compound.create([centroid.id()])
    centroid.set_compound(compound.id())

    # Setup gear
    compound_gear = BasicCompoundHousekeeping()
    compound_gear.options.model = model
    compound_engine = Engine(manager.get_credentials(), fork=False)
    compound_engine.set_gear(compound_gear)

    # Run a single loop
    compound_engine.run(single=True)

    # Checks
    assert len(compound.get_structures()) == 2
    assert compound.get_structures()[1].string() == structure.id().string()
    assert compound.get_centroid().string() == centroid.id().string()
    assert structure.has_compound()
    assert structure.get_compound().string() == compound.id().string()
    assert structure.get_label() == db.Label.MINIMUM_OPTIMIZED

    # Cleaning
    manager.wipe()


def test_intermediate_deduplication():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_intermediate_deduplication")

    # Get collections
    model = db.Model("FAKE", "", "")
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")

    # Add fake data
    model = db.Model("FAKE", "", "")
    rr = resources_root_path()
    centroid = db.Structure()
    centroid.link(structures)
    centroid.create(os.path.join(rr, "water.xyz"), 0, 1)
    centroid.set_label(db.Label.MINIMUM_OPTIMIZED)
    centroid.set_graph("masm_cbor_graph",
                       "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWF"
                       "zAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                       )
    centroid.set_graph("masm_decision_list", "(10,20,40,1):(110,120,140,1)")
    structure = db.Structure()
    structure.link(structures)
    structure.create(os.path.join(rr, "water.xyz"), 0, 1)
    structure.set_label(db.Label.MINIMUM_OPTIMIZED)
    structure.set_graph("masm_cbor_graph",
                        "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWF"
                        "zAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                        )
    structure.set_graph("masm_decision_list", "(10,30,40,1):(110,130,140,1)")
    compound = db.Compound()
    compound.link(compounds)
    compound.create([centroid.id()])
    centroid.set_compound(compound.id())

    # Setup gear
    compound_gear = BasicCompoundHousekeeping()
    compound_gear.options.model = model
    compound_engine = Engine(manager.get_credentials(), fork=False)
    compound_engine.set_gear(compound_gear)

    # Run a single loop
    compound_engine.run(single=True)

    # Checks
    assert len(compound.get_structures()) == 2
    assert compound.get_structures()[1].string() == structure.id().string()
    assert compound.get_centroid().string() == centroid.id().string()
    assert structure.has_compound()
    assert structure.get_compound().string() == compound.id().string()
    assert structure.get_label() == db.Label.DUPLICATE

    # Cleaning
    manager.wipe()


def test_intermediate_deduplication_empty_dlist():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_intermediate_deduplication")

    # Get collections
    model = db.Model("FAKE", "", "")
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")

    # Add fake data
    model = db.Model("FAKE", "", "")
    rr = resources_root_path()
    centroid = db.Structure()
    centroid.link(structures)
    centroid.create(os.path.join(rr, "water.xyz"), 0, 1)
    centroid.set_label(db.Label.MINIMUM_OPTIMIZED)
    centroid.set_graph("masm_cbor_graph",
                       "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWFzAWF"
                       "jD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                       )
    centroid.set_graph("masm_decision_list", "")
    structure = db.Structure()
    structure.link(structures)
    structure.create(os.path.join(rr, "water.xyz"), 0, 1)
    structure.set_label(db.Label.MINIMUM_OPTIMIZED)
    structure.set_graph("masm_cbor_graph",
                        "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWFzAWF"
                        "jD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                        )
    structure.set_graph("masm_decision_list", "")
    compound = db.Compound()
    compound.link(compounds)
    compound.create([centroid.id()])
    centroid.set_compound(compound.id())

    # Setup gear
    compound_gear = BasicCompoundHousekeeping()
    compound_gear.options.model = model
    compound_engine = Engine(manager.get_credentials(), fork=False)
    compound_engine.set_gear(compound_gear)

    # Run a single loop
    compound_engine.run(single=True)

    # Checks
    assert len(compound.get_structures()) == 2
    assert compound.get_structures()[1].string() == structure.id().string()
    assert compound.get_centroid().string() == centroid.id().string()
    assert structure.has_compound()
    assert structure.get_compound().string() == compound.id().string()
    assert structure.get_label() == db.Label.DUPLICATE

    # Cleaning
    manager.wipe()


def test_irrelevant_structure():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_irrelevant_structure")

    # Get collections
    structures = manager.get_collection("structures")

    # Add structure data
    model = db.Model("FAKE", "", "")
    rr = resources_root_path()
    structure = db.Structure()
    structure.link(structures)
    structure.create(os.path.join(rr, "water.xyz"), 0, 1)
    structure.set_label(db.Label.MINIMUM_OPTIMIZED)
    structure.set_graph("masm_cbor_graph",
                        "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWF"
                        "zAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA;"
                        "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWF"
                        "zAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                        )
    structure.set_graph("masm_decision_list", "(10,30,40,1)")

    # Setup gear
    compound_gear = BasicCompoundHousekeeping()
    compound_gear.options.model = model
    compound_gear.options.graph_job = db.Job("testy_mac_test_face")
    compound_engine = Engine(manager.get_credentials(), fork=False)
    compound_engine.set_gear(compound_gear)

    # Run a single loop
    compound_engine.run(single=True)

    assert not structure.has_compound()
    assert structure.get_label() == db.Label.IRRELEVANT

    # Cleaning
    manager.wipe()


def test_graph_job_setup():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_graph_job_setup")

    # Get collections
    structures = manager.get_collection("structures")
    calculations = manager.get_collection("calculations")

    # Add structure data
    model = db.Model("FAKE", "", "")
    rr = resources_root_path()
    structure = db.Structure()
    structure.link(structures)
    structure.create(os.path.join(rr, "water.xyz"), 0, 1)
    structure.set_label(db.Label.MINIMUM_OPTIMIZED)
    structure.add_property("bond_orders", db.ID())

    # Setup gear
    compound_gear = BasicCompoundHousekeeping()
    compound_gear.options.model = model
    compound_gear.options.graph_job = db.Job("testy_mac_test_face")
    compound_engine = Engine(manager.get_credentials(), fork=False)
    compound_engine.set_gear(compound_gear)

    # Run a single loop
    compound_engine.run(single=True)

    # Checks
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 1

    calculation = db.Calculation(hits[0].id())
    calculation.link(calculations)

    assert len(calculation.get_structures()) == 1
    assert calculation.get_structures()[0].string() == structure.id().string()
    assert calculation.get_status() == db.Status.HOLD
    assert calculation.get_job().order == "testy_mac_test_face"

    # Run a second time
    compound_engine.run(single=True)
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 1

    # Rerun with a different model
    model2 = db.Model("FAKE2", "", "")
    compound_gear.options.model = model2
    compound_engine.run(single=True)
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 2

    # Cleaning
    manager.wipe()


def test_bo_job_setup():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_bo_job_setup")

    # Get collections
    structures = manager.get_collection("structures")
    calculations = manager.get_collection("calculations")

    # Add structure data
    model = db.Model("FAKE", "", "")
    rr = resources_root_path()
    structure = db.Structure()
    structure.link(structures)
    structure.create(os.path.join(rr, "water.xyz"), 0, 1)
    structure.set_label(db.Label.MINIMUM_OPTIMIZED)

    # Setup gear
    compound_gear = BasicCompoundHousekeeping()
    compound_gear.options.model = model
    compound_gear.options.bond_order_job = db.Job("eggs_bacon_and_spam")
    compound_engine = Engine(manager.get_credentials(), fork=False)
    compound_engine.set_gear(compound_gear)

    # Run a single loop
    compound_engine.run(single=True)

    # Checks
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 1

    calculation = db.Calculation(hits[0].id())
    calculation.link(calculations)

    assert len(calculation.get_structures()) == 1
    assert calculation.get_structures()[0].string() == structure.id().string()
    assert calculation.get_status() == db.Status.HOLD
    assert calculation.get_job().order == "eggs_bacon_and_spam"

    # Run a second time
    compound_engine.run(single=True)
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 1

    # Rerun with a different model
    model2 = db.Model("FAKE2", "", "")
    compound_gear.options.model = model2
    compound_engine.run(single=True)
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 2

    # Cleaning
    manager.wipe()
