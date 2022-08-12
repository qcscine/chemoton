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
import scine_utilities as utils

# Local application tests imports
from .. import test_database_setup as db_setup
from ..resources import resources_root_path

# Local application imports
from ...engine import Engine
from ...gears.thermo import BasicThermoDataCompletion


def test_hessian_calculation():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_hessian_calculation")

    # Get collections
    structures = manager.get_collection("structures")
    calculations = manager.get_collection("calculations")

    # Add structure data
    model = db.Model("FAKE", "FAKE", "F-AKE")
    rr = resources_root_path()
    structure = db.Structure()
    structure.link(structures)
    structure.create(os.path.join(rr, "water.xyz"), 0, 1)
    structure.set_label(db.Label.USER_OPTIMIZED)
    structure.set_aggregate(db.ID())

    # Setup gear
    thermo_gear = BasicThermoDataCompletion()
    thermo_gear.options.model = model
    thermo_gear.options.job = db.Job("fake_hessian")
    thermo_gear.options.settings = utils.ValueCollection({"some_thing": "else"})
    thermo_engine = Engine(manager.get_credentials(), fork=False)
    thermo_engine.set_gear(thermo_gear)

    # Run a single loop
    thermo_engine.run(single=True)

    # Checks
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 1

    calculation = db.Calculation(hits[0].id())
    calculation.link(calculations)

    assert len(calculation.get_structures()) == 1
    assert calculation.get_structures()[0].string() == structure.id().string()
    assert calculation.get_status() == db.Status.HOLD
    assert calculation.get_job().order == "fake_hessian"

    # Run a second loop
    thermo_engine.run(single=True)

    # Check again
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 1

    # Rerun with a different model
    model2 = db.Model("FAKE2", "", "")
    thermo_gear.options.model = model2
    thermo_engine.run(single=True)
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 2

    # Cleaning
    manager.wipe()


def test_hessian_skip():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_hessian_skip")

    # Get collections
    structures = manager.get_collection("structures")
    calculations = manager.get_collection("calculations")
    properties = manager.get_collection("properties")

    # Add structure data
    model = db.Model("FAKE", "FAKE", "F-AKE")
    rr = resources_root_path()
    structure = db.Structure()
    structure.link(structures)
    structure.create(os.path.join(rr, "water.xyz"), 0, 1)
    structure.set_label(db.Label.MINIMUM_OPTIMIZED)
    structure.set_aggregate(db.ID())

    # Add property
    gibbs_energy_correction = db.NumberProperty()
    gibbs_energy_correction.link(properties)
    gibbs_energy_correction.create(model, "gibbs_energy_correction", 13.37)
    structure.add_property("gibbs_energy_correction", gibbs_energy_correction.id())

    # Setup gear
    thermo_gear = BasicThermoDataCompletion()
    thermo_gear.options.model = model
    thermo_engine = Engine(manager.get_credentials(), fork=False)
    thermo_engine.set_gear(thermo_gear)

    # Run a single loop
    thermo_engine.run(single=True)

    # Checks
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 0

    # Cleaning
    manager.wipe()


def test_hessian_skip_no_aggregate():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_hessian_skip")

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
    # no aggregate set

    # Setup gear
    thermo_gear = BasicThermoDataCompletion()
    thermo_gear.options.model = model
    thermo_engine = Engine(manager.get_credentials(), fork=False)
    thermo_engine.set_gear(thermo_gear)

    # Run a single loop
    thermo_engine.run(single=True)

    # Checks
    hits = calculations.query_calculations(dumps({}))
    assert len(hits) == 0

    # Cleaning
    manager.wipe()
