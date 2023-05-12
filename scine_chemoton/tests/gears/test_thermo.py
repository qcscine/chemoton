#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
import unittest
from json import dumps

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application tests imports
from .. import test_database_setup as db_setup
from ...gears import HoldsCollections
from ..resources import resources_root_path

# Local application imports
from ...engine import Engine
from ...gears.elementary_steps.aggregate_filters import MolecularWeightFilter
from ...gears.thermo import BasicThermoDataCompletion


class ThermoTests(unittest.TestCase, HoldsCollections):

    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "reactions", "compounds", "flasks",
                                      "properties", "calculations"]
        self.initialize_collections(manager)

    def tearDown(self) -> None:
        self._manager.wipe()

    def test_hessian_calculation(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_hessian_calculation")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()
        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(db.Label.USER_OPTIMIZED)
        compound = db.Compound(db.ID(), self._compounds)
        compound.create([structure.id()])
        structure.set_aggregate(compound.id())

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
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 1

        calculation = db.Calculation(hits[0].id())
        calculation.link(self._calculations)

        assert len(calculation.get_structures()) == 1
        assert calculation.get_structures()[0].string() == structure.id().string()
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "fake_hessian"

        # Run a second loop
        thermo_gear.clear_cache()
        thermo_engine.set_gear(thermo_gear)
        thermo_engine.run(single=True)

        # Check again
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 1

        # Rerun with a different model
        model2 = db.Model("FAKE2", "", "")
        thermo_gear.options.model = model2
        thermo_gear.clear_cache()
        thermo_engine.set_gear(thermo_gear)
        thermo_engine.run(single=True)
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 2

    def test_hessian_skip(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_hessian_skip")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()
        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        compound = db.Compound(db.ID(), self._compounds)
        compound.create([structure.id()])
        structure.set_aggregate(compound.id())

        # Add property
        gibbs_energy_correction = db.NumberProperty()
        gibbs_energy_correction.link(self._properties)
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
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 0

    def test_hessian_skip_no_aggregate(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_hessian_skip")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "", "")
        rr = resources_root_path()
        structure = db.Structure()
        structure.link(self._structures)
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
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 0

    def test_hessian_skip_with_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_hessian_skip_with_filter")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()
        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(db.Label.USER_OPTIMIZED)
        compound = db.Compound(db.ID(), self._compounds)
        compound.create([structure.id()])
        structure.set_aggregate(compound.id())

        # Setup gear
        thermo_gear = BasicThermoDataCompletion()
        thermo_gear.aggregate_filter = MolecularWeightFilter(1.0)
        thermo_gear.options.model = model
        thermo_gear.options.job = db.Job("fake_hessian")
        thermo_gear.options.settings = utils.ValueCollection({"some_thing": "else"})
        thermo_engine = Engine(manager.get_credentials(), fork=False)
        thermo_engine.set_gear(thermo_gear)

        # Run a single loop with filter -> no result
        thermo_engine.run(single=True)
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 0

        # Run a single loop after filter has been adjusted
        thermo_gear.aggregate_filter = MolecularWeightFilter(100.0)
        thermo_gear.clear_cache()
        thermo_engine.set_gear(thermo_gear)
        thermo_engine.run(single=True)

        # Checks
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 1

        calculation = db.Calculation(hits[0].id(), self._calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_structures()[0].string() == structure.id().string()
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "fake_hessian"

        # Run a second loop
        thermo_engine.run(single=True)

        # Check again
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 1

        # Rerun with a different model
        model2 = db.Model("FAKE2", "", "")
        thermo_gear.options.model = model2
        thermo_gear.clear_cache()
        thermo_engine.set_gear(thermo_gear)
        thermo_engine.run(single=True)
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 2

    def test_hessian_with_structure_model(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_hessian_with_structure_model")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()
        # First structure
        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(db.Label.USER_OPTIMIZED)
        structure.set_model(model)
        # Second structure
        model2 = db.Model("fake", "fake", "f-ake")
        structure2 = db.Structure()
        structure2.link(self._structures)
        structure2.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure2.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure2.set_model(model2)

        compound = db.Compound(db.ID(), self._compounds)
        compound.create([structure.id()])
        structure.set_aggregate(compound.id())
        compound.add_structure(structure2.id())
        structure2.set_aggregate(compound.id())

        # Setup gear
        thermo_gear = BasicThermoDataCompletion()
        thermo_gear.options.model = model
        thermo_gear.options.structure_model = model
        thermo_engine = Engine(manager.get_credentials(), fork=False)
        thermo_engine.set_gear(thermo_gear)

        # Run a single loop
        thermo_engine.run(single=True)

        # Checks
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 1
        assert hits[0].get_structures()[0] == structure.id()


def test_wrong_structure_model():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_wrong_structure_model")

    # Get collections
    structures = manager.get_collection("structures")
    calculations = manager.get_collection("calculations")

    # Add structure data
    model = db.Model("FAKE", "", "")
    rr = resources_root_path()
    structure = db.Structure()
    structure.link(structures)
    structure.create(os.path.join(rr, "water.xyz"), 0, 1)
    structure.set_label(db.Label.TS_OPTIMIZED)
    structure.set_aggregate(db.ID())

    # Setup gear
    thermo_gear = BasicThermoDataCompletion()
    thermo_gear.options.model = model
    thermo_gear.options.structure_model = model
    thermo_gear.options.structure_model.solvent = "solvent"
    thermo_engine = Engine(manager.get_credentials(), fork=False)
    thermo_engine.set_gear(thermo_gear)

    # Run a single loop
    thermo_engine.run(single=True)

    # Checks
    assert calculations.count(dumps({})) == 0

    # Cleaning
    manager.wipe()
