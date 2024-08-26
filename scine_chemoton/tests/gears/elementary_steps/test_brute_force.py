#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
import json
import unittest

# Third party imports
import scine_database as db
from scine_database import test_database_setup as db_setup

# Local application tests imports
from ....gears import HoldsCollections
from ...resources import resources_root_path

# Local application imports
from ....engine import Engine
from ....gears.elementary_steps.brute_force import BruteForceElementarySteps
from .mock_trial_generator import MockGenerator


class ElementaryStepBruteForceTests(unittest.TestCase, HoldsCollections):

    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "calculations",
                                      "reactions", "compounds", "flasks", "properties"]
        self.initialize_collections(manager)

    def tearDown(self) -> None:
        self._manager.wipe()

    def test_bimol(self):
        """
        Test whether the correct number of bimolecular combinations is probed
        """
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_minimal_bimol")
        self.custom_setup(manager)

        # Add fake data
        rr = resources_root_path()
        cheap_model = db.Model("cheap", "CHEAP", "")
        for mol in ["hydrogenperoxide", "water"]:
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([])
            for i in range(3):
                graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
                structure = db.Structure()
                structure.link(self._structures)
                structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
                structure.set_label(db.Label.USER_OPTIMIZED)
                structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
                structure.set_graph("masm_idx_map", graph["masm_idx_map"])
                structure.set_graph("masm_decision_list", str(i))
                structure.set_model(cheap_model)
                compound.add_structure(structure.id())
                structure.set_aggregate(compound.id())

        # Setup gear
        es_gear = BruteForceElementarySteps()
        es_gear.trial_generator = MockGenerator()
        es_gear.options.model = cheap_model
        es_gear.options.enable_unimolecular_trials = False
        es_gear.options.enable_bimolecular_trials = True
        es_gear.options.structure_model = db.Model("expensive", "EXPENSIVE", "")
        es_engine = Engine(manager.get_credentials(), fork=False)
        es_engine.set_gear(es_gear)

        # Run a single loop
        es_engine.run(single=True)

        # Expected numbers due to model mismatch:
        # Unimolecular: 0
        # Bimolecular: 0
        assert es_gear.trial_generator.unimol_counter == 0
        assert es_gear.trial_generator.bimol_counter == 0

        # Run again with proper model
        es_gear.options.structure_model = cheap_model
        es_engine.run(single=True)

        # Expected numbers:
        # Unimolecular: 0
        # Bimolecular: 2 compound combinations with 2 structures per compound: 4 choose 2 with repetition = 10
        assert es_gear.trial_generator.unimol_counter == 0
        assert es_gear.trial_generator.bimol_counter == 21

    def test_unimol(self):
        """
        Test whether the correct number of unimolecular combinations is probed
        """
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_minimal_bimol")
        self.custom_setup(manager)

        # Add fake data
        rr = resources_root_path()
        cheap_model = db.Model("cheap", "CHEAP", "")
        for mol in ["hydrogenperoxide", "water", "cyclohexene"]:
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([])
            for i in range(4):
                graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
                structure = db.Structure()
                structure.link(self._structures)
                structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
                structure.set_label(db.Label.USER_OPTIMIZED)
                structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
                structure.set_graph("masm_idx_map", graph["masm_idx_map"])
                structure.set_graph("masm_decision_list", str(i))
                structure.set_model(cheap_model)
                compound.add_structure(structure.id())
                structure.set_aggregate(compound.id())

        # Setup gear
        es_gear = BruteForceElementarySteps()
        es_gear.trial_generator = MockGenerator()
        es_gear.options.model = cheap_model
        es_gear.options.enable_unimolecular_trials = True
        es_gear.options.enable_bimolecular_trials = False
        es_gear.options.structure_model = db.Model("expensive", "EXPENSIVE", "")
        es_engine = Engine(manager.get_credentials(), fork=False)
        es_engine.set_gear(es_gear)

        # Run a single loop
        es_engine.run(single=True)

        # Expected numbers due to model mismatch:
        # Unimolecular: 0
        # Bimolecular: 0
        assert es_gear.trial_generator.unimol_counter == 0
        assert es_gear.trial_generator.bimol_counter == 0

        # Run again with proper model
        es_gear.options.structure_model = cheap_model
        es_engine.run(single=True)

        # Expected numbers:
        # Unimolecular: 3 compounds with 4 structures each: 3*4 = 12
        # Bimolecular: 0
        assert es_gear.trial_generator.unimol_counter == 12
        assert es_gear.trial_generator.bimol_counter == 0

    def test_unimol_bimol(self):
        """
        Test whether the correct number of unimolecular and bimolecular combinations
        is probed
        """
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_minimal_unimol_bimol")
        self.custom_setup(manager)

        # Add fake data
        rr = resources_root_path()
        for mol in ["hydrogenperoxide", "cyclohexene", "water"]:
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([])
            # Adding more structures per compound should not have an effect
            for i in range(3):
                graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
                structure = db.Structure()
                structure.link(self._structures)
                structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
                structure.set_label(db.Label.USER_OPTIMIZED)
                structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
                structure.set_graph("masm_idx_map", graph["masm_idx_map"])
                structure.set_graph("masm_decision_list", str(i))
                compound.add_structure(structure.id())
                structure.set_aggregate(compound.id())

        # Setup gear
        es_gear = BruteForceElementarySteps()
        es_gear.trial_generator = MockGenerator()
        cheap_model = db.Model("cheap", "CHEAP", "")
        es_gear.options.model = cheap_model
        es_gear.options.enable_unimolecular_trials = True
        es_gear.options.enable_bimolecular_trials = True
        es_engine = Engine(manager.get_credentials(), fork=False)
        es_engine.set_gear(es_gear)

        # Run a single loop
        es_engine.run(single=True)

        # Expected numbers:
        # Unimolecular: 3 compounds with 3 structures: 9
        # Bimolecular: 9 choose 2 with repetion: 45
        assert es_gear.trial_generator.unimol_counter == 9
        assert es_gear.trial_generator.bimol_counter == 45
