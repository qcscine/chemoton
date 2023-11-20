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
from ....gears.elementary_steps.minimum_energy_conformer import MinimumEnergyConformerElementarySteps
from .mock_trial_generator import MockGenerator


class ElementaryStepMinimumEnergyConformerTests(unittest.TestCase, HoldsCollections):

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
        manager = db_setup.get_clean_db("chemoton_minimum_energy_bimol")
        self.custom_setup(manager)

        # Add fake data
        rr = resources_root_path()
        fake_model = db.Model("FAKE", "FAKE", "F-AKE")
        compound_list = []
        for mol in ["hydrogenperoxide", "water"]:
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([])
            compound_list.append(compound)
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
                structure.set_compound(compound.id())
                energy_property = db.NumberProperty.make("electronic_energy", fake_model, -i, self._properties)
                structure.add_property("electronic_energy", energy_property.id())
                boltzmann_property = db.NumberProperty.make("boltzmann_weight", fake_model, i, self._properties)
                structure.add_property("boltzmann_weight", boltzmann_property.id())
                structure.set_model(fake_model)

        # Setup gear
        es_gear = MinimumEnergyConformerElementarySteps()
        es_gear.options.model = fake_model
        es_gear.options.energy_upper_bound = 0.0
        es_gear.options.max_number_structures = 1
        es_gear.trial_generator = MockGenerator()
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
        es_gear.options.structure_model = fake_model
        es_engine.run(single=True)

        # Expected numbers:
        # Unimolecular: 0
        # Bimolecular: 2 choose 2 with repetition: 3 (H2O + H2O, H2O + H2O2, H2O2)
        assert es_gear.trial_generator.unimol_counter == 0
        assert es_gear.trial_generator.bimol_counter == 3

    def test_unimol(self):
        """
        Test whether the correct number of unimolecular combinations is probed
        """
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_minimum_energy_unimol")
        self.custom_setup(manager)

        # Add fake data
        rr = resources_root_path()
        fake_model = db.Model("FAKE", "FAKE", "F-AKE")
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
                compound.add_structure(structure.id())
                structure.set_compound(compound.id())
                energy_property = db.NumberProperty.make("electronic_energy", fake_model, -i, self._properties)
                structure.add_property("electronic_energy", energy_property.id())
                boltzmann_property = db.NumberProperty.make("boltzmann_weight", fake_model, i, self._properties)
                structure.add_property("boltzmann_weight", boltzmann_property.id())
                structure.set_model(fake_model)

        # Setup gear
        es_gear = MinimumEnergyConformerElementarySteps()
        es_gear.options.model = fake_model
        es_gear.options.energy_upper_bound = 0.0
        es_gear.options.max_number_structures = 1
        es_gear.trial_generator = MockGenerator()
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
        es_gear.options.structure_model = fake_model
        es_engine.run(single=True)

        # Expected numbers:
        # Unimolecular: 2 (once for every compound)
        # Bimolecular: 0
        assert es_gear.trial_generator.unimol_counter == 2
        assert es_gear.trial_generator.bimol_counter == 0

    def test_unimol_bimol(self):
        """
        Test whether the correct number of unimolecular combinations is probed
        """
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_minimum_energy_unimol_bimol")
        self.custom_setup(manager)

        # Add fake data
        rr = resources_root_path()
        fake_model = db.Model("FAKE", "FAKE", "F-AKE")
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
                compound.add_structure(structure.id())
                structure.set_compound(compound.id())
                energy_property = db.NumberProperty.make("electronic_energy", fake_model, -i, self._properties)
                structure.add_property("electronic_energy", energy_property.id())
                boltzmann_property = db.NumberProperty.make("boltzmann_weight", fake_model, i, self._properties)
                structure.add_property("boltzmann_weight", boltzmann_property.id())
                structure.set_model(fake_model)

        # Setup gear
        es_gear = MinimumEnergyConformerElementarySteps()
        es_gear.options.model = fake_model
        es_gear.options.energy_upper_bound = 0.0
        es_gear.options.max_number_structures = 1
        es_gear.trial_generator = MockGenerator()
        es_gear.options.enable_unimolecular_trials = True
        es_gear.options.enable_bimolecular_trials = True
        es_engine = Engine(manager.get_credentials(), fork=False)
        es_engine.set_gear(es_gear)

        # Run a single loop
        es_engine.run(single=True)

        assert es_gear.trial_generator.unimol_counter == 2
        assert es_gear.trial_generator.bimol_counter == 3

        # Setup gear
        es_gear = MinimumEnergyConformerElementarySteps()
        es_gear.options.model = fake_model
        es_gear.options.energy_upper_bound = 3000.0
        es_gear.options.max_number_structures = 5
        es_gear.trial_generator = MockGenerator()
        es_gear.options.enable_unimolecular_trials = False
        es_gear.options.enable_bimolecular_trials = True
        es_engine = Engine(manager.get_credentials(), fork=False)
        es_engine.set_gear(es_gear)

        # Run a single loop
        es_engine.run(single=True)

        assert es_gear.trial_generator.unimol_counter == 0
        assert es_gear.trial_generator.bimol_counter == 7
