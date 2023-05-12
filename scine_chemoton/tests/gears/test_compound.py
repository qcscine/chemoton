#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
from json import dumps, load
import unittest

# Third party imports
import pytest
import scine_database as db

# Local application tests imports
from .. import test_database_setup as db_setup
from ...gears import HoldsCollections
from ..resources import resources_root_path

# Local application imports
from ...engine import Engine
from ...gears.compound import BasicAggregateHousekeeping


class CompoundTests(unittest.TestCase, HoldsCollections):

    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "calculations",
                                      "reactions", "compounds", "flasks", "properties"]
        self.initialize_collections(manager)

    def tearDown(self) -> None:
        self._manager.wipe()

    def test_compound_creation(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_compound_creation")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()
        for label in [db.Label.MINIMUM_OPTIMIZED, db.Label.USER_OPTIMIZED]:
            # Setup clean database
            manager.init()

            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, "water.xyz"), 0, 1)
            structure.set_label(label)
            structure.set_graph("masm_cbor_graph",
                                "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAW"
                                "FzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                                )
            structure.set_graph("masm_decision_list", "")
            structure.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")

            structure_2 = db.Structure()
            structure_2.link(self._structures)
            structure_2.create(os.path.join(rr, "water.xyz"), 1, 2)
            structure_2.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure_2.set_graph("masm_cbor_graph",
                                  "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAW"
                                  "FzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                                  )
            structure_2.set_graph("masm_decision_list", "(181,182,183,1)")
            structure_2.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")

            structure_3 = db.Structure()
            structure_3.link(self._structures)
            structure_3.create(os.path.join(rr, "water.xyz"), 0, 3)
            structure_3.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure_3.set_graph("masm_cbor_graph",
                                  "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAW"
                                  "FzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                                  )
            structure_3.set_graph("masm_decision_list", "(181,182,183,1)")
            structure_3.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")

            # Setup gear
            compound_gear = BasicAggregateHousekeeping()
            compound_gear.options.model = model
            compound_engine = Engine(manager.get_credentials(), fork=False)
            compound_engine.set_gear(compound_gear)

            # Run a single loop
            compound_engine.run(single=True)

            # Checks
            hits = self._compounds.query_compounds(dumps({}))
            assert len(hits) == 3

            all_structure_ids = [structure.id(), structure_2.id(), structure_3.id()]
            for compound in self._compounds.iterate_all_compounds():
                compound.link(self._compounds)
                assert len(compound.get_structures()) == 1
                assert compound.get_structures()[0] in all_structure_ids
                assert compound.get_centroid() in all_structure_ids
            for struc in [structure, structure_2, structure_3]:
                assert struc.has_aggregate()

            # Wipe database before starting next loop
            manager.wipe()

    def test_compound_extension(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_compound_extension")
        self.custom_setup(manager)

        # Add fake data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()
        centroid = db.Structure()
        centroid.link(self._structures)
        centroid.create(os.path.join(rr, "water.xyz"), 0, 1)
        centroid.set_label(db.Label.USER_OPTIMIZED)
        centroid.set_graph("masm_cbor_graph",
                           "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAW"
                           "FzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                           )
        centroid.set_graph("masm_decision_list", "(1,2,3,1)")
        centroid.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")
        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph",
                            "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAW"
                            "FzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                            )
        structure.set_graph("masm_decision_list", "(181,182,183,1)")
        structure.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")

        compound = db.Compound()
        compound.link(self._compounds)
        compound.create([centroid.id()])
        centroid.set_aggregate(compound.id())

        # Setup gear
        compound_gear = BasicAggregateHousekeeping()
        compound_gear.options.model = model
        compound_engine = Engine(manager.get_credentials(), fork=False)
        compound_engine.set_gear(compound_gear)

        # Run a single loop
        compound_engine.run(single=True)

        # Checks
        assert len(compound.get_structures()) == 2
        assert compound.get_structures()[1].string() == structure.id().string()
        assert compound.get_centroid().string() == centroid.id().string()
        assert structure.has_aggregate()
        assert structure.get_aggregate().string() == compound.id().string()
        assert structure.get_label() == db.Label.MINIMUM_OPTIMIZED

    def test_intermediate_deduplication(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_intermediate_deduplication")
        self.custom_setup(manager)

        # Add fake data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()
        centroid = db.Structure()
        centroid.link(self._structures)
        centroid.create(os.path.join(rr, "water.xyz"), 0, 1)
        centroid.set_label(db.Label.MINIMUM_OPTIMIZED)
        centroid.set_graph("masm_cbor_graph",
                           "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWF"
                           "zAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                           )
        centroid.set_graph("masm_decision_list", "(10,20,40,1):(110,120,140,1)")
        centroid.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")
        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph",
                            "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWF"
                            "zAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                            )
        structure.set_graph("masm_decision_list", "(10,30,40,1):(110,130,140,1)")
        structure.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")
        compound = db.Compound()
        compound.link(self._compounds)
        compound.create([centroid.id()])
        centroid.set_aggregate(compound.id())

        # Setup gear
        compound_gear = BasicAggregateHousekeeping()
        compound_gear.options.model = model
        compound_engine = Engine(manager.get_credentials(), fork=False)
        compound_engine.set_gear(compound_gear)

        # Run a single loop
        compound_engine.run(single=True)

        # Checks
        assert len(compound.get_structures()) == 1  # duplicate are not added to aggregate
        assert compound.get_centroid().string() == centroid.id().string()
        # but we can still use aggregate methods on the structure
        assert not structure.has_aggregate(recursive=False)
        assert structure.has_aggregate()
        assert structure.get_aggregate().string() == compound.id().string()
        assert structure.get_label() == db.Label.DUPLICATE
        assert structure.get_original() == centroid.id()

    def test_intermediate_deduplication_empty_dlist(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_intermediate_deduplication_empty_dlist")
        self.custom_setup(manager)

        # Add fake data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()
        centroid = db.Structure()
        centroid.link(self._structures)
        centroid.create(os.path.join(rr, "water.xyz"), 0, 1)
        centroid.set_label(db.Label.MINIMUM_OPTIMIZED)
        centroid.set_graph("masm_cbor_graph",
                           "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWFzAWF"
                           "jD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                           )
        centroid.set_graph("masm_decision_list", "")
        centroid.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")
        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph",
                            "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWFzAWF"
                            "jD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                            )
        structure.set_graph("masm_decision_list", "")
        structure.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")
        compound = db.Compound()
        compound.link(self._compounds)
        compound.create([centroid.id()])
        centroid.set_aggregate(compound.id())

        # Setup gear
        compound_gear = BasicAggregateHousekeeping()
        compound_gear.options.model = model
        compound_engine = Engine(manager.get_credentials(), fork=False)
        compound_engine.set_gear(compound_gear)

        # Run a single loop
        compound_engine.run(single=True)

        # Checks
        assert len(compound.get_structures()) == 1  # duplicate are not added to aggregate
        assert compound.get_centroid().string() == centroid.id().string()
        # but we can still use aggregate methods on the structure
        assert structure.has_aggregate()
        assert structure.get_aggregate().string() == compound.id().string()
        assert structure.get_label() == db.Label.DUPLICATE
        assert structure.get_original() == centroid.id()

    @pytest.mark.filterwarnings("ignore:.+received incorrect label:UserWarning")
    def test_irrelevant_structure(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_irrelevant_structure")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()
        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph",
                            "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWF"
                            "zAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                            )
        structure.set_graph("masm_decision_list", "(10,30,40,1)")
        structure.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")

        starting_structure = db.Structure(db.ID(), self._structures)
        starting_structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        starting_structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        starting_structure.set_graph("masm_cbor_graph", "original")
        starting_structure.set_aggregate(db.ID())

        minimization = db.Calculation(db.ID(), manager.get_collection("calculations"))
        minimization.create(model, db.Job("scine_geometry_optimization"), [starting_structure.id()])
        results = db.Results()
        results.add_structure(structure.id())
        minimization.set_results(results)
        minimization.set_status(db.Status.COMPLETE)

        # Setup gear
        compound_gear = BasicAggregateHousekeeping()
        compound_gear.options.model = model
        compound_gear.options.graph_job = db.Job("testy_mac_test_face")
        compound_engine = Engine(manager.get_credentials(), fork=False)
        compound_engine.set_gear(compound_gear)

        # Run a single loop
        compound_engine.run(single=True)

        assert not structure.has_aggregate()
        assert structure.get_label() == db.Label.IRRELEVANT

    def test_graph_job_setup(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_graph_job_setup")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()
        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.add_property("bond_orders", db.ID())

        # Setup gear
        compound_gear = BasicAggregateHousekeeping()
        compound_gear.options.model = model
        compound_gear.options.graph_job = db.Job("testy_mac_test_face")
        compound_engine = Engine(manager.get_credentials(), fork=False)
        compound_engine.set_gear(compound_gear)

        # Run a single loop
        compound_engine.run(single=True)

        # Checks
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 1

        calculation = db.Calculation(hits[0].id())
        calculation.link(self._calculations)

        assert len(calculation.get_structures()) == 1
        assert calculation.get_structures()[0].string() == structure.id().string()
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "testy_mac_test_face"

        # Run a second time
        compound_engine.run(single=True)
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 1

        # Rerun with a different model
        model2 = db.Model("FAKE2", "", "")
        compound_gear.options.model = model2
        compound_engine.run(single=True)
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 2

    def test_bo_job_setup(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_bo_job_setup")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()
        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)

        # Setup gear
        compound_gear = BasicAggregateHousekeeping()
        compound_gear.options.model = model
        compound_gear.options.bond_order_job = db.Job("eggs_bacon_and_spam")
        compound_engine = Engine(manager.get_credentials(), fork=False)
        compound_engine.set_gear(compound_gear)

        # Run a single loop
        compound_engine.run(single=True)

        # Checks
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 1

        calculation = db.Calculation(hits[0].id())
        calculation.link(self._calculations)

        assert len(calculation.get_structures()) == 1
        assert calculation.get_structures()[0].string() == structure.id().string()
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "eggs_bacon_and_spam"

        # Run a second time
        compound_engine.run(single=True)
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 1

        # Rerun with a different model
        model2 = db.Model("FAKE2", "", "")
        compound_gear.options.model = model2
        compound_engine.run(single=True)
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 2

    def test_flask_creation(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_flask_creation")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "", "")
        rr = resources_root_path()
        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "proline_propanal_complex.xyz"), 0, 1)
        structure.set_label(db.Label.COMPLEX_OPTIMIZED)
        graph = load(open(os.path.join(rr, "proline_propanal_complex.json"), "r"))
        structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", graph["masm_idx_map"])
        structure.set_graph("masm_decision_list", graph["masm_decision_list"])

        # Setup gear
        compound_gear = BasicAggregateHousekeeping()
        compound_gear.options.model = model
        compound_engine = Engine(manager.get_credentials(), fork=False)
        compound_engine.set_gear(compound_gear)

        # Run a single loop
        compound_engine.run(single=True)

        # Checks
        hits = self._flasks.query_flasks(dumps({}))
        assert len(hits) == 1

        flask = db.Flask(hits[0].id())
        flask.link(self._flasks)

        assert len(flask.get_structures()) == 1
        assert flask.get_structures()[0].string() == structure.id().string()
        assert flask.get_centroid().string() == structure.id().string()
        assert structure.has_aggregate()
        assert structure.get_aggregate().string() == flask.id().string()

    def test_flask_to_compound_mapping(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_flask_to_compound_mapping")
        self.custom_setup(manager)

        _, s_id1 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        _, s_id2 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        structure_1 = db.Structure(s_id1, self._structures)
        structure_2 = db.Structure(s_id2, self._structures)
        graph_1 = "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWFzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
        graph_2 = ("pGFhgqRhYQBhYwJhcqNhbIKBAYEDYmxygoEAgQFhc4KBAYEDYXMBpGFhAGFjA2"
                   "Fyo2FsgoEAgQJibHKCgQCBAWFzgoEAgQJhcwFhYw9hZ6JhRYODAAMAgwECAIMC"
                   "AwBhWoQBAQgIYXaDAQAA")
        structure_1.set_graph("masm_cbor_graph", graph_1)
        structure_1.set_graph("masm_decision_list", "(10,30,40,1)")
        structure_1.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")
        structure_2.set_graph("masm_cbor_graph", graph_2)
        structure_2.set_graph("masm_decision_list", "(10,30,40,1):(1, 3, 5, 8)")
        structure_2.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()
        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(db.Label.COMPLEX_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", graph_1 + ";" + graph_2)
        structure.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2), (1, 3), (1, 4), (1, 5)")

        # Setup gear
        compound_gear = BasicAggregateHousekeeping()
        compound_gear.options.model = model
        compound_engine = Engine(manager.get_credentials(), fork=False)
        compound_engine.set_gear(compound_gear)

        assert self._flasks.count(dumps({})) == 0
        # Run a single loop
        compound_engine.run(single=True)

        # Checks
        assert self._flasks.count(dumps({})) == 1
        assert self._flasks.count(dumps({"compounds": {"$size": 0}})) == 1

        # Run a second time
        compound_engine.run(single=True)
        assert self._flasks.count(dumps({})) == 1
        assert self._flasks.count(dumps({"compounds": {"$size": 0}})) == 1

        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(db.Label.COMPLEX_OPTIMIZED)
        graph_3 = ("pWFhhqRhYQBhYwphcqRhbIOBAYELgQ1jbG5rgaJhcIIBAmNzZXGGCgsMDw4NYmxyg4EAgQGB"
                   "AmFzg4EBgQuBDWFzAqRhYQBhYwthcqRhbIOBAIEKgQxjbG5rgaJhcIIBAmNzZXGGCwoNDg8M"
                   "Ymxyg4EAgQGBAmFzg4EAgQqBDGFzAqRhYQBhYwxhcqRhbISBCIEJgQuBD2NsbmuBomFwggID"
                   "Y3NlcYYMCwoNDg9ibHKDggABgQKBA2Fzg4IICYELgQ9hcwWkYWEAYWMNYXKkYWyEgQaBB4EK"
                   "gQ5jbG5rgaJhcIICA2NzZXGGDQoLDA8OYmxyg4IAAYECgQNhc4OCBgeBCoEOYXMFpGFhAGFj"
                   "DmFypGFshIEEgQWBDYEPY2xua4GiYXCCAgNjc2Vxhg4NCgsMD2JscoOCAAGBAoEDYXODggQF"
                   "gQ2BD2FzBaRhYQBhYw9hcqRhbISBAoEDgQyBDmNsbmuBomFwggIDY3NlcYYPDAsKDQ5ibHKD"
                   "ggABgQKBA2Fzg4ICA4EMgQ5hcwVhYoGiYWEAYWWCCgthYw9hZ6JhRZCDAAsAgwEKAIMCDwCD"
                   "Aw8AgwQOAIMFDgCDBg0AgwcNAIMIDACDCQwAgwoLAYMKDQCDCwwAgwwPAIMNDgCDDg8AYVqQ"
                   "AQEBAQEBAQEBAQYGBgYGBmF2gwEAAA==")
        structure.set_graph("masm_cbor_graph", graph_1 + ";" + graph_3)
        structure.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2), (1, 3), (1, 4), (1, 5)")
        compound_engine.run(single=True)
        assert self._flasks.count(dumps({})) == 2
        assert self._flasks.count(dumps({"compounds": {"$size": 0}})) == 2

    def test_unique_structures_loading(self):
        """
        Test if the loading of compound already in the database works.
        This test will set up two compounds with structures, decision lists, and graphs. The
        structures are then cached as unique structures for each compound. We then check if
        a third structures is correctly identified as a duplicate.
        """
        manager = db_setup.get_clean_db("chemoton_test_unique_structures_loading")
        self.custom_setup(manager)

        _, s_id1 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        _, s_id2 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        structure_1 = db.Structure(s_id1, self._structures)
        structure_2 = db.Structure(s_id2, self._structures)
        graph_1 = "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAWFzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
        graph_2 = ("pGFhgqRhYQBhYwJhcqNhbIKBAYEDYmxygoEAgQFhc4KBAYEDYXMBpGFhAGFjA2"
                   "Fyo2FsgoEAgQJibHKCgQCBAWFzgoEAgQJhcwFhYw9hZ6JhRYODAAMAgwECAIMC"
                   "AwBhWoQBAQgIYXaDAQAA")
        structure_1.set_graph("masm_cbor_graph", graph_1)
        structure_1.set_graph("masm_decision_list", "(10,30,40,1)")
        structure_1.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")
        structure_2.set_graph("masm_cbor_graph", graph_2)
        structure_2.set_graph("masm_decision_list", "(10,30,40,1):(1, 3, 5, 8)")
        structure_2.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()
        duplicate_1 = db.Structure()
        duplicate_1.link(self._structures)
        duplicate_1.create(os.path.join(rr, "water.xyz"), 0, 1)
        duplicate_1.set_label(db.Label.MINIMUM_OPTIMIZED)
        duplicate_1.set_graph("masm_cbor_graph", graph_1)
        duplicate_1.set_graph("masm_decision_list", "(10,30,40,1)")
        duplicate_1.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")
        duplicate_1.set_model(model)

        duplicate_2 = db.Structure()
        duplicate_2.link(self._structures)
        duplicate_2.create(os.path.join(rr, "water.xyz"), 0, 1)
        duplicate_2.set_label(db.Label.MINIMUM_OPTIMIZED)
        duplicate_2.set_graph("masm_cbor_graph", graph_2)
        duplicate_2.set_graph("masm_decision_list", "(29, 39, 49, 1):(2, 4, 7, 8)")
        duplicate_2.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")
        duplicate_2.set_model(model)

        # Setup gear
        compound_gear = BasicAggregateHousekeeping()
        compound_gear.options.model = model
        compound_engine = Engine(manager.get_credentials(), fork=False)
        compound_engine.set_gear(compound_gear)

        # Run a single loop
        compound_engine.run(single=True)

        assert duplicate_1.get_label() == db.Label.DUPLICATE
        assert duplicate_2.get_label() == db.Label.DUPLICATE
