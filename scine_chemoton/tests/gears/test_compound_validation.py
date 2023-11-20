__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
from json import dumps, load
import unittest
import numpy as np

# Third party imports
import pytest
import scine_database as db
from scine_database import test_database_setup as db_setup
import scine_utilities as utils

# Local application tests imports
from ...gears import HoldsCollections
from ..resources import resources_root_path

# Local application imports
from ...engine import Engine
from ...gears.compound import ThermoAggregateHousekeeping


class CompoundTests(unittest.TestCase, HoldsCollections):

    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "calculations",
                                      "reactions", "compounds", "flasks", "properties"]
        self.initialize_collections(manager)

    def tearDown(self) -> None:
        self._manager.wipe()

    def store_property(self, property_name: str, property_type: str, data, model, structure) -> None:
        class_ = getattr(db, property_type)
        db_property = class_()
        db_property.link(self._properties)
        db_property.create(model, property_name, structure.id(), db.ID(), data)
        structure.add_property(property_name, db_property.id())

    def test_compound_creation(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_compound_creation")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()
        # looping twice over database
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

            freq = np.asarray([1.0, 2.0, 3.0])
            self.store_property("frequencies", "VectorProperty", freq, model, structure)

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

            freq_2 = np.asarray([1.1, 2.1, 3.1])
            self.store_property("frequencies", "VectorProperty", freq_2, model, structure_2)

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

            freq_3 = np.asarray([1.2, 2.2, 3.2])
            self.store_property("frequencies", "VectorProperty", freq_3, model, structure_3)

            # Setup gear
            compound_gear = ThermoAggregateHousekeeping()
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

            manager.wipe()

    def test_validation_job_setup(self):

        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_validation_job_setup")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()

        # Setup clean database
        manager.init()

        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph",
                            "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAW"
                            "FzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                            )
        structure.set_graph("masm_decision_list", "")
        structure.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")

        settings = utils.ValueCollection({
            "optimization_attempts": 2,
        })

        # Setup gear
        compound_gear = ThermoAggregateHousekeeping()
        compound_gear.options.model = model
        compound_gear.options.validation_job = db.Job("birchermuesli")
        compound_gear.options.validation_settings = settings
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
        assert calculation.get_job().order == "birchermuesli"
        assert calculation.get_settings() == settings

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

    def test_valid_minimum_threshold(self):

        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_valid_minimum_threshold")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()

        # Setup clean database
        manager.init()

        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph",
                            "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAW"
                            "FzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                            )
        structure.set_graph("masm_decision_list", "(201,298,854,1)")
        structure.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")

        # considered a minimum
        freq = np.asarray([-14.0, 2.0, 3.0]) * utils.HARTREE_PER_INVERSE_CENTIMETER / (2 * utils.PI)
        self.store_property("frequencies", "VectorProperty", freq, model, structure)

        structure_2 = db.Structure()
        structure_2.link(self._structures)
        structure_2.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure_2.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure_2.set_graph("masm_cbor_graph",
                              "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAW"
                              "FzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                              )
        structure_2.set_graph("masm_decision_list", "(181,182,183,1)")
        structure_2.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")

        # considered a minimum
        freq_2 = np.asarray([-16.0, 2.0, 3.0]) * utils.HARTREE_PER_INVERSE_CENTIMETER / (2 * utils.PI)
        self.store_property("frequencies", "VectorProperty", freq_2, model, structure_2)

        # Setup gear
        compound_gear = ThermoAggregateHousekeeping()
        compound_gear.options.model = model
        compound_gear.options.validation_job = db.Job("birchermuesli")
        compound_gear.options.absolute_frequency_threshold = 15.0
        compound_engine = Engine(manager.get_credentials(), fork=False)
        compound_engine.set_gear(compound_gear)

        # Run a single loop
        compound_engine.run(single=True)

        # Checks
        calc_hits = self._calculations.query_calculations(dumps({}))
        assert len(calc_hits) == 1

        # Checks
        cmp_hits = self._compounds.query_compounds(dumps({}))
        assert len(cmp_hits) == 1
        cmp = cmp_hits[0]
        assert len(cmp.get_structures()) == 1
        assert cmp.get_structures()[0] == structure.id()

        # Run once more
        compound_engine.run(single=True)

        # Checks
        calc_hits = self._calculations.query_calculations(dumps({}))
        assert len(calc_hits) == 1

        # Complete calculation and disable structure 2
        calc = calc_hits[0]
        calc.set_status(db.Status.COMPLETE)

        # Run once more
        compound_engine.run(single=True)

        assert not structure_2.analyze()
        assert not structure_2.explore()
        assert structure_2.get_comment() == "Structure has imaginary frequencies larger than " + \
            str(int(compound_gear.options.absolute_frequency_threshold)) + " cm^-1\nDisabled."

    def test_valid_minimum_check_for_present_aggregate(self):

        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_valid_minimum_check_for_present_aggregate")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()

        # Setup clean database
        manager.init()

        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph",
                            "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAW"
                            "FzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                            )
        structure.set_graph("masm_decision_list", "(201,298,854,1)")
        structure.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")

        # considered a minimum
        freq = np.asarray([1.0, 2.0, 3.0]) * utils.HARTREE_PER_INVERSE_CENTIMETER / (2 * utils.PI)
        self.store_property("frequencies", "VectorProperty", freq, model, structure)

        # Setup gear
        compound_gear = ThermoAggregateHousekeeping()
        compound_gear.options.model = model
        compound_gear.options.validation_job = db.Job("birchermuesli")
        compound_gear.options.absolute_frequency_threshold = 15.0
        compound_engine = Engine(manager.get_credentials(), fork=False)
        compound_engine.set_gear(compound_gear)

        # Run a single loop
        compound_engine.run(single=True)

        structure_2 = db.Structure()
        structure_2.link(self._structures)
        structure_2.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure_2.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure_2.set_graph("masm_cbor_graph",
                              "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAW"
                              "FzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                              )
        structure_2.set_graph("masm_decision_list", "(181,182,183,1)")
        structure_2.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")

        # considered a minimum
        freq_2 = np.asarray([1.1, 2.2, 3.3]) * utils.HARTREE_PER_INVERSE_CENTIMETER / (2 * utils.PI)
        self.store_property("frequencies", "VectorProperty", freq_2, model, structure_2)

        compound_engine.run(single=True)

        # Checks
        hits = self._compounds.query_compounds(dumps({}))
        assert len(hits) == 1

        cmp = hits[0]
        assert len(cmp.get_structures()) == 2

    def test_single_atom_is_valid_minimum(self):

        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_single_atom_is_valid_minimum")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        rr = resources_root_path()

        # Setup clean database
        manager.init()

        structure = db.Structure()
        structure.link(self._structures)
        structure.create(os.path.join(rr, "iodine_atom.xyz"), 0, 2)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph",
                            "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAW"
                            "FzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                            )
        structure.set_graph("masm_decision_list", "")
        structure.set_graph("masm_idx_map", "")

        # Setup gear
        compound_gear = ThermoAggregateHousekeeping()
        compound_gear.options.model = model
        compound_gear.options.validation_job = db.Job("birchermuesli")
        compound_engine = Engine(manager.get_credentials(), fork=False)
        compound_engine.set_gear(compound_gear)

        # Run a single loop
        compound_engine.run(single=True)

        # Checks
        hits = self._compounds.query_compounds(dumps({}))
        assert len(hits) == 1
        cmp = hits[0]
        assert len(cmp.get_structures()) == 1
        assert cmp.get_structures()[0] == structure.id()

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
        # consider centroid a minimum
        freq = np.asarray([1.0, 2.0, 3.0]) * utils.HARTREE_PER_INVERSE_CENTIMETER / (2 * utils.PI)
        self.store_property("frequencies", "VectorProperty", freq, model, centroid)

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
        self.store_property("frequencies", "VectorProperty", freq, model, structure)

        compound = db.Compound()
        compound.link(self._compounds)
        compound.create([centroid.id()])
        centroid.set_aggregate(compound.id())

        # Setup gear
        compound_gear = ThermoAggregateHousekeeping()
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
        # consider centroid a minimum
        freq = np.asarray([1.0, 2.0, 3.0]) * utils.HARTREE_PER_INVERSE_CENTIMETER / (2 * utils.PI)
        self.store_property("frequencies", "VectorProperty", freq, model, structure)

        starting_structure = db.Structure(db.ID(), self._structures)
        starting_structure.create(os.path.join(rr, "water.xyz"), 0, 1)
        starting_structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        starting_structure.set_graph("masm_cbor_graph", "original")
        starting_structure.set_aggregate(db.ID())
        self.store_property("frequencies", "VectorProperty", freq, model, starting_structure)

        minimization = db.Calculation(db.ID(), manager.get_collection("calculations"))
        minimization.create(model, db.Job("scine_geometry_optimization"), [starting_structure.id()])
        results = db.Results()
        results.add_structure(structure.id())
        minimization.set_results(results)
        minimization.set_status(db.Status.COMPLETE)

        # Setup gear
        compound_gear = ThermoAggregateHousekeeping()
        compound_gear.options.model = model
        compound_gear.options.graph_job = db.Job("testy_mac_test_face")
        compound_engine = Engine(manager.get_credentials(), fork=False)
        compound_engine.set_gear(compound_gear)

        # Run a single loop
        compound_engine.run(single=True)

        assert not structure.has_aggregate()
        assert structure.get_label() == db.Label.IRRELEVANT

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

        freq = np.asarray([1.0, 2.0, 3.0]) * utils.HARTREE_PER_INVERSE_CENTIMETER / (2 * utils.PI)
        self.store_property("frequencies", "VectorProperty", freq, model, structure)

        # Setup gear
        compound_gear = ThermoAggregateHousekeeping()
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
