__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
from json import dumps
import unittest
import numpy as np

# Third party imports
import scine_database as db
from scine_database import test_database_setup as db_setup
import scine_utilities as utils

# Local application tests imports
from scine_chemoton.gears import HoldsCollections
from ...resources import resources_root_path

# Local application imports
from scine_chemoton.engine import Engine
from scine_chemoton.gears.network_refinement.structure_refinement.minimum_structure_reoptimization import (
    MinimumStructureReoptimization
)


class MinimumStructureReoptimizationTests(unittest.TestCase, HoldsCollections):

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

    def create_comp_flask(self):
        rr = resources_root_path()
        model = db.Model("FAKE", "FAKE", "F-AKE")
        # Setup clean database
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

        freq = np.asarray([1.0, 2.0, 3.0])
        self.store_property("frequencies", "VectorProperty", freq, model, structure)

        c = db.Compound(db.ID(), self._compounds)
        c.create([])
        c.add_structure(structure.id())
        c.disable_exploration()

        structure_2 = db.Structure()
        structure_2.link(self._structures)
        structure_2.create(os.path.join(rr, "h4o2.xyz"), 1, 2)
        structure_2.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure_2.set_graph("masm_cbor_graph",
                              "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAW"
                              "FzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA;"
                              "pGFhgaRhYQBhYwJhcqNhbIKBAIEBYmxygYIAAWFzgYIAAW"
                              "FzAWFjD2FnomFFgoMAAgCDAQIAYVqDAQEIYXaDAQAA"
                              )
        structure_2.set_graph("masm_decision_list", "(181,182,183,1)")
        structure_2.set_graph("masm_idx_map", "(0, 0), (0, 1), (0, 2)")

        freq_2 = np.asarray([1.1, 2.1, 3.1])
        self.store_property("frequencies", "VectorProperty", freq_2, model, structure_2)

        f = db.Flask(db.ID(), self._flasks)
        f.create([], [])
        f.add_structure(structure_2.id())
        f.disable_exploration()

    def test_default_method(self):
        manager = db_setup.get_clean_db("chemoton_test_both_creation")
        self.custom_setup(manager)
        model = db.Model("FAKE", "FAKE", "F-AKE")

        self.create_comp_flask()

        structure_gear = MinimumStructureReoptimization()
        structure_gear.options.model = model
        structure_gear.options.refine_model = db.Model("gfn2", "gfn2", "")
        structure_gear.options.reoptimization_job_settings = utils.ValueCollection(
            {"convergence_max_iterations": 100}
        )
        structure_engine = Engine(manager.get_credentials(), fork=False)
        structure_engine.set_gear(structure_gear)

        structure_engine.run(single=True)

        # checks if 2 calculations are set up
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 2

        # check calculation methods
        for calc in self._calculations.iterate_all_calculations():
            calc.link(self._calculations)
            assert calc.get_model().method_family == 'gfn2'
            assert calc.get_job().order == "scine_geometry_optimization"

        structure_engine.run(single=True)

        # should still be 2 calculations
        hits2 = self._calculations.query_calculations(dumps({}))
        assert len(hits2) == 2

        structure_gear.clear_cache()
        structure_engine.run(single=True)

        hits3 = self._calculations.query_calculations(dumps({}))
        assert len(hits3) == 2

        manager.wipe()

    def test_compound_only(self):
        manager = db_setup.get_clean_db("chemoton_test_compound_creation")
        self.custom_setup(manager)
        model = db.Model("FAKE", "FAKE", "F-AKE")

        self.create_comp_flask()

        structure_gear = MinimumStructureReoptimization()
        structure_gear.options.model = model
        structure_gear.options.refine_model = db.Model("gfn2", "gfn2", "")
        structure_gear.options.reoptimize_flasks = False
        structure_gear.options.reoptimize_centroid_only = True
        structure_engine = Engine(manager.get_credentials(), fork=False)
        structure_engine.set_gear(structure_gear)

        structure_engine.run(single=True)

        # checks if 1 calculations is set up
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 1

        manager.wipe()

    def test_flasks_only(self):
        manager = db_setup.get_clean_db("chemoton_test_flask_creation")
        self.custom_setup(manager)
        model = db.Model("FAKE", "FAKE", "F-AKE")

        self.create_comp_flask()

        structure_gear = MinimumStructureReoptimization()
        structure_gear.options.model = model
        structure_gear.options.refine_model = db.Model("gfn2", "gfn2", "")
        structure_gear.options.reoptimize_compounds = False
        structure_engine = Engine(manager.get_credentials(), fork=False)
        structure_engine.set_gear(structure_gear)

        structure_engine.run(single=True)

        # checks if 1 calculation is set up
        hits = self._calculations.query_calculations(dumps({}))
        assert len(hits) == 1

        manager.wipe()

    def test_model_error(self):
        manager = db_setup.get_clean_db("chemoton_test_error_creation")
        self.custom_setup(manager)
        model = db.Model("gfn2", "gfn2", "")

        self.create_comp_flask()

        structure_gear = MinimumStructureReoptimization()
        structure_gear.options.model = model
        structure_gear.options.refine_model = model
        structure_engine = Engine(manager.get_credentials(), fork=False)
        structure_engine.set_gear(structure_gear)

        self.assertRaises(RuntimeError, structure_engine.run)

        manager.wipe()
