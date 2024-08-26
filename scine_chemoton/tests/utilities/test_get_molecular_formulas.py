#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
import json

import unittest
from scine_chemoton.gears import HoldsCollections

import scine_database as db
from scine_database import test_database_setup as db_setup

# local imports
from ..resources import resources_root_path
from ...utilities.get_molecular_formula import (
    get_molecular_formula_of_structure,
    get_molecular_formula_of_compound,
    get_molecular_formula_of_flask,
    get_molecular_formula_of_aggregate
)


class GetMolecularFormulaTest(unittest.TestCase, HoldsCollections):

    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "reactions", "compounds", "flasks",
                                      "properties"]
        self.initialize_collections(manager)

    def tearDown(self) -> None:
        self._manager.wipe()

    def test_get_molecular_formula_of_structure(self):
        manager = db_setup.get_clean_db("chemoton_test_get_molecular_formula_of_structure")
        self.custom_setup(manager)
        # Add structure to database
        structure = db.Structure()
        structure.link(self._structures)
        rr = resources_root_path()
        structure.create(os.path.join(rr, "proline_acid_propanal_product.xyz"), 0, 1)

        assert get_molecular_formula_of_structure(structure.id(), self._structures) == "C8H15NO3(c:0, m:1)"
        structure.set_charge(-99)
        assert get_molecular_formula_of_structure(structure.id(), self._structures) == "C8H15NO3(c:-99, m:1)"
        structure.set_multiplicity(99)
        assert get_molecular_formula_of_structure(structure.id(), self._structures) == "C8H15NO3(c:-99, m:99)"

    def test_get_molecular_formula_of_compound(self):
        manager = db_setup.get_clean_db("chemoton_test_get_molecular_formula_of_compound")
        self.custom_setup(manager)
        compound_id, _ = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)

        assert get_molecular_formula_of_compound(compound_id, self._compounds, self._structures) == "H2O(c:0, m:1)"

    def test_get_molecular_formula_of_flask(self):
        manager = db_setup.get_clean_db("chemoton_test_get_molecular_formula_of_flask_and_aggregates")
        self.custom_setup(manager)

        flask = db.Flask()
        flask.link(self._flasks)
        flask.create([], [], exploration_disabled=True)
        # Structure of flask
        flask_structure = db.Structure()
        flask_structure.link(self._structures)
        flask_structure.create(os.path.join(resources_root_path(), "h4o2.xyz"), 0, 1)
        flask_structure.set_label(db.Label.COMPLEX_OPTIMIZED)
        flask_structure.set_aggregate(flask.get_id())
        graph = json.load(open(os.path.join(resources_root_path(), "water" + ".json"), "r"))
        flask_structure.set_graph("masm_cbor_graph", ";".join([graph['masm_cbor_graph']] * 2))
        flask.add_structure(flask_structure.get_id())

        water, _ = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)

        ref_flask_string = "H4O2(c:0, m:1) [H2O | H2O]"
        assert get_molecular_formula_of_flask(flask.id(), self._flasks, self._structures) == ref_flask_string

        assert get_molecular_formula_of_aggregate(water, db.CompoundOrFlask.COMPOUND,
                                                  self._compounds, self._flasks, self._structures) == "H2O(c:0, m:1)"
        assert get_molecular_formula_of_aggregate(flask.id(), db.CompoundOrFlask.FLASK,
                                                  self._compounds, self._flasks, self._structures) == ref_flask_string
