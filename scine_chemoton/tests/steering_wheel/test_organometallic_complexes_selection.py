#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import load
import os
import unittest

# Third party imports
import scine_database as db
from scine_database import test_database_setup as db_setup

# Local application tests imports
from ..resources import resources_root_path

# Local application imports
from scine_chemoton.steering_wheel.selections.organometallic_complexes import CentralMetalSelection
from scine_chemoton.utilities import connect_to_db
from scine_chemoton.filters.aggregate_filters import CatalystFilter
from scine_chemoton.filters.reactive_site_filters import CentralSiteFilter


class OrganometallicComplexesSelectionTests(unittest.TestCase):

    def setUp(self) -> None:
        manager = db_setup.get_clean_db(f"chemoton_{self.__class__.__name__}")
        self.credentials = manager.get_credentials()
        self.manager = connect_to_db(self.credentials)
        self.structures = self.manager.get_collection("structures")
        self.compounds = self.manager.get_collection("compounds")
        rr = resources_root_path()
        self.model = db_setup.get_fake_model()

        self.cat_name = "grubbs"
        self.other_name = "proline_acid_propanal_product"

        self.example_structures = {}
        self.example_compounds = {}
        for name in [self.cat_name, self.other_name]:

            path = os.path.join(rr, f"{name}.xyz")

            structure = db.Structure(db.ID(), self.structures)
            structure.create(path, 0, 1, self.model, db.Label.MINIMUM_OPTIMIZED)

            compound = db.Compound(db.ID(), self.compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())

            graph = load(open(os.path.join(rr, f"{name}.json"), "r"))
            structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
            structure.set_graph("masm_idx_map", graph["masm_idx_map"])
            structure.set_graph("masm_decision_list", graph["masm_decision_list"])

            self.example_structures[name] = structure
            self.example_compounds[name] = compound

    def tearDown(self) -> None:
        self.manager.wipe()

    def test_result_access_fails(self):
        sele = CentralMetalSelection(self.model, "Ru", ligand_without_metal_reactive=False)
        with self.assertRaises(PermissionError) as context:
            _ = sele.get_step_result()
        self.assertTrue('may not access the step_result member' in str(context.exception))

    def test_ligand_ligand(self):
        sele = CentralMetalSelection(self.model, "Ru", ligand_without_metal_reactive=False)
        result = sele(self.credentials)
        assert len(result.structures) == 0
        assert isinstance(result.aggregate_filter, CatalystFilter)
        result.aggregate_filter.initialize_collections(self.manager)
        for k, v in self.example_compounds.items():
            assert result.aggregate_filter.filter(v) == (k == self.cat_name)
        assert isinstance(result.reactive_site_filter, CentralSiteFilter)
