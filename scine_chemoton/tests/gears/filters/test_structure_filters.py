#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
import unittest

# Third party imports
import scine_database as db
from scine_database import test_database_setup as db_setup

# Local application tests imports
from ....gears import HoldsCollections
from ...resources import resources_root_path
from scine_chemoton.filters.structure_filters import (
    StructureFilter,
    StructureLabelFilter,
    StructureFilterOrArray,
    StructureFilterAndArray
)
from scine_chemoton.filters.reactive_site_filters import ReactiveSiteFilter


class StructureFiltersTests(unittest.TestCase, HoldsCollections):

    def custom_setup(self, manager: db.Manager):
        self._required_collections = ["manager", "elementary_steps", "structures", "reactions", "compounds", "flasks",
                                      "properties"]
        self.initialize_collections(manager)

    def tearDown(self) -> None:
        if hasattr(self, "_manager"):
            self._manager.wipe()

    def add_structures_to_db(self):
        rr = resources_root_path()
        test_structures = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_structures[xyz] = structure
        return test_structures

    def test_default_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_default_filter")
        self.custom_setup(manager)

        # Add structure data
        test_structures = self.add_structures_to_db()

        # Setup filter
        f = StructureFilter()

        # Filter and check
        for i in ["water", "arginine", "cyclohexene"]:
            assert f.filter(test_structures[i])
            for j in ["water", "arginine", "cyclohexene"]:
                assert f.filter(test_structures[i], test_structures[j])

    def test_filter_chain_failure_logic_and(self):
        with self.assertRaises(TypeError) as context:
            _ = (StructureFilter() & ReactiveSiteFilter())
        self.assertTrue('StructureFilter' in str(context.exception))

    def test_filter_chain_failure_logic_or(self):
        with self.assertRaises(TypeError) as context:
            _ = (StructureFilter() | ReactiveSiteFilter())
        self.assertTrue('StructureFilter' in str(context.exception))

    def test_filter_chain_failure_class_and(self):
        with self.assertRaises(TypeError) as context:
            StructureFilterAndArray([StructureFilter(), ReactiveSiteFilter()])
        self.assertTrue('StructureFilterAndArray' in str(context.exception))

    def test_filter_chain_failure_class_or(self):
        with self.assertRaises(TypeError) as context:
            StructureFilterOrArray([StructureFilter(), ReactiveSiteFilter()])
        self.assertTrue('StructureFilterOrArray' in str(context.exception))

    def test_filter_chain_derived_classes_or(self):
        _ = StructureFilterOrArray([StructureFilter(), StructureLabelFilter([db.Label.MINIMUM_OPTIMIZED])])
        _ = StructureFilter() & StructureLabelFilter([db.Label.MINIMUM_OPTIMIZED])
        _ = StructureFilter() and StructureLabelFilter([db.Label.MINIMUM_OPTIMIZED])

    def test_structure_label_filter(self):
        manager = db_setup.get_clean_db("chemoton_test_structure_label_filter")
        self.custom_setup(manager)
        test_structures = self.add_structures_to_db()
        f = StructureLabelFilter([db.Label.MINIMUM_OPTIMIZED])

        for i in ["water", "arginine", "cyclohexene"]:
            assert f.filter(test_structures[i])
            for j in ["water", "arginine", "cyclohexene"]:
                assert f.filter(test_structures[i], test_structures[j])

        for i in ["water", "arginine", "cyclohexene"]:
            test_structures[i].set_label(db.Label.COMPLEX_OPTIMIZED)

        for i in ["water", "arginine", "cyclohexene"]:
            assert not f.filter(test_structures[i])
            for j in ["water", "arginine", "cyclohexene"]:
                assert not f.filter(test_structures[i], test_structures[j])

        test_structures["water"].set_label(db.Label.MINIMUM_OPTIMIZED)
        assert f.filter(test_structures["water"])
        assert not f.filter(test_structures["water"], test_structures["arginine"])
