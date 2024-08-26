#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import unittest
from json import dumps

# Third party imports
import scine_database as db
import scine_utilities as utils
from scine_database import test_database_setup as db_setup

# Local application tests imports
from ...gears import HoldsCollections
from scine_chemoton.gears.single_point import SinglePoint
from scine_chemoton.engine import Engine
from scine_chemoton.filters.structure_filters import StructureLabelFilter, StructureFilter, ModelFilter


class SinglePointTests(unittest.TestCase, HoldsCollections):

    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "calculations", "structures"]
        self.initialize_collections(manager)

    def tearDown(self) -> None:
        self._manager.wipe()

    def test_single_point_gear(self):
        manager = db_setup.get_clean_db("chemoton_test_single_point_gear")
        self.custom_setup(manager)
        _, s1_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.USER_GUESS)
        _, __ = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_GUESS)
        _, __ = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)

        gear = SinglePoint()
        gear.options.model = db_setup.get_fake_model()
        gear.structure_filter = StructureLabelFilter([db.Label.COMPLEX_OPTIMIZED])
        engine = Engine(manager.get_credentials(), fork=False)
        engine.set_gear(gear)
        for _ in range(2):
            engine.run(single=True)

        assert self._calculations.count(dumps({})) == 0

        gear.structure_filter = StructureFilter()
        gear.options.allowed_labels = [db.Label.USER_GUESS]
        engine.run(single=True)

        assert self._calculations.count(dumps({})) == 1
        engine.run(single=True)
        assert self._calculations.count(dumps({})) == 1
        calculation = self._calculations.get_one_calculation(dumps({}))
        calculation.link(self._calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_structures()[0] == s1_id

        gear.options.allowed_labels = None
        engine.run(single=True)

        assert self._calculations.count(dumps({})) == 3

        gear.options.job_settings = utils.ValueCollection({"some_settings": "some_value"})
        engine.run(single=True)

        assert self._calculations.count(dumps({})) == 6
        assert self._calculations.count(dumps({"settings.some_settings": "some_value"})) == 3

        other_model = db.Model("DFT", "any", "any")
        _, s4_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        s4 = db.Structure(s4_id, self._structures)
        gear.structure_filter = ModelFilter(other_model)
        engine.run(single=True)
        # unchanged
        assert self._calculations.count(dumps({})) == 6
        assert self._calculations.count(dumps({"settings.some_settings": "some_value"})) == 3
        # fitting model
        s4.set_model(other_model)
        engine.run(single=True)
        assert self._calculations.count(dumps({})) == 7
        assert self._calculations.count(dumps({"settings.some_settings": "some_value"})) == 4
