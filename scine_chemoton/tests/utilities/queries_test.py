#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import unittest

import scine_database as db
from ...utilities.queries import identical_reaction, model_query
from .. import test_database_setup as db_setup


class QueriesTest(unittest.TestCase):
    def test_model_query(self):
        model = db.Model("PM6", "PM6", "")
        q = model_query(model)
        assert {"model.version": "any"} not in q
        assert {'model.method': 'PM6'} in q

    def test_identical_reaction_query(self):
        manager = db_setup.get_clean_db("chemoton_test_identical_reaction_query")
        reactions = manager.get_collection("reactions")
        # set up compounds
        c1 = db_setup.insert_single_empty_structure_compound(manager, db.Label.USER_OPTIMIZED)[0]
        c2 = db_setup.insert_single_empty_structure_compound(manager, db.Label.USER_OPTIMIZED)[0]
        c3 = db_setup.insert_single_empty_structure_compound(manager, db.Label.MINIMUM_OPTIMIZED)[0]
        c4 = db_setup.insert_single_empty_structure_compound(manager, db.Label.MINIMUM_OPTIMIZED)[0]
        # set up reactions
        r1 = db.Reaction()
        r1.link(reactions)
        r1.create([c1], [c3])
        assert identical_reaction([c1], [c3], reactions)
        assert identical_reaction([c3], [c1], reactions)
        assert not identical_reaction([c1], [c3, c4], reactions)
        assert not identical_reaction([c3, c4], [c1], reactions)
        r2 = db.Reaction()
        r2.link(reactions)
        r2.create([c1], [c3, c4])
        assert identical_reaction([c1], [c3, c4], reactions)
        assert identical_reaction([c3, c4], [c1], reactions)
        r3 = db.Reaction()
        r3.link(reactions)
        r3.create([c2, c1], [c3, c4])
        assert identical_reaction([c1, c2], [c3, c4], reactions)
        assert identical_reaction([c3, c4], [c1, c2], reactions)
        assert identical_reaction([c3, c4], [c2, c1], reactions)
        assert not identical_reaction([c3], [c2, c1], reactions)
        assert not identical_reaction([c2, c1], [c3], reactions)
