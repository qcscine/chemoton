#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import unittest

import scine_database as db
from .. import test_database_setup as db_setup
from ...utilities.queries import (
    identical_reaction,
    model_query,
    calculation_exists_in_structure,
    get_calculation_id_from_structure,
    get_calculation_id
)


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
        c1 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.USER_OPTIMIZED)[0]
        c2 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.USER_OPTIMIZED)[0]
        c3 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)[0]
        c4 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)[0]
        # set up reactions
        r1 = db.Reaction()
        r1.link(reactions)
        r1.create([c1], [c3])
        cl1 = [db.CompoundOrFlask.COMPOUND]
        cl2 = [db.CompoundOrFlask.COMPOUND, db.CompoundOrFlask.COMPOUND]
        assert identical_reaction([c1], [c3], cl1, cl1, reactions)
        assert identical_reaction([c3], [c1], cl1, cl1, reactions)
        assert not identical_reaction([c1], [c3, c4], cl1, cl2, reactions)
        assert not identical_reaction([c3, c4], [c1], cl2, cl1, reactions)
        r2 = db.Reaction()
        r2.link(reactions)
        r2.create([c1], [c3, c4])
        assert identical_reaction([c1], [c3, c4], cl1, cl2, reactions)
        assert identical_reaction([c3, c4], [c1], cl2, cl1, reactions)
        r3 = db.Reaction()
        r3.link(reactions)
        r3.create([c2, c1], [c3, c4])
        assert identical_reaction([c1, c2], [c3, c4], cl2, cl2, reactions)
        assert identical_reaction([c3, c4], [c1, c2], cl2, cl2, reactions)
        assert identical_reaction([c3, c4], [c2, c1], cl2, cl2, reactions)
        assert not identical_reaction([c3], [c2, c1], cl1, cl2, reactions)
        assert not identical_reaction([c2, c1], [c3], cl2, cl1, reactions)
        # Cleaning
        manager.wipe()

    def test_calculation_exists_in_structure(self):
        import scine_utilities as utils
        manager = db_setup.get_clean_db("chemoton_test_calculation_exists_in_structure")
        calculations = manager.get_collection("calculations")
        structures = manager.get_collection("structures")

        _, s_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_GUESS)
        _, s_id_2 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_GUESS)

        model = db.Model("FAKE", "FAKE", "F-AKE")
        wrong_model = db.Model("WRONG", "wrong", "wrong")
        calculation = db.Calculation(db.ID(), calculations)
        job = db.Job("some_job")
        calculation.create(model, job, [s_id])
        settings = {
            "some_settings": "some_value"
        }
        settings_collection = utils.ValueCollection(settings)
        calculation.set_settings(settings_collection)
        assert get_calculation_id_from_structure(job.order, [s_id], model, structures, calculations) is None

        structure = db.Structure(s_id, structures)
        structure.add_calculation(calculation.job.order, calculation.id())

        assert get_calculation_id_from_structure(job.order, [s_id], model, structures, calculations) == calculation.id()
        assert get_calculation_id_from_structure(job.order, [s_id], model, structures, calculations, settings)
        assert get_calculation_id_from_structure(job.order, [s_id], model, structures, calculations,
                                                 settings_collection)
        assert get_calculation_id_from_structure(job.order, [s_id_2], model, structures, calculations, settings) is None
        assert get_calculation_id_from_structure(job.order, [s_id], wrong_model, structures, calculations) is None
        assert not calculation_exists_in_structure("wrong_order", [s_id], wrong_model, structures, calculations)
        # without structure shortcut
        assert get_calculation_id(job.order, [s_id], model, calculations) == calculation.id()
        assert get_calculation_id(job.order, [s_id], model, calculations, settings)
        assert get_calculation_id(job.order, [s_id], model, calculations, settings_collection)
        assert get_calculation_id(job.order, [s_id_2], model, calculations, settings) is None
        assert get_calculation_id(job.order, [s_id], wrong_model, calculations) is None
        assert get_calculation_id("wrong_order", [s_id], wrong_model, calculations) is None
        # Cleaning
        manager.wipe()
