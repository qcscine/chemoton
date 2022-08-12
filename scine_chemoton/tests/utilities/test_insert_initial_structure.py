#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""


# Standard library imports
import os
import warnings

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application tests imports
from .. import test_database_setup as db_setup
from ..resources import resources_root_path

# Local application imports
from ...utilities.insert_initial_structure import insert_initial_structure


def test_insert_initial_structure():

    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_insert_initial_structure")

    # Add structure data
    rr = resources_root_path()

    # Fake model
    model = db.Model("FAKE", "FAKE", "F-AKE")
    # Insert structure from file with default settings
    water_path = os.path.join(rr, "water.xyz")
    struct, calc = insert_initial_structure(manager, water_path, 0, 1, model)

    assert struct.get_charge() == 0
    assert struct.get_multiplicity() == 1
    assert struct.get_label().name == "USER_GUESS"
    assert struct.get_atoms().size() == 3
    assert calc.get_job().order == "scine_geometry_optimization"
    assert calc.get_model().method_family == model.method_family
    assert calc.get_status() == db.Status.NEW
    assert calc.get_priority() == 1
    assert calc.get_structures()[0].string() == struct.get_id().string()
    assert not calc.get_settings()

    # Insert structure from atom collection with non-default settings
    arginine = utils.io.read(os.path.join(rr, "arginine.xyz"))[0]
    fake_settings = utils.ValueCollection({"fake_setting": "fake_value"})
    with warnings.catch_warnings(record=True) as warning:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        struct, calc = insert_initial_structure(
            manager, arginine, 1, 2, model, db.Label.IRRELEVANT, db.Job("fake_job"), fake_settings
        )
        assert len(warning) == 1
        assert "user_guess" in str(warning[-1].message)

    assert struct.get_charge() == 1
    assert struct.get_multiplicity() == 2
    assert struct.get_label().name == "IRRELEVANT"
    assert struct.get_atoms().size() == 26
    assert calc.get_job().order == "fake_job"
    assert calc.get_model().method_family == model.method_family
    assert calc.get_status() == db.Status.NEW
    assert calc.get_priority() == 1
    assert calc.get_structures()[0].string() == struct.get_id().string()
    assert dict(calc.get_settings()) == dict(fake_settings)

    # Cleaning
    manager.wipe()
