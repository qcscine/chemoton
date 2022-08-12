#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Third party imports
import scine_database as db

# Local application tests imports
from .. import test_database_setup as db_setup

# Local application imports
from ...engine import Engine
from ...gears.scheduler import Scheduler


def test_counts_and_priorities():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_scheduling")

    # Initialize some test settings
    job_creation = {
        "clean_laundry": 2,
        "mow_the_lawn": 11,
        "build_flux_compensator": 7,
        "schroedingers_job": 1,
    }
    job_counts = {
        "clean_laundry": 0,
        "mow_the_lawn": 9,
        "build_flux_compensator": 5,
    }
    job_priorities = {
        "clean_laundry": 9,
        "mow_the_lawn": 5,
        "build_flux_compensator": 1,
    }

    # Add fake calculations
    model = db.Model("FAKE", "FAKE", "F-AKE")
    calculations = manager.get_collection("calculations")
    for k, v in job_creation.items():
        job = db.Job(k)
        for _ in range(v + 2):
            new_calc = db.Calculation()
            new_calc.link(calculations)
            new_calc.create(model, job, [])
            new_calc.set_status(db.Status.HOLD)
            new_calc.set_priority(10)

    # Setup gear
    schedule_gear = Scheduler()
    schedule_gear.options.job_counts = job_counts
    schedule_gear.options.job_priorities = job_priorities
    schedule_engine = Engine(manager.get_credentials(), fork=False)
    schedule_engine.set_gear(schedule_gear)

    # Run a single loop
    schedule_engine.run(single=True)

    # Checks
    actual_counts = {
        "clean_laundry": 0,
        "mow_the_lawn": 0,
        "build_flux_compensator": 0,
        "schroedingers_job": 0,
    }
    for calc in calculations.iterate_all_calculations():
        calc.link(calculations)
        order = calc.get_job().order
        assert order in actual_counts
        if db.Status.NEW == calc.get_status():
            actual_counts[order] += 1
            assert calc.get_priority() == job_priorities[order]

    for k, v in job_counts.items():
        assert v == actual_counts[k]
    assert 0 == actual_counts["schroedingers_job"]

    # Cleaning
    manager.wipe()
