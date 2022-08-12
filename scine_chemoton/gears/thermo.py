#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application imports
from ..gears import Gear
from ..utilities.queries import model_query, stationary_points, stop_on_timeout
from ..utilities.calculation_creation_helpers import finalize_calculation


class BasicThermoDataCompletion(Gear):
    """
    This Gear will autocomplete the thermochemistry data for optimized minimum
    energy structures and optimized transition states.

    Attributes
    ----------
    options :: BasicThermoDataCompletion.Options
        The options for the BasicThermoDataCompletion Gear.

    Notes
    -----
    The logic checks for 'user_optimized', 'minimum_optimized' and 'ts_optimized' structures.
    For the optimized minima only those assigned to a compound will be queried.
    If no 'gibbs_energy_correction' with the given model is present, then a
    calculation generating that data is set up (on hold).
    """

    class Options:
        """
        The options for the BasicThermoDataCompletion Gear.
        """

        __slots__ = ("cycle_time", "model", "job", "settings")

        def __init__(self):
            self.cycle_time = 101
            """
            int
                The minimum number of seconds between two cycles of the Gear.
                Cycles are finished independent of this option, thus if a cycle
                takes longer than the cycle_time will effectively lead to longer
                cycle times and not cause multiple cycles of the same Gear.
            """
            self.model: db.Model = db.Model("PM6", "PM6", "")
            """
            db.Model (Scine::Database::Model)
                The Model used for the Hessian/thermo chemistry calculations.
                The default is: PM6 using Sparrow.
            """
            self.job: db.Job = db.Job("scine_hessian")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for the Hessian/thermo chemistry calculations.
                The default is: the 'scine_hessian' order on a single core.
            """
            self.settings: utils.ValueCollection = utils.ValueCollection()
            """
            utils.ValueCollection
                Additional settings passed to the Hessian/thermo chemistry
                calculations.
                Empty by default.
            """

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self._required_collections = ["calculations", "properties", "structures"]

    def _loop_impl(self):
        # Setup query for optimized structures linked to a compound and transition states
        selection = stationary_points()
        # Loop over all results
        for structure in stop_on_timeout(self._structures.iterate_structures(dumps(selection))):
            structure.link(self._structures)
            if structure.has_property("gibbs_energy_correction"):
                if len(structure.query_properties("gibbs_energy_correction", self.options.model, self._properties)) > 0:
                    continue
            # Check if a calculation for this is already scheduled
            selection = {
                "$and": [
                    {"job.order": self.options.job.order},
                    {"structures": {"$oid": structure.id().string()}},
                ]
                + model_query(self.options.model)
            }
            if len(self._calculations.query_calculations(dumps(selection))) > 0:
                continue
            hessian = db.Calculation()
            hessian.link(self._calculations)
            hessian.create(self.options.model, self.options.job, [structure.id()])
            if self.options.settings:
                hessian.set_settings(self.options.settings)
            finalize_calculation(hessian, self._structures)
