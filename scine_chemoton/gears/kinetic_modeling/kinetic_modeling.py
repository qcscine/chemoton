#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""
# Standard library imports
from time import sleep
from json import dumps
from copy import deepcopy
# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application imports
from ...utilities.queries import (
    model_query,
    stop_on_timeout
)
from .. import Gear
from ...utilities.energy_query_functions import get_barriers_for_elementary_step_by_type
from .prepare_kinetic_modeling_job import KineticModelingJobFactory


class KineticModeling(Gear):
    """
    This gear sets up kinetic modeling jobs. The frequency of the kinetic modeling jobs.

    Attributes
    ----------
    options :: KineticModeling.Options
        The options for the KineticModeling gear.

    Notes
    -----
    The kinetic modeling can be run in a sleepy mode. In this mode, kinetic modeling jobs are only set up
    if all other jobs are terminated (not new, hold, or pending).
    """
    class Options(Gear.Options):
        """
        The options for the KineticModeling Gear.
        """

        __slots__ = (
            "cycle_time",
            "elementary_step_interval",
            "energy_label",
            "job",
            "time_step",
            "solver",
            "batch_interval",
            "n_batches",
            "convergence",
            "sleeper_mode",
            "ts_energy_threshold_deduplication",
            "rate_from_lowest_conformer",
            "use_spline_barrier",
            "max_barrier",
            "min_barrier_intermolecular",
            "min_barrier_intramolecular",
            "min_flux_truncation",
            "diffusion_controlled_barrierless",
            "use_max_flux_for_truncation"
        )

        def __init__(self):
            super().__init__()
            self.cycle_time = 30
            """
            int
                The minimum number of seconds between two cycles of the Gear.
                Cycles are finished independent of this option, thus if a cycle
                takes longer than the cycle_time will effectively lead to longer
                cycle times and not cause multiple cycles of the same Gear.
            """
            self.job = db.Job('kinetx_kinetic_modeling')
            """
            db.Job
                The kinetic modeling job to set up.
            """
            self.elementary_step_interval = 100
            """
            int
                Set up a kinetic modeling job if at least this number of new
                elementary steps were added to the network and are eligible according
                to reaction-barrier cut-off, and electronic structure model.
            """
            self.energy_label = "electronic_energy"
            """
            str
                The property lable to be used as the energy for the reaction rate determination.
            """
            self.time_step = 1e-8
            """
            float
                The kinetic modeling time step. This may be overwritten if an automatic time-step
                selection is used in the kinetic modeling solver.
            """
            self.solver = "cash_karp_5"
            """
            str
                The label of the numerical integration scheme for the kinetic model.
            """
            self.batch_interval = 1000
            """
            int
                The numerical integration may be performed batch-wise. This is the number of
                time steps to be integrated in each batch.
            """
            self.n_batches = 100000
            """
            int
                The maximum number of integration batches.
            """
            self.convergence = 1e-10
            """
            float
                Stop the numerical integration if the maximum concentration change between two
                consecutive batches is lower than this threshold.
            """
            self.sleeper_mode = False
            """
            bool
                Run the kinetic modeling only if all other jobs are terminated.
            """
            self.ts_energy_threshold_deduplication = 1e-5
            """
            float
                If two reactions are mirrors/inverses of one another all elementary
                steps with the same transition state energy are eliminated. This
                threshold determines the energy tolerance for the elimination.
            """
            self.rate_from_lowest_conformer = True
            """
            bool
                Calculate the reaction rate always with respect to the energy of the
                lowest energy conformer without any reweighing. Alternatively a Boltzmann
                distribution is assumed for all conformers.
            """
            self.use_spline_barrier = False
            """
            bool
                If true, the reaction barrier is calculated from the spline interpolation.
            """
            self.max_barrier = 1000
            """
            float
                The maximum allowed barrier in kJ/mol.
            """
            self.min_barrier_intermolecular = 0.0  # 15.943960 max rate of 1e+10
            """
            float
                The minimum allowed barrier in kJ/mol for intermolecular reactions.
            """
            self.min_barrier_intramolecular = 0.0
            """
            float
                The minimum allowed barrier in kJ/mol for intramolecular reactions.
            """
            self.min_flux_truncation = 1e-9
            """
            float
                Minimum flux of all aggregates in a reaction in a previous kinetic modeling job. If the flux is lower
                than this threshold, the reaction is excluded from the kinetic modeling.
            """
            self.diffusion_controlled_barrierless = True
            """
            bool
                If true, all barrierless reactions are assigned the maximum rate according to the
                min_barrier_intramolecular and min_barrier_intermolecular settings.
            """
            self.use_max_flux_for_truncation = False
            """
            bool
                If true, the maximum entry of all concentration fluxes for a given reaction is used for the flux based
                truncation.
            """

    def __init__(self):
        super().__init__()
        self._required_collections = ["calculations", "elementary_steps", "properties", "reactions", "structures",
                                      "flasks", "compounds"]
        self._n_reactions_started_last = 0

    def _loop_impl(self):
        # Allow only one kinetic modeling calculation to be queuing at a time
        n_already_set_up = self._get_n_queuing_kinetic_modeling_calculations()
        if n_already_set_up > 0:
            return
        # Run the kinetic modeling if all other calculations are done, independent on the elementary-step interval.
        n_calc_still_waiting = self._get_n_queuing_calculations()
        if self.options.sleeper_mode and n_calc_still_waiting > 0:
            return
        if self.options.sleeper_mode and not self._reaction_gear_finished():
            return
        n_qualified_reactions = 0
        if n_calc_still_waiting > 0 and not self.options.sleeper_mode:
            n_qualified_reactions = self._get_n_qualified_reactions()

        if (n_qualified_reactions - self._n_reactions_started_last) // self.options.elementary_step_interval > 0:
            self._set_up_job(n_qualified_reactions)
        elif n_calc_still_waiting == 0:
            sleep(self.options.cycle_time)
            n_calc_still_waiting = self._get_n_queuing_calculations()
            if n_calc_still_waiting > 0:
                return
            self._set_up_job(n_qualified_reactions)

    def _get_n_queuing_kinetic_modeling_calculations(self):
        selection = {
            "$and": [
                {"job.order": {"$eq": self.options.job.order}},
                {"$or": [{"status": "new"}, {"status": "hold"}, {"status": "pending"}]},
            ]
            + model_query(self.options.model)
        }
        return self._calculations.count(dumps(selection))

    def _get_n_queuing_calculations(self) -> int:
        selection = {
            "$and": [
                {"$or": [{"status": "new"}, {"status": "hold"}, {"status": "pending"}]},
            ]
        }
        return self._calculations.count(dumps(selection))

    def _reaction_gear_finished(self) -> bool:
        selection = {
            "reaction": ""
        }
        return self._elementary_steps.count(dumps(selection)) == 0

    def _get_n_qualified_reactions(self):
        n_qualified_reactions = 0
        for reaction in stop_on_timeout(self._reactions.iterate_reactions(dumps({}))):
            reaction.link(self._reactions)
            elementary_steps = reaction.get_elementary_steps()
            for elementary_step_id in elementary_steps:
                elementary_step = db.ElementaryStep(elementary_step_id, self._elementary_steps)
                lhs, rhs = get_barriers_for_elementary_step_by_type(elementary_step,
                                                                    self.options.energy_label,
                                                                    self.options.model,
                                                                    self._structures,
                                                                    self._properties)
                if lhs is None or rhs is None:
                    continue
                else:
                    n_qualified_reactions += 1
        return n_qualified_reactions

    def _set_up_job(self, n_qualified_reactions: int):
        if self.stop_at_next_break_point:
            return
        self._n_reactions_started_last = n_qualified_reactions
        job_factory = KineticModelingJobFactory(deepcopy(self.options.model), self._manager,
                                                self.options.energy_label, self.options.job,
                                                self.options.ts_energy_threshold_deduplication,
                                                self.options.rate_from_lowest_conformer,
                                                self.options.use_spline_barrier, self.options.max_barrier,
                                                self.options.min_barrier_intermolecular,
                                                self.options.min_barrier_intramolecular,
                                                self.options.min_flux_truncation, "",
                                                self.options.diffusion_controlled_barrierless,
                                                self.options.use_max_flux_for_truncation)
        settings = utils.ValueCollection({})
        settings["time_step"] = self.options.time_step
        settings["solver"] = self.options.solver
        settings["batch_interval"] = self.options.batch_interval
        settings["n_batches"] = self.options.n_batches
        settings["energy_label"] = self.options.energy_label
        settings["convergence"] = self.options.convergence
        job_factory.create_kinetic_modeling_job(settings)
