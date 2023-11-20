#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""
# Standard library imports
from time import sleep
from json import dumps
from typing import Optional

# Third party imports
import scine_database as db
import scine_utilities as utils
from scine_database.queries import model_query

# Local application imports
from .. import Gear
from .prepare_kinetic_modeling_job import KineticModelingJobFactory
from .rms_kinetic_modeling import RMSKineticModelingJobFactory
from .kinetx_kinetic_modeling import KinetxKineticModelingJobFactory
from .thermodynamic_properties import ReferenceState


class KineticModeling(Gear):
    """
    This gear sets up kinetic modeling jobs as soon as no other jobs are pending/waiting.

    Attributes
    ----------
    options :: KineticModeling.Options
        The options for the KineticModeling gear.
    """
    class Options(Gear.Options):
        """
        The options for the KineticModeling Gear.
        """

        __slots__ = (
            "electronic_model",
            "hessian_model",
            "elementary_step_interval",
            "job",
            "sleeper_mode",
            "max_barrier",
            "min_flux_truncation",
            "reference_state",
            "job_settings",
            "only_electronic"
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
            int
                Set up a kinetic modeling job if at least this number of new
                elementary steps were added to the network and are eligible according
                to reaction-barrier cut-off, and electronic structure model.
            """
            self.electronic_model = db.Model("PM6", "", "")
            """
            db.Model
                The electronic structure model used for the electronic energy calculations.
            """
            self.hessian_model = db.Model("PM6", "", "")
            """
            db.Model
                The electronic structure model used for the hessian calculations.
            """
            self.max_barrier: float = 300.0  # in kJ/mol
            """
            float
                The maximum barrier to consider in the kinetic modeling.
            """
            self.min_flux_truncation: float = 1e-9
            """
            float
                Minimum flux of all aggregates in a reaction in a previous kinetic modeling job. If the flux is lower
                than this threshold, the reaction is excluded from the kinetic modeling.
            """
            self.reference_state: Optional[ReferenceState] = None
            """
            ReferenceState
                The thermodynamic reference state defined by temperature (in K) and pressure (in Pa).
            """
            self.job_settings = utils.ValueCollection({})
            """
            ValueCollection
                The job settings. See get_default_settings(job) for default settings.
            """
            self.only_electronic = False
            """
            bool
                If true, only the electronic energies are used for the rate constant calculations.
            """

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self._required_collections = ["calculations", "elementary_steps"]
        self._job_factory: Optional[KineticModelingJobFactory] = None

    def _get_job_factory(self) -> KineticModelingJobFactory:
        if self._job_factory is None:
            if self.options.job == RMSKineticModelingJobFactory.get_job():
                self._job_factory = RMSKineticModelingJobFactory(self.options.electronic_model,
                                                                 self.options.hessian_model,
                                                                 self._manager, self.options.only_electronic)
            elif self.options.job == KinetxKineticModelingJobFactory.get_job():
                self._job_factory = KinetxKineticModelingJobFactory(self.options.electronic_model,
                                                                    self.options.hessian_model,
                                                                    self._manager, self.options.only_electronic)
            else:
                raise RuntimeError("Error: The given kinetic modeling job is not supported. Options are:\n"
                                   + RMSKineticModelingJobFactory.get_job().order + "\n"
                                   + KinetxKineticModelingJobFactory.get_job().order)
        if self.options.reference_state is None:
            self.options.reference_state = ReferenceState(float(self.options.electronic_model.temperature),
                                                          float(self.options.electronic_model.pressure))
        self._job_factory.reference_state = self.options.reference_state
        self._job_factory.max_barrier = self.options.max_barrier
        self._job_factory.min_flux_truncation = self.options.min_flux_truncation
        return self._job_factory

    @staticmethod
    def get_default_settings(job: db.Job) -> utils.ValueCollection:
        if job.order == RMSKineticModelingJobFactory.get_job().order:
            return RMSKineticModelingJobFactory.get_default_settings()
        elif job.order == KinetxKineticModelingJobFactory.get_job().order:
            return KinetxKineticModelingJobFactory.get_default_settings()
        else:
            raise RuntimeError("Error: The given kinetic modeling job is not supported. Options are:\n"
                               + RMSKineticModelingJobFactory.get_job().order + "\n"
                               + KinetxKineticModelingJobFactory.get_job().order)

    def start_conditions_are_met(self) -> bool:
        # Allow only one kinetic modeling calculation to be queuing at a time
        if self._get_n_queuing_kinetic_modeling_calculations() > 0:
            return False
        # Run the kinetic modeling if all other calculations are done, independent on the elementary-step interval.
        if self._get_n_queuing_calculations() > 0:
            return False
        if not self._reaction_gear_finished():
            return False
        return True

    def _loop_impl(self):
        if not self.start_conditions_are_met() or self.stop_at_next_break_point:
            return
        sleep(self.options.cycle_time)  # make sure that there is not just one gear lagging behind.
        if not self.start_conditions_are_met() or self.stop_at_next_break_point:
            return
        self._get_job_factory().create_kinetic_modeling_job(self.options.job_settings)

    def _get_n_queuing_kinetic_modeling_calculations(self):
        selection = {
            "$and": [
                {"job.order": {"$eq": self.options.job.order}},
                {"$or": [{"status": "new"}, {"status": "hold"}, {"status": "pending"}]},
            ]
            + model_query(self.options.electronic_model)
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
