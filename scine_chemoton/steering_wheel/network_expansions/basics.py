#!/usr/bin/env python3
from __future__ import annotations
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from json import dumps
from time import sleep
from typing import Any, Dict, List, Optional

import scine_database as db
from scine_utilities import ValueCollection
from scine_database.queries import optimized_labels_enums, lastmodified_since

from scine_chemoton.default_settings import default_opt_settings
from scine_chemoton.gears.compound import BasicAggregateHousekeeping
from scine_chemoton.gears.kinetics import MinimalConnectivityKinetics
from ..datastructures import NetworkExpansionResult, ProtocolEntry, GearOptions, StopPreviousProtocolEntries
from . import NetworkExpansion, thermochemistry_job_wrapper


class SimpleOptimization(NetworkExpansion):
    """
    Carries out a structure optimization and aggregation of the given individual structures.
    Hence, it must receive a selection result that holds individual structures
    """

    class Options(NetworkExpansion.Options):

        def __init__(self,
                     model: db.Model,
                     status_cycle_time: float,
                     include_thermochemistry: bool,
                     gear_options: Optional[GearOptions],
                     general_settings: Optional[Dict[str, Any]],
                     minimization_job_order: str,
                     *args, **kwargs):
            super().__init__(model, status_cycle_time, include_thermochemistry, gear_options, general_settings,
                             *args, **kwargs)
            self.minimization_job = db.Job(minimization_job_order)
            self.minimization_job_settings = default_opt_settings()
            if self.general_settings is not None:
                self.minimization_job_settings.update({**self.general_settings})

    options: SimpleOptimization.Options  # required for mypy checks, so it knows which options object to check

    def __init__(self, model: db.Model,  # pylint: disable=keyword-arg-before-vararg
                 gear_options: Optional[GearOptions] = None,
                 status_cycle_time: float = 60.0,
                 include_thermochemistry: bool = False,
                 general_settings: Optional[Dict[str, Any]] = None,
                 minimization_job: str = "scine_geometry_optimization", *args, **kwargs):
        super().__init__(model, gear_options, status_cycle_time, include_thermochemistry, general_settings,
                         minimization_job, *args, **kwargs)

    @thermochemistry_job_wrapper
    def _relevant_puffin_jobs(self) -> List[str]:
        return [self.options.minimization_job.order] + self._aggregation_necessary_jobs()

    def _set_protocol(self, credentials: db.Credentials) -> None:
        self.protocol.append(ProtocolEntry(credentials, self._prepare_scheduler()))
        self._potential_thermochemistry_protocol_addition(credentials)
        self.protocol.append(ProtocolEntry(credentials, MinimalConnectivityKinetics()))
        # double aggregate to avoid skipping problems with too short cycle times
        housekeeping = BasicAggregateHousekeeping()
        housekeeping.options.cycle_time = 1
        self.protocol.append(ProtocolEntry(credentials, housekeeping, fork=False, n_runs=3,
                                           wait_for_calculation_finish=True))
        # stop everything and run kinetics gear once more to make sure new compounds are activated
        self.protocol.append(StopPreviousProtocolEntries())
        self.protocol.append(ProtocolEntry(credentials, MinimalConnectivityKinetics(), n_runs=1, fork=False))

    def _execute(self, n_already_executed_protocol_steps: int) -> NetworkExpansionResult:
        if self._selection is None:
            raise RuntimeError(f"{self.name} requires a given selection.")
        if not self._selection.structures:
            raise RuntimeError(f"{self.name} requires a given selection with specific structures.")

        settings = ValueCollection(
            {**self.options.general_settings, **self.options.minimization_job_settings}  # type: ignore
        )
        calculations_we_want_result_from = []
        if n_already_executed_protocol_steps == 0:
            skip_labels = optimized_labels_enums() + [db.Label.TS_OPTIMIZED]
            for sid in self._selection.structures:
                if db.Structure(sid, self._structures).get_label() in skip_labels:
                    continue
                calculation = db.Calculation(db.ID(), self._calculations)
                calculation.create(self.options.model, self.options.minimization_job, [sid])
                calculation.set_settings(settings)
                calculation.set_status(db.Status.HOLD)
                calculations_we_want_result_from.append(calculation.id())
                sleep(0.01)
        else:
            selection = {"job.order": self.options.minimization_job.order}
            for calc in self._calculations.iterate_calculations(dumps(selection)):
                calc.link(self._calculations)
                if calc.get_structures()[0] in self._selection.structures and calc.get_settings() == settings:
                    calculations_we_want_result_from.append(calc.id())

        self._basic_execute(n_already_executed_protocol_steps)

        selection = lastmodified_since(self._start_time)
        compounds = [c.id() for c in self._compounds.query_compounds(dumps(selection))]
        flasks = [f.id() for f in self._flasks.query_flasks(dumps(selection))]
        structures = []
        for cid in calculations_we_want_result_from:
            calculation = db.Calculation(cid, self._calculations)
            result_structures = calculation.get_results().structure_ids
            if result_structures:
                structures.append(result_structures[0])

        return NetworkExpansionResult(compounds=compounds,
                                      flasks=flasks,
                                      structures=structures)


class ThermochemistryGeneration(NetworkExpansion):
    """
    Produces thermochemistry results for the given selection result.
    Can be used to add thermochemistry after the fact after exploring faster without it.
    """

    class Options(NetworkExpansion.Options):

        def __init__(self,
                     model: db.Model,
                     status_cycle_time: float,
                     include_thermochemistry: bool,
                     gear_options: Optional[GearOptions],
                     general_settings: Optional[Dict[str, Any]],
                     hessian_job_order: str,
                     *args, **kwargs):
            super().__init__(model, status_cycle_time, include_thermochemistry, gear_options, general_settings,
                             *args, **kwargs)
            self.include_thermochemistry = True
            self.hessian_job = db.Job(hessian_job_order)

    options: ThermochemistryGeneration.Options  # required for mypy checks, so it knows which options object to check

    def __init__(self, model: db.Model,  # pylint: disable=keyword-arg-before-vararg
                 gear_options: Optional[GearOptions] = None,
                 status_cycle_time: float = 60.0,
                 include_thermochemistry: bool = False,
                 general_settings: Optional[Dict[str, Any]] = None,
                 hessian_job: str = "scine_hessian", *args, **kwargs):
        super().__init__(model, gear_options, status_cycle_time, include_thermochemistry, general_settings,
                         hessian_job, *args, **kwargs)

    def _relevant_puffin_jobs(self) -> List[str]:
        return [self.options.hessian_job.order]

    def _set_protocol(self, credentials: db.Credentials) -> None:
        self.protocol.append(ProtocolEntry(credentials, self._prepare_scheduler()))
        thermo = self.thermochemistry_gear()
        thermo.options.job = self.options.hessian_job
        thermo.options.ignore_explore_bool = True
        thermo.options.settings = ValueCollection(self.options.general_settings)
        self.protocol.append(ProtocolEntry(credentials, thermo, wait_for_calculation_finish=True))

    def _execute(self, n_already_executed_protocol_steps: int) -> NetworkExpansionResult:
        self._basic_execute(n_already_executed_protocol_steps)

        selection = lastmodified_since(self._start_time)
        compounds = [c.id() for c in self._compounds.query_compounds(dumps(selection))]
        flasks = [f.id() for f in self._flasks.query_flasks(dumps(selection))]

        return NetworkExpansionResult(compounds=compounds,
                                      flasks=flasks)
