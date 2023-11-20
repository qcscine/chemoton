#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import abstractmethod
from copy import deepcopy
from datetime import datetime
from functools import wraps
from json import dumps
from warnings import warn
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Union

import scine_database as db
from scine_database.queries import model_query, lastmodified_since, stop_on_timeout, optimized_labels
from scine_utilities import ValueCollection

from scine_chemoton.gears import Gear
from scine_chemoton.gears.elementary_steps import ElementaryStepGear
from scine_chemoton.gears.compound import BasicAggregateHousekeeping
from scine_chemoton.gears.reaction import BasicReactionHousekeeping
from scine_chemoton.gears.kinetics import MinimalConnectivityKinetics
from scine_chemoton.gears.scheduler import Scheduler
from scine_chemoton.gears.thermo import BasicThermoDataCompletion
from scine_chemoton.utilities import connect_to_db
from ..selections import SelectionResult
from ..datastructures import (
    NetworkExpansionResult,
    GearOptions,
    ProtocolEntry,
    ExplorationSchemeStep,
    Status,
    StopPreviousProtocolEntries,
)


class NetworkExpansion(ExplorationSchemeStep):
    """
    The base class for operations to expand the network with new information.
    It specifies the common __call__ execution and holds 3 abstract methods that
    must be implemented by each implementation.
    Additionally it holds some common functionalities for execution and querying
    to simplify future implementations of new expansions.
    """
    class Options(ExplorationSchemeStep.Options):
        def __init__(self,
                     model: db.Model,
                     status_cycle_time: float,
                     include_thermochemistry: bool,
                     gear_options: Optional[GearOptions],
                     general_settings: Optional[Dict[str, Any]],
                     *args, **kwargs):
            super().__init__(model, *args, **kwargs)
            self.status_cycle_time = status_cycle_time
            self.include_thermochemistry = include_thermochemistry
            self.gear_options = gear_options
            if general_settings is None:
                self.general_settings: Dict[str, Any] = {}
            else:
                self.general_settings = general_settings

    options: Options  # required for mypy checks, so it knows which options object to check

    def __init__(self, model: db.Model,  # pylint: disable=keyword-arg-before-vararg
                 gear_options: Optional[GearOptions] = None,
                 status_cycle_time: float = 1.0,
                 include_thermochemistry: bool = False,
                 general_settings: Optional[Dict[str, Any]] = None,
                 *args, **kwargs):
        """
        Generate a new network expansion with the most important options.
        Any detailed options can still be specified on gear-level in GearOptions.

        Parameters
        ----------
        model : db.Model
            The model of the network expansion.
        status_cycle_time : float, optional
            How often the status of the individual gears and subsequently the own status
            should be updated. Small expansions on small networks benefit from shorter
            times, while the querying on larger database may become expensive and
            short expansion can cause race condition problems with the gears, hence
            longer times are safer, by default 1.0 (very short).
        include_thermochemistry : bool, optional
            If thermochemistry calculations (require Hessian calculations) should
            be added, by default False
        general_settings : Optional[Dict[str, Any]], optional
            Settings to be added to each separate job (beside graph job), because
            this only accepts special options and is at time of writing not created
            anymore, because the graph is generated within the bond orders job, which
            also accepts calculator specific settings.
            Due to being added to many jobs, we recommend only calculator settings,
            such as 'max_scf_iterations', by default None.
        add_default_chemoton_nt_settings : bool, optional
            Whether the default react job setting of Chemoton should be added
            to all react jobs, by default False.
        """
        super().__init__(model, status_cycle_time, include_thermochemistry, gear_options, general_settings,
                         *args, **kwargs)
        self.energy_type = "gibbs_free_energy" if include_thermochemistry else "electronic_energy"
        self.protocol: List[Union[ProtocolEntry, StopPreviousProtocolEntries]] = []
        self._selection = SelectionResult()
        self._start_id_timestamp: Optional[db.ID] = None
        self._start_time: datetime = datetime.now()
        self._result: Optional[NetworkExpansionResult] = None

    def dry_setup_protocol(self, credentials: db.Credentials, selection: Optional[SelectionResult] = None) -> None:
        """
        Sets up the protocol (individual gears) as currently specified in the options without running
        anything. This is useful to get some preemptive information about the gears to be run.

        Notes
        -----
        Protocol should be cleared afterwards if the step is not immediately executed to avoid
        problems with multithreaded / multiprocessed code due to presence of potentially
        forked objects in the protocol.

        Parameters
        ----------
        credentials : db.Credentials
            The credentials of the database to connect to. Only used for retrieving collections, not manipulation.
        selection : Optional[SelectionResult], optional
            A selection to be given to the expansion, can manipulate the individual gears, by default None.
        """
        manager = connect_to_db(credentials)
        if selection is not None:
            self._selection = selection
        self.initialize_collections(manager)
        self._propagate_db_manager(manager)
        self._set_protocol_wrap(manager)
        self._prepare_engines()

    def current_gears(self) -> List[Gear]:
        """
        The gears currently in the protocol. Allows `ProtocolEntry` agnostic access.

        Returns
        -------
        List[Gear]
            List of gears.
        """
        return [p.gear for p in self.protocol if isinstance(p, ProtocolEntry)]

    def get_result(self) -> Optional[NetworkExpansionResult]:
        return self._result

    def set_result(self, result: Optional[NetworkExpansionResult]) -> None:  # type: ignore[override]
        self._result = result

    def __call__(
            self,
            credentials: db.Credentials,
            selection: Optional[SelectionResult] = None) -> NetworkExpansionResult:
        """
        Execution of the network expansion. Usually executed by SteeringWheel.

        Parameters
        ----------
        credentials : db.Credentials
            The credentials of the database.
        selection : Optional[SelectionResult], optional
            The previous selection, by default None

        Returns
        -------
        NetworkExpansionResult
            The result holding the new / modified IDs.
        """
        if selection is None:
            self._selection = SelectionResult()
        else:
            self._selection = selection

        # Get required collections
        manager = connect_to_db(credentials)
        self.initialize_collections(manager)
        self._propagate_db_manager(manager)

        # create gears
        self._set_protocol_wrap(manager)

        # execute
        self.status = Status.CALCULATING
        self._result = self._wrapped_execute()
        if self._result is None:
            self._result = NetworkExpansionResult()
        assert isinstance(self._result, NetworkExpansionResult)

        # adapting status and returning result
        self.status = Status.FINISHED if self._result else Status.FAILED
        return self._result

    def _set_protocol_wrap(self, manager: db.Manager) -> None:
        """
        Makes sure that protocol and gears are handled correctly, so that not every implementation has
        to handle this.

        Parameters
        ----------
        manager : db.Manager
            The manager of the database we are working on.
        """
        self.protocol = []  # empty protocol to be sure that nothing is duplicated
        self._set_protocol(manager.get_credentials())  # abstract method to be implemented
        # make sure that gears already hold their collections
        for p in self.protocol:
            if not isinstance(p, ProtocolEntry):
                continue
            p.gear.initialize_collections(manager)

    def _propagate_db_manager(self, manager: db.Manager):
        """
        Make sure our selection (previous result), is correctly set-up.
        """
        if not isinstance(self._selection, SelectionResult):
            raise TypeError(f"{self.name} expects SelectionResult (or derived class) "
                            f"to select relevant chemical structures and not {type(self._selection)}.")
        self._selection.aggregate_filter.initialize_collections(manager)
        self._selection.reactive_site_filter.initialize_collections(manager)
        self._selection.further_exploration_filter.initialize_collections(manager)

    def _determine_current_status(self) -> None:
        """
        Update our own status based on a database query
        """
        if self._calculations.get_one_calculation(dumps({"status": "pending"})) is None:
            self.status = Status.STALLING
        else:
            self.status = Status.CALCULATING
        # finished states (FAILED and FINISHED) are handled by our forked process finishing in the main call

    def _unfinished_calculations_exist(self) -> bool:
        unfinished_selection = {"status": {"$in": ["hold", "new", "pending"]}}
        return self._calculations.get_one_calculation(dumps(unfinished_selection)) is not None

    def _wrapped_execute(self) -> NetworkExpansionResult:
        """
        Handles the correct set-up of ourself, the engines (and their cleanup) and some useful querying
        information to ease implementation of future implementations.

        Returns
        -------
        NetworkExpansionResult
            The result of our execution
        """
        self._give_current_process_own_name()
        self._start_id_timestamp = db.ID()  # we are using the time relation of mongodb IDs to hack a fast time look-up
        self._start_time = datetime.utcnow()  # for slower 'lastmodified' look-up
        self._prepare_engines()
        result = self._execute()
        self._stop_engines()
        return result

    def _prepare_scheduler(self, max_n_jobs: int = 1000) -> Scheduler:
        """
        Creates a Scheduler gear so that calculations can be accessed by Puffins based on the expected
        jobs in our protocol.

        Parameters
        ----------
        max_n_jobs : int, optional
            The maximum number of jobs, by default 1000

        Returns
        -------
        Scheduler
            The scheduler gear.

        Raises
        ------
        RuntimeError
            If we want a job that is not implemented in the Scheduler. Implementation error on Scheduler's side
            or user specified a job in special settings and misspelled it.
        """
        scheduling_gear = Scheduler()
        for k in scheduling_gear.options.job_counts.keys():
            scheduling_gear.options.job_counts[k] = 0
        relevant = self._relevant_puffin_jobs()
        for k in relevant:
            if k not in scheduling_gear.options.job_counts:
                raise RuntimeError(f"{self.name} specified '{k}' as a relevant job, "
                                   f"but this is not present in '{scheduling_gear.name}' job_counts options.")
            scheduling_gear.options.job_counts[k] = max_n_jobs
        return scheduling_gear

    def _prepare_engines(self) -> None:
        """
        Handles the set-up of engines with their gears and their correct options.
         - Basic sanity checks of the holding members and if Scheduler exists.
         - Propagate our model to all gears.
         - Apply our GearOptions.
         - Apply our general settings.
         - Give gears filters whenever possible.
        """
        if self._manager is None:
            raise RuntimeError(f"Engines cannot be started before initializing '{self.name}'")
        if not self.protocol:
            raise RuntimeError(f"Engines cannot be started before setting up the protocol of '{self.name}'")
        # sanity check if we have a scheduler
        if not any(isinstance(entry.gear, Scheduler) for entry in self.protocol if isinstance(entry, ProtocolEntry)):
            warn(f"No scheduling gear present in {self.name}, "
                 f"adding the default scheduler to ensure calculations are run")
            credentials = self._manager.get_credentials()
            # place at the front to ensure unforking engines are not blocking the scheduler from ever starting
            self.protocol.insert(0, ProtocolEntry(credentials, self._prepare_scheduler(), fork=True, n_runs=0))
        # sanity check and set model
        for entry in self.protocol:
            if not isinstance(entry, ProtocolEntry):
                continue
            gear = entry.gear
            GearOptions.sanity_check(gear)
            if hasattr(gear.options, "model"):
                gear.options.model = self.options.model
            if isinstance(gear, ElementaryStepGear):
                if gear.trial_generator is not None:
                    gear.trial_generator.options.model = self.options.model
        # overwrite with provided options
        if self.options.gear_options is not None:
            self._apply_gear_options(self.options.gear_options)
        # apply general settings
        if self.options.general_settings is not None:
            for entry in self.protocol:
                if not isinstance(entry, ProtocolEntry):
                    continue
                gear = entry.gear
                for attr in dir(gear):
                    if not attr.startswith("_") and "settings" in attr and attr != "graph_settings":
                        existing_settings = getattr(gear, attr)
                        if isinstance(existing_settings, dict) or isinstance(existing_settings, ValueCollection):
                            combined = {**existing_settings, **self.options.general_settings}  # type: ignore
                            if isinstance(existing_settings, ValueCollection):
                                setattr(gear, attr, ValueCollection(combined))
                            else:
                                setattr(gear, attr, combined)
                    if attr.endswith("options"):
                        for option_attr in dir(getattr(gear, attr)):
                            if not option_attr.startswith("_") and "settings" in option_attr and \
                                    option_attr != "graph_settings":
                                existing_settings = getattr(getattr(gear, attr), option_attr)
                                if not isinstance(existing_settings, dict) \
                                        and not isinstance(existing_settings, ValueCollection):
                                    continue
                                combined = {**existing_settings, **self.options.general_settings}  # type: ignore
                                if isinstance(existing_settings, ValueCollection):
                                    setattr(getattr(gear, attr), option_attr, ValueCollection(combined))
                                else:
                                    setattr(getattr(gear, attr), option_attr, combined)

        # set filters
        for entry in self.protocol:
            if not isinstance(entry, ProtocolEntry):
                continue
            gear = entry.gear
            if hasattr(gear, "aggregate_filter"):
                # setattr from here on to avoid linter cries
                setattr(gear, "aggregate_filter", self._selection.aggregate_filter)
            if hasattr(gear, "trial_generator") and getattr(gear, "trial_generator") is not None:
                setattr(getattr(gear, "trial_generator"), "reactive_site_filter",
                        self._selection.reactive_site_filter)
                if hasattr(getattr(gear, "trial_generator"), "further_exploration_filter"):
                    setattr(getattr(gear, "trial_generator"), "further_exploration_filter",
                            self._selection.further_exploration_filter)

    def _apply_gear_options(self, gear_options: GearOptions):
        """
        Apply the given options to our holded gears.

        Parameters
        ----------
        gear_options : GearOptions
            The options.

        Raises
        ------
        TypeError
            We are not holding a specified gear
        NotImplementedError
            We cannot propagate a given second value in the options tuple.
        """
        for name, options in gear_options.items():
            gears_to_change = [entry.gear for entry in self.protocol if isinstance(entry, ProtocolEntry)
                               and entry.name == name]
            if not gears_to_change:
                raise TypeError(f"Specified options for '{name}', but this gear is not present in the "
                                f"step protocol {self.protocol} of {self.name}")
            for gear in gears_to_change:
                gear.options = options[0]
                if gear.options.model != self.options.model:
                    warn(f"The gear '{gear.name}' has received the model:\n"
                         f"{str(gear.options.model)}\n"
                         f"via the provided gear options which is different to the model of {self.name}:\n"
                         f"{str(self.options.model)}")
                if options[1] is not None:
                    if not isinstance(gear, ElementaryStepGear):
                        raise NotImplementedError(f"Gear options for '{name}' hold another set of options, "
                                                  f"but this gear is not an elementary step gear. "
                                                  f"This is not supported.")
                    gear.trial_generator.options = options[1]

    def _wait_for_calculations_to_finish(self) -> None:
        """
        Method to be applied if we should wait for a gear to finish. It waits until no more unfinished
        calculations exist, then it checks if we have infinitely looping gears and to make sure we are
        avoiding race conditions by finding no unfinished calculations, but a gear setting some up shortly after,
        we are waiting for two more loops of each infinitely looping gear and then checking again for
        calculations.
        """
        while True:
            # wait until we have at one point no open calculations
            while self._unfinished_calculations_exist():
                sleep(self.options.status_cycle_time)
            # wait for all infinitely looping gears to loop two more times and check again
            # to make sure we haven't missed anything due to cycle times
            finish_counters = self._get_gear_counters()
            if not finish_counters:
                # no infinitely looping gears anyway
                break
            counters = deepcopy(finish_counters)
            last_counters = deepcopy(finish_counters)
            last_counting_loop_id = db.ID()  # for faster time lookup with indexed IDs
            while not all(c > fc + 1 for c, fc in zip(counters, finish_counters)):
                counters = self._get_gear_counters()
                # if any gear has looped once, we check for new calculations
                if any(c > last for c, last in zip(counters, last_counters)):
                    if self._calculations.get_one_calculation(dumps(self._newer_id(last_counting_loop_id))) is not None:
                        # some calculation was set up in the meantime, increase our finish_counters
                        finish_counters = [f + 2 for f in finish_counters]
                    last_counting_loop_id = db.ID()
                    last_counters = deepcopy(counters)
                sleep(self.options.status_cycle_time)
            # check if we still have no new calculations after gears have looped
            if not self._unfinished_calculations_exist():
                break

    def _get_gear_counters(self) -> List[int]:
        """
        Makes logic easier readable by hiding ProtocolEntry logic
        """
        return [p.engine.get_number_of_gear_loops() for p in self.protocol if isinstance(p, ProtocolEntry)
                and not p.limited_runs and not p.was_stopped]

    def _basic_execute(self):
        """
        A common way of producing our data. This can be used by our implementations, if this works for them
        and then they only have to implement the database queries for results.
        This method is not used by the base class and is an optional suggestion to avoid handling the
        different protocol entries.

        Note
        ----
        If this method is not used by a child class, the handling of StopPreviousProtocolEntries
        has to be implemented in the execution in the child class.
        """
        for i, entry in enumerate(self.protocol):
            if isinstance(entry, StopPreviousProtocolEntries):
                self._stop_engines(until_index=i)
                continue
            n = max(entry.n_runs, 1)
            for _ in range(n):
                entry.run()
                self._give_current_process_own_name()  # change process name back in case gear was not forked
                if entry.wait_for_calculation_finish:
                    self._wait_for_calculations_to_finish()

    def _set_default_gear_options(self):
        """
        Construct options to write our model into the options, this does not overwrite the given gear_options.
        Protocol must have been constructed
        """
        if not self.protocol:
            raise RuntimeError(f"The protocol has not be specified for {self.name}")
        gears = [d.gear for d in self.protocol if isinstance(d, ProtocolEntry)]
        default_options = GearOptions(gears, self.options.model)
        self._apply_gear_options(default_options)

    def _stop_engines(self, until_index: Optional[int] = None):
        """
        Stop and join holding engines.

        Parameters
        ----------
        until_index : Optional[int]
            An optional index to specify until which engine we should stop
        """
        if until_index is None:
            # stop whole protocol
            for entry in self.protocol:
                entry.stop()
        else:
            for entry in self.protocol[:until_index]:
                entry.stop()

    def _add_basic_chemoton_gears_to_protocol(self, credentials: db.Credentials):
        """
        Add some basic gears responsible for aggregate sorting or activations to our protocol.
        Convenience method for implementations.

        Parameters
        ----------
        credentials : db.Credentials
            The credentials of the database to be given to the created gears.
        """
        self.protocol.append(ProtocolEntry(credentials, self._prepare_scheduler()))
        self.protocol.append(ProtocolEntry(credentials, MinimalConnectivityKinetics()))
        self.protocol.append(ProtocolEntry(credentials, BasicAggregateHousekeeping()))
        self.protocol.append(ProtocolEntry(credentials, BasicReactionHousekeeping()))
        self._potential_thermochemistry_protocol_addition(credentials)

    def _potential_thermochemistry_protocol_addition(self, credentials: db.Credentials):
        if self.options.include_thermochemistry:
            self.protocol.append(ProtocolEntry(credentials, BasicThermoDataCompletion()))

    def _extra_manual_cycles_to_avoid_race_condition(self, credentials: db.Credentials,
                                                     aggregate_reactions: bool,
                                                     additional_entries: Optional[List[ProtocolEntry]] = None) -> None:
        # ensure that we stop previous engines
        self.protocol.append(StopPreviousProtocolEntries())
        scheduler = self._prepare_scheduler()

        # take care of aggregates
        for _ in range(3):  # 3 because of potential bo/graph calculations
            self.protocol.append(ProtocolEntry(credentials, BasicAggregateHousekeeping(), n_runs=1, fork=False))
            self.protocol.append(ProtocolEntry(credentials, scheduler, n_runs=1, fork=False,
                                               wait_for_calculation_finish=True))  # ensure we can loop again
        # take care of thermochemistry
        if self.options.include_thermochemistry:
            for _ in range(2):  # 2 just to be sure
                self.protocol.append(ProtocolEntry(credentials, BasicThermoDataCompletion(), n_runs=1, fork=False))
                self.protocol.append(ProtocolEntry(credentials, scheduler, n_runs=1, fork=False,
                                                   wait_for_calculation_finish=True))

        if aggregate_reactions:
            # take care of reactions
            self.protocol.append(ProtocolEntry(credentials, BasicReactionHousekeeping(), n_runs=2, fork=False))

        # take care of aggregate activation
        self.protocol.append(ProtocolEntry(credentials, MinimalConnectivityKinetics(), n_runs=4, fork=False))
        if additional_entries is not None:
            self.protocol.extend(additional_entries)

        self._set_default_gear_options()

    @staticmethod
    def _aggregation_necessary_jobs():
        return ["scine_bond_orders", "graph"]

    def _new_id(self) -> Dict[str, Any]:
        """
        Returns query dictionary to query for all IDs that were newly generated by us.

        Notes
        -----
        Querying utility
        """
        if self._start_id_timestamp is None:
            raise RuntimeError("Start ID has never been set")
        return self._newer_id(self._start_id_timestamp)

    @staticmethod
    def _newer_id(compared_id: db.ID) -> Dict[str, Any]:
        return {"_id": {"$gt": {"$oid": str(compared_id)}}}

    def _new_entry_with_model(self) -> Dict[str, Any]:
        """
        Returns query dictionary to query for all IDs that were newly generated by us with a specific model.

        Notes
        -----
        Querying utility
        """
        return {
            "$and": [
                self._new_id(),
                *model_query(self.options.model)
            ]
        }

    def _new_minimum_structures(self) -> Dict[str, Any]:
        """
        Returns query dictionary for all new minimum structures added by us with a specific model.

        Notes
        -----
        Querying utility
        """
        return {
            "$and": [
                self._new_id(),
                {"label": {"$in": optimized_labels()}},
                *model_query(self.options.model)
            ]
        }

    def _modified_entry_with_model(self) -> Dict[str, Any]:
        """
        Returns query dictionary for all modified entries by us with a specific model.

        Notes
        -----
        Querying utility
        """
        return {
            "$and": [
                lastmodified_since(self._start_time),
                *model_query(self.options.model)
            ]
        }

    def _add_modified_reactions_to_results(self, result: NetworkExpansionResult, include_reactants: bool):
        """
        Adds all modified reactions (and optionally their reactants) to the given result.

        Parameters
        ----------
        result : NetworkExpansionResult
            The result we are adding to.
        include_reactants : bool
            If we should add the reacting compounds / flasks.

        """
        selection = lastmodified_since(self._start_time)
        for reaction in stop_on_timeout(self._reactions.iterate_reactions(dumps(selection))):
            reaction.link(self._reactions)
            result.reactions.append(reaction.id())
            if not include_reactants:
                continue
            types = reaction.get_reactant_types(db.Side.BOTH)
            reactants = reaction.get_reactants(db.Side.BOTH)
            for type_side, reactant_side in zip(types, reactants):
                for type, reactant in zip(type_side, reactant_side):
                    if type == db.CompoundOrFlask.COMPOUND:
                        result.compounds.append(reactant)
                    elif type == db.CompoundOrFlask.FLASK:
                        result.flasks.append(reactant)
                    else:
                        raise TypeError("Unknown aggregate type")

    @abstractmethod
    def _relevant_puffin_jobs(self) -> List[str]:
        """
        Method to be implemented. Specifies list of all job names that we expect to set-up during the
        network expansion.

        Returns
        -------
        List[str]
            List of job names.
        """

    @abstractmethod
    def _set_protocol(self, credentials: db.Credentials) -> None:
        """
        Method to be implemented. Set-up our Protocol of specific gears and how to run them.
        See `ProtocolEntry` for details or convenience methods in this base class for examples.

        Parameters
        ----------
        credentials : db.Credentials
            The credentials of the database.
        """

    @abstractmethod
    def _execute(self) -> NetworkExpansionResult:
        """
        Method to be implemented. Specifies the execution of our gears (see basic_execute for
        convenience method for basic tasks) and then query the database to fill our result.

        Returns
        -------
        NetworkExpansionResult
            The result of the execution.
        """


def thermochemistry_job_wrapper(fun: Callable):
    """
    Potential decorator for `_relevant_puffin_jobs` to make sure we are adhering to the thermochemistry option.

    Parameters
    ----------
    fun : Callable
        The function to be decorated.
    """
    @wraps(fun)
    def _impl(self, *args, **kwargs) -> List[str]:
        relevant = fun(self, *args, **kwargs)
        if self.options.include_thermochemistry:
            relevant.append('scine_hessian')
        return relevant

    return _impl


class GiveWholeDatabaseWithModelResult(NetworkExpansion):
    """
    The most basic implementation. Meant for testing and debugging and it is the only network expansion
    that is not manipulating the network in any way.
    """

    def _relevant_puffin_jobs(self) -> List[str]:
        return []

    def _set_protocol(self, credentials: db.Credentials) -> None:
        # dummy protocol that does not do anything
        self.protocol = [ProtocolEntry(credentials, self._prepare_scheduler(), fork=False, n_runs=1)]

    def _execute(self) -> NetworkExpansionResult:
        """
        Only gathers information, no manipulation happening here.
        """
        selection = {"$and": model_query(self.options.model)}
        structures = []
        flasks = []
        compounds = []
        for structure in stop_on_timeout(self._structures.iterate_structures(dumps(selection))):
            structure.link(self._structures)
            structures.append(structure.id())
            if structure.has_aggregate() and structure.has_graph("masm_cbor_graph"):
                g = structure.get_graph("masm_cbor_graph")
                if ";" in g:
                    flasks.append(structure.get_aggregate())
                else:
                    compounds.append(structure.get_aggregate())
        reactions = [r.id() for r in self._reactions.query_reactions("{}")]
        return NetworkExpansionResult(compounds=compounds,
                                      flasks=flasks,
                                      reactions=reactions,
                                      structures=structures)
