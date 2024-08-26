#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from copy import copy
from functools import wraps
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from shutil import rmtree
from os import path, mkdir
from uuid import uuid1
import pickle
from typing import Dict, List, Optional, Tuple, Union, Callable
from warnings import warn

import scine_database as db

from scine_chemoton.gears import connect_to_db, HasName
from scine_chemoton.utilities import yes_or_no_question, integer_question
from scine_chemoton.utilities.datastructure_transfer import (
    make_picklable,
    read_connection,
    ReadAble,
    MultiProcessingConnectionsWithProxyThread,
    StopReading
)
from .network_expansions import NetworkExpansion, GiveWholeDatabaseWithModelResult
from .selections import (
    Selection,
    AllCompoundsSelection,
    SelectionAndArray,
    SafeFirstSelection,
    SelectionOrArray,
    PredeterminedSelection,
)
from .selections.input_selections import InputSelection
from .datastructures import (
    Status,
    ExplorationSchemeStep,
    ExplorationResult,
    LogicCoupling,
    RestartPartialExpansionInfo,
    NoRestartInfoPresent,
)
from .result_transfer import receive_multiple_results_from_pipe, send_multiple_results_in_pipe, WaitForFurtherResults


class FailedSaveException(Exception):
    """
    Signals that a save operation failed
    """


class SteeringWheel(HasName):
    """
    The class managing and executing an exploration protocol that allows an active steering of the exploration
    in a reproducible manner.
    The exploration scheme steps can already be given upon initialization or added
    afterwards with `+=`.
    Each addition is sanity checked by the class and might be altered based on that.
    In some cases the class will ask for user input via the provided callback
    """

    def __init__(self, credentials: db.Credentials, exploration_scheme: List[ExplorationSchemeStep],
                 global_selection: Optional[Selection] = None, global_for_first_selection: bool = False,
                 restart_file: Optional[str] = None, callable_input: Callable = input) -> None:
        """
        Initialize the steering wheel to a database with the given credentials and an exploration scheme.
        The exploration scheme can still be empty and added later with `+=`.

        Parameters
        ----------
        credentials : db.Credentials
            The credentials to connect to the database
        exploration_scheme : List[ExplorationSchemeStep]
            The exploration scheme to be executed, will be sanity checked already
        global_selection : Optional[Selection], optional
            A selection to be applied on top of all selections in the scheme, by default None
        global_for_first_selection : bool, optional
            If the global selection should be applied to the first selection as well, by default False
        restart_file : Optional[str], optional
            The restart file, by default None
        callable_input : Callable, optional
            The function the steering wheel relies on to get user feedback, by default input
        """
        super().__init__()
        self.credentials = credentials
        self.manager: Optional[db.Manager] = connect_to_db(credentials)
        self._partial_protocol_restart_info: Optional[RestartPartialExpansionInfo] = None
        self._default_name = copy(self._name)
        self._external_restart_file: Optional[str] = restart_file
        self._input = callable_input
        self._restart_info_file = ".chemoton_wheel_restart_info.pkl"
        self._save_dir = "chemoton_protocol_saves"
        self._results: Optional[List[ExplorationResult]] = None
        self._global_selection = global_selection
        self._global_for_first_selection = global_for_first_selection
        self._process: Optional[Process] = None
        self._process_manager: Optional[Connection] = None
        self._results_connection_recv: Optional[ReadAble] = None
        self._partial_steps_connection_recv: Optional[ReadAble] = None
        self._status_connection_recv: Optional[Connection] = None
        self._status_connection_send: Optional[Connection] = None
        self._reusing_prev_selection: Dict[int, int] = {}
        # determine steps from given exploration scheme
        self.scheme: List[ExplorationSchemeStep] = []
        if exploration_scheme:
            last_selection = self._first_step_sanity_checks(exploration_scheme)
            # the first step has been added with if/else before
            self._add_exploration_steps_to_scheme(exploration_scheme, last_selection, first_has_been_added=True)

    def _first_step_sanity_checks(self, exploration_scheme: List[ExplorationSchemeStep]) \
            -> Tuple[Optional[Selection], int]:
        """
        Method to check the first step of the exploration scheme and add it to the scheme if it is valid.
        The first step is special because it must fulfill some requirements,
        e.g., it must be a selection that inputs structures
        or the database must already contain structures.

        Returns
        -------
        Tuple[Optional[Selection], int]
            The last selection and its index in the scheme.

        Raises
        ------
        TypeError
            If the first step is not a selection that inputs structures and
            the database does not contain any structures.
        """
        first_step = exploration_scheme[0]

        def step_is_selection_type(step: ExplorationSchemeStep, sele_type: type) -> bool:
            return isinstance(step, sele_type) \
                or (isinstance(step, SelectionAndArray) and
                    all(isinstance(s, sele_type) for s in step)) \
                or (isinstance(step, SelectionOrArray) and
                    any(isinstance(s, sele_type) for s in step))

        if self.manager is None:
            self.manager = connect_to_db(self.credentials)
        if self.manager is None \
                or (self.manager.get_collection("structures").count("{}") == 0
                    and not step_is_selection_type(first_step, InputSelection)):
            raise TypeError(f"The first entry in your exploration scheme is not a selection that inputs structures, "
                            f"but '{type(first_step)}' and the database does not include any structures")
        last_selection: Tuple[Optional[Selection], int] = (None, 0)
        if isinstance(first_step, Selection):
            index = 0
            if not step_is_selection_type(first_step, SafeFirstSelection):
                warning = f"The first entry in your exploration scheme ({first_step.name}) is not a safe first " \
                          f"selection, i.e., it accesses a previous result.\n" \
                          f"We are giving it a result, that selects everything from the database with the same model."
                warn(warning)
                if yes_or_no_question(f"{warning}\nDo you want to continue", self._input):
                    model = first_step.options.model
                    self.scheme.append(GiveWholeDatabaseWithModelResult(model))
                    index = 1
                else:
                    self.stop(save_progress=False)
            sele = self._selection_construction(first_step, add_global=self._global_for_first_selection)
            self.scheme.append(sele)
            last_selection = (sele, index)
        elif self.manager is None or self.manager.get_collection("structures").count("{}") == 0:
            raise TypeError(f"The first entry in your exploration scheme is not a selection, "
                            f"but '{type(first_step)}' and the database does not include any structures")
        elif isinstance(first_step, GiveWholeDatabaseWithModelResult):
            self.scheme.append(first_step)
        else:
            warning = f"The first entry in your exploration scheme is not a selection, but '{type(first_step)}'. " \
                      f"We are assuming that all structures in the database are valid"
            warn(warning)
            if yes_or_no_question(f"{warning}\nDo you want to continue", self._input):
                model = first_step.options.model
                default_first = self._selection_construction(AllCompoundsSelection(model))
                self.scheme.append(default_first)
                last_selection = (default_first, 0)
            else:
                self.stop(save_progress=False)
            # the first step in the given scheme is now safe to add
            self.scheme.append(first_step)
        if last_selection is not None and last_selection[0] is not None:
            result = last_selection[0].get_result()
            if result is not None:
                self._set_results([result])
        return last_selection

    @property
    def name(self) -> str:
        if self._name != self._default_name:
            return self._name
        return "ChemotonProtocol-" + ("-".join(i.name for i in self.scheme))

    @name.setter
    def name(self, n: str):
        self._name = n

    @property
    def restart_file(self) -> str:
        if self._external_restart_file is None:
            return f".chemoton.restart-{self.name}.pkl"
        return self._external_restart_file

    @restart_file.setter
    def restart_file(self, filename: str):
        self._external_restart_file = filename

    def _add_exploration_steps_to_scheme(self,
                                         exploration_scheme: List[ExplorationSchemeStep],
                                         last_selection: Optional[Tuple[Optional[Selection], int]],
                                         first_has_been_added: bool) -> None:
        """
        This method sequentially adds the given exploration steps to the scheme, while checking for validity.
        It requires to know if the first step has been added already and the last selection and its index in the scheme.
        to carry out the checks.

        Parameters
        ----------
        exploration_scheme : List[ExplorationSchemeStep]
            The added scheme.
        last_selection : Optional[Tuple[Optional[Selection], int]]
            The last selection in the existing scheme and its index. None if no selection yet.
        first_has_been_added : bool
            If the first step has been added already.

        Raises
        ------
        RuntimeError
            The existing logic failed.
        TypeError
            The sanity checks failed. Reason is given in the error message.
        """
        if not exploration_scheme:
            return
        if not self.scheme:
            if first_has_been_added:
                raise RuntimeError("InternalError, something went wrong with scheme management")
            last_selection = self._first_step_sanity_checks(exploration_scheme)
            first_has_been_added = True
        if not self.scheme:
            raise RuntimeError("InternalError, something went wrong with scheme management")
        start_index = 1 if first_has_been_added else 0
        if last_selection is None or last_selection[0] is None:
            last_selection = self._determine_last_selection_in_scheme()
        for i, step in enumerate(exploration_scheme[start_index:], len(self.scheme)):
            last_entry_was_selection = isinstance(self.scheme[-1], Selection)
            if not last_entry_was_selection and isinstance(step, NetworkExpansion):
                if last_selection[0] is None:
                    raise TypeError("Exploration scheme could not be deduced into proper steps and selections.")
                warn(f"The given NetworkExpansion {step.name} is not following after a selection. We are assuming that "
                     f"this step should get the same Selection ({last_selection[0].name}) as the NetworkExpansion "
                     f"{self.scheme[-1].name} before")
                self._reusing_prev_selection[i] = last_selection[1]
                self.scheme.append(step)
            elif last_entry_was_selection and isinstance(step, Selection):
                if last_selection[0] is None:
                    raise TypeError("Exploration scheme could not be deduced into proper steps and selections.")
                warn(f"You gave two selections ({last_selection[0].name} - {step.name}) back to back. "
                     f"Based on the assigned logic coupling '{step.logic_coupling.value}', "
                     f"we assume you want to combine them as a logic '{step.logic_coupling.name}' selection.")
                self.scheme.pop(-1)
                add_global = self._global_for_first_selection if not self.scheme else True
                if step.logic_coupling == LogicCoupling.AND:
                    new_sele = self._selection_construction(SelectionAndArray([last_selection[0], step]), add_global)
                elif step.logic_coupling == LogicCoupling.OR:
                    new_sele = self._selection_construction(SelectionOrArray([last_selection[0], step]), add_global)
                else:
                    raise TypeError(f"Unknown logic coupling '{step.logic_coupling.name}'")
                self.scheme.append(new_sele)
            elif isinstance(step, Selection):
                self._check_new_step_for_existing_results(step, i)
                add_global = self._global_for_first_selection if not self.scheme else True
                self.scheme.append(self._selection_construction(step, add_global))
            else:
                self._check_new_step_for_existing_results(step, i)
                self.scheme.append(step)

            last_step = self.scheme[-1]
            if isinstance(last_step, Selection):
                last_selection = (last_step, i)

    def _check_new_step_for_existing_results(self, step: ExplorationSchemeStep, scheme_index: int) -> None:
        """
        Check if the step to be added has a result and if this fits with the existing scheme and its results.

        Parameters
        ----------
        step : ExplorationSchemeStep
            The step to be added.
        scheme_index : int
            The index of the step in the scheme.
        """
        result = step.get_result()
        if result is not None:
            total_results = self.get_results()
            if isinstance(step, PredeterminedSelection):
                # adapt status to what is expected depending on the steps before
                all_finished = all(s.status in [Status.FINISHED, Status.FAILED] for s in self.scheme[:scheme_index])
                step.status = Status.FINISHED if all_finished else Status.WAITING
            if total_results is None:
                if not self.scheme:
                    # fine for first entry if we don't have any results yet, but added step has a result
                    self._set_results([result])
                elif isinstance(step, PredeterminedSelection):
                    # fine that this one is holding a results object
                    pass
                else:
                    warn(f"Added step {step.name} to scheme, which has a result, but the total scheme "
                         f"does not have any results yet. We are discarding the result of this step.")
                    step.status = Status.WAITING
                    step.set_result(None)
            elif len(total_results) == scheme_index:
                # fine for last entry if we have results and added step has a result
                total_results.append(result)
            elif isinstance(step, PredeterminedSelection):
                # fine that this one is holding a results object
                pass
            else:
                warn(f"Added step {step.name} to scheme at index {scheme_index}, which has a result, "
                     f"but the total scheme only has {len(total_results)} results. "
                     f"We are discarding the result of this step.")
                step.status = Status.WAITING
                step.set_result(None)

    def __getstate__(self) -> Tuple[db.Credentials, List[ExplorationSchemeStep], Optional[Selection], bool,
                                    Optional[str], Callable, Optional[List[ExplorationResult]],
                                    Optional[RestartPartialExpansionInfo]]:
        """
        Defined dunder method to make the ExplorationProtocol picklable.

        Returns
        -------
        Tuple[db.Credentials, List[ExplorationSchemeStep], Optional[Selection], bool,
              Optional[str], Callable, Optional[List[ExplorationResult]],
              Optional[RestartPartialExpansionInfo]]
            The state information to be pickled.
        """
        results = self.get_results()
        partial_steps = self.get_partial_restart_info()
        self.manager = None  # cannot be pickled
        make_picklable(self)
        # we only want to specify the restart file if it was something special, otherwise keep the rolling restart name
        restart_file = self.restart_file if self._external_restart_file is not None else None
        return self.credentials, self.scheme, self._global_selection, \
            self._global_for_first_selection, restart_file, self._input, results, partial_steps  # type: ignore

    def __setstate__(self, state: Tuple[db.Credentials, List[ExplorationSchemeStep], Optional[Selection], bool,
                                        Optional[str], Callable, Optional[List[ExplorationResult]],
                                        Optional[RestartPartialExpansionInfo]]) -> None:
        """
        Defined dunder method to make the ExplorationProtocol picklable.

        Parameters
        ----------
        state : Tuple[db.Credentials, List[ExplorationSchemeStep], Optional[Selection], bool,
                      Optional[str], Callable, Optional[List[ExplorationResult]], Optional[RestartPartialExpansionInfo]]
            The state information to be unpickled.
        """
        self.__init__(credentials=state[0], exploration_scheme=state[1], global_selection=state[2],  # type: ignore
                      global_for_first_selection=state[3], restart_file=state[4], callable_input=state[5])
        self._results = state[6]
        self.set_partial_restart_info(state[7])

    def set_global_selection(self, global_selection: Selection, global_for_first_selection: bool = False) -> None:
        """
        Allows to add a global selection to the exploration protocol.

        Notes
        -----
        This will stop a running protocol and restart it with the new global selection.

        Parameters
        ----------
        global_selection : Selection
            The global selection to be added.
        global_for_first_selection : bool, optional
            If the selection should be added to the first selection as well, by default False
        """
        self.get_results()
        self._global_selection = global_selection
        self._global_for_first_selection = global_for_first_selection
        if self.is_running():
            self.stop(save_progress=True)
            self.run(allow_restart=True, add_global_selection=True)
        else:
            self._add_global_selection_to_selections()

    def _add_global_selection_to_selections(self) -> None:
        """
        Implements the logic to add the global selection to the selections in the scheme.
        """
        new_scheme: List[ExplorationSchemeStep] = []
        for step in self.scheme:
            if isinstance(step, Selection):
                if len(new_scheme) < 2:
                    # we have the first selection, since if the length is 1,
                    # the existing scheme step is either not a selection or if it is a selection, they will be combined
                    # then the length will still be 1 afterwards, so this works for any number of selections
                    new_scheme.append(self._selection_construction(step, add_global=self._global_for_first_selection))
                else:
                    new_scheme.append(self._selection_construction(step, add_global=True))
            else:
                new_scheme.append(step)
        self.scheme = new_scheme

    def remove_global_selection(self, to_remove: Selection) -> None:
        """
        Allows to remove a selection that has been added to all selections.

        Notes
        -----
        This will remove the given selection from all AndSelections in the scheme independently if the given selection
        was added as a global one or just regularly.

        Parameters
        ----------
        to_remove : Selection
            The selection to be removed.
        """
        new_scheme: List[ExplorationSchemeStep] = []
        for step in self.scheme:
            if isinstance(step, SelectionAndArray):
                new_subselections: List[Selection] = [sele for sele in step.selections if sele is not to_remove]
                if len(new_subselections) == 0:
                    new_scheme.append(AllCompoundsSelection(step.options.model))
                else:
                    new_scheme.append(SelectionAndArray(new_subselections))
            else:
                new_scheme.append(step)
        self.scheme = new_scheme

    def __iadd__(self, other: Union[List[ExplorationSchemeStep], ExplorationSchemeStep]):
        """
        The method to add new steps after the wheel has already been initialized.

        Parameters
        ----------
        other : Union[List[ExplorationSchemeStep], ExplorationSchemeStep]
            The new step(s) to be added.

        Returns
        -------
        SteeringWheel
            The updated instance of the steering wheel.

        Raises
        ------
        TypeError
            If the given exploration scheme is not valid.
        """
        self.get_results()
        previous_scheme_length = len(self.scheme)
        if isinstance(other, ExplorationSchemeStep):
            other = [other]
        if not other or not isinstance(other, list) or not all(isinstance(o, ExplorationSchemeStep) for o in other):
            raise TypeError(f"Given exploration scheme {other} is not valid")
        if self.is_running():
            self.stop(save_progress=True)
            self.run(allow_restart=True, additional_steps=other)
        else:
            self._add_exploration_steps_to_scheme(other, None, first_has_been_added=False)
        if len(self.scheme) == previous_scheme_length and \
                self._results is not None and len(self._results) == previous_scheme_length:
            # the addition was combined into a new step, remove result of previous last step
            self._results.pop(-1)
        return self

    def pop(self, index: int, ignore_results: bool = False) -> ExplorationSchemeStep:
        """
        Method to remove an exploration step from the scheme by indexing.
        Negative indices are supported, but slicing is not.

        Parameters
        ----------
        index : int
            The index of the step to be removed.
        ignore_results : bool, optional
            If the results should be ignored, by default False

        Returns
        -------
        ExplorationSchemeStep
            The removed step.

        Raises
        ------
        IndexError
            The given index is out of range.
        RuntimeError
            The given index has already been explored and the results are not ignored.
        """
        # emulate standard python negative index
        if index < 0:
            index += len(self.scheme)
        if not (0 <= index < len(self.scheme)):
            raise IndexError(f"Invalid index {index} for SteeringWheel with an exploration scheme "
                             f"of length {len(self.scheme)}")
        self.get_results()
        if not ignore_results and self._results is not None and len(self._results) > index:
            raise RuntimeError(f"Cannot remove index '{index}', " +
                               "this exploration step has already been carried out!")
        while_running = self.is_running()
        if while_running:
            self.stop(save_progress=True)
        new_scheme = self.scheme
        step = new_scheme.pop(index)
        self.scheme = []
        self._add_exploration_steps_to_scheme(new_scheme, last_selection=None, first_has_been_added=False)
        if while_running:
            self.run(allow_restart=True)
        return step

    def _determine_last_selection_in_scheme(self) -> Tuple[Optional[Selection], int]:
        """
        Find out what the last selection in the current scheme is.

        Returns
        -------
        Tuple[Optional[Selection], int]
            The last selection in the scheme and its index in the scheme.
        """
        last_selection: Tuple[Optional[Selection], int] = (None, 0)  # (Selection, index in scheme)
        for i, step in enumerate(reversed(self.scheme)):
            if isinstance(step, Selection):
                last_selection = (step, len(self.scheme) - i - 1)
                break
        return last_selection

    def run(self, allow_restart: Optional[bool] = None,
            additional_steps: Optional[List[ExplorationSchemeStep]] = None,
            add_global_selection: bool = False,
            ask_for_how_many_results: bool = False) -> None:
        """
        Start the exploration protocol. Additional steps or selections can be added just before starting.

        Parameters
        ----------
        allow_restart : Optional[bool], optional
            If the protocol may be restarted if a restart state exists, if None the input function defined
            upon initialization is used to ask if a restart state exists, by default None
        additional_steps : Optional[List[ExplorationSchemeStep]], optional
            The additional steps to be added before the run. They are sanity checked before the run, by default None
        add_global_selection : bool, optional
            Whether the existing global selection should be added to the additional steps, by default False
        ask_for_how_many_results : bool, optional
            If the wheel should ask the user how many results it should take over from the restart
            if a restart state exists, by default False

        Raises
        ------
        RuntimeError
            The exploration protocol is already running.
        """
        self.get_results()
        self.get_partial_restart_info()
        if add_global_selection:
            # make sure that set global selection is not overwritten back by loading previous wheel
            current_global = self._global_selection
            current_global_first = self._global_for_first_selection
        if self.is_running():
            raise RuntimeError(f"Already running the exploration protocol {self.name}")
        allow_restart, restart_file = self._restart_check(allow_restart)
        if allow_restart:
            self._load_impl(restart_file, ask_for_how_many_results)
            if add_global_selection:
                # set back to before the loading
                self._global_selection = current_global
                self._global_for_first_selection = current_global_first
        if additional_steps is not None:
            self._add_exploration_steps_to_scheme(additional_steps, None, first_has_been_added=False)
        if add_global_selection is not None:
            self._add_global_selection_to_selections()
        # we build non-duplex connections, so we can use our own ProxyWrapper that avoids stalling behavior
        # due to too large messages within the connection
        # therefore we need have separate objects depending on the direction of the connection
        # we do not use our ProxyWrapper for the status for now, because this should always be very small
        # this might change for very long protocols, then simply construct as results objects below
        # we need status connection, to get the status of each step and send a stop signal to the worker
        self._status_connection_recv, status_send = Pipe(duplex=False)
        status_recv, self._status_connection_send = Pipe(duplex=False)
        # we need results connections to receive the results of the steps to the worker
        # and send the already existing results before starting the worker
        self._results_connection_recv, results_send = \
            MultiProcessingConnectionsWithProxyThread.construct_connections()
        # ensure that we restart where we left off
        if self._results is not None:
            # we only send the results to the worker once, so this connection is not a class member
            # we use a normal pipe here, because our Proxy is not safe for multiprocessing after creation
            # however it might be so large, that the send is blocking until the other ends receives
            # to avoid race conditions, we first put a signal in the pipe, that the other end should wait for
            # the results to be sent, create the worker and then send the results
            existing_results_recv, existing_results_send = Pipe(duplex=False)
            existing_results_send.send(WaitForFurtherResults())
        else:
            existing_results_recv = None
        # same for partial steps, but we don't need waiting procedure, because we are sending only a single integer
        # we are constructing it with our ProxyWrapper, because the worker might spam the same integer multiple times
        self._partial_steps_connection_recv, worker_partial_step_send = \
            MultiProcessingConnectionsWithProxyThread.construct_connections()
        worker_partial_steps_recv, initial_partial_step_send = Pipe(duplex=False)
        if self._partial_protocol_restart_info is None:
            initial_partial_step_send.send(NoRestartInfoPresent())
        else:
            initial_partial_step_send.send(self._partial_protocol_restart_info)
        self._process = Process(name=self.name, target=_worker,
                                args=(self.credentials, self.scheme, self.name, self._reusing_prev_selection,
                                      existing_results_recv, results_send,
                                      status_recv, status_send, worker_partial_steps_recv, worker_partial_step_send))
        self._process.start()
        if self._results is not None:
            send_multiple_results_in_pipe(self._results, existing_results_send)

    @wraps(run)
    def start(self, *args, **kwargs) -> None:
        """
        Identical to :meth:`run`, but with a different name for convenience.
        """
        self.run(*args, **kwargs)

    def _restart_check(self, allow_restart: Optional[bool]) -> Tuple[bool, str]:
        """
        Implementation of the check for existing restart states.

        Parameters
        ----------
        allow_restart : Optional[bool]
            If a restart is even allowed, if None, the input function is used to ask if a restart state exists,
            by default None

        Returns
        -------
        Tuple[bool, str]
            If a restart is allowed and the restart file name.
        """
        have_triggered_warning = False
        restart_info = self._get_restart_info()
        restart_file = restart_info.get(self.restart_file)
        if restart_file is None:
            if allow_restart:
                warn(f"Specified restart allowance, but restart file '{self.restart_file}' does not exists.")
                have_triggered_warning = True
            cut_restart = self._cut_last_exploration_step_from_fileinfo(self.restart_file)
            restart_file = restart_info.get(cut_restart, self._cut_last_exploration_step_from_fileinfo(cut_restart))
        assert isinstance(restart_file, str)
        while True:
            # look iteratively for the restart file in the restart info by cutting off the last exploration step.
            assert isinstance(restart_file, str)
            if path.exists(path.join(self._save_dir, restart_file)):
                protocol_name = self._protocol_name_from_restart_file(restart_file, restart_info)
                if allow_restart is None:
                    split_protocol_name = self._split_protocol_name(protocol_name)
                    allow_restart = yes_or_no_question(f"Found restart file '{split_protocol_name}'.\n"
                                                       f"Do you want to continue with this", self._input)
                if have_triggered_warning:
                    split_protocol_name = self._split_protocol_name(protocol_name)
                    allow_restart = yes_or_no_question(f"Could now find the restart file '{split_protocol_name}'"
                                                       f".\nDo you want to load it", self._input)
                break
            elif not restart_file:
                # no names left to search for
                allow_restart = False
                break
            elif allow_restart:
                if not have_triggered_warning:
                    warn(f"Specified restart allowance, but restart file '{restart_file}' does not exists.")
                    have_triggered_warning = True
                # remove last step to see if we find a name from before a wheel addition
                try:
                    protocol_name = self._protocol_name_from_restart_file(restart_file, restart_info)
                except ValueError:
                    allow_restart = False
                    break
                restart_file = restart_info.get(self._cut_last_exploration_step_from_fileinfo(protocol_name), "")
            else:
                break
        assert isinstance(restart_file, str)
        if allow_restart is None:
            allow_restart = False
        return allow_restart, restart_file

    def get_results(self) -> Optional[List[ExplorationResult]]:
        """
        The results of the exploration protocol. This will also update the results of the holding exploration steps.

        Returns
        -------
        Optional[List[ExplorationResult]]
            The results of the exploration protocol, None if no results are available.
        """
        if self._results_connection_recv is None:
            return self._results
        read_results = receive_multiple_results_from_pipe(self._results_connection_recv)
        if not read_results:
            if self._results_connection_recv.was_closed():  # pylint: disable=(no-member)
                # we cannot read from our proxy thread anymore, so we avoid senselessly trying to read
                self._results_connection_recv = None
            return self._results
        self._set_results(read_results)  # this updates our own members
        return read_results

    def _set_results(self, results: Optional[List[ExplorationResult]]) -> None:
        """
        Set the results of the protocol and update the results of the holding exploration steps.

        Parameters
        ----------
        results : Optional[List[ExplorationResult]]
            The results to set.
        """
        self._results = results
        if results is not None:
            for r, step, in zip(results, self.scheme):
                step.set_result(r)

    def delete_results(self) -> None:
        """
        Delete all results. Also updates the holding steps and resets their statusD
        """
        if self.is_running():
            self.terminate(try_save_progress=False)
        self._set_results(None)
        self.set_partial_restart_info(None)
        for step in self.scheme:
            step.set_result(None)
            step.status = Status.WAITING
        # remove information written to disk
        self.clear_cache()

    def get_partial_restart_info(self) -> Optional[RestartPartialExpansionInfo]:
        if self._partial_steps_connection_recv is None:
            return self._partial_protocol_restart_info
        read_steps = read_connection(self._partial_steps_connection_recv)
        if read_steps is None:
            if self._partial_steps_connection_recv.was_closed():  # pylint: disable=(no-member)
                # we cannot read from our proxy thread anymore, so we avoid senselessly trying to read
                self._partial_steps_connection_recv = None
            return self._partial_protocol_restart_info
        if isinstance(read_steps, NoRestartInfoPresent):
            read_steps = None
        self.set_partial_restart_info(read_steps)  # this updates our own members
        return read_steps

    def set_partial_restart_info(self, restart_info: Optional[RestartPartialExpansionInfo]) -> None:
        if restart_info is not None and self._results is not None and len(self._results) == len(self.scheme):
            raise RuntimeError(
                f"Wanted to set partial protocol steps to {restart_info}, but all exploration steps have a "
                f"result already")
        self._partial_protocol_restart_info = restart_info

    def clear_cache(self) -> None:
        if path.exists(self._save_dir):
            rmtree(self._save_dir, ignore_errors=True)

    def stop(self, save_progress: bool = True):
        """
        If the protocol is running, stop it and save the progress depending on the input.

        Parameters
        ----------
        save_progress : bool, optional
            If the results should be saved to file, by default True
        """
        if self.is_running() and self._process is not None:
            self._results = self.get_results()
            if self._status_connection_send is not None:
                # try graceful exit
                self._status_connection_send.send(Status.STOPPING)
            else:
                self._process.terminate()
            self._process.join()
            try:
                self._results = self.get_results()
                self.get_partial_restart_info()
                self.get_status_report()
            except ConnectionResetError:
                pass
            if save_progress:
                self.save()
            self._close_receiving_connections()
        else:
            warn(f"Tried to stop steering wheel {self.name}, but it was not running anyways")

    def save(self):
        """
        Save the current exploration progress.
        """
        try:
            self.get_partial_restart_info()
            self._results = self.get_results()
            self.get_status_report()
        except ConnectionResetError:
            pass
        # copy things before we wipe them
        credentials = copy(self.credentials)
        results = self._results.copy() if self._results is not None else None
        partial_restart_info = self._partial_protocol_restart_info
        # save
        self._save_impl()
        # load back
        self.credentials = credentials
        self.manager = connect_to_db(credentials)
        self._results = results
        self._set_results(results)
        self.set_partial_restart_info(partial_restart_info)

    def _save_impl(self):
        """
        The implementation of the save function that handles the two-file system.

        Notes
        -----
        We require two save files, one that contains the translation of the unique id file names to the name of
        their exploration protocols they contain and then the individual files that contain the exploration protocols.
        We do this because naming the restart file after the protocol name would fail at some stage
        due to too long file names.

        Raises
        ------
        FailedSaveException
            If the save fails.
        """
        data = self._get_restart_info()
        if not path.exists(self._save_dir):
            mkdir(self._save_dir)
        # we only save the file name without the directory in the info file
        uid_file = f".chemoton_{uuid1()}.pkl"
        data[self.restart_file] = uid_file
        try:
            with open(path.join(self._save_dir, uid_file), "wb") as f:
                pickle.dump(self, f)
        except TypeError as e:
            raise FailedSaveException(f"Failed to save {self.name} to {uid_file} because {e}") from e
        with open(self._restart_info_file, "wb") as f:
            pickle.dump(data, f)

    def _load_impl(self, file_name: str, ask_for_how_many_results: bool) -> None:
        """
        Implementation of the loading process from a unique id file name with the information if we should ask if we
        want all results from the file.

        Parameters
        ----------
        file_name : str
            The unique id file name.
        ask_for_how_many_results : bool
            If we should ask for how many results we want to load.
        """
        protocol_name = self._protocol_name_from_restart_file(file_name)
        with open(path.join(self._save_dir, file_name), "rb") as f:
            obj = pickle.load(f)
        obj_results = obj.get_results()
        if self._results is not None and obj_results is None:
            warn(f"Don't remove existing results while loading {protocol_name}")
        elif obj_results is not None and obj_results:
            partial_restart = obj.get_partial_restart_info()
            if ask_for_how_many_results:
                split_protocol_name = self._split_protocol_name(protocol_name)
                n_avail = len(obj_results)
                sure = False
                how_many_results = n_avail
                while not sure:
                    q = f"Found results for {n_avail} steps, from the protocol\n{split_protocol_name}\n" \
                        f"that contains {len(obj)} exploration steps.\nHow many results do you want to load"
                    how_many_results = integer_question(q, limits=(0, len(obj_results)), callable_input=self._input)
                    if how_many_results == n_avail:
                        # assume that it is safe to pick all the results and don't ask twice
                        break
                    q = f"Only selected {how_many_results} results from {n_avail} available results\n" \
                        f"Are you sure"
                    sure = yes_or_no_question(q, callable_input=self._input)

                if how_many_results < n_avail and partial_restart is not None:
                    warn("Did not load all results, "
                         "hence we restart the partially executed exploration step from scratch")
                    wanted_partial_restart = None
                elif partial_restart is not None:
                    q = f"The exploration protocol also contains a partially executed step with " \
                        f"{partial_restart.n_already_executed_protocol_steps} engines of its protocol executed.\n" \
                        f"Do you want to continue from the partial execution (y) or restart the step (n)"
                    honor_partial_restart = yes_or_no_question(q, callable_input=self._input)
                    wanted_partial_restart = partial_restart if honor_partial_restart else None
                else:
                    wanted_partial_restart = None
                wanted_results = obj_results[:how_many_results]
            else:
                wanted_results = obj_results
                wanted_partial_restart = partial_restart
            if self._results is not None and len(wanted_results) < len(self._results):
                kept_results = self._results[len(wanted_results):]
            else:
                kept_results = []
            self.__dict__.update(obj.__dict__)
            self._set_results(wanted_results + kept_results)
            self.set_partial_restart_info(wanted_partial_restart)
        else:
            self.__dict__.update(obj.__dict__)
            self._set_results(obj_results)

    @staticmethod
    def _split_protocol_name(protocol_name: str, split_size: int = 5) -> str:
        split_protocol_name = ""
        for i in range(0, protocol_name.count('-'), split_size):
            split_protocol_name += "-".join(protocol_name.split("-")[i:i + split_size]) + "-\n"
        return split_protocol_name

    def _protocol_name_from_restart_file(self, restart_file: str, restart_info: Optional[Dict[str, str]] = None) \
            -> str:
        """
        Deduce the protocol name from the restart file name.

        Parameters
        ----------
        restart_file : str
            The name of the unique id file.
        restart_info : Optional[Dict[str, str]], optional
            The restart info dictionary, if None, we read it from our restart file, by default None

        Returns
        -------
        str
            The protocol name.

        Raises
        ------
        ValueError
            The restart file is not saved in our restart info file.
        """
        if restart_info is None:
            restart_info = self._get_restart_info()
        if restart_file not in restart_info.values():
            raise ValueError(f"Could not find {restart_file} in {restart_info}")
        return list(restart_info.keys())[list(restart_info.values()).index(restart_file)]

    def _get_restart_info(self) -> Dict[str, str]:
        """
        Implementation to load our restart info dictionary from the restart file name.

        Returns
        -------
        Dict[str, str]
            The restart info dictionary, translating protocol names to unique ids.
        """
        if not path.exists(self._restart_info_file):
            return {}
        with open(self._restart_info_file, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _cut_last_exploration_step_from_fileinfo(filename: str) -> str:
        """
        Remove the last step from the file name.
        """
        filename, suffix = path.splitext(filename)
        return "-".join(filename.split("-")[:-1]) + suffix

    def terminate(self, try_save_progress: bool = True, suppress_warning: bool = False) -> None:
        """
        Immediately terminate the exploring subprocess and don't wait for the current step to finish.

        Parameters
        ----------
        try_save_progress : bool, optional
            Whether we should still try to save the current state, by default True
        suppress_warning : bool, optional
            Whether we should suppress the warning if the wheel was not running, by default False
        """
        if self.is_running() and self._process is not None:
            if try_save_progress:
                try:
                    self._results = self.get_results()
                    self.get_partial_restart_info()
                except ConnectionResetError:
                    pass
                self._save_impl()
            self._process.terminate()
            self._close_receiving_connections()
            self._process.join()
        elif not suppress_warning:
            warn(f"Tried to stop steering wheel {self.name}, but it was not running anyways")

    def join(self):
        """
        Joins the exploring subprocess.
        """
        if self._process is not None:
            self._process.join()

    def is_running(self) -> bool:
        """
        Whether we are currently running an exploration.
        """
        return self._process is not None and self._process.is_alive()

    def _construct_status_report(self) -> Dict[str, Status]:
        """
        Build a status report from the currently holding steps.

        Returns
        -------
        Dict[str, Status]
            The status report.
        """
        return {f"{count}: {step.name}": step.status for count, step in enumerate(self.scheme)}

    def _update_status(self, status: List[Status]) -> None:
        """
        Change the status of each currently holding step.

        Parameters
        ----------
        status : List[Status]
            The new status for each step.
        """
        for step, stat in zip(self.scheme, status):
            step.status = stat

    def get_status_report(self) -> Dict[str, Status]:
        """
        Update the status of the currently holding steps and return a status report.
        The report contains the step names prepended with their index (to ensure unique keys) and the values are the
        status of the step.

        Returns
        -------
        Dict[str, Status]
            The status report.
        """
        self.get_partial_restart_info()  # this makes sure that status and partial steps are not conflicting
        if self._status_connection_recv is None:
            return self._construct_status_report()
        status = read_connection(self._status_connection_recv)
        if status is not None:
            self._update_status(status)
        return self._construct_status_report()

    def _selection_construction(self, selection: Selection, add_global: bool = True) -> Selection:
        """
        Implementation to construct a selection from the given selection and the global selection.

        Parameters
        ----------
        selection : Selection
            The selection to add the global selection to if needed.
        add_global : bool, optional
            Whether the global selection should be added if one exists, by default True

        Returns
        -------
        Selection
            The constructed selection.
        """
        if self._global_selection is None or not add_global:
            return selection
        return SelectionAndArray([self._global_selection, selection])

    def __bool__(self) -> bool:
        """
        Shortcut to evaluate existing exploration scheme.
        """
        return len(self.scheme) > 0

    def __len__(self) -> int:
        """
        Length of the exploration scheme.
        """
        return len(self.scheme)

    def _close_receiving_connections(self) -> None:
        if self._results_connection_recv is not None:
            self._results_connection_recv.close()  # pylint: disable=(no-member)
            self._results_connection_recv = None
        if self._partial_steps_connection_recv is not None:
            self._partial_steps_connection_recv.close()  # pylint: disable=(no-member)
            self._partial_steps_connection_recv = None

    def __del__(self) -> None:
        try:
            self._close_receiving_connections()
            if self.is_running():
                self.terminate(try_save_progress=False)
            else:
                self.join()
        except BaseException:
            pass


def _worker(credentials: db.Credentials, scheme: List[ExplorationSchemeStep], name: str,
            reusing_prev_selection: Dict[int, int],
            results_connection_recv: Optional[ReadAble], results_connection_send: Connection,
            stop_status_connection: Connection, status_connection: Connection,
            worker_partial_steps_recv: Connection, worker_partial_step_send: Connection
            ) -> None:
    """
    The worker function for the exploration process, that is forked to execute the exploration protocol.

    Parameters
    ----------
    credentials : db.Credentials
        The credentials to connect to the database.
    scheme : List[ExplorationSchemeStep]
        The exploration scheme.
    name : str
        The name of the Steering Wheel.
    reusing_prev_selection : Dict[int, int]
        A dictionary telling at which step we should reuse the previous selection from the given index.
    results_connection_recv : Optional[ReadAble]
        The connection to receive existing results from.
    results_connection_send : Connection
        The connection to send the results to.
    stop_status_connection : Connection
        The connection to receive potential stop signal from at end of each step.
    status_connection : Connection
        The connection to send the status to.

    Raises
    ------
    RuntimeError
        The existing results are not compatible with the status of the step or the reusing of previous results
        is not possible.
    """
    partial_restart_info = read_connection(worker_partial_steps_recv)
    if partial_restart_info is None:
        raise RuntimeError("Received no information about the partial restart state")
    if isinstance(partial_restart_info, NoRestartInfoPresent):
        partial_restart_info = None
    results: List[ExplorationResult] = receive_multiple_results_from_pipe(results_connection_recv)
    if results_connection_recv is not None:
        results_connection_recv.close()
    last_output = None if not results else results[-1]
    n = len(results)
    status_connection.send([s.status for s in scheme])
    # safety check for steps which are already done
    for s in scheme[:n]:
        if s.status in [Status.CALCULATING, Status.WAITING]:
            if isinstance(s, PredeterminedSelection):
                s.status = Status.FINISHED
                status_connection.send([s.status for s in scheme])
            else:
                raise RuntimeError(f"ExplorationScheme {name} was incorrectly deduced.\n"
                                   f"Step {s.name} has already results, but its status is {s.status}")

    def _send_partial_steps(restart_info_: Union[NoRestartInfoPresent, RestartPartialExpansionInfo]) -> None:
        # we don't want to send the step index but the number of executed steps,
        # so we add 1
        worker_partial_step_send.send(restart_info_)

    have_given_partial_steps = False
    for i, step in enumerate(scheme[n:], n):
        step.status = Status.CALCULATING
        status_connection.send([s.status for s in scheme])
        have_expansion = isinstance(step, NetworkExpansion)
        callback = _send_partial_steps if have_expansion else None
        restart_info = partial_restart_info if not have_given_partial_steps and have_expansion else None
        have_given_partial_steps = have_expansion
        if i in reusing_prev_selection:
            wanted_result_index = reusing_prev_selection[i]
            if wanted_result_index >= len(results):
                raise RuntimeError(f"ExplorationScheme {name} was incorrectly deduced.\n"
                                   f"Wanted to take the selection result {wanted_result_index} "
                                   f"({scheme[wanted_result_index].name}) for step {i} ({step.name}), "
                                   f"but we only have {len(results)} so far.")
            last_output = step(credentials, results[wanted_result_index], callback, restart_info)
        elif last_output is None:
            last_output = step(credentials, notify_partial_steps_callback=callback,
                               restart_information=restart_info)
        else:
            last_output = step(credentials, last_output, callback, restart_info)
        if not isinstance(last_output, ExplorationResult):
            raise TypeError(f"ExplorationScheme {name} was incorrectly deduced.\n"
                            f"Step {step.name} did not return an ExplorationResult, but a {type(last_output)}")
        _send_partial_steps(NoRestartInfoPresent())  # because step is finished
        results.append(last_output)
        # copy because we remove things for pickling which we do for sending across connections
        send_multiple_results_in_pipe(results.copy(), results_connection_send)
        # update status
        status_connection.send([s.status for s in scheme])
        # check for potential exit
        status_received = read_connection(stop_status_connection)
        if status_received is not None:
            if status_received == Status.STOPPING:
                # got signal to stop gracefully
                return
            # got signal, but not the expected signal
            warn(f"Received unknown signal {status_received.name}, "
                 f"SteeringWheel worker process only expects {Status.STOPPING.name} status")
            return
    # we are finished
    # we send signal to stop main process from querying the result connection permanently
    # this relies on the indirect coupling that the main process will have all the results it needs via the memory
    # of the proxy object and will not query the connection anymore
    # and reestablish a new connection if more results are needed
    results_connection_send.send(StopReading())
