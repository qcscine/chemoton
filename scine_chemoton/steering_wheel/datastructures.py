#!/usr/bin/env python3
from __future__ import annotations

# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import datetime
from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.connection import Connection
from typing import Any, List, Optional, Union, Tuple, Type, Dict, Callable

import scine_database as db
import scine_utilities as utils

from scine_chemoton.engine import Engine
from scine_chemoton.gears import Gear, HasName, HoldsCollections
from scine_chemoton.gears.elementary_steps import ElementaryStepGear
from scine_chemoton.filters.aggregate_filters import AggregateFilter
from scine_chemoton.filters.reactive_site_filters import ReactiveSiteFilter
from scine_chemoton.filters.further_exploration_filters import FurtherExplorationFilter
from scine_chemoton.gears.elementary_steps.trial_generator import TrialGenerator
from scine_chemoton.gears.kinetics import KineticsBase
from scine_chemoton.utilities.datastructure_transfer import ReadAble
from scine_chemoton.utilities.options import BaseOptions


class Status(Enum):
    """
    Member of each Exploration step to indicate the current status of the step.
    """
    WAITING = "waiting"  # step waits for previous step
    STALLING = "stalling"  # step has to be calculated but no calculations are running
    CALCULATING = "calculating"  # calculations are running
    FINISHED = "finished"  # successfully finished
    INTERACTION_REQUIRED = "interaction required"  # need some interactive input
    FAILED = "failed"  # technical or conceptual failure
    STOPPING = "stopping"  # have to abort


class LogicCoupling(Enum):
    """
    Signals how multiple selection are combined, helps to define this coupling in Heron.
    """
    AND = "and"
    OR = "or"


class ExplorationResult:
    """
    Result of a single step in the exploration scheme. This is the base class for all results.
    """

    @classmethod
    @abstractmethod
    def from_split_results(cls, split_results: List[Any]):
        pass

    @abstractmethod
    def to_split_results(self) -> List[Any]:
        pass

    def send_in_pipe(self, pipe: Connection) -> None:
        split_results = self.to_split_results()
        pipe.send(self.__class__)
        for result in split_results:
            pipe.send(result)
        pipe.send(StopSplitCommunicationMethod())

    @classmethod
    def receive_from_pipe(cls, pipe: Union[Connection, ReadAble]) -> Any:
        results = []
        while True:
            result = pipe.recv()
            if result == cls:
                # ignore class information in case it is present
                continue
            if result is None or isinstance(result, StopSplitCommunicationMethod):
                break
            if isinstance(result, type) and result != cls:
                raise ValueError(f"The connection does not contain the class information {cls} from which"
                                 f"the receive method was called, but {result} instead."
                                 f"This could be caused if this classmethod was called by the base class."
                                 f"If you don't know which result type to expect, make sure to use the class-agnostic "
                                 f"functions in the result_transfer module.")
            results.append(result)
        return cls.from_split_results(results)

    @staticmethod
    def _split_list_into_chunks(lst: List[Any], n: int) -> List[List[Any]]:
        len_list = len(lst)
        return [lst[i * n:(i + 1) * n] for i in range((len_list + n - 1) // n)]


@dataclass
class SelectionResult(ExplorationResult):
    """
    The result of a selection step. It holds varies filters to be taken over by the next step
    and / or individual structure IDs.
    """
    aggregate_filter: AggregateFilter = AggregateFilter()
    reactive_site_filter: ReactiveSiteFilter = ReactiveSiteFilter()
    further_exploration_filter: FurtherExplorationFilter = FurtherExplorationFilter()
    structures: List[db.ID] = field(default_factory=list)

    def to_split_results(self) -> List[Tuple[Optional[AggregateFilter],
                                             Optional[ReactiveSiteFilter],
                                             Optional[FurtherExplorationFilter],
                                             List[db.ID]]]:
        results: List[Tuple[Optional[AggregateFilter], Optional[ReactiveSiteFilter],
                            Optional[FurtherExplorationFilter], List[db.ID]]] = []
        # split structures into chunks of 1000 entries
        split_structures = self._split_list_into_chunks(self.structures, 1000)
        if not split_structures:
            results.append((self.aggregate_filter,
                            self.reactive_site_filter,
                            self.further_exploration_filter,
                            []))
        for i, structures in enumerate(split_structures):
            if i == 0:
                results.append((self.aggregate_filter,
                                self.reactive_site_filter,
                                self.further_exploration_filter,
                                structures))
            else:
                results.append((None, None, None, structures))
        return results

    @classmethod
    def from_split_results(cls, split_results: List[Any]):
        inst = cls()
        if not split_results:
            return inst
        inst.aggregate_filter, inst.reactive_site_filter, inst.further_exploration_filter, inst.structures \
            = split_results[0]
        for _, _, _, structures in split_results[1:]:
            inst.structures.extend(structures)
        return inst


@dataclass
class NetworkExpansionResult(ExplorationResult):
    """
    The result of a network expansion step. It holds different list of database IDs, that were
    created / modified by the network expansion.
    """
    reactions: List[db.ID] = field(default_factory=list)
    compounds: List[db.ID] = field(default_factory=list)
    flasks: List[db.ID] = field(default_factory=list)
    structures: List[db.ID] = field(default_factory=list)

    def __bool__(self):
        """
        Short notation if found / changed anything. Can be used to evaluate success / failure.
        """
        return any(bool(entry) for entry in [self.reactions, self.compounds, self.flasks, self.structures])

    def to_split_results(self) -> List[Any]:
        split_reactions = self._split_list_into_chunks(self.reactions, 1000)
        split_compounds = self._split_list_into_chunks(self.compounds, 1000)
        split_flasks = self._split_list_into_chunks(self.flasks, 1000)
        split_structures = self._split_list_into_chunks(self.structures, 1000)
        # extend all lists to the same length of the longest list
        max_len = max(len(split_reactions), len(split_compounds), len(split_flasks), len(split_structures))
        split_reactions.extend([[]] * (max_len - len(split_reactions)))
        split_compounds.extend([[]] * (max_len - len(split_compounds)))
        split_flasks.extend([[]] * (max_len - len(split_flasks)))
        split_structures.extend([[]] * (max_len - len(split_structures)))
        # write chunked lists to returned list
        results = []
        for reactions, compounds, flasks, structures in \
                zip(split_reactions, split_compounds, split_flasks, split_structures):
            results.append((reactions, compounds, flasks, structures))
        return results

    @classmethod
    def from_split_results(cls, split_results: List[Any]):
        inst = cls()
        for reactions, compounds, flasks, structures in split_results:
            inst.reactions.extend(reactions)
            inst.compounds.extend(compounds)
            inst.flasks.extend(flasks)
            inst.structures.extend(structures)
        return inst


class ExplorationSchemeStep(ABC, HasName, HoldsCollections):
    """
    The base class for a step in an exploration protocol. It defines the common handling of
    status, model, result access, name handling, and the initialization of required collections of the DB.
    """

    class Options(BaseOptions):
        """
        Options for the exploration scheme step.
        """

        def __init__(self, model: db.Model, *args, **kwargs):
            """
            Construct it with a given model

            Parameters
            ----------
            model : db.Model
                The model given to all subclasses.
            """
            if not isinstance(model, db.Model):
                raise TypeError("model must be a Model object")
            self.model = model
            super().__init__(*args, **kwargs)

    options: ExplorationSchemeStep.Options

    def __init__(self, model: db.Model, *args, **kwargs):
        super().__init__()
        self.options = self.Options(model, *args, **kwargs)
        self.status = Status.WAITING
        self._remove_chemoton_from_name()
        self._required_collections = self.possible_attributes()
        self._result: Optional[ExplorationResult] = None

    def get_result(self) -> Optional[ExplorationResult]:
        return self._result

    def set_result(self, result: Optional[ExplorationResult]) -> None:
        self._result = result

    @abstractmethod
    def __call__(
            self,
            credentials: db.Credentials,
            last_output: Optional[Any] = None,
            notify_partial_steps_callback:
            Optional[Callable[[Union[NoRestartInfoPresent, RestartPartialExpansionInfo]], None]] = None,
            restart_information: Optional[RestartPartialExpansionInfo] = None) -> Any:
        pass

    def __str__(self):
        return self.name

    def __eq__(self, other) -> bool:
        """
        Compare type and options.
        """
        return self.name == other.name and self.options == other.options


@dataclass
class ProtocolEntry:
    """
    A container for executions carried out by a network expansion.
    This holds the combined information about a gear, the database and how the gear should be executed.

    Notes
    -----
    The default execution mode, if only credentials and a gear are given, is forking the gear and running it
    indefinitely.

    Raises
    ------
    RuntimeError
        If the specified execution behavior (forking and number of runs) does not make sense.
    """
    credentials: db.Credentials
    gear: Gear

    fork: bool = field(default=True)
    n_runs: int = field(default=0)  # 0 means run endless
    wait_for_calculation_finish: bool = field(default=False)

    was_stopped: bool = field(init=False)
    name: str = field(init=False)
    limited_runs: bool = field(init=False)
    engine: Engine = field(init=False)

    _gears_that_can_stop_themselves: List[Type] = field(init=False, repr=False, default_factory=lambda: [KineticsBase])
    _methods_to_ensure_that_gear_stops_itself: Dict[Type[Gear], Callable[[Gear], None]] = \
        field(init=False, repr=False, default_factory=(lambda: {
            KineticsBase:  # type: ignore
            lambda gear: setattr(gear.options, "stop_if_no_new_aggregates_are_activated", True)
        }))

    def __post_init__(self):
        assert len(self._gears_that_can_stop_themselves) == len(self._methods_to_ensure_that_gear_stops_itself)
        self.was_stopped = False
        self.name = self.gear.name
        self.limited_runs = self.n_runs > 0
        if self.fork and self.n_runs > 1:
            raise RuntimeError(f"Invalid protocol for {self.name}, "
                               f"multiple forked runs would most likely lead to an inconsistent database.")
        if not self.fork and not self.limited_runs:
            for gear_type, method in self._methods_to_ensure_that_gear_stops_itself.items():
                if isinstance(self.gear, gear_type):
                    method(self.gear)
                    break
            else:
                raise RuntimeError(f"Invalid protocol for {self.name}, "
                                   f"unforked engines must not run indefinitely")
        self.engine = Engine(self.credentials, fork=self.fork)
        self.engine.set_gear(self.gear)

    def run(self) -> None:
        """
        Run the gear in the specified mode.
        Due to better flexibility, this class only holds the information about the number of runs,
        and this method still executes the gear only once.
        """
        self.engine.set_gear(self.gear)
        self.engine.run(single=self.limited_runs)
        if not self.limited_runs and not self.fork:
            # gear stopped itself
            self.was_stopped = True

    def is_running(self) -> bool:
        return self.engine.is_running()

    def stop(self) -> None:
        self.engine.stop()
        self.engine.join()
        self.was_stopped = True

    def terminate(self) -> None:
        self.engine.terminate()
        self.was_stopped = True


class StopPreviousProtocolEntries:
    """
    An instance can be put into a protocol of a Network Expansion
    to signal that all previous protocol entries should be stopped and waited for
    """

    def stop(self):
        """
        Dummy method
        """


class GearOptions(UserDict):
    """
    A container with additional sanity checks to hold and modify the options of each gear.
    The keys are the names of the gears and the values are a tuple of the gear's options and the
    TrialGenerator's options in case the gear holds a TrialGenerator, otherwise the second entry is None.

    Notes
    -----
    Currently does not support initialization from existing dictionary.
    """

    def __init__(self, gears_and_indices: Optional[List[Tuple[Gear, Optional[int]]]] = None,
                 model: Optional[db.Model] = None):
        """
        Convenience method to only plug-in gears and an optional for model, that is then already added
        to each option. This builds the respective dictionary based on the common API of the gears.

        Parameters
        ----------
        gears_and_indices : Optional[List[Tuple[Gear, Optional[int]]]], optional
            List of gears to generate options for, by default None.
            The optional integer represents if this options should apply to all gears of this type (None)
            or only to the gear with the given index (starting at zero).
        model : Optional[db.Model], optional
            Optional model to add to the options, by default None.
        """
        super().__init__()
        if gears_and_indices is not None:
            self._loop_impl(gears_and_indices, model)

    def _loop_impl(self, gears_and_indices: List[Tuple[Gear, Optional[int]]], model: Optional[db.Model] = None):
        """
        Implementation for adding options by giving gears.
        """
        for gear, index in gears_and_indices:
            if model is not None:
                gear.options.model = model
            self.data[(gear.name, index)] = self.generate_value(gear)

    def __iadd__(self, other: Union[List[Tuple[Gear, Optional[int]]], Tuple[Gear, Optional[int]]]):
        """
        Addition method to add gears to the options.
        """
        if isinstance(other, tuple) and len(other) == 2 and isinstance(other[0], Gear):
            add: List[Tuple[Gear, Optional[int]]] = [other]
        else:
            add = other  # type: ignore
        if not isinstance(add, list) or not add or not isinstance(add[0], tuple) or not len(add[0]) == 2 \
                or not isinstance(add[0][0], Gear):
            raise TypeError(f"Can only add a single gear or a list of gears to {self.__class__.__name__}, "
                            f"you added {add}")
        self._loop_impl(add)
        return self

    @staticmethod
    def generate_value(gear: Gear) -> Tuple[Gear.Options, Optional[TrialGenerator.Options]]:
        """
        Builds a single tuple value from a gear based on known Chemoton API.
        """
        if isinstance(gear, ElementaryStepGear):
            return gear.options, gear.trial_generator.options
        else:
            return gear.options, None

    def key_check(self, key: Tuple[str, Optional[int]]) -> None:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(f"{self.__class__.__name__} requires a tuple of length 2 as key")

    def __getitem__(self, key: Tuple[str, Optional[int]]) -> Tuple[Gear.Options, Optional[TrialGenerator.Options]]:
        self.key_check(key)
        return super().__getitem__(key)

    def __setitem__(self, key: Tuple[str, Optional[int]],
                    value: Tuple[Gear.Options, Optional[TrialGenerator.Options]]) -> None:
        self.key_check(key)
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(f"Can only set a tuple of length 2 as value for {self.__class__.__name__}, "
                            f"you added {value}")
        super().__setitem__(key, value)

    def apply_to_protocol(self, protocol: List[Union[ProtocolEntry, StopPreviousProtocolEntries]]) -> None:
        if not self.data:
            return
        loop_count: Dict[str, int] = defaultdict(int)
        used_options = set()
        for entry in protocol:
            if isinstance(entry, StopPreviousProtocolEntries):
                continue
            gear = entry.gear
            this_gear_count = loop_count[gear.name]
            loop_count[gear.name] += 1
            # check for keys with None that signal that these options apply to all gears of this type
            key: Tuple[str, Optional[int]] = (gear.name, None)
            options = self.data.get(key)
            if options is not None:
                self._give_gear_option(gear, options)
                used_options.add(key)
            # check for keys with index that signal that these options apply to this specific gear
            key = (gear.name, this_gear_count)
            options = self.data.get(key)
            if options is not None:
                self._give_gear_option(gear, options)
                used_options.add(key)
        if len(used_options) != len(self.data):
            unused_options = set(self.data.keys()) - used_options
            description = "these gears" if len(unused_options) > 1 else "this gear"
            raise TypeError(f"Specified options for '{unused_options}, but {description} is not present in the "
                            f"step protocol {protocol}")

    def _give_gear_option(self, gear: Gear, options: Tuple[Gear.Options, Optional[TrialGenerator.Options]]) -> None:
        """
        Convenience method to give the options to a gear.
        """
        gear.options = options[0]
        if options[1] is not None:
            if not isinstance(gear, ElementaryStepGear):
                raise NotImplementedError(f"Gear options for '{gear.name}' hold another set of options, "
                                          f"but this gear is not an elementary step gear. "
                                          f"This is not supported.")
            gear.trial_generator.options = options[1]


class StopSplitCommunicationMethod:
    """
    Signals in a Connection that a split-up communication is finished
    """


@dataclass
class StructureInformation:
    """
    Container to hold a chemical system with fully specified potential energy surface information.
    Common API to be generated by implementations of InputSelections, so that basic input behavior
    is deduplicated and specified in base class.
    """
    geometry: Union[utils.AtomCollection, utils.PeriodicSystem]
    charge: int
    multiplicity: int


@dataclass
class RestartPartialExpansionInfo:
    protocol_step_index: int
    start_id: Optional[db.ID]
    start_time: datetime.datetime
    n_already_executed_protocol_steps: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_already_executed_protocol_steps = self.protocol_step_index + 1


class NoRestartInfoPresent:
    pass
