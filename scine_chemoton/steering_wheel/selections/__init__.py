#!/usr/bin/env python3
from __future__ import annotations
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""
from abc import abstractmethod, ABC, ABCMeta
from functools import wraps
from typing import Callable, List, Optional, Set, Union

import scine_database as db
from scine_database.energy_query_functions import get_barriers_for_elementary_step_by_type

from scine_chemoton.filters.aggregate_filters import (
    AggregateFilter,
    AggregateFilterAndArray,
    AggregateFilterOrArray
)
from scine_chemoton.filters.reactive_site_filters import (
    ReactiveSiteFilter,
    ReactiveSiteFilterAndArray,
    ReactiveSiteFilterOrArray,
)
from scine_chemoton.filters.further_exploration_filters import (
    FurtherExplorationFilterAndArray,
    FurtherExplorationFilterOrArray,
)
from scine_chemoton.utilities import connect_to_db
from ..datastructures import (
    NetworkExpansionResult,
    SelectionResult,
    ExplorationSchemeStep,
    Status,
    LogicCoupling,
    NoRestartInfoPresent,
    RestartPartialExpansionInfo
)


def status_wrap(fun: Callable):
    """
    Decorator to wrap a function and set the status of the class to calculating while the function is running
    and to finished after the function has finished.
    """
    @wraps(fun)
    def _impl(self, *args, **kwargs):
        self.status = Status.CALCULATING
        result = fun(self, *args, **kwargs)
        if self.status != Status.FAILED:
            self.status = Status.FINISHED
        return result

    return _impl


class Selection(ExplorationSchemeStep, metaclass=ABCMeta):
    """
    The base class for selecting aggregates, individual structures, and/or reactive sites.
    It specifies the common __call__ execution and holds 1 abstract methods that
    must be implemented by each implementation.
    Additionally, it holds some common functionalities for querying
    to simplify future implementation of new selections.
    """

    options: Selection.Options

    def __init__(self, model: db.Model,  # pylint: disable=keyword-arg-before-vararg
                 additional_aggregate_filters: Optional[List[AggregateFilter]] = None,
                 additional_reactive_site_filters: Optional[List[ReactiveSiteFilter]] = None,
                 logic_coupling: Union[str, LogicCoupling] = LogicCoupling.AND,
                 *args, **kwargs
                 ) -> None:
        """
        Initialize the selection with the given parameters.

        Parameters
        ----------
        model : db.Model
            The model to use for the selection.
        additional_aggregate_filters : Optional[List[AggregateFilter]], optional
            An optional list of aggregate filters to further limit selection. They are combined by an 'and' logic step.
            By default, None.
        additional_reactive_site_filters : Optional[List[ReactiveSiteFilter]], optional
            An optional list of reactive site filters to further limit selection. They are combined by an 'and' logic
            step. By default, None
        logic_coupling : Union[str, LogicCoupling], optional
            Define how this selection may be coupled together with other selections, by default LogicCoupling.AND
        """
        super().__init__(model, *args, **kwargs)
        self._add_aggregate_filter = additional_aggregate_filters if additional_aggregate_filters is not None else []
        self._add_site_filter = additional_reactive_site_filters if additional_reactive_site_filters is not None else []
        self._step_result: Optional[NetworkExpansionResult] = None
        if isinstance(logic_coupling, str):
            self.logic_coupling = LogicCoupling(logic_coupling.replace("LogicCoupling.", "").lower())
        else:
            self.logic_coupling = logic_coupling
        self._result: Optional[SelectionResult] = None

    def get_step_result(self) -> NetworkExpansionResult:
        """
        Get the result of the previous expansion step.

        Returns
        -------
        NetworkExpansionResult
            The result of the previous expansion step.

        Raises
        ------
        RuntimeError
            If the selection did not receive a previous step result.
        """
        if self._step_result is None:
            raise RuntimeError(f"The selection {self.name} did not receive a previous step result, "
                               f"but wanted to access it")
        return self._step_result

    def set_step_result(self, step_result: Optional[NetworkExpansionResult]) -> None:
        self._step_result = step_result

    def _not_implemented_arguments_sanity_check(
            self,
            notify_partial_steps_callback:
            Optional[Callable[[Union[NoRestartInfoPresent, RestartPartialExpansionInfo]], None]] = None,
            restart_information: Optional[RestartPartialExpansionInfo] = None) -> None:
        if notify_partial_steps_callback is not None:
            raise NotImplementedError("The notify_partial_steps_callback is not implemented for Selections.")
        if restart_information is not None:
            raise NotImplementedError("The restart_information is not implemented for Selections.")

    @status_wrap
    def __call__(self, credentials: db.Credentials, step_result: Optional[NetworkExpansionResult] = None,
                 notify_partial_steps_callback: Optional[
                     Callable[[Union[
                         NoRestartInfoPresent, RestartPartialExpansionInfo]], None]] = None,
                 restart_information: Optional[RestartPartialExpansionInfo] = None) \
            -> SelectionResult:
        self._not_implemented_arguments_sanity_check(notify_partial_steps_callback, restart_information)
        manager = connect_to_db(credentials)
        self.initialize_collections(manager)
        if step_result is not None:
            self.set_step_result(step_result)
        try:
            self._result = self._select()
        except BaseException as e:
            print(f"Selection {self.name} failed with error {e}")
            self.status = Status.FAILED
            raise e
        if self._add_aggregate_filter:
            self._result.aggregate_filter = AggregateFilterAndArray(self._add_aggregate_filter +
                                                                    [self._result.aggregate_filter])
        if self._add_site_filter:
            self._result.reactive_site_filter = ReactiveSiteFilterAndArray(self._add_site_filter +
                                                                           [self._result.reactive_site_filter])
        return self._result

    def lowest_barrier_per_reaction(self, step_result: NetworkExpansionResult,
                                    energy_type: str) -> List[Optional[float]]:
        """
        Convenience method to get the lowest barrier for each reaction in the given step result.

        Parameters
        ----------
        step_result : NetworkExpansionResult
            The step result to get the lowest barrier for.
        energy_type : str
            The energy type to use for the barrier lookup.

        Returns
        -------
        List[Optional[float]]
            A list of the lowest barrier for each reaction in the given step result.
        """
        barriers: List[Optional[float]] = []
        for rid in step_result.reactions:
            reaction = db.Reaction(rid, self._reactions)
            step_barriers = []
            for sid in reaction.get_elementary_steps():
                step = db.ElementaryStep(sid, self._elementary_steps)
                _, rhs = get_barriers_for_elementary_step_by_type(step, energy_type, self.options.model,
                                                                  self._structures, self._properties)
                if rhs is not None:
                    step_barriers.append(rhs)
            barriers.append(min(step_barriers) if step_barriers else None)
        return barriers

    def get_result(self) -> Optional[SelectionResult]:
        return self._result

    def set_result(self, result: Optional[SelectionResult]):  # type: ignore[override]
        self._result = result

    @abstractmethod
    def _select(self) -> SelectionResult:
        """
        Abstract method to be implemented by each selection.
        The handling of the database initialization and the addition of the given additional filters is already
        covered by the base class.

        Returns
        -------
        SelectionResult
            The result of the selection.
        """


class SafeFirstSelection(Selection, metaclass=ABCMeta):
    """
    A selection that is safe to use as the first selection in a network expansion.
    This is still an abstract class and should not be used directly.
    Instead, classes that can be used as the first selection should inherit from this class.
    """

    def get_step_result(self) -> NetworkExpansionResult:
        """
        Ensures that the step_result member is not accessed.
        """
        raise PermissionError(f"The class {self.__class__.__name__} may not access the step_result member,\n"
                              f"because it inherits from 'SafeFirstSelection' and must therefore give a\n"
                              f"selection without a previous step result.")

    def set_step_result(self, step_result: Union[NetworkExpansionResult, None]) -> None:
        pass


class AllCompoundsSelection(SafeFirstSelection):
    """
    Most basic selection that selects all compounds in the database, which is defined by an empty
    result because the default filters do not filter anything.
    However, the structures are empty to avoid transferring huge lists.
    """

    def _select(self) -> SelectionResult:
        return SelectionResult()


class PredeterminedSelection(SafeFirstSelection):
    """
    A selection that is predetermined by the user.
    """

    options: PredeterminedSelection.Options

    def __init__(self, model: db.Model, result: SelectionResult,  # pylint: disable=keyword-arg-before-vararg
                 additional_aggregate_filters: Optional[List[AggregateFilter]] = None,
                 additional_reactive_site_filters: Optional[List[ReactiveSiteFilter]] = None,
                 logic_coupling: Union[str, LogicCoupling] = LogicCoupling.AND,
                 *args, **kwargs) -> None:
        super().__init__(model, additional_aggregate_filters, additional_reactive_site_filters, logic_coupling,
                         *args, **kwargs)
        self._result = result

    def _select(self) -> SelectionResult:
        assert self._result is not None
        return self._result


class _MultipleSelections(Selection, ABC):
    """
    A base class for the classes that combine multiple selections.

    Notes
    -----
    * It requires to be initialized with a list of selections.
    * It receives its model from the first selection in the list.
    * It is not possible to use this class directly, but it should be inherited from.
    """

    options: _MultipleSelections.Options

    def __init__(self, selections: List[Selection], *args, **kwargs) -> None:
        if not selections:
            raise TypeError(f"Cannot give empty list of selections to {self.__class__.__name__}")
        super().__init__(selections[0].options.model, *args, **kwargs)
        self.selections = selections

    def initialize_collections(self, manager: db.Manager) -> None:
        super().initialize_collections(manager)
        for selection in self.selections:
            selection.initialize_collections(manager)

    def _call_selections(self, credentials: db.Credentials, step_result: Optional[NetworkExpansionResult] = None) \
            -> List[SelectionResult]:
        """
        Handles the call for each selection and their combination based on the fact whether they are
        safe to use with a previous expansion result or not.

        Parameters
        ----------
        credentials : db.Credentials
            The credentials to the database we are selecting from
        step_result : Optional[NetworkExpansionResult]
            The optional previous network expansion result

        Returns
        -------
        List[SelectionResult]
            The results of each individual selection. Their logical combination is handled by the
            inheriting class.
        """
        manager = connect_to_db(credentials)
        self.initialize_collections(manager)
        results = [sele(credentials, step_result) for sele in self.selections
                   if not isinstance(sele, SafeFirstSelection)]
        results += [sele(credentials) for sele in self.selections
                    if isinstance(sele, SafeFirstSelection)]
        return results

    def _gather_structures_from_filter(self, aggregate_filter: AggregateFilter,
                                       excluded_aggregates: Set[str]) -> List[db.ID]:
        res = []
        # TODO extend to flasks if flasks relevant and all aggregate filters safe for flasks
        for compound in self._compounds.iterate_all_compounds():
            compound.link(self._compounds)
            if str(compound.id()) in excluded_aggregates or not aggregate_filter.filter(compound):
                continue
            res += compound.get_structures()
        return res

    def __len__(self) -> int:
        return len(self.selections)

    def __iter__(self):
        return (s for s in self.selections)

    def __setitem__(self, key, value):
        self.selections[key] = value


class SelectionAndArray(_MultipleSelections):
    """
    Combines multiple selections with a logical AND.
    """

    options: SelectionAndArray.Options

    def __init__(self, selections: List[Selection], *args, **kwargs) -> None:
        super().__init__(selections, *args, **kwargs)
        self.name = "AndSelection[" + ("-".join(s.name for s in selections)) + "]"

    @status_wrap
    def __call__(self, credentials: db.Credentials, step_result: Optional[NetworkExpansionResult] = None,
                 notify_partial_steps_callback: Optional[
                     Callable[[Union[NoRestartInfoPresent, RestartPartialExpansionInfo]], None]] = None,
                 restart_information: Optional[RestartPartialExpansionInfo] = None) \
            -> SelectionResult:
        self._not_implemented_arguments_sanity_check(notify_partial_steps_callback, restart_information)
        results = self._call_selections(credentials, step_result)
        structures = self._combine_structures(results)
        return SelectionResult(
            AggregateFilterAndArray([r.aggregate_filter for r in results]),
            ReactiveSiteFilterAndArray([r.reactive_site_filter for r in results]),
            FurtherExplorationFilterAndArray([r.further_exploration_filter for r in results]),
            [db.ID(s) for s in structures]
        )

    def _select(self) -> SelectionResult:
        raise NotImplementedError

    @staticmethod
    def _combine_structures(results: List[SelectionResult]) -> Set[str]:
        structures = [set(str(ss) for ss in r.structures) for r in results]
        intersect_structures = set.intersection(*structures)
        return intersect_structures


class SelectionOrArray(_MultipleSelections):
    """
    Combines multiple selections with a logical OR.
    """

    options: SelectionOrArray.Options

    def __init__(self, selections: List[Selection], *args, **kwargs) -> None:
        super().__init__(selections, *args, **kwargs)
        self.name = "OrSelection[" + ("-".join(s.name for s in selections)) + "]"

    @status_wrap
    def __call__(self, credentials: db.Credentials, step_result: Optional[NetworkExpansionResult] = None,
                 notify_partial_steps_callback: Optional[
                     Callable[[Union[
                         NoRestartInfoPresent, RestartPartialExpansionInfo]], None]] = None,
                 restart_information: Optional[RestartPartialExpansionInfo] = None) \
            -> SelectionResult:
        self._not_implemented_arguments_sanity_check(notify_partial_steps_callback, restart_information)
        results = self._call_selections(credentials, step_result)
        structures = self._combine_structures(results)
        return SelectionResult(
            AggregateFilterOrArray([r.aggregate_filter for r in results]),
            ReactiveSiteFilterOrArray([r.reactive_site_filter for r in results]),
            FurtherExplorationFilterOrArray([r.further_exploration_filter for r in results]),
            [db.ID(s) for s in structures]
        )

    def _select(self) -> SelectionResult:
        raise NotImplementedError

    def _combine_structures(self, results: List[SelectionResult]) -> Set[str]:
        # if one result does not specify structures
        # gather structures based on the aggregate filter
        no_structure_results = [r for r in results if not r.structures]
        if no_structure_results:
            covered_aggregates = set()
            for r in results:
                if not r.structures:
                    continue
                for s in r.structures:
                    structure = db.Structure(s, self._structures)
                    if structure.has_aggregate():
                        covered_aggregates.add(str(structure.get_aggregate()))
            agg_filter = AggregateFilterOrArray([r.aggregate_filter for r in no_structure_results])
            structures = self._gather_structures_from_filter(agg_filter, covered_aggregates)
        else:
            structures = []
        for r in results:
            structures += r.structures
        return set(str(ss) for ss in structures)
