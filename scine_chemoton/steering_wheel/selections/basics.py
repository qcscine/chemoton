#!/usr/bin/env python3
from __future__ import annotations
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from json import dumps
from typing import List, Optional, Union

import scine_database as db

from scine_chemoton.filters.aggregate_filters import (
    AggregateFilter,
    SelectedAggregateIdFilter
)
from scine_chemoton.filters.reactive_site_filters import (
    ReactiveSiteFilter,
)
from ..datastructures import SelectionResult, LogicCoupling
from . import Selection, SafeFirstSelection


class AllFromPreviousResultSelection(Selection):
    """
    Selects all aggregates and structures from the previous step result.
    """

    options: AllFromPreviousResultSelection.Options

    def _select(self) -> SelectionResult:
        step_result = self.get_step_result()
        if step_result is None:
            raise RuntimeError(f"{self.name} did not receive a proper step result.")
        str_result = list(set([str(i) for i in step_result.compounds + step_result.flasks]))
        return SelectionResult(
            aggregate_filter=SelectedAggregateIdFilter(str_result),
            structures=step_result.structures
        )


class ProductsSelection(Selection):
    """
    Selects all aggregates and structures that are products of the reactions in the previous step result.
    """

    options: ProductsSelection.Options

    def _select(self) -> SelectionResult:
        step_result = self.get_step_result()
        if step_result is None:
            raise RuntimeError(f"{self.name} did not receive a proper step result.")
        aggregates = []
        for rid in step_result.reactions:
            reaction = db.Reaction(rid, self._reactions)
            aggregates += [str(i) for i in reaction.get_reactants(db.Side.RHS)[1]]

        return SelectionResult(
            aggregate_filter=SelectedAggregateIdFilter(list(set(aggregates))),
            structures=step_result.structures
        )


class LowestBarrierSelection(Selection):
    """
    Selects the n lowest barrier aggregates and structures from the previous step result.
    """

    options: LowestBarrierSelection.Options

    def __init__(self, model: db.Model, n_lowest: int,  # pylint: disable=keyword-arg-before-vararg
                 include_thermochemistry: bool = False,
                 additional_aggregate_filters: Optional[List[AggregateFilter]] = None,
                 additional_reactive_site_filters: Optional[List[ReactiveSiteFilter]] = None,
                 logic_coupling: Union[str, LogicCoupling] = LogicCoupling.AND,
                 *args, **kwargs) -> None:
        super().__init__(model, additional_aggregate_filters, additional_reactive_site_filters, logic_coupling,
                         *args, **kwargs)
        self._n_lowest = n_lowest
        self._energy_type = "gibbs_free_energy" if include_thermochemistry else "electronic_energy"

    def _select(self) -> SelectionResult:
        step_result = self.get_step_result()
        if step_result is None:
            raise RuntimeError(f"{self.name} did not receive a proper step result.")
        aggregates = []
        barriers = self.lowest_barrier_per_reaction(step_result, self._energy_type)
        if barriers:
            assert len(barriers) == len(step_result.reactions)
            for barrier in sorted([b for b in barriers if b is not None])[:self._n_lowest]:
                i = barriers.index(barrier)
                reaction = db.Reaction(step_result.reactions[i], self._reactions)
                aggregates += [str(rid) for rid in reaction.get_reactants(db.Side.RHS)[1]]

        return SelectionResult(
            aggregate_filter=SelectedAggregateIdFilter(list(set(aggregates))),
            structures=step_result.structures
        )


class BarriersWithinRangeSelection(Selection):
    """
    Selects all aggregates and structures that are products of the reactions in the previous step result and
    have a barrier lower than the given maximum barrier.
    """

    options: BarriersWithinRangeSelection.Options

    def __init__(self, model: db.Model, max_barrier: float,  # pylint: disable=keyword-arg-before-vararg
                 include_thermochemistry: bool = False,
                 additional_aggregate_filters: Optional[List[AggregateFilter]] = None,
                 additional_reactive_site_filters: Optional[List[ReactiveSiteFilter]] = None,
                 logic_coupling: Union[str, LogicCoupling] = LogicCoupling.AND,
                 *args, **kwargs) -> None:
        super().__init__(model, additional_aggregate_filters, additional_reactive_site_filters, logic_coupling,
                         *args, **kwargs)
        self._max_barrier = max_barrier  # in kJ/mol
        self._energy_type = "gibbs_free_energy" if include_thermochemistry else "electronic_energy"

    def _select(self) -> SelectionResult:
        step_result = self.get_step_result()
        if step_result is None:
            raise RuntimeError(f"{self.name} did not receive a proper step result.")
        aggregates = []
        barriers = self.lowest_barrier_per_reaction(step_result, self._energy_type)
        if barriers:
            assert len(barriers) == len(step_result.reactions)
            for i, barrier in enumerate(barriers):
                if barrier is not None and barrier <= self._max_barrier:
                    reaction = db.Reaction(step_result.reactions[i], self._reactions)
                    aggregates += [str(i) for i in reaction.get_reactants(db.Side.RHS)[1]]

        return SelectionResult(
            aggregate_filter=SelectedAggregateIdFilter(list(set(aggregates))),
            structures=step_result.structures
        )


class AllUserInputsSelection(SafeFirstSelection):
    """
    Selects all aggregates and structures that are user inputs.
    """

    options: AllUserInputsSelection.Options

    def _select(self) -> SelectionResult:
        query = {
            "label": {
                "$in": [
                    "user_optimized",
                    "user_surface_optimized",
                    "user_complex_optimized",
                    "user_surface_complex_optimized"]}}
        aggregate_hits = []
        structures_hits = []
        for structure in self._structures.iterate_structures(dumps(query)):
            structure.link(self._structures)
            structures_hits.append(structure.id())
            if structure.has_aggregate():
                aggregate_hits.append(str(structure.get_aggregate()))

        return SelectionResult(
            aggregate_filter=SelectedAggregateIdFilter(aggregate_hits),
            structures=structures_hits
        )
