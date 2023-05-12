#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import Set, Optional, Dict, List, Union

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application imports
from ..gears import Gear
from ..gears.elementary_steps.aggregate_filters import AggregateFilter
from ..utilities.queries import stop_on_timeout, calculation_exists_in_structure
from ..utilities.calculation_creation_helpers import finalize_calculation


class BasicThermoDataCompletion(Gear):
    """
    This Gear will autocomplete the thermochemistry data for optimized minimum
    energy structures and optimized transition states.

    Attributes
    ----------
    options :: BasicThermoDataCompletion.Options
        The options for the BasicThermoDataCompletion Gear.

    aggregate_filter :: AggregateFilter
        A possible aggregate filter to select certain aggregates.

    Notes
    -----
    The logic checks for 'user_optimized', 'minimum_optimized' and 'ts_optimized' structures.
    For the optimized minima only those assigned to a compound will be queried.
    If no 'gibbs_energy_correction' with the given model is present, then a
    calculation generating that data is set up (on hold).
    """

    class Options(Gear.Options):
        """
        The options for the BasicThermoDataCompletion Gear.
        """

        __slots__ = ("cycle_time", "job", "settings", "structure_model", "ignore_explore_bool")

        def __init__(self):
            super().__init__()
            self.cycle_time = 101
            """
            int
                The minimum number of seconds between two cycles of the Gear.
                Cycles are finished independent of this option, thus if a cycle
                takes longer than the cycle_time will effectively lead to longer
                cycle times and not cause multiple cycles of the same Gear.
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
            self.structure_model: Optional[db.Model] = None
            """
            Optional[db.Model (Scine::Database::Model)]
                Hessian calculations are only started for structures with the given model.
            """
            self.ignore_explore_bool: bool = False
            """
            bool
                If True, the gear will ignore the explore field of the aggregates and reactions in the database.
            """

    def __init__(self):
        super().__init__()
        self.aggregate_filter = AggregateFilter()
        self._flask_cache: Dict[str, int] = {}
        self._compound_cache: Dict[str, int] = {}
        self._reaction_cache: Dict[str, int] = {}
        self._structure_cache: Set[str] = set()  # structure_id in cache --> no more investigation
        self._required_collections = [
            "calculations", "compounds", "flasks", "reactions", "elementary_steps", "properties", "structures"
        ]

    def _propagate_db_manager(self, manager: db.Manager):
        self._sanity_check_configuration()
        self.aggregate_filter.initialize_collections(manager)

    def _sanity_check_configuration(self):
        if not isinstance(self.aggregate_filter, AggregateFilter):
            raise TypeError(f"Expected a AggregateFilter (or a class derived "
                            f"from it) in {self.name}.aggregate_filter.")

    def clear_cache(self):
        self._structure_cache = set()
        self._flask_cache = {}
        self._compound_cache = {}
        self._reaction_cache = {}

    @staticmethod
    def _hash_id_lists(id_list: List[db.ID]) -> int:
        return hash(','.join([i.string() for i in id_list]))

    def _check_aggregate(
        self,
        aggregate: Union[db.Flask, db.Compound],
        collection: db.Collection,
        cache: Dict[str, int]
    ) -> None:
        aggregate.link(collection)
        if not self.options.ignore_explore_bool and not aggregate.explore():
            return
        current_id = aggregate.get_id().string()
        current_id_list = aggregate.get_structures()
        current_hash = self._hash_id_lists(current_id_list)
        if cache.get(current_id) == current_hash:
            return
        for structure_id in current_id_list:
            self._check_structure(structure_id)
        cache[current_id] = current_hash

    def _loop_impl(self):
        # Check only aggregated TSs and minima.
        # Cache if an aggregate has not changed and skip it all together
        for flask in stop_on_timeout(self._flasks.iterate_flasks('{}')):
            if self.stop_at_next_break_point:
                return
            self._check_aggregate(flask, self._flasks, self._flask_cache)
        for compound in stop_on_timeout(self._compounds.iterate_compounds('{}')):
            if self.stop_at_next_break_point:
                return
            self._check_aggregate(compound, self._compounds, self._compound_cache)
        for reaction in stop_on_timeout(self._reactions.iterate_reactions('{}')):
            if self.stop_at_next_break_point:
                return
            reaction.link(self._reactions)
            if not self.options.ignore_explore_bool and not reaction.explore():
                continue
            current_id = reaction.get_id().string()
            current_id_list = reaction.get_elementary_steps()
            current_hash = self._hash_id_lists(current_id_list)
            if self._reaction_cache.get(current_id) == current_hash:
                continue
            for step_id in current_id_list:
                step = db.ElementaryStep(step_id)
                step.link(self._elementary_steps)
                if not step.has_transition_state():
                    continue
                structure_id = step.get_transition_state()
                self._check_structure(structure_id)
            self._reaction_cache[current_id] = current_hash

    def _check_structure(self, structure_id: db.ID) -> None:
        structure = db.Structure(structure_id)
        sid = structure_id.string()
        if self.stop_at_next_break_point:
            return
        if sid in self._structure_cache:
            return
        structure.link(self._structures)
        if self.options.structure_model is not None:
            if structure.get_model() != self.options.structure_model:
                return
        # AggregateFilter does not filter, so skip check if filter has not been changed
        have_to_filter: bool = self.aggregate_filter is not AggregateFilter
        if have_to_filter and self._cancelled_by_filter(structure):
            self._structure_cache.add(sid)
            return
        if structure.has_property("gibbs_energy_correction"):
            if len(structure.query_properties("gibbs_energy_correction", self.options.model, self._properties)) > 0:
                self._structure_cache.add(sid)
                return
        # Check if a calculation for this is already scheduled
        settings = self.options.settings if self.options.settings else None
        if calculation_exists_in_structure(self.options.job.order, [structure_id], self.options.model,
                                           self._structures, self._calculations, settings):
            self._structure_cache.add(sid)
            return
        hessian = db.Calculation()
        hessian.link(self._calculations)
        hessian.create(self.options.model, self.options.job, [structure_id])
        if self.options.settings:
            hessian.set_settings(self.options.settings)
        self._structure_cache.add(sid)
        finalize_calculation(hessian, self._structures)

    def _cancelled_by_filter(self, structure: db.Structure) -> bool:
        label = structure.get_label()
        if label == db.Label.TS_OPTIMIZED:
            # TS has no aggregate, so AggregateFilters cannot filter
            return False
        agg_id = structure.get_aggregate()
        if label == db.Label.COMPLEX_OPTIMIZED:
            aggregate = db.Flask(agg_id, self._flasks)
        else:
            aggregate = db.Compound(agg_id, self._compounds)  # type: ignore
        return not self.aggregate_filter.filter(aggregate)
