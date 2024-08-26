#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, Dict, Set
from json import dumps
import os
import pickle

# Third party imports
import scine_database as db

from .refinement import NetworkRefinement
from scine_database.queries import (
    model_query,
    stop_on_timeout,
)


class CalculationBasedRefinement(NetworkRefinement):
    """
    This gear allows the refinement of results from previous calculations that produced elementary steps. For instance,
    one could recalculate the single point energies for all structures in the elementary step or rerun the transition
    state optimization with a different electronic structure model.
    """

    class Options(NetworkRefinement.Options):
        """
        See NetworkRefinement for options.
        """

    options: Options

    def __init__(self) -> None:
        super().__init__()
        self._calculation_id_cache: Dict[str, Set[str]] = {
            "refine_single_points": set(),
            "refine_optimizations": set(),
            "double_ended_refinement": set(),
            "double_ended_new_connections": set(),
            "refine_single_ended_search": set(),
            "refine_structures_and_irc": set(),
        }

    def _loop(self, job_label: str):
        """
        Create refinement calculations under the condition that there is a calculation that produced an elementary
        step that fulfills the elementary step filters.

        Parameters
        ----------
        job_label: str
            The label for the refinement to be executed.
        """
        self._load_calculation_id_cache()
        cache = self._calculation_id_cache[job_label]
        selection = {
            "$and": [
                {"status": "complete"},
                {"results.elementary_steps.0": {"$exists": True}},
            ]
            + model_query(self.options.calculation_model)  # type: ignore
        }
        cache_update: List[str] = list()
        for calculation in stop_on_timeout(self._calculations.iterate_calculations(dumps(selection))):
            if self.have_to_stop_at_next_break_point():
                cache.update(set(cache_update))
                self._save_calculation_id_cache()
                return
            str_id = calculation.id().string()
            if str_id in cache:
                continue
            calculation.link(self._calculations)
            elementary_steps = [db.ElementaryStep(step_id, self._elementary_steps) for step_id in
                                calculation.get_results().get_elementary_steps()]
            any_qualified = any(self.elementary_step_filter.filter(step) for step in elementary_steps)
            if not any_qualified:
                continue
            calculation_fully_done = self._set_up_calculation(job_label, calculation)
            if calculation_fully_done:
                cache_update.append(str_id)
        cache.update(set(cache_update))
        self._save_calculation_id_cache()

    def _save_calculation_id_cache(self) -> None:
        # save dictionary to pickle file
        with open(self.options.caching_file_name, 'wb') as f:
            pickle.dump(self._calculation_id_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_calculation_id_cache(self) -> None:
        if os.path.exists(self.options.caching_file_name) and os.path.getsize(self.options.caching_file_name) > 0:
            with open(self.options.caching_file_name, "rb") as f:
                load_cache = pickle.load(f)
                if load_cache:
                    self._calculation_id_cache.update(load_cache)
