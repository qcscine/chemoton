#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from json import dumps
from copy import deepcopy
from typing import List, Dict, Set, Tuple, Optional, Any

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application imports
from ..gears import Gear
from ..utilities.queries import model_query, calculation_exists_in_id_set, stop_on_timeout
from ..utilities.calculation_creation_helpers import finalize_calculation


class RerunCalculations(Gear):
    """
    This gear re-starts (failed) calculations with different settings, a new model, or a different job based
    on calculations which were already run. The set of calculations to "re-run" can be characterized through
    setting, the mode, the job-order, the resulting calculation status, and the comment.
    """

    __slots__ = ("options")

    class Options:
        """
        The options for the RerunCalculations Gear.
        """

        def __init__(self, parent: Optional[Any] = None):
            self._parent = parent  # best be first member to be set because of __setattr__
            self.cycle_time = 101
            """
            int
                The minimum number of seconds between two cycles of the Gear.
                Cycles are finished independent of this option, thus if a cycle
                takes longer than the cycle_time will effectively lead to longer
                cycle times and not cause multiple cycles of the same Gear.
            """
            self.old_job_settings: Dict[str, Any] = dict()
            """
            dict
                The settings of the original calculation. This dictionary does not
                have to be complete. It is only used to reduce the number of calculations
                re-run by the gear.
            """
            self.new_job_settings: Dict[str, Any] = dict()
            """
            dict
                New settings for the calculation. The settings from this dictionary are used to update
                the original calculations settings.
            """
            self.old_status: str = "failed"
            """
            str
                The calculation status of the calculations to be re-run.
            """
            self.old_job: db.Job = db.Job("scine_react_complex_nt2")
            """
            db.Job
                The original job of the calculations.
            """
            self.new_job: db.Job = db.Job("scine_react_complex_nt2")
            """
            db.Job
                The new job to re-run the calculations with.
            """
            self.change_model: bool = False
            """
            bool
                If true, the model for the newly set up calculation is updated.
            """
            self.old_model: db.Model = db.Model("pm6", "pm6", "")
            """
            db.Model
                The original model of the calculations.
            """
            self.new_model: db.Model = db.Model("dftb3", "dftb3", "")
            """
            db.Model
                The new calculation model. The keyword change_model must be set to True in order to set the new model.
            """
            self.comment_filter: List[str] = []
            """
            List[str]
                A list of comments that is used to further identify calculations tht should be re-run.
                Example comments:
                * No more negative eigenvalues
                * TS has incorrect number of imaginary frequencies.
                * Self consistent charge iterator did not converge
            """
            self.calculation_id_list: Optional[List[db.ID]] = None
            """
            Optional[List[db.ID]]
                A list of calculation ids to consider for rerunning. If None, all calculations are looped.
            """

        def __setattr__(self, item, value):
            """
            Overwritten standard method to mark the cache as out of date if any option is changed.
            """
            super().__setattr__(item, value)
            if self._parent is not None:
                self._parent._recreate_cache = True
                self._parent._calculation_cache = list()

    def __init__(self):
        super().__init__()
        self.options = self.Options(parent=self)
        self._required_collections = ["calculations", "structures"]
        self._calculation_cache = list()
        self._already_set_up_calculations: Dict[Tuple, Set[str]] = dict()
        self._recreate_cache = True

    def _loop_impl(self):
        if self._identical_calculation_characterization():
            return
        if self._recreate_cache:
            self._create_cache_of_set_up_calculations()
        cache_update = dict()
        if self.options.calculation_id_list:
            print("Test")
            for calculation_id in self.options.calculation_id_list:
                old_calculation = db.Calculation(calculation_id, self._calculations)
                self._rerun_calculation(old_calculation, cache_update)
        else:
            for old_calculation in stop_on_timeout(self._calculations.iterate_calculations(
                    dumps(self._get_calculation_selection()))):
                old_calculation.link(self._calculations)
                self._rerun_calculation(old_calculation, cache_update)
        # Update the cache only after completing the cycle.
        self._update_already_set_up_cache(cache_update)

    def _rerun_calculation(self, old_calculation: db.Calculation, cache_update: Dict):
        self._calculation_cache.append(old_calculation.id())
        # check if the comment given for the calculation corresponds to the problem that should be fixed.
        if not self._check_comment(old_calculation):
            return
        # get old structures and settings
        old_structures = old_calculation.get_structures()
        new_settings = self._build_new_settings(old_calculation.get_settings().as_dict())
        auxiliaries = old_calculation.get_auxiliaries()
        # create new calculation
        model = self.options.new_model if self.options.change_model else self.options.old_model
        id_selection = self._calculation_ids_already_set_up(old_structures)
        if not calculation_exists_in_id_set(id_selection, len(old_structures), self._calculations,
                                            old_structures, new_settings, auxiliaries):
            new_calculation = db.Calculation()
            new_calculation.link(self._calculations)
            new_calculation.create(model, self.options.new_job, old_structures)
            new_calculation.set_settings(utils.ValueCollection(new_settings))
            new_calculation.set_auxiliaries(auxiliaries)
            self._add_to_already_set_up_calculations(new_calculation, cache_update)
            finalize_calculation(new_calculation, self._structures)

    def _check_comment(self, old_calculation: db.Calculation) -> bool:
        comment = old_calculation.get_comment()
        for message in self.options.comment_filter:
            if message in comment:
                return True
        return False

    def _build_new_settings(self, old_settings: dict) -> dict:
        new_settings = deepcopy(old_settings)
        new_settings.update(self.options.new_job_settings)
        return new_settings

    def _get_calculation_selection(self):
        order = self.options.old_job.order
        calc_id_str = [{"$oid": str_id.string()} for str_id in self._calculation_cache]
        selection = {
            "$and": [
                {"_id": {"$nin": calc_id_str}},
                {"status": str(self.options.old_status)},
                {"job.order": order},
            ]
            + self._expand_settings_query()
            + model_query(self.options.old_model)
        }
        return selection

    def _expand_settings_query(self) -> List[dict]:
        query_list = []
        for key in self.options.old_job_settings.keys():
            item = {"settings." + str(key): self.options.old_job_settings[key]}
            query_list.append(item)
        return query_list

    def _identical_calculation_characterization(self) -> bool:
        if self.options.old_model != self.options.new_model and self.options.change_model:
            return False
        if self.options.old_job != self.options.new_job:
            return False
        for key in self.options.new_job_settings:
            if key not in self.options.old_job_settings:
                return False
            if self.options.old_job_settings[key] != self.options.new_job_settings[key]:
                return False
        print("The calculation rerun gear detected that it would set up identical calculations, e.g.,")
        print("the characterization of the new calculation does not change the characterization of the")
        print("original one! The gear will do nothing.")
        return True

    @staticmethod
    def _get_caching_key(structure_id_list: List[db.ID]) -> Tuple:
        s_id_str = [s_id.string() for s_id in structure_id_list]
        return tuple((*s_id_str, ))

    def _add_to_already_set_up_calculations(self, calculation: db.Calculation, caching_map: Dict[Tuple, Set[str]]):
        key = self._get_caching_key(calculation.get_structures())
        if key in caching_map:
            caching_map[key].add(calculation.id().string())
        else:
            caching_map[key] = set([calculation.id().string()])

    def _create_cache_of_set_up_calculations(self):
        model = self.options.new_model if self.options.change_model else self.options.old_model
        self._already_set_up_calculations = dict()
        selection = {
            "$and": [
                {"job.order": self.options.new_job.order},
            ] + model_query(model)
        }
        for calculation in stop_on_timeout(self._calculations.iterate_calculations(dumps(selection))):
            calculation.link(self._calculations)
            self._add_to_already_set_up_calculations(calculation, self._already_set_up_calculations)
        self._recreate_cache = False

    def _calculation_ids_already_set_up(self, structure_ids: List[db.ID]) -> Set[str]:
        calculation_str_ids: Set[str] = set()
        key = self._get_caching_key(structure_ids)
        if key in self._already_set_up_calculations:
            calculation_str_ids = self._already_set_up_calculations[key]
        return calculation_str_ids

    def _update_already_set_up_cache(self, update: Dict[Tuple, Set[str]]):
        for key in update.keys():
            if key in self._already_set_up_calculations:
                self._already_set_up_calculations[key] = self._already_set_up_calculations[key].union(update[key])
            else:
                self._already_set_up_calculations[key] = update[key]
