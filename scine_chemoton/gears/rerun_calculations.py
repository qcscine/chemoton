#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from json import dumps
from copy import deepcopy
from typing import List, Dict, Set, Tuple, Optional, Any

# Third party imports
import scine_database as db
from scine_database.queries import (
    model_query, get_calculation_id_from_structure, stop_on_timeout, query_calculation_in_id_set
)
import scine_utilities as utils

# Local application imports
from scine_chemoton.gears import Gear
from scine_chemoton.utilities.calculation_creation_helpers import finalize_calculation
from scine_chemoton.gears.network_refinement.enabling import EnableCalculationResults, PlaceHolderCalculationEnabling
from scine_chemoton.utilities.place_holder_model import (
    ModelNotSetError,
    construct_place_holder_model,
    PlaceHolderModelType
)


class RerunCalculations(Gear):
    """
    This gear re-starts (failed) calculations with different settings, a new model, or a different job based
    on calculations which were already run. The set of calculations to "re-run" can be characterized through
    setting, the mode, the job-order, the resulting calculation status, and the comment.
    """

    restart_info_key = "restart_ids"

    class Options(Gear.Options):
        """
        The options for the RerunCalculations Gear.
        """

        __slots__ = ("_parent", "old_job_settings", "new_job_settings", "old_status", "old_job",
                     "new_job", "change_model", "new_model", "comment_filter", "calculation_id_list",
                     "old_settings_to_remove", "legacy_existence_check")

        def __init__(self, parent: Optional[Any] = None) -> None:
            self._parent = parent  # best be first member to be set because of __setattr__
            super().__init__()
            self.old_job_settings: utils.ValueCollection = utils.ValueCollection()
            """
            utils.ValueCollection
                The settings of the original calculation. This dictionary does not
                have to be complete. It is only used to reduce the number of calculations
                re-run by the gear.
            """
            self.new_job_settings: utils.ValueCollection = utils.ValueCollection()
            """
            utils.ValueCollection
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
            self.new_model: db.Model = construct_place_holder_model()
            """
            db.Model
                The new calculation model. The keyword change_model must be set to True in order to set the new model.
            """
            self.comment_filter: List[str] = []
            """
            List[str]
                A list of comments that is used to further identify calculations that should be re-run.
                Example comments:
                * No more negative eigenvalues
                * TS has incorrect number of imaginary frequencies.
                * Self consistent charge iterator did not converge
            """
            self.calculation_id_list: List[db.ID] = []
            """
            List[db.ID]
                A list of calculation ids to consider for rerunning. If empty, all calculations are looped.
            """
            self.old_settings_to_remove: List[str] = []
            """
            List[str]
                A list of settings to remove from the original calculation settings.
            """
            self.legacy_existence_check = False
            """
            bool
                If True, the gear will check if the calculation already exists in the database without looking up
                the restart_information field. This is the old behavior of the gear and should only be used if
                the restart_information field is not available or re-runs have been carried out with the old gear.
            """

        def __setattr__(self, item, value):
            """
            Overwritten standard method to mark the cache as out of date if any option is changed.
            """
            super().__setattr__(item, value)
            if self._parent is not None:
                self._parent._recreate_cache = True
                self._parent._calculation_cache = list()

    options: Options

    def __init__(self) -> None:
        super().__init__()
        self.options = self.Options(parent=self)
        self._required_collections = ["calculations", "structures"]
        self._calculation_cache: List[db.ID] = list()
        self._already_set_up_calculations: Dict[Tuple, Set[str]] = dict()
        self.result_enabling: EnableCalculationResults = PlaceHolderCalculationEnabling()
        """
        Optional[EnableCalculationResults]
            If this calculation result enabling policy is given, the result of an already existing calculation
            is enabled again (if disabled previously).
        """
        self.__have_printed_note: bool = False

    def _loop_impl(self):
        if self.options.change_model and isinstance(self.options.new_model, PlaceHolderModelType):
            raise ModelNotSetError("Specified to change the model, but have not specified the new model")
        if not self.options.legacy_existence_check and not self.__have_printed_note:
            print("Note: The calculation Re-Run gear will check the existence of old calculations by their restart\n"
                  "information. Note that this could lead to duplicated calculations if the already existing"
                  " calculation was\n"
                  "created without registering in the restart information, e.g., through another gear or an older\n"
                  "chemoton version. Use the option 'legacy_existence_check' to ensure that this cannot happen.")
            self.__have_printed_note = True
        if self._identical_calculation_characterization():
            return
        if not isinstance(self.result_enabling, PlaceHolderCalculationEnabling):
            self.result_enabling.initialize_collections(self._manager)
        cache_update = dict()
        encountered_identical_settings = False
        if self.options.calculation_id_list:
            for calculation_id in self.options.calculation_id_list:
                if self.have_to_stop_at_next_break_point():
                    break
                old_calculation = db.Calculation(calculation_id, self._calculations)
                encountered_identical_settings = self._rerun_calculation(old_calculation, cache_update)
        else:
            for old_calculation in stop_on_timeout(self._calculations.iterate_calculations(
                    dumps(self._get_calculation_selection()))):
                if self.have_to_stop_at_next_break_point():
                    break
                old_calculation.link(self._calculations)
                encountered_identical_settings = self._rerun_calculation(old_calculation, cache_update)
        if encountered_identical_settings:
            print("Encountered identical settings when re-running calculations.\n"
                  "This is a sign that 1) the new and old settings and model are identical, which does not make sense,"
                  "\nor 2) the calculation with the new settings encountered the same problem (e.g. convergence issue) "
                  "as the old calculation.\n"
                  "Hence, the new settings could not solve the problem.\n"
                  "In either case this gear has not set up any new calculations."
                  )

        # Update the cache only after completing the cycle.
        self._update_already_set_up_cache(cache_update)

    def _rerun_calculation(self, old_calculation: db.Calculation, cache_update: Dict[Tuple[str, ...], Set[str]]) \
            -> bool:
        self._calculation_cache.append(old_calculation.id())
        # check if the comment given for the calculation corresponds to the problem that should be fixed.
        if not self._check_comment(old_calculation):
            return False
        # get old structures and settings
        old_structures = old_calculation.get_structures()
        new_settings = self._build_new_settings(old_calculation.get_settings().as_dict())
        if not self.options.change_model and new_settings == old_calculation.get_settings().as_dict():
            # identical settings
            return True
        auxiliaries = old_calculation.get_auxiliaries()
        # create new calculation
        model = self.options.new_model if self.options.change_model else self.options.model
        if self.options.legacy_existence_check:
            calculation_id = get_calculation_id_from_structure(self.options.new_job.order, old_structures, model,
                                                               self._structures, self._calculations, new_settings,
                                                               auxiliaries)
        else:
            id_selection = set([str(i) for i in self.get_restart_ids(old_calculation)])
            calculation_id = query_calculation_in_id_set(id_selection, len(old_structures), self._calculations,
                                                         old_structures, new_settings, auxiliaries,
                                                         self.options.new_job.order)
        if calculation_id is None:
            new_calculation = db.Calculation()
            new_calculation.link(self._calculations)
            new_calculation.create(model, self.options.new_job, old_structures)
            new_calculation.set_settings(utils.ValueCollection(new_settings))
            new_calculation.set_auxiliaries(auxiliaries)
            if not self.options.legacy_existence_check:
                counter = len(self.get_restart_ids(old_calculation))
                old_calculation.set_restart_information(f"{RerunCalculations.restart_info_key}_{counter}",
                                                        new_calculation.id())
            self._add_to_already_set_up_calculations(new_calculation, cache_update)
            finalize_calculation(new_calculation, self._structures)
        else:
            calculation = db.Calculation(calculation_id, self._calculations)  # type: ignore
            self._add_to_already_set_up_calculations(calculation, cache_update)
            if not isinstance(self.result_enabling, PlaceHolderCalculationEnabling):
                self.result_enabling.process(calculation)
        return False

    def _check_comment(self, old_calculation: db.Calculation) -> bool:
        if not self.options.comment_filter:
            # no comments specified, we are not filtering based on comments
            return True
        comment = old_calculation.get_comment()
        for message in self.options.comment_filter:
            if message in comment:
                return True
        return False

    def _build_new_settings(self, old_settings: Dict[str, Any]) -> Dict[str, Any]:
        new_settings = deepcopy(old_settings)
        for setting in self.options.old_settings_to_remove:
            new_settings.pop(setting, None)
        new_settings.update(self.options.new_job_settings)
        return new_settings

    def _get_calculation_selection(self):
        calc_id_str = [{"$oid": str_id.string()} for str_id in self._calculation_cache]
        selection = {
            "$and": [
                {"_id": {"$nin": calc_id_str}},
                {"status": str(self.options.old_status)},
                {"job.order": self.options.old_job.order},
                {"analysis_disabled": False},
            ]
            + self._expand_settings_query(self.options.old_job_settings)
            + model_query(self.options.model)
        }
        return selection

    @staticmethod
    def _expand_settings_query(settings: utils.ValueCollection) -> List[Dict[str, Any]]:
        query_list = []
        for key in settings.keys():
            item = {"settings." + str(key): settings[key]}
            query_list.append(item)
        return query_list

    def _identical_calculation_characterization(self) -> bool:
        if self.options.model != self.options.new_model and self.options.change_model:
            return False
        if self.options.old_job != self.options.new_job:
            return False
        for key in self.options.new_job_settings.keys():
            if key not in self.options.old_job_settings.keys():
                return False
            if self.options.old_job_settings[key] != self.options.new_job_settings[key]:  # type: ignore
                return False
        if self.options.old_settings_to_remove and any(setting in self.options.old_job_settings
                                                       for setting in self.options.old_settings_to_remove):
            return False
        print("The calculation rerun gear detected that it would set up identical calculations, e.g.,")
        print("the characterization of the new calculation does not change the characterization of the")
        print("original one! The gear will do nothing.")
        return True

    @staticmethod
    def _get_caching_key(structure_id_list: List[db.ID]) -> Tuple[str, ...]:
        s_id_str = [s_id.string() for s_id in structure_id_list]
        return tuple((*s_id_str, ))

    def _add_to_already_set_up_calculations(self, calculation: db.Calculation,
                                            caching_map: Dict[Tuple[str, ...], Set[str]]):
        key = self._get_caching_key(calculation.get_structures())
        if key in caching_map:
            caching_map[key].add(calculation.id().string())
        else:
            caching_map[key] = {calculation.id().string()}

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

    @staticmethod
    def get_restart_ids(calculation: db.Calculation) -> List[db.ID]:
        restart_info = calculation.get_restart_information()
        counter = 0
        ids: List[db.ID] = []
        while True:
            i = restart_info.get(f"{RerunCalculations.restart_info_key}_{counter}", None)
            if i is None:
                return ids
            ids.append(i)
            counter += 1
