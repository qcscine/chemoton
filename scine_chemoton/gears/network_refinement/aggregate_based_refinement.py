#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Optional, Dict, Union, List
import os
import pickle

# Third party imports
import scine_database as db
import scine_utilities as utils
from scine_database.queries import get_calculation_id_from_structure, stop_on_timeout

from scine_chemoton.gears import Gear, HoldsCollections
from scine_chemoton.filters.aggregate_filters import AggregateFilter, PlaceHolderAggregateFilter
from scine_chemoton.utilities.db_object_wrappers.aggregate_cache import AggregateCache
from scine_chemoton.utilities.db_object_wrappers.wrapper_caches import MultiModelCacheFactory
from scine_chemoton.utilities.db_object_wrappers.thermodynamic_properties import (
    ReferenceState, PlaceHolderReferenceState
)
from scine_chemoton.utilities.calculation_creation_helpers import finalize_calculation
from scine_chemoton.utilities.model_combinations import ModelCombination
from scine_chemoton.utilities.place_holder_model import (
    construct_place_holder_model,
    PlaceHolderModelType
)
from scine_chemoton.gears.network_refinement.enabling import (
    AggregateEnabling, EnableCalculationResults, PlaceHolderAggregateEnabling, PlaceHolderCalculationEnabling
)


class AggregateBasedRefinement(Gear):
    """
    Run refinement calculations for structures based on aggregates.
    For instance, the 10 lowest energy structures in an aggregate may be reoptimized
    or reevaluated with a different electronic structure method.
    The selection which aggregates to refine can be steered through the aggregate_filter system
    used for the elementary step trial selection.

    Attributes
    ----------
    options : AggregateBasedRefinement.Options
    """
    class Options(Gear.Options):
        __slots__ = {
            "refinement",
            "post_refine_model",
            "hessian_model",
            "n_lowest",
            "energy_window",
            "sp_job",
            "sp_job_settings",
            "opt_job",
            "opt_job_settings",
            "only_electronic_energies",
            "reference_state",
            "cache_file_name",
        }

        def __init__(self) -> None:
            super().__init__()
            self.refinement: Dict[str, bool] = {
                "refine_single_points": False,
                "refine_optimizations": False
            }
            self.post_refine_model: db.Model = db.Model("DFT", "", "")
            """
            db.Model (Scine::Database::Model)
                The Model used for the refinement
                The default is: DFT
            """
            self.hessian_model: db.Model = construct_place_holder_model()
            """
            db.Model
                A second electronic structure model along side the pre_refine_model. With this model, the free
                energy corrections are evaluated. If only a place-holder is given, the pre_refine_model is used.
            """
            self.n_lowest: int = 20
            """
            int
                Set up refinement calculations for the n structures with the lowest energy in the given aggregate.
            """
            self.energy_window: float = 20.0  # kJ/mol
            """
            float
                Set up refinement calculations for structures with a free energy difference lower than the given value
                to the most stable structure in the aggregate. Energy value in kJ/mol.
            """
            self.sp_job: db.Job = db.Job("scine_single_point")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for calculating new single point energies.
                The default is: the 'scine_single_point' order on a single core.
            """
            self.sp_job_settings: utils.ValueCollection = utils.ValueCollection()
            """
            utils.ValueCollection
                Additional settings for calculating new single point energies.
                Empty by default.
            """
            self.opt_job: db.Job = db.Job("scine_geometry_optimization")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for optimizing all minima.
                The default is: the 'scine_geometry_optimization' order on a single core.
            """
            self.opt_job_settings: utils.ValueCollection = utils.ValueCollection()
            """
            utils.ValueCollection
                Additional settings for optimizing all minima.
                Empty by default.
            """
            self.only_electronic_energies: bool = False
            """
            bool
                If true, all electronic energies are used instead of free energies to evaluate energy differences in
                the screening.
            """
            self.reference_state: ReferenceState = PlaceHolderReferenceState()
            """
            ReferenceState
                The thermodynamic reference state (temperature, pressure). If only a place-holder is given, the
                temperature and pressure are taken from the pre-refine model.
            """
            self.cache_file_name: str = ".chemoton_aggregate_based_refinement_structure_ids.pickle"
            """
            str
                The pickle file's name for caching.
            """

    options: Options

    def __init__(self) -> None:
        super().__init__()
        self._required_collections = ["calculations", "compounds", "flasks", "properties", "structures"]
        self._aggregate_cache: Optional[AggregateCache] = None
        self._calculation_already_set_up: Dict[str, Dict[int, str]] = {}
        self._known_keys = ["refine_single_points", "refine_optimizations"]

        self.aggregate_filter: AggregateFilter = AggregateFilter()
        """
        AggregateFilter
            Refine only aggregates that pass the given filter.
        """
        self.aggregate_enabling: AggregateEnabling = PlaceHolderAggregateEnabling()
        """
        AggregateEnabling
            If given (none place-holder), the aggregate enabling policy is applied and further refinement is skipped if
            the aggregate_validation succeeds.
        """
        self.aggregate_validation: AggregateFilter = PlaceHolderAggregateFilter()
        """
        AggregateFilter
            If this filter succeeds after applying the aggregate_enabling policy, no further refinement is done
            for the given aggregate.
        """
        self.result_enabling: EnableCalculationResults = PlaceHolderCalculationEnabling()
        """
        EnableCalculationResults
            If this calculation result enabling policy is given (none place-holder), the result of an already existing
            calculation is enabled again (if disabled previously).
        """

    def _propagate_db_manager(self, manager: db.Manager) -> None:
        self.__initialize_helper_attributes_and_filters()

        if isinstance(self.aggregate_enabling, HoldsCollections):
            self.aggregate_enabling.initialize_collections(manager)
        self.aggregate_validation.initialize_collections(manager)
        self.result_enabling.initialize_collections(manager)
        self.aggregate_filter.initialize_collections(manager)

    def __initialize_helper_attributes_and_filters(self) -> None:
        """
        Initialize some attributes that are only known as soon as the _loop_impl is called and all options are final for
        the loop.
        """
        model = self.options.model
        if isinstance(self.options.hessian_model, PlaceHolderModelType):
            self.options.hessian_model = self.options.model
        if self._aggregate_cache is None:
            h_model = self.options.hessian_model
            self._aggregate_cache = MultiModelCacheFactory().get_aggregates_cache(self.options.only_electronic_energies,
                                                                                  ModelCombination(model, h_model),
                                                                                  self._manager)
        if isinstance(self.options.reference_state, PlaceHolderReferenceState):
            self.options.reference_state = ReferenceState(float(model.temperature), float(model.pressure))

        if (not isinstance(self.aggregate_enabling, PlaceHolderAggregateEnabling)
                and isinstance(self.aggregate_enabling, PlaceHolderAggregateEnabling)):
            raise RuntimeError("Error: If an aggregate enabling policy is supplied to the AggregateBasedRefinement,"
                               " the aggregate_validation option must be set, too.")

    def _loop_impl(self):
        if self.options.model == self.options.post_refine_model:
            raise RuntimeError("model and post_refine_model must be different!")
        for key in self.options.refinement.keys():
            if key not in self._known_keys:
                raise RuntimeError("Unknown network_refinement option for AggregateBasedRefinement. Options are "
                                   + str(self._known_keys))
        self._load_structure_id_cache()
        if self.options.refinement["refine_single_points"]:
            self._loop_aggregates(self.options.sp_job, self.options.sp_job_settings, self._compounds,
                                  db.CompoundOrFlask.COMPOUND)
            self._loop_aggregates(self.options.sp_job, self.options.sp_job_settings, self._flasks,
                                  db.CompoundOrFlask.FLASK)
        if self.options.refinement["refine_optimizations"]:
            self._loop_aggregates(self.options.opt_job, self.options.opt_job_settings, self._compounds,
                                  db.CompoundOrFlask.COMPOUND)
            self._loop_aggregates(self.options.opt_job, self.options.opt_job_settings, self._flasks,
                                  db.CompoundOrFlask.FLASK)

    def _loop_aggregates(self, job: db.Job, settings: utils.ValueCollection, collection: db.Collection,
                         agg_type: db.CompoundOrFlask) -> None:
        if agg_type == db.CompoundOrFlask.COMPOUND:
            iterator = collection.iterate_all_compounds()
        elif agg_type == db.CompoundOrFlask.FLASK:
            iterator = collection.iterate_all_flasks()
        else:
            raise RuntimeError(f"Unknown aggregate type {agg_type}")
        if job.order not in self._calculation_already_set_up:
            self._calculation_already_set_up[job.order] = {}
        for aggregate in stop_on_timeout(iterator):
            if self.have_to_stop_at_next_break_point():
                return
            aggregate.link(collection)
            self._set_up_aggregate(job, settings, aggregate)
        self._save_structure_id_cache()

    def _select_structures(self, db_aggregate: Union[db.Compound, db.Flask]) -> List[db.ID]:
        # TODO: We could add other filter criteria here in the future.
        aggregate = self._aggregate_cache.get_or_produce(db_aggregate.id())  # type: ignore
        return aggregate.get_lowest_n_structures(self.options.n_lowest, self.options.energy_window,
                                                 self.options.reference_state)

    def _set_up_aggregate(self, job: db.Job, settings: utils.ValueCollection,
                          db_aggregate: Union[db.Compound, db.Flask]) -> None:
        # We can remove the mypy ignore statement once the aggregate filter fully support flasks.
        if not self.aggregate_filter.filter(db_aggregate):  # type: ignore
            return
        # Enable aggregates/structures if desired and return if the aggregate is now valid.
        if not isinstance(self.aggregate_enabling, PlaceHolderAggregateEnabling):
            self.aggregate_enabling.process(db_aggregate)
            # We can remove the mypy ignore statement once the aggregate filter fully support flasks.
            if self.aggregate_validation.filter(db_aggregate):  # type: ignore
                return
        structure_ids = self._select_structures(db_aggregate)
        for s_id in structure_ids:
            int_id = int(s_id.string(), 16)
            if int_id in self._calculation_already_set_up[job.order]:
                calc_id: Optional[db.ID] = db.ID(self._calculation_already_set_up[job.order][int_id])
            else:
                calc_id = get_calculation_id_from_structure(job.order, [s_id], self.options.post_refine_model,
                                                            self._structures, self._calculations, settings.as_dict())
            if calc_id is None:
                calc = db.Calculation()
                calc.link(self._calculations)
                calc.create(self.options.post_refine_model, job, [s_id])
                calc.set_settings(settings)
                finalize_calculation(calc, self._structures, [s_id])
                self._calculation_already_set_up[job.order][int_id] = calc.id().string()
            else:
                if not isinstance(self.result_enabling, PlaceHolderCalculationEnabling):
                    self.result_enabling.process(db.Calculation(calc_id, self._calculations))

    def _save_structure_id_cache(self) -> None:
        with open(self.options.cache_file_name, 'wb') as f:
            pickle.dump(self._calculation_already_set_up, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_structure_id_cache(self) -> None:
        if os.path.exists(self.options.cache_file_name) and os.path.getsize(self.options.cache_file_name) > 0:
            with open(self.options.cache_file_name, "rb") as file:
                load_cache = pickle.load(file)
                if load_cache:
                    self._calculation_already_set_up.update(load_cache)

            correct_format = True
            if self._calculation_already_set_up:
                if not isinstance(self._calculation_already_set_up, dict):
                    correct_format = False
                else:
                    zero_key = list(self._calculation_already_set_up.keys())[0]
                    if (not isinstance(self._calculation_already_set_up[zero_key], dict)
                            or not isinstance(zero_key, str)):
                        correct_format = False
            if not correct_format:
                raise RuntimeError("The object type in the aggregate network_refinement caching file must be"
                                   " Dict[str, Dict[int, str]]")
