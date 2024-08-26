#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import abstractmethod, ABC
from collections import defaultdict
from json import dumps
from typing import Callable, Dict, Iterator, List, Set, Optional, Tuple, Union
from warnings import warn

from numpy import ndarray
import scine_database as db
from scine_database.queries import stop_on_timeout
from scine_utilities import ValueCollection

# Local application imports
from scine_chemoton.filters.aggregate_filters import AggregateFilter
from scine_chemoton.filters.reactive_site_filters import ReactiveSiteFilter
from scine_chemoton.utilities.place_holder_model import (
    construct_place_holder_model,
    PlaceHolderModelType
)
from .trial_generator import TrialGenerator
from .trial_generator.bond_based import BondBased
from .. import Gear, _initialize_a_gear_to_a_db
from scine_chemoton.utilities.warnings import ModelChangedWarning, SettingsChangedWarning


class ElementaryStepGear(Gear, ABC):
    """
    Base class for elementary step reaction generators
    """

    class Options(Gear.Options):
        """
        The options for an ElementarySteps Gear.
        """

        __slots__ = (
            "_parent",
            "enable_unimolecular_trials",
            "enable_bimolecular_trials",
            "run_one_cycle_with_settings_enhancement",
            "base_job_settings",
            "structure_model",
            "looped_collection"
        )

        def __init__(self, _parent: Optional[Gear] = None) -> None:
            self._parent = _parent
            super().__init__()
            self.enable_unimolecular_trials = True
            """
            bool
                If `True`, enables the exploration of unimolecular reactions.
            """
            self.enable_bimolecular_trials = True
            """
            bool
                If `True`, enables the exploration of bimolecular reactions.
            """
            self.run_one_cycle_with_settings_enhancement = False
            """
            bool
                If `True`, enables the enhancement of the settings for the next cycle.
            """
            self.base_job_settings: ValueCollection = ValueCollection({})
            """
            ValueCollection
                The base settings for the jobs. Duplicate keys are overwritten by the settings of the TrialGenerator.
            """
            self.structure_model: db.Model = construct_place_holder_model()
            """
            Optional[db.Model]
                If not None, calculations are only started for structures with the given model.
            """
            self.looped_collection: str = "compounds"
            """
            str
                The collection to loop over. Can be "compounds" or "flasks".
            """

        def __setattr__(self, item, value) -> None:
            """
            Overwritten standard method to synchronize model option
            """
            model_case = bool(
                item == "model" and
                hasattr(self, "model") and
                self.model != value and
                hasattr(self, "_parent") and
                self._parent is not None and
                hasattr(self._parent, "trial_generator")
            )
            if item == "looped_collection" and value not in ["compounds", "flasks"]:
                raise ValueError(f"Invalid value for {item}: '{value}'. Only 'compounds' and 'flasks' are allowed.")
            if item == "base_job_settings":
                if not isinstance(value, ValueCollection):
                    raise TypeError(f"The {item} must be a ValueCollection.")
                if hasattr(self, "_parent") and self._parent is not None and hasattr(self._parent, "trial_generator"):
                    if self._parent.trial_generator.options.base_job_settings:
                        warn("The base job settings of the trial generator are overwritten by the gear.",
                             category=SettingsChangedWarning)
                    self._parent.trial_generator.options.base_job_settings = value
            super().__setattr__(item, value)
            if model_case:
                if not isinstance(self._parent.trial_generator.options.model, PlaceHolderModelType):  # type: ignore
                    warn("The model of the trial generator is overwritten by the gear.",
                         category=ModelChangedWarning)
                self._parent.trial_generator.options.model = value  # type: ignore
                self._parent.clear_cache()  # type: ignore

    options: Options

    def __init__(self) -> None:
        super().__init__()
        self._required_collections = ["calculations", "compounds", "flasks", "properties", "reactions", "structures"]
        self.options = self.Options(_parent=self)
        self.trial_generator: TrialGenerator = BondBased()
        self.trial_generator.options.base_job_settings = self.options.base_job_settings
        self.aggregate_filter: AggregateFilter = AggregateFilter()
        self._cache: Set[str] = set()
        self._rebuild_cache = True

    def __setattr__(self, item, value) -> None:
        """
        Overwritten standard method to synchronize model option
        """
        super().__setattr__(item, value)
        if isinstance(value, TrialGenerator):
            if isinstance(self.options.model, PlaceHolderModelType) \
                    and not isinstance(value.options.model, PlaceHolderModelType):
                warn("The model of the gear is overwritten by the given trial generator.",
                     category=ModelChangedWarning)
                self.options.model = value.options.model
            else:
                if not isinstance(value.options.model, PlaceHolderModelType):
                    warn("The model of the trial generator is overwritten by the gear.",
                         category=ModelChangedWarning)
                self.trial_generator.options.model = self.options.model
                self.trial_generator._parent = self
            if hasattr(self, "aggregate_filter"):
                self._check_filters_for_flask_compatibility()
            self.clear_cache()
        if item == "aggregate_filter":
            if not isinstance(value, AggregateFilter):
                raise TypeError(f"The {item} must be an AggregateFilter.")
            if hasattr(self, "options") and self.options.looped_collection == "flasks" and not value.supports_flasks():
                raise ValueError(f"The aggregate filter {value.name} does not support flasks.")

    def clear_cache(self) -> None:
        self._cache = set()
        self._rebuild_cache = True

    def disable_caching(self) -> None:
        self.clear_cache()
        self._rebuild_cache = False

    def enable_caching(self) -> None:
        self._rebuild_cache = True

    def unimolecular_coordinates(self, credentials: db.Credentials,
                                 observer: Optional[Callable[[], None]] = None) \
            -> Dict[str,
                    Dict[str,
                         List[Tuple[List[List[Tuple[int, int]]], int]]]]:
        """
        Returns the reaction coordinates allowed for unimolecular reactions for the whole database based on
        the set options and filters. This method does not set up new calculations.
        The returned object is a dictionary of dictionaries containing list of tuple.
        The dictionary holds the aggregate IDs, the next dictionary holds then the structures of each aggregate
        with the reaction coordinate information.
        The first argument in the tuple is a list of reaction coordinates.
        The second argument in the tuple is the number of dissociations.

        Parameters
        ----------
        credentials : db.Credentials
            The credentials of the database.
        observer : Optional[Callable[[], None]]
            A function that is called after each aggregate to count the number of aggregates processed.
        """
        _initialize_a_gear_to_a_db(self, credentials)
        return self._internal_loop_impl(setup_calculations=False,
                                        loop_unimolecular=self.options.enable_unimolecular_trials,
                                        loop_bimolecular=False,
                                        observer=observer)[0]

    def bimolecular_coordinates(self, credentials: db.Credentials,
                                observer: Optional[Callable[[], None]] = None) -> \
            Dict[str,
                 Dict[str,
                      Dict[Tuple[List[Tuple[int, int]], int],
                           List[Tuple[ndarray, ndarray, float, float]]
                           ]
                      ]
                 ]:
        """
        Returns the reaction coordinates allowed for bimolecular reactions for the whole database based on
        the set options and filters. This method does not set up new calculations.
        The returned object is a dictionary of dictionaries containing a dictionary specifying the coordinates.
        The dictionary holds the aggregate IDs, the next dictionary holds then the structures of each aggregate
        with the reaction coordinate information.
        The keys are a tuple containing a reaction coordinates and the number of dissociations.
        The values hold a list of instructions. Each entry in this list allows to construct a reactive complex.
        Therefore, the number of reactive complexes per reaction coordinate can also be inferred.

        Notes
        -----
        The index basis (total system or separate systems) of the returned indices in the reaction coordinates
        varies between different TrialGenerator implementations!

        Parameters
        ----------
        credentials : db.Credentials
            The credentials of the database.
        observer : Optional[Callable[[], None]]
            A function that is called after each aggregate to count the number of aggregates processed.
        """
        _initialize_a_gear_to_a_db(self, credentials)
        return self._internal_loop_impl(setup_calculations=False,
                                        loop_unimolecular=False,
                                        loop_bimolecular=self.options.enable_bimolecular_trials,
                                        observer=observer)[1]

    def _sanity_check_configuration(self):
        if not isinstance(self.aggregate_filter, AggregateFilter):
            raise TypeError(f"Expected a AggregateFilter (or a class derived "
                            f"from it) in {self.name}.aggregate_filter.")
        if hasattr(self.trial_generator, 'reactive_site_filter'):
            if not isinstance(getattr(self.trial_generator, 'reactive_site_filter'), ReactiveSiteFilter):
                raise TypeError(f"Expected a ReactiveSiteFilter (or a class derived "
                                f"from it) in {self.name}.trial_generator.reactive_site_filter.")

    def _propagate_db_manager(self, manager: db.Manager):
        self._sanity_check_configuration()
        self.trial_generator.initialize_collections(manager)
        if hasattr(self, 'aggregate_filter'):
            self.aggregate_filter.initialize_collections(manager)
        if hasattr(self.trial_generator, 'reactive_site_filter'):
            self.trial_generator.reactive_site_filter.initialize_collections(manager)

    def _loop_impl(self):
        if self.options.run_one_cycle_with_settings_enhancement:
            self.clear_cache()
        self._internal_loop_impl(setup_calculations=True,
                                 loop_unimolecular=self.options.enable_unimolecular_trials,
                                 loop_bimolecular=self.options.enable_bimolecular_trials)
        self.options.run_one_cycle_with_settings_enhancement = False

    def _internal_loop_impl(self, setup_calculations: bool, loop_unimolecular: bool, loop_bimolecular: bool,
                            observer: Optional[Callable[[], None]] = None) \
            -> Tuple[Dict[str,
                          Dict[str,
                               List[Tuple[List[List[Tuple[int, int]]], int]]
                               ]
                          ],
                     Dict[str,
                          Dict[str,
                               Dict[Tuple[List[Tuple[int, int]], int],
                                    List[Tuple[ndarray, ndarray, float, float]]
                                    ]
                               ]
                          ]
                     ]:

        if self.options.model != self.trial_generator.options.model:
            raise TypeError(f"Elementary step gear {self.name} and trial generator "
                            f"{self.trial_generator.__class__.__name__} have diverging models")
        uni_result: Dict[str,
                         Dict[str,
                              List[Tuple[List[List[Tuple[int, int]]], int]]]] \
            = defaultdict(lambda: defaultdict(list))
        bi_result: Dict[str,
                        Dict[str,
                             Dict[Tuple[List[Tuple[int, int]], int],
                                  List[Tuple[ndarray, ndarray, float, float]]
                                  ]
                             ]
                        ] \
            = defaultdict(lambda: defaultdict(dict))
        # Loop over all aggregates
        collection, iterator = self._get_collection_iterator()
        for aggregate_one in stop_on_timeout(iterator):
            aggregate_one.link(collection)
            if self.have_to_stop_at_next_break_point():
                return {}, {}
            if observer is not None:
                observer()
            eligible_sid_one = None
            if loop_unimolecular and self.aggregate_filter.filter(aggregate_one):
                eligible_sid_one = sorted(self._get_eligible_structures(aggregate_one))
                for sid_one in eligible_sid_one:
                    if self.have_to_stop_at_next_break_point():
                        return {}, {}
                    if not self.options.run_one_cycle_with_settings_enhancement and sid_one.string() in self._cache:
                        continue
                    structure_one = db.Structure(sid_one, self._structures)
                    if not self._check_structure_model(structure_one):
                        continue
                    if self._rebuild_cache:
                        if sid_one.string() not in self._cache:
                            self._update_cache(
                                structure_one,
                                self.trial_generator.get_unimolecular_job_order(),
                                self.trial_generator.options.model
                            )
                        if not self.options.run_one_cycle_with_settings_enhancement and sid_one.string() in self._cache:
                            continue
                    if setup_calculations:
                        self.trial_generator.unimolecular_reactions(
                            structure_one, self.options.run_one_cycle_with_settings_enhancement)
                    else:
                        uni_result[str(aggregate_one.id())][str(sid_one)] = \
                            self.trial_generator.unimolecular_coordinates(
                                structure_one, self.options.run_one_cycle_with_settings_enhancement)
                    self._cache.add(sid_one.string())
            # Get intermolecular reaction partners
            if not loop_bimolecular:
                continue
            if eligible_sid_one is None:
                eligible_sid_one = sorted(self._get_eligible_structures(aggregate_one))
            if not eligible_sid_one:
                continue
            c_id_one = aggregate_one.id().string()
            _, second_iterator = self._get_collection_iterator()
            for aggregate_two in stop_on_timeout(second_iterator):
                aggregate_two.link(collection)
                if self.have_to_stop_at_next_break_point():
                    return {}, {}
                # Make this loop run lower triangular + diagonal only
                c_id_two = aggregate_two.id().string()
                sorted_ids = sorted([c_id_one, c_id_two])
                # Second criterion needed to not exclude diagonal
                if sorted_ids[0] == c_id_two and c_id_one != c_id_two:
                    continue
                # Filter
                if not self.aggregate_filter.filter(aggregate_one, aggregate_two):
                    continue
                eligible_sid_two = sorted(self._get_eligible_structures(aggregate_two))
                if not eligible_sid_two:
                    continue
                same_compounds = c_id_one == c_id_two
                for i, sid_one in enumerate(eligible_sid_one):
                    for j, sid_two in enumerate(eligible_sid_two):
                        if self.have_to_stop_at_next_break_point():
                            return {}, {}
                        if same_compounds and j > i:
                            break
                        joined_ids = ';'.join(sorted([sid_one.string(), sid_two.string()]))
                        if not self.options.run_one_cycle_with_settings_enhancement and joined_ids in self._cache:
                            continue
                        structure_one = db.Structure(sid_one, self._structures)
                        structure_two = db.Structure(sid_two, self._structures)
                        if not self._check_structure_model(structure_one) or \
                                not self._check_structure_model(structure_two):
                            continue
                        if self._rebuild_cache:
                            if i == 0 and (sid_one.string() not in self._cache):
                                self._update_cache(
                                    structure_one,
                                    self.trial_generator.get_bimolecular_job_order(),
                                    self.trial_generator.options.model
                                )
                            if sid_two.string() not in self._cache:
                                self._update_cache(
                                    structure_two,
                                    self.trial_generator.get_bimolecular_job_order(),
                                    self.trial_generator.options.model
                                )
                            if not self.options.run_one_cycle_with_settings_enhancement and joined_ids in self._cache:
                                continue
                        if setup_calculations:
                            self.trial_generator.bimolecular_reactions(
                                [structure_one, structure_two], self.options.run_one_cycle_with_settings_enhancement)
                        else:
                            # split to make more readable
                            compound_key = f"{str(aggregate_one.id())}-{str(aggregate_two.id())}"
                            structure_key = f"{str(sid_one)}-{str(sid_two)}"
                            bi_result[compound_key][structure_key] = \
                                self.trial_generator.bimolecular_coordinates(
                                    [structure_one, structure_two],
                                    self.options.run_one_cycle_with_settings_enhancement)
                        self._cache.add(joined_ids)
        if self._rebuild_cache:
            self._rebuild_cache = False
        return uni_result, bi_result

    def _update_cache(self, structure: db.Structure, job_order: str, model: db.Model) -> None:
        calc_ids = structure.get_calculations(job_order)
        if not calc_ids:
            return
        for calc_id in calc_ids:
            calculation = db.Calculation(calc_id, self._calculations)
            if calculation.get_model() != model:
                continue
            structures_in_calc_ids = calculation.get_structures()
            joined_ids = ';'.join(sorted([s.string() for s in structures_in_calc_ids]))
            self._cache.add(joined_ids)

    @abstractmethod
    def _get_eligible_structures(self, aggregate: Union[db.Compound, db.Flask]) -> List[db.ID]:
        pass

    def _check_structure_model(self, structure: db.Structure) -> bool:
        # Only check the model if the option for the structure model is checked
        if not isinstance(self.options.structure_model, PlaceHolderModelType):
            return self.options.structure_model == structure.get_model()
        return True

    def _get_collection_iterator(self) -> Tuple[db.Collection, Iterator[Union[db.Compound, db.Flask]]]:
        selection = {"exploration_disabled": {"$ne": True}}
        if self.options.looped_collection == "compounds":
            return self._compounds, self._compounds.iterate_compounds(dumps(selection))
        if self.options.looped_collection == "flasks":
            self._check_filters_for_flask_compatibility()
            return self._flasks, self._flasks.iterate_flasks(dumps(selection))
        raise ValueError(f"Invalid value for looped_collection: '{self.options.looped_collection}'. "
                         f"Only 'compounds' and 'flasks' are allowed.")

    def _check_filters_for_flask_compatibility(self) -> None:
        """
        Checks if the aggregate filter and the trial generator are compatible with flasks.

        Raises
        ------
        ValueError
            If the aggregate filter or the trial generator are not compatible with flasks.
        """
        if not self.aggregate_filter.supports_flasks():
            raise ValueError(f"The aggregate filter {self.aggregate_filter.name} does not support flasks.")
        if not self.trial_generator.reactive_site_filter.supports_flasks():
            raise ValueError(f"The reactive site filter {self.trial_generator.reactive_site_filter.name} does not "
                             f"support flasks.")
        further_filter = getattr(self.trial_generator, "further_reactive_site_filter", None)
        if isinstance(further_filter, ReactiveSiteFilter) and not further_filter.supports_flasks():
            raise ValueError(f"The further reactive site filter "
                             f"{getattr(self.trial_generator, 'further_reactive_site_filter').name} "
                             f"does not support flasks.")
