#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from abc import abstractmethod, ABC
from collections import defaultdict
from json import dumps
from typing import Dict, List, Set, Optional, Tuple

from numpy import ndarray
import scine_database as db
from scine_utilities import ValueCollection

# Local application imports
from .aggregate_filters import AggregateFilter
from .reactive_site_filters import ReactiveSiteFilter
from .trial_generator import TrialGenerator
from .trial_generator.bond_based import BondBased
from .. import Gear, _initialize_a_gear_to_a_db
from scine_chemoton.utilities.queries import stop_on_timeout


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
            "cycle_time",
            "enable_unimolecular_trials",
            "enable_bimolecular_trials",
            "run_one_cycle_with_settings_enhancement",
            "base_job_settings"
        )

        def __init__(self, _parent: Optional[Gear] = None):
            self._parent = _parent
            super().__init__()
            self.cycle_time = 10
            """
            int
                Sleep time between cycles, in seconds.
            """
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

        def __setattr__(self, item, value):
            """
            Overwritten standard method to synchronize model option
            """
            model_case = bool(
                item == "model" and hasattr(
                    self,
                    "model") and self.model != value and hasattr(
                    self,
                    "_parent") and self._parent is not None and hasattr(
                    self._parent,
                    "trial_generator"))
            super().__setattr__(item, value)
            if model_case:
                self._parent.trial_generator.options.model = value
                self._parent.clear_cache()

    def __init__(self):
        super().__init__()
        self._required_collections = ["calculations", "compounds", "properties", "reactions", "structures"]
        self.options: ElementaryStepGear.Options = self.Options(_parent=self)
        self.trial_generator: TrialGenerator = BondBased()
        self.trial_generator.options.base_job_settings = self.options.base_job_settings
        self.aggregate_filter: AggregateFilter = AggregateFilter()
        self._cache: Set[str] = set()
        self._rebuild_cache = True

    def __setattr__(self, item, value):
        """
        Overwritten standard method to synchronize model option
        """
        super().__setattr__(item, value)
        if isinstance(value, TrialGenerator):
            self.trial_generator.options.model = self.options.model
            self.trial_generator._parent = self
            self.clear_cache()
        if item == "base_job_settings":
            if not isinstance(value, ValueCollection):
                raise TypeError(f"The {item} must be a ValueCollection.")
            self.trial_generator.options.base_job_settings = value

    def clear_cache(self):
        self._cache = set()
        self._rebuild_cache = True

    def unimolecular_coordinates(self, credentials: db.Credentials) \
            -> Dict[str,
                    Dict[str,
                         List[Tuple[List[List[Tuple[int, int]]], int]]]]:
        """
        Returns the reaction coordinates allowed for unimolecular reactions for the whole database based on
        the set options and filters. This method does not set up new calculations.
        The returned object is a dictionary of dictionaries containing list of tuple.
        The dictionary holds the compounds IDs, the next dictionary holds then the structures of each compound
        with the reaction coordinate information.
        The first argument in the tuple is a list of reaction coordinates.
        The second argument in the tuple is the number of dissociations.

        Parameters
        ----------
        credentials :: db.Credentials
            The credentials of the database.
        """
        _initialize_a_gear_to_a_db(self, credentials)
        with self._DelayedKeyboardInterrupt(callable=self.stop):
            return self._internal_loop_impl(setup_calculations=False,
                                            loop_unimolecular=self.options.enable_unimolecular_trials,
                                            loop_bimolecular=False)[0]

    def bimolecular_coordinates(self, credentials: db.Credentials) -> \
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
        The dictionary holds the compounds IDs, the next dictionary holds then the structures of each compound
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
        credentials :: db.Credentials
            The credentials of the database.
        """
        _initialize_a_gear_to_a_db(self, credentials)
        with self._DelayedKeyboardInterrupt(callable=self.stop):
            return self._internal_loop_impl(setup_calculations=False,
                                            loop_unimolecular=False,
                                            loop_bimolecular=self.options.enable_bimolecular_trials)[1]

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

    def _internal_loop_impl(self, setup_calculations: bool, loop_unimolecular: bool, loop_bimolecular: bool) \
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
        # Loop over all compounds
        selection = {"exploration_disabled": {"$ne": True}}
        for compound_one in stop_on_timeout(self._compounds.iterate_compounds(dumps(selection))):
            compound_one.link(self._compounds)
            if self.stop_at_next_break_point:
                return {}, {}
            eligible_sid_one = None
            if loop_unimolecular and self.aggregate_filter.filter(compound_one):
                eligible_sid_one = sorted(self._get_eligible_structures(compound_one))
                for sid_one in eligible_sid_one:
                    if self.stop_at_next_break_point:
                        return {}, {}
                    if not self.options.run_one_cycle_with_settings_enhancement and sid_one.string() in self._cache:
                        continue
                    structure_one = db.Structure(sid_one, self._structures)
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
                        uni_result[str(compound_one.id())][str(sid_one)] = \
                            self.trial_generator.unimolecular_coordinates(
                                structure_one, self.options.run_one_cycle_with_settings_enhancement)
                    self._cache.add(sid_one.string())
            # Get intermolecular reaction partners
            if not loop_bimolecular:
                continue
            if eligible_sid_one is None:
                eligible_sid_one = sorted(self._get_eligible_structures(compound_one))
            if not eligible_sid_one:
                continue
            c_id_one = compound_one.id().string()
            selection = {"exploration_disabled": {"$ne": True}}
            for compound_two in stop_on_timeout(self._compounds.iterate_compounds(dumps(selection))):
                compound_two.link(self._compounds)
                if self.stop_at_next_break_point:
                    return {}, {}
                # Make this loop run lower triangular + diagonal only
                c_id_two = compound_two.id().string()
                sorted_ids = sorted([c_id_one, c_id_two])
                # Second criterion needed to not exclude diagonal
                if sorted_ids[0] == c_id_two and c_id_one != c_id_two:
                    continue
                # Filter
                if not self.aggregate_filter.filter(compound_one, compound_two):
                    continue
                eligible_sid_two = sorted(self._get_eligible_structures(compound_two))
                if not eligible_sid_two:
                    continue
                same_compounds = c_id_one == c_id_two
                for i, sid_one in enumerate(eligible_sid_one):
                    for j, sid_two in enumerate(eligible_sid_two):
                        if self.stop_at_next_break_point:
                            return {}, {}
                        if same_compounds and j > i:
                            break
                        joined_ids = ';'.join(sorted([sid_one.string(), sid_two.string()]))
                        if not self.options.run_one_cycle_with_settings_enhancement and joined_ids in self._cache:
                            continue
                        structure_one = db.Structure(sid_one, self._structures)
                        structure_two = db.Structure(sid_two, self._structures)
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
                            compound_key = f"{str(compound_one.id())}-{str(compound_two.id())}"
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
            calculation = db.Calculation(calc_id)
            calculation.link(self._calculations)
            if calculation.get_model() != model:
                continue
            structures_in_calc_ids = calculation.get_structures()
            joined_ids = ';'.join(sorted([s.string() for s in structures_in_calc_ids]))
            self._cache.add(joined_ids)

    @abstractmethod
    def _get_eligible_structures(self, compound: db.Compound) -> List[db.ID]:
        pass
