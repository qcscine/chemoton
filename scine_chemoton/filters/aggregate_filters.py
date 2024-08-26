#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
from os import path, walk
from json import dumps
import math

# Third party imports
import numpy as np
import scine_database as db
import scine_molassembler as masm
import scine_utilities as utils
from scine_database.concentration_query_functions import query_concentration_with_object
from scine_database.energy_query_functions import get_energy_for_structure
from scine_database.queries import model_query

from scine_chemoton.gears import HoldsCollections, HasName
from scine_chemoton.filters.structure_filters import (
    AtomNumberFilter as StructureAtomNumberFilter,
    SpecificChargeFilter as StructureSpecificChargeFilter,
    ChargeCombinationFilter as StructureChargeCombinationFilter,
    CatalystFilter as StructureCatalystFilter,
    TrueMinimumFilter as StructureTrueMinimumFilter,
    ElementCountFilter as StructureElementCountFilter,
    ElementSumCountFilter as StructureElementSumCountFilter,
    MolecularWeightFilter as StructureMolecularWeightFilter,
)
from scine_chemoton.utilities.masm import deserialize_molecules
from scine_chemoton.utilities.get_molecular_formula import get_elements_in_structure, combine_element_counts


class _AbstractFilter(ABC):

    @abstractmethod
    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Union[db.Compound, db.Flask, None] = None) -> bool:
        pass

    @abstractmethod
    def supports_flasks(self) -> bool:
        pass


class AggregateFilter(HoldsCollections, HasName, _AbstractFilter):
    """
    The base and default class for all filters. The default is to allow/pass all
    given checks.

    CompoundFilters are optional barriers in Chemoton that allow the user to cut down
    the exponential growth of the combinatorial explosion. The different
    subclasses of this main AggregateFilter allow for a tailored choice of which
    additional branches of the network to allow.

    There are some predefined filters that will be given in Chemoton, however,
    it should be simple to extend as needed even on a per-project basis.
    The key element when sub-classing this interface is to override the `filter`
    functions as defined here. When sub-classing please be aware that these
    filters are expected to be called often. Having each call do loops over
    entire collections is not wise.

    For the latter reason, user defined subclasses are intended to be more
    complex, allowing for non-database stored/cached data across a run.
    This can be a significant speed-up and allow for more intricate filtering.
    """

    def __init__(self) -> None:
        super().__init__()
        self._remove_chemoton_from_name()
        self._required_collections = ["calculations", "compounds", "elementary_steps",
                                      "flasks", "properties", "structures", "reactions"]
        self._can_cache: bool = True
        self._currently_caches: bool = True
        self._cache: Dict[int, bool] = {}

    def __and__(self, o):
        if not isinstance(o, AggregateFilter):
            raise TypeError("AggregateFilter expects AggregateFilter "
                            "(or derived class) to chain with.")
        return AggregateFilterAndArray([self, o])

    def __or__(self, o):
        if not isinstance(o, AggregateFilter):
            raise TypeError("AggregateFilter expects AggregateFilter "
                            "(or derived class) to chain with.")
        return AggregateFilterOrArray([self, o])

    def disable_caching(self) -> None:
        """
        Disables the cache in this filter and flushes the existing cache.
        """
        self._currently_caches = False
        self._cache = {}

    def enable_caching(self) -> None:
        """
        Enables the cache in this filter.
        """
        if self._can_cache:
            self._currently_caches = True
        else:
            raise RuntimeError("Cannot enable_caching in filter that cannot cache.")

    def currently_caches(self) -> bool:
        """
        Returns
        -------
        result : bool
            True if the filter is currently set to cache results, and return
            results from cache if possible.
        """
        return self._currently_caches

    def can_cache(self) -> bool:
        """
        Returns
        -------
        result : bool
            True if the filter can possibly cache results.
        """
        return self._can_cache

    @staticmethod
    def _hash_aggregates(aggregate: Union[db.Compound, db.Flask],
                         optional_aggregate: Optional[Union[db.Compound, db.Flask]] = None) -> int:
        if optional_aggregate is not None:
            return hash(';'.join(sorted([aggregate.get_id().string(), optional_aggregate.get_id().string()])))
        else:
            return hash(aggregate.get_id().string())

    @staticmethod
    def _id_sanity_check(ids: List[str]) -> None:
        for i in ids:
            if not isinstance(i, str):
                raise TypeError(f"Expected str, got {type(i)}")
            try:
                db.ID(i)
            except RuntimeError as e:
                raise ValueError(f"The entry {i} is not a valid database ID.") from e

    def _get_centroids(self, aggregate_one: Union[db.Compound, db.Flask],
                       aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> Tuple[db.Structure,
                                                                                              Union[db.Structure,
                                                                                                    None]]:
        structure_one = db.Structure(aggregate_one.get_centroid(), self._structures)
        if aggregate_two is not None:
            structure_two = db.Structure(aggregate_two.get_centroid(), self._structures)
        else:
            structure_two = None

        return structure_one, structure_two

    def filter(self, aggregate: Union[db.Compound, db.Flask],
               optional_aggregate: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        """
        The filter function, checking if both of the aggregates
        are allowed to be used in an exploration (logical and). If only one
        aggregate is given a check using only that one aggregate is performed.
        This default AggregateFilter accepts all aggregates.

        Attributes
        ----------
        aggregate : Union[db.Compound, db.Flask]
            The aggregate to be checked.
        optional_aggregate : Union[db.Compound, db.Flask]
            The other aggregate to be checked in the case of bimolecular reactions.

        Returns
        -------
        result : bool
            True if the aggregate passed the check/filter, False if the aggregate
            did not pass the check and should not be used.
        """
        if self._can_cache and self._currently_caches:
            key = self._hash_aggregates(aggregate, optional_aggregate)
            result = self._cache.get(key)
            if result is None:
                result = self._filter_impl(aggregate, optional_aggregate)
                self._cache[key] = result
            return result
        else:
            return self._filter_impl(aggregate, optional_aggregate)

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        """
        The blueprint for a filter implementation function.
        See public function for detailed description.
        This default AggregateFilter accepts all aggregates.

        Attributes
        ----------
        aggregate_one : Union[db.Compound, db.Flask]
            The aggregate to be checked.
        aggregate_two : Union[db.Compound, db.Flask]
            The other aggregate to be checked in the case of bimolecular reactions.

        Returns
        -------
        result : bool
            True if the aggregate passed the check/filter, False if the aggregate
            did not pass the check and should not be used.
        """
        return True

    def supports_flasks(self) -> bool:
        return True


class PlaceHolderAggregateFilter(AggregateFilter):
    """
    A place-holder for aggregate filters. This can be used instead of a None default argument.
    """

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        raise NotImplementedError(f"The {self.name} is not meant to be used as a filter, but has to be replaced.")


class AggregateFilterAndArray(AggregateFilter):
    """
    An array of logically 'and' connected filters.
    """

    def __init__(self, filters: Optional[List[AggregateFilter]] = None) -> None:
        """
        Parameters
        ----------
        filters : Optional[List[AggregateFilter]]
            A list of filters to be combined.
        """
        super().__init__()
        if filters is None:
            filters = []
        self._filters = filters
        for f in filters:
            if not isinstance(f, AggregateFilter):
                raise TypeError("AggregateFilterAndArray expects AggregateFilter "
                                "(or derived class) to chain with.")
        self._join_names(self._filters)
        # Check if caching is an option
        self._can_cache = True
        self._currently_caches = True
        for f in self._filters:
            if not f.can_cache():
                self._can_cache = False
                self._currently_caches = False
                break
        # Disable all caches if this array can cache
        #   If this filter can not cache, lower filters
        #   that can cache are still allowed to
        if self._can_cache:
            for f in self._filters:
                f.disable_caching()

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        return all(f.filter(aggregate_one, aggregate_two) for f in self._filters)

    def initialize_collections(self, manager: db.Manager) -> None:
        for f in self._filters:
            f.initialize_collections(manager)

    def __iter__(self):
        return (f for f in self._filters)

    def __setitem__(self, key, value):
        self._filters[key] = value

    def supports_flasks(self) -> bool:
        return all(f.supports_flasks() for f in self._filters)


class AggregateFilterOrArray(AggregateFilter):
    """
    An array of logically 'or' connected filters.
    """

    def __init__(self, filters: Optional[List[AggregateFilter]] = None) -> None:
        """
        Parameters
        ----------
        filters : Optional[List[AggregateFilter]]
            A list of filters to be combined.
        """
        super().__init__()
        if filters is None:
            filters = []
        self._filters = filters
        for f in filters:
            if not isinstance(f, AggregateFilter):
                raise TypeError("AggregateFilterOrArray expects AggregateFilter "
                                "(or derived class) to chain with.")
        self._join_names(self._filters)
        # Check if caching is an option
        self._can_cache = True
        self._currently_caches = True
        for f in self._filters:
            if not f.can_cache():
                self._can_cache = False
                self._currently_caches = False
                break
        # Disable all caches if this array can cache
        #   If this filter can not cache, lower filters
        #   that can cache are still allowed to
        if self._can_cache:
            for f in self._filters:
                f.disable_caching()

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        return any(f.filter(aggregate_one, aggregate_two) for f in self._filters)

    def initialize_collections(self, manager: db.Manager) -> None:
        for f in self._filters:
            f.initialize_collections(manager)

    def __iter__(self):
        return (f for f in self._filters)

    def __setitem__(self, key, value):
        self._filters[key] = value

    def supports_flasks(self) -> bool:
        return all(f.supports_flasks() for f in self._filters)


class ElementCountFilter(AggregateFilter):
    """
    Filters by atom counts. All given aggregates must resolve to structures
    that separately fall below or equal the given threshold. No assumptions about atom
    counts of possible combinations/products are made in this filter.
    """

    def __init__(self, atom_type_count: Dict[str, int], allow_above_limit: bool = False) -> None:
        """
        Construct the filter with the allowed element counts.

        Parameters
        ----------
        atom_type_count : Dict[str,int]
            A dictionary giving the number (values) of allowed occurrences of each
            atom type (atom symbol string given as keys). Atom symbols not given
            as keys are interpreted as forbidden.
        allow_above_limit : bool
            If false the element counts must be equal or below the given limit.
            If true the element counts must be equal or above the given limit.
        """
        super().__init__()
        self._structure_element_count_filter = StructureElementCountFilter(atom_type_count, allow_above_limit)

    def initialize_collections(self, manager: db.Manager) -> None:
        super().initialize_collections(manager)
        self._structure_element_count_filter.initialize_collections(manager)

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        structure_one, structure_two = self._get_centroids(aggregate_one, aggregate_two)
        return self._structure_element_count_filter.filter(structure_one, structure_two)


class ElementSumCountFilter(AggregateFilter):
    """
    Filters by atom counts. All given aggregates must resolve to structures
    that together fall below or equal the given threshold.
    """

    def __init__(self, atom_type_count: Dict[str, int], allow_above_limit: bool = False) -> None:
        """
        Construct the filter with the allowed element counts.

        Parameters
        ----------
        atom_type_count : Dict[str,int]
            A dictionary giving the number (values) of allowed occurrences of each
            atom type (atom symbol string given as keys). Atom symbols not given
            as keys are interpreted as forbidden.
        allow_above_limit : bool
            If false the element counts must be equal or below the given limit.
            If true the element counts must be above the given limit.
        """
        super().__init__()
        self._structure_element_sum_count_filter = StructureElementSumCountFilter(atom_type_count, allow_above_limit)

    def initialize_collections(self, manager: db.Manager) -> None:
        super().initialize_collections(manager)
        self._structure_element_sum_count_filter.initialize_collections(manager)

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        structure_one, structure_two = self._get_centroids(aggregate_one, aggregate_two)
        return self._structure_element_sum_count_filter.filter(structure_one, structure_two)


class MolecularWeightFilter(AggregateFilter):
    """
    Filters by molecular weight. All given aggregates must resolve to structures
    that separately fall below the given threshold. No assumptions about weights
    of possible combinations/products are made in this filter.
    """

    def __init__(self, max_weight: float, allow_additions_above_limit: bool = True) -> None:
        """
        Construct the filter with the weight options.

        Parameters
        ----------
        max_weight : float
            The maximum weight to be allowed, given in unified atomic mass units (u).
            For example, dinitrogen has a weight of about 28 u.
        allow_additions_above_limit : bool
            If true only checks if the reactants are each above the given limit.
            If false, assumes complete additions may happen and restricts all
            combinations where the sum of weights is above the given limit.
        """
        super().__init__()
        self._structure_molecular_weight_filter = StructureMolecularWeightFilter(max_weight,
                                                                                 allow_additions_above_limit)

    def initialize_collections(self, manager: db.Manager) -> None:
        super().initialize_collections(manager)
        self._structure_molecular_weight_filter.initialize_collections(manager)

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        structure_one, structure_two = self._get_centroids(aggregate_one, aggregate_two)
        return self._structure_molecular_weight_filter.filter(structure_one, structure_two)


class IdFilter(AggregateFilter):
    """
    Filters by aggregate id. Returns true only for the specified aggregates.
    """

    def __init__(self, ids: List[str]) -> None:
        """
        Construct the filter with the allowed IDs.

        Parameters
        ----------
        reactive_ids : List[str]
            The IDs of the aggregates to be considered as reactive.
        """
        super().__init__()
        self._id_sanity_check(ids)
        self.reactive_ids = set(ids)

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        # Get aggregate ids
        aggregate_ids = set()
        aggregate_ids.add(aggregate_one.get_id().string())
        if aggregate_two is not None:
            aggregate_ids.add(aggregate_two.get_id().string())
        return aggregate_ids.issubset(self.reactive_ids)


class SelfReactionFilter(AggregateFilter):
    """
    Filters out bimolecular reactions of aggregates with themselves.
    """

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        if aggregate_two is None:
            # Do not rule out unimolecular reactions
            return True
        # Get compound ids
        return aggregate_one.get_id() != aggregate_two.get_id()


class TrueMinimumFilter(AggregateFilter):
    """
    Filters by checking if aggregate fulfills requirement to be considered a
    true minimum. This includes having frequencies and all of them are above a
    given threshold.
    All given aggregates must resolve to structures that separately only have
    frequencies above the given threshold. Single atoms or ions are considered
    as minima automatically.
    Frequencies must be calculated with the ThermoGear during an exploration,
    otherwise all aggregates and combinations of them are filtered out.
    """

    def __init__(self, imaginary_frequency_threshold: float = 0.0) -> None:
        """
        Construct filter with the wanted threshold

        Parameters
        ----------
        imaginary_frequency_threshold : float
            The frequency in atomic units above which a structure is considered a
            minimum structure.
            For example, a molecule with one imaginary frequency of -1e-4 (-138 cm^-1)
            can be considered a minimum by setting the threshold to -1.1e-4 (-152 cm^-1)
        """
        super().__init__()
        self._structure_true_minimum_filter = StructureTrueMinimumFilter(imaginary_frequency_threshold)

    def initialize_collections(self, manager: db.Manager) -> None:
        super().initialize_collections(manager)
        self._structure_true_minimum_filter.initialize_collections(manager)

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        structure_one, structure_two = self._get_centroids(aggregate_one, aggregate_two)
        return self._structure_true_minimum_filter.filter(structure_one, structure_two)


class CatalystFilter(AggregateFilter):
    """
    Filters by defining an important structural motive.
    A user defined set of atoms present in a structure identify it to be the, or
    a version of the catalyst. This simple check should work great for common
    transition metal catalysts.

    Only specific reactions revolving around a catalytic cycle are then allowed:
    i) reactions that feature the catalyst and one other aggregate that is not
    the catalyst.
    ii) reactions that only involve a single aggregate (not restricted to the
    catalyst, unless specified otherwise with flag (see parameters))
    """

    def __init__(self, atom_type_count: Dict[str, int], restrict_unimolecular_to_catalyst: bool = False) -> None:
        """
        Construct the filter with the allowed element counts.

        Parameters
        ----------
        atom_type_count : Dict[str,int]
            A dictionary giving the number (values) of atoms that are expected to be
            only present in - and thus defining - the catalyst. Atom type (atom
            symbol strings) are given as keys. Atom symbols not given are considered
            to be irrelevant to the check and may be present in the catalyst. In
            order to ban atoms, set their count to zero.
        restrict_unimolecular_to_catalyst : bool
            Whether unimolecular reactions should also be limited to the catalyst.
        """
        super().__init__()
        self._structure_catalyst_filter = StructureCatalystFilter(atom_type_count, restrict_unimolecular_to_catalyst)

    def initialize_collections(self, manager: db.Manager) -> None:
        super().initialize_collections(manager)
        self._structure_catalyst_filter.initialize_collections(manager)

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        structure_one, structure_two = self._get_centroids(aggregate_one, aggregate_two)
        return self._structure_catalyst_filter.filter(structure_one, structure_two)


class AtomNumberFilter(AggregateFilter):
    """
    Filters out all aggregates with a total number of atoms larger than the given
    value. For multiple aggregates the total number of atoms has to be equal or
    smaller than the given value.
    The optional minimum limit is also defined as equal or smaller, i.e., 1 allows aggregates with 1 atom.
    """

    def __init__(self, max_n_atoms: int, min_n_atoms: int = 0) -> None:
        """
        Construct the filter with the limits.

        Parameters
        ----------
        max_n_atoms : int
            The maximum number of allowed atoms.
        min_n_atoms : int
            The minimum number of allowed atoms.
        """
        super().__init__()
        self._structure_atom_number_filter = StructureAtomNumberFilter(max_n_atoms, min_n_atoms)

    def initialize_collections(self, manager: db.Manager) -> None:
        super().initialize_collections(manager)
        self._structure_atom_number_filter.initialize_collections(manager)

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        structure_one, structure_two = self._get_centroids(aggregate_one, aggregate_two)
        return self._structure_atom_number_filter.filter(structure_one, structure_two)


class OneAggregateIdFilter(AggregateFilter):
    """
    Filters all aggregates that are not present on a given "white list" of IDs.
    In the case of multiple aggregates, at least one has to be present in the
    list. Note that this is identical to the IdFilter in the case of only one
    aggregate.
    """

    def __init__(self, ids: List[str]) -> None:
        """
        Construct the filter with the allowed IDs.

        Parameters
        ----------
        reactive_ids : List[str]
            The IDs of the aggregates to be considered as reactive.
        """
        super().__init__()
        self.reactive_ids = set(ids)

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        if aggregate_one.get_id().string() in self.reactive_ids:
            return True
        if aggregate_two is not None:
            if aggregate_two.get_id().string() in self.reactive_ids:
                return True
        return False


class SelectedAggregateIdFilter(AggregateFilter):
    """
    Filters all aggregates for which one needs to be one a given "white list"
    of reactive aggregates and the other has to be either on a list of
    aggregates of interest or on the list of reactive aggregates.
    """

    def __init__(self, reactive_ids: List[str], selected_ids: Optional[List[str]] = None) -> None:
        """
        Construct the filter with the allowed IDs.

        Parameters
        ----------
        reactive_ids : List[str]
            The IDs of the aggregates to be considered as reactive.
        selected_ids : Optional[List[str]]
            The IDs of the aggregates to be of interest.
        """
        super().__init__()
        self._id_sanity_check(reactive_ids)
        self.reactive_ids = set(reactive_ids)
        if selected_ids is None:
            selected_ids = []
        else:
            self._id_sanity_check(selected_ids)
        self.selected_ids = set(selected_ids)

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        # One aggregate case: the id has to be a member of the reactive_ids.
        # Two aggregate case: one of the ids has to be a member of the reactive_ids.
        #                   The other one has to be either a member of the reactive_ids or the selected_ids.
        one_is_reactive = aggregate_one.get_id().string() in self.reactive_ids
        # One aggregate case
        if aggregate_two is None:
            return one_is_reactive
        # Two aggregate case
        two_is_reactive = aggregate_two.get_id().string() in self.reactive_ids
        one_is_selected = aggregate_one.get_id().string() in self.selected_ids or one_is_reactive
        two_is_selected = aggregate_two.get_id().string() in self.selected_ids or two_is_reactive
        return (one_is_reactive and two_is_selected) or (two_is_reactive and one_is_selected)


class ConcentrationPropertyFilter(AggregateFilter):
    """
    Filters all aggregates that have a concentration larger than a given value. For the two aggregate case,
    the product of both concentrations has to be larger than this value. The filter may be set to only
    filter for the two-aggregate case. Furthermore, the filter can include error information on the concentration
    properties through their variance if variance_labels are supplied.

    Filter condition for one aggregate:\n
        [c^n + \\sqrt(c^n_var)] > \\tau\n
    Filter condition for two aggregates:\n
        [c^n + \\sqrt(c^n_var)] * [c^m + \\sqrt(c^m_var)] > \tau
    """

    def __init__(self, property_labels: List[str], min_value: float, filter_only_pairs: bool,
                 variance_labels: Optional[List[Optional[str]]] = None) -> None:
        """
        Construct the filter with the allowed labels and values.

        Parameters
        ----------
        property_labels : List[str]
            The labels for the concentration to filter with.
        min_value : float
            The minimum concentration/concentration product.
        filter_only_pairs : bool
            If true, the filter will always return True for the single aggregate case.
        variance_labels : Optional[List[Optional[str]]]
            If a variance for the concentration is known, its property label can be given here.
        """
        super().__init__()
        self._property_labels = property_labels
        self._min_value = min_value
        self._filter_only_paris = filter_only_pairs
        self._can_cache: bool = False
        self._currently_caches: bool = False
        if variance_labels is None:
            variance_labels = [None for _ in self._property_labels]
        self._variance_labels: List[Optional[str]] = variance_labels
        if len(self._variance_labels) != len(self._property_labels):
            raise RuntimeError("The number of variance labels and property labels must match.")

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        if self._filter_only_paris and aggregate_two is None:
            return True
        concentration = self._get_concentration(aggregate_one)
        if aggregate_two is not None:
            concentration *= self._get_concentration(aggregate_two)
        return concentration > self._min_value

    def _get_concentration(self, aggregate: Union[db.Compound, db.Flask]):
        concentrations: List[float] = list()
        for label, variance_label in zip(self._property_labels, self._variance_labels):
            concentration = query_concentration_with_object(label, aggregate, self._properties, self._structures)
            if variance_label is not None:
                concentration += np.sqrt(query_concentration_with_object(variance_label, aggregate, self._properties,
                                                                         self._structures))
            if math.isnan(concentration):
                raise RuntimeError("Error: NaN was detected during concentration-filter evaluation for aggregate: "
                                   + aggregate.id().string() + "\n Please verify that its concentration is non-"
                                   "negative.")
            concentrations.append(concentration)
        return max(concentrations)


class ChargeCombinationFilter(AggregateFilter):
    """
    Avoid combination of two aggregates that both have negative charges or both have positive charges.
    """

    def __init__(self) -> None:
        super().__init__()
        self._structure_charge_combination_filter = StructureChargeCombinationFilter()

    def initialize_collections(self, manager: db.Manager) -> None:
        super().initialize_collections(manager)
        self._structure_charge_combination_filter.initialize_collections(manager)

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        structure_one, structure_two = self._get_centroids(aggregate_one, aggregate_two)
        return self._structure_charge_combination_filter.filter(structure_one, structure_two)


class SpecificChargeFilter(AggregateFilter):
    """
    Either allow or exclude aggregates with the specified charge(s).
    """

    def __init__(self, charges: List[int], allow: bool = True, both_charges_must_be_valid: bool = True) -> None:
        """
        Parameters
        ----------
        charges : List[int]
            The charges to be allowed or excluded.
        allow : bool
            If true, the charges will be allowed, otherwise excluded.
        both_charges_must_be_valid : bool
            If true, both charges must be valid for bimolecular reactions, otherwise only one of them.
        """
        super().__init__()
        self._structure_specific_charge_filter = StructureSpecificChargeFilter(charges, allow,
                                                                               both_charges_must_be_valid)

    def initialize_collections(self, manager: db.Manager) -> None:
        super().initialize_collections(manager)
        self._structure_specific_charge_filter.initialize_collections(manager)

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        structure_one, structure_two = self._get_centroids(aggregate_one, aggregate_two)
        return self._structure_specific_charge_filter.filter(structure_one, structure_two)


class LastKineticModelingFilter(AggregateFilter):
    """
    Allow only aggregates that were handled in the last kinetic modeling calculation or have a non-zero start
    concentration.
    """

    def __init__(self, kinetic_modeling_job_order: Optional[str] = None,
                 aggregate_settings_key: Optional[str] = None) -> None:
        """
        Construct the filter with optional job information and settings.

        Parameters
        ----------
        kinetic_modeling_job_order : Optional[str]
            The kinetic modeling job order. By default, "kinetx_kinetic_modeling".
        aggregate_settings_key : Optional[str]
            The key to the aggregate ids in the kinetic modeling job. By default, "aggregate_ids".
        """
        super().__init__()
        self._start_structure: Union[None, db.Structure] = None
        if not kinetic_modeling_job_order:
            kinetic_modeling_job_order = "kinetx_kinetic_modeling"
        self._kinetic_modeling_job_order: str = kinetic_modeling_job_order
        if not aggregate_settings_key:
            aggregate_settings_key = "aggregate_ids"
        self._aggregate_settings_key = aggregate_settings_key
        self._n_calculations_last = 0
        self._aggregate_str_ids_in_last_job: List[db.ID] = list()
        self._can_cache: bool = False
        self._currently_caches: bool = False

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        if not self._start_structure:
            self._initialize_start_structure()
        self._update_aggregate_ids_in_last_job()
        if aggregate_one.id().string() not in self._aggregate_str_ids_in_last_job:
            start_concentration = query_concentration_with_object(
                "start_concentration", aggregate_one, self._properties, self._structures)
            if start_concentration <= 0.0:
                return False
        if aggregate_two:
            if aggregate_two.id().string() not in self._aggregate_str_ids_in_last_job:
                start_concentration = query_concentration_with_object(
                    "start_concentration", aggregate_two, self._properties, self._structures)
                if start_concentration <= 0.0:
                    return False
        return True

    def _initialize_start_structure(self) -> None:
        for aggregates, iterator in zip([self._compounds, self._flasks],
                                        [self._compounds.iterate_all_compounds(), self._flasks.iterate_all_flasks()]):
            for aggregate in iterator:
                aggregate.link(aggregates)
                c_start = query_concentration_with_object(
                    "start_concentration", aggregate, self._properties, self._structures)
                if c_start > 0.0:
                    self._start_structure = db.Structure(aggregate.get_centroid(), self._structures)
                    break

        if not self._start_structure:
            raise RuntimeError("LastKineticModelingFilter: No aggregate with a non-zero starting concentration is"
                               " given! This may prevent the exploration of further species and is not allowed.")

    def _update_aggregate_ids_in_last_job(self) -> None:
        assert self._start_structure
        if self._start_structure.has_calculations(self._kinetic_modeling_job_order):
            calc_ids = self._start_structure.get_calculations(self._kinetic_modeling_job_order)
            if len(calc_ids) == self._n_calculations_last:
                return
            for i, calc_id in enumerate(reversed(calc_ids)):
                calculation = db.Calculation(calc_id, self._calculations)
                if calculation.exists() and calculation.get_status() == db.Status.COMPLETE:
                    aggregate_str_ids = calculation.get_settings()[self._aggregate_settings_key]
                    self._aggregate_str_ids_in_last_job = aggregate_str_ids  # type: ignore
                    self._n_calculations_last = len(calc_ids) - i
                    break


class SelectedStructureIdFilter(SelectedAggregateIdFilter):
    """
    See SelectedAggregateIdFilter, but filters for aggregates that include the given Structure IDs.
    """

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        aggregate_one_structures = [str(sid) for sid in aggregate_one.get_structures()]
        one_is_reactive = any(sid in self.reactive_ids for sid in aggregate_one_structures)
        # One aggregate case
        if aggregate_two is None:
            return one_is_reactive
        aggregate_two_structures = [str(sid) for sid in aggregate_two.get_structures()]
        two_is_reactive = any(sid in self.reactive_ids for sid in aggregate_two_structures)
        one_is_selected = any(sid in self.selected_ids for sid in aggregate_one_structures)
        two_is_selected = any(sid in self.selected_ids for sid in aggregate_two_structures)
        return (one_is_reactive and two_is_selected) or (two_is_reactive and one_is_selected)


class CompoundCostPropertyFilter(AggregateFilter):
    """
    Filters by aggregate cost. Any given aggregate must have an aggregate cost
    below the given threshold.
    For any pair of aggregates, the sum of their aggregate costs must be below
    the threshold as well.

    Notes
    -----
    Always the last aggregate cost entry is considered.
    If an aggregate has no aggregate cost assigned, it is considered to have a
    aggregate cost of 1e12. This corresponds to +inf in the Pathfinder logic.
    """

    def __init__(self, max_aggregate_cost: float) -> None:
        """
        Construct the filter with the maximum cost.

        Parameters
        ----------
        max_aggregate_cost : float
            The threshold for the allowed overall aggregate costs of one or the
            sum of two aggregates.
        """
        super().__init__()
        self._max_aggregate_cost = max_aggregate_cost
        # remembers the last aggregate_one to save time in bimolecular loop
        self._partial_cache: Tuple[db.ID, float] = (db.ID(), 0.0)

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        cid = aggregate_one.id()
        if cid == self._partial_cache[0]:
            cc_one = self._partial_cache[1]
        else:
            cc_one = self._get_aggregate_cost(aggregate_one)
            self._partial_cache = (cid, cc_one)
        # Unimolecular case
        if aggregate_two is None:
            return cc_one < self._max_aggregate_cost
        # Bimolecular case
        cc_two = self._get_aggregate_cost(aggregate_two)
        return cc_one + cc_two < self._max_aggregate_cost

    def _get_aggregate_cost(self, aggregate: Union[db.Compound, db.Flask]) -> float:
        centroid = db.Structure(aggregate.get_centroid(), self._structures)
        property_list = centroid.get_properties("compound_cost")
        # Return +inf for aggregate cost has no aggregate_cost yet
        if len(property_list) < 1:
            return 1e12
        # # # Pick last entry of list
        prop = db.NumberProperty(property_list[-1], self._properties)
        return prop.get_data()


class CompleteSubstructureFilter(AggregateFilter):
    """
    Matches an aggregate if a given substructure is completely present in it.
    If multiple substructures are given, any match suffices.
    """

    def __init__(self, file_or_directory_with_files: str, require_both_match_bimolecular: bool = False,
                 exclude_mode: bool = False) -> None:
        """
        Parameters
        ----------
        file_or_directory_with_files : str
            The path to a file or directory containing files with substructures.
            The files may be in .xyz, .mol, .cbor or .bson format and are read in by Molassembler.
        require_both_match_bimolecular : bool
            If True, both compounds of a bimolecular reaction have to match a substructure.
            If False, one match suffices.
        exclude_mode : bool
            If True, the filter will exclude compounds that match the substructure.
            If False, the filter will only include compounds that match the substructure.
        """
        super().__init__()
        self._require_both_match_bimolecular = require_both_match_bimolecular
        self._exclude_mode = exclude_mode
        self._molecules: List[masm.Molecule] = []
        self._aggregate_to_molecules_map: Dict[str, List[masm.Molecule]] = {}
        if path.isdir(file_or_directory_with_files):
            for file_path, _, files in walk(file_or_directory_with_files):
                for file_name in files:
                    if file_name.endswith(".xyz") or file_name.endswith(".mol") or file_name.endswith(".cbor") \
                            or file_name.endswith(".bson"):
                        # todo replace once Molassembler has been updated
                        # self._molecules.extend(masm.io.split(str(path.expanduser(path.join(file_path, file_name)))))
                        ac, bo = utils.io.read(str(path.expanduser(path.join(file_path, file_name))))
                        if bo.empty():
                            bo = utils.BondDetector.detect_bonds(ac)
                        self._molecules.extend(masm.interpret.molecules(
                            ac, bo, masm.interpret.BondDiscretization.RoundToNearest).molecules)
        elif path.isfile(file_or_directory_with_files):
            # todo replace once Molassembler has been updated
            # self._molecules = masm.io.split(str(path.expanduser(file_or_directory_with_files)))
            ac, bo = utils.io.read(str(path.expanduser(file_or_directory_with_files)))
            if bo.empty():
                bo = utils.BondDetector.detect_bonds(ac)
            self._molecules = masm.interpret.molecules(
                ac, bo, masm.interpret.BondDiscretization.RoundToNearest).molecules
        else:
            raise ValueError(f"Given path '{file_or_directory_with_files}' is neither a file nor a directory.")
        for m in self._molecules:
            m.canonicalize()

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        if aggregate_two is None:
            return self._check_if_substructure(aggregate_one)
        if self._require_both_match_bimolecular:
            return self._check_if_substructure(aggregate_one) and self._check_if_substructure(aggregate_two)
        return self._check_if_substructure(aggregate_one) or self._check_if_substructure(aggregate_two)

    def _check_if_substructure(self, aggregate: Union[db.Compound, db.Flask]) -> bool:
        aggregate_molecules = self._get_molecules(aggregate)
        for aggregate_molecule in aggregate_molecules:
            for molecule in self._molecules:
                if aggregate_molecule.graph.V < molecule.graph.V:
                    # no need to check if the aggregate is smaller
                    continue
                if masm.subgraphs.complete(molecule, aggregate_molecule,
                                           edge_strictness=masm.subgraphs.EdgeStrictness.BondType):
                    return not self._exclude_mode
        return self._exclude_mode

    def _get_molecules(self, aggregate: Union[db.Compound, db.Flask]) -> List[masm.Molecule]:
        if self.currently_caches():
            saved_molecules = self._aggregate_to_molecules_map.get(aggregate.get_id().string())
            if saved_molecules is not None:
                return saved_molecules
        molecules = deserialize_molecules(db.Structure(aggregate.get_centroid(), self._structures))
        self._aggregate_to_molecules_map[aggregate.get_id().string()] = molecules
        return molecules

    def disable_caching(self) -> None:
        super().disable_caching()
        self._aggregate_to_molecules_map = {}


class HasStructureWithModel(AggregateFilter):
    """
    Returns true if the aggregate has a structure with the given electronic structure model.
    """

    def __init__(self, model: db.Model, check_only_energies: bool = False) -> None:
        """
        Construct the filter with the model.

        Parameters
        ----------
        model : db.Model
            The electronic structure model.
        check_only_energies : bool
            If true, the aggregate must only contain a structure with an energy
            that was calculated with the given electronic structure model.
        """
        super().__init__()
        self._model = model
        self._check_only_energies = check_only_energies
        self._can_cache: bool = False
        self._currently_caches: bool = False

    def _check_aggregate(self, aggregate: Union[db.Compound, db.Flask]) -> bool:
        if self._check_only_energies:
            return any(get_energy_for_structure(db.Structure(s_id), "electronic_energy", self._model, self._structures,
                                                self._properties) is not None for s_id in aggregate.get_structures())
        else:
            return any(db.Structure(s_id, self._structures).get_model() == self._model
                       for s_id in aggregate.get_structures())

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        if not self._check_aggregate(aggregate_one):
            return False
        if aggregate_two is not None and not self._check_aggregate(aggregate_two):
            return False
        return True


class StopDuringExploration(AggregateFilter):
    """
    This filter returns false as long as calculations with a given job order are on status hold, new, or pending.
    """

    def __init__(self, orders_to_wait_for: Optional[List[str]] = None, model: Optional[db.Model] = None) -> None:
        """
        Parameters
        ----------
        orders_to_wait_for : Optional[List[str]]
            The list of job orders to wait for. If none is given,
            ["scine_react_complex_nt2", "scine_single_point"] is used as default.
        model : Optional[db.Model]
            The electronic structure model associated to the calculations.
            If none is given, any calculations with the given
            job orders and status will make this filter return false.
            Otherwise, the model of the calculation must match this model.
        """
        super().__init__()
        if orders_to_wait_for is None:
            orders_to_wait_for = ["scine_react_complex_nt2", "scine_single_point", "scine_hessian"]
        self._orders_to_wait_for = orders_to_wait_for
        self._model = model
        self._can_cache: bool = False
        self._currently_caches: bool = False

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        selection = {
            "$and": [
                {"status": {"$in": ["hold", "new", "pending"]}},
                {"job.order": {"$in": self._orders_to_wait_for}}
            ]
        }
        if self._model is not None:
            selection["$and"] += model_query(self._model)
        return self._calculations.get_one_calculation(dumps(selection)) is None


class OnePotentialEnergySurface(AggregateFilter):
    """
    This filter ensures that all aggregates were generated during the exploration on the same potential energy surface
    (no constraints on the spin state). This condition is ensured by the following conditions:
    1. The total charge is conserved.
    2. The total sum formula is conserved.
    3. Aggregates must have been produced by the same reaction in the case of multiple aggregates, i.e., they are
    associated with the same reaction and occur on the same side of the reaction.
    """

    def __init__(self, total_charge: int, element_counts: Dict[str, int]) -> None:
        """
        Construct the filter with the PES definition.

        Parameters
        ----------
        total_charge : int
            The total charge of the system.
        element_counts : Dict[str, int]
            A dictionary with the element symbol and the number of atom of this type in the system.
        """
        super().__init__()
        if total_charge is None or element_counts is None:
            raise ValueError("Either a reference structure or the total charge and element counts must be provided.")
        self.__ref_charge: int = total_charge
        self.__ref_element_composition: Dict[str, int] = element_counts

    @classmethod
    def from_structure(cls, structure: db.Structure) -> 'OnePotentialEnergySurface':
        """
        Create the filter with a structure as reference to obtain the element counts and total charge.
        Note: This factory methods avoids using the db.Structure object in the __init__ to ensure compatibility with
        Heron.

        Parameters
        ----------
        structure : db.Structure
            The reference structure.
        """
        ref_charge = structure.get_charge()
        ref_element_composition = get_elements_in_structure(structure)
        return cls(ref_charge, ref_element_composition)

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        centroid_one = db.Structure(aggregate_one.get_centroid(), self._structures)
        element_counts = get_elements_in_structure(centroid_one)
        charge = centroid_one.get_charge()
        has_joined_reaction = True
        if aggregate_two is not None:
            centroid_two = db.Structure(aggregate_two.get_centroid(), self._structures)
            element_counts = combine_element_counts(get_elements_in_structure(centroid_two), element_counts)
            charge += centroid_two.get_charge()
            reactions_one = set([r_id.string() for r_id in aggregate_one.get_reactions()])
            reactions_two = set([r_id.string() for r_id in aggregate_two.get_reactions()])
            joined_reactions = reactions_one & reactions_two
            has_joined_reaction = len(reactions_one) + len(reactions_two) == 0  # the molecules are starting aggregates.
            for reaction_str_id in joined_reactions:
                reaction = db.Reaction(db.ID(reaction_str_id), self._reactions)
                lhs, rhs = reaction.get_reactants(db.Side.BOTH)
                has_joined_reaction = ((aggregate_one.id() in lhs and aggregate_two.id() in lhs)
                                       or (aggregate_one.id() in rhs and aggregate_two.id() in rhs))
        return charge == self.__ref_charge and element_counts == self.__ref_element_composition and has_joined_reaction


class ActivatedAggregateFilter(AggregateFilter):
    """
    Filter all aggregates that have the exploration or analysis disabled flag.

    This filter is not particularly useful for the exploration itself but may help in filtering results in the GUI.
    """

    def __init__(self) -> None:
        super().__init__()
        # The exploration status of the aggregates may/will constantly be changed by the kinetics gear.
        self._can_cache = False

    def _filter_impl(self, aggregate_one: Union[db.Compound, db.Flask],
                     aggregate_two: Optional[Union[db.Compound, db.Flask]] = None) -> bool:
        if aggregate_two is None:
            return aggregate_one.explore() and aggregate_one.analyze()
        return aggregate_one.explore() and aggregate_one.analyze() and \
            aggregate_two.explore() and aggregate_two.analyze()
