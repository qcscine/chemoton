#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import Dict, List, Optional, Union
from collections import Counter
import numpy as np

# Third party imports
import scine_database as db
import scine_utilities as utils

from scine_chemoton.gears import HoldsCollections, HasName
from ..kinetic_modeling.concentration_query_functions import query_concentration_with_object


class AggregateFilter(HoldsCollections, HasName):
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

    def __init__(self):
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
        result :: bool
            True if the filter is currently set to cache results, and return
            results from cache if possible.
        """
        return self._currently_caches

    def can_cache(self) -> bool:
        """
        Returns
        -------
        result :: bool
            True if the filter can possibly cache results.
        """
        return self._can_cache

    @staticmethod
    def _hash_compounds(compound: db.Compound, optional_compound: Optional[db.Compound] = None) -> int:
        if optional_compound is not None:
            return hash(';'.join(sorted([compound.get_id().string(), optional_compound.get_id().string()])))
        else:
            return hash(compound.get_id().string())

    def filter(self, compound: db.Compound, optional_compound: Optional[db.Compound] = None) -> bool:
        """
        The filter function, checking if both of the compounds
        are allowed to be used in an exploration (logical and). If only one
        compound is given a check using only that one compound is performed.
        This default AggregateFilter accepts all compounds.

        Attributes
        ----------
        compound :: db.Compound
            The compound to be checked.
        optional_compound :: db.Compound
            The other compound to be checked in the case of bimolecular reactions.

        Returns
        -------
        result :: bool
            True if the compound passed the check/filter, False if the compound
            did not pass the check and should not be used.
        """
        if self._can_cache and self._currently_caches:
            key = self._hash_compounds(compound, optional_compound)
            result = self._cache.get(key)
            if result is None:
                result = self._filter_impl(compound, optional_compound)
                self._cache[key] = result
            return result
        else:
            return self._filter_impl(compound, optional_compound)

    def _filter_impl(self, _: db.Compound, __: Optional[db.Compound] = None) -> bool:
        """
        The blueprint for a filter implementation function.
        See public function for detailed description.
        This default AggregateFilter accepts all compounds.

        Attributes
        ----------
        _ :: db.Compound
            The compound to be checked.
        __ :: db.Compound
            The other compound to be checked in the case of bimolecular reactions.

        Returns
        -------
        result :: bool
            True if the compound passed the check/filter, False if the compound
            did not pass the check and should not be used.
        """
        return True


class AggregateFilterAndArray(AggregateFilter):
    """
    An array of logically 'and' connected filters.

    Attributes
    ----------
    filters :: List[AggregateFilter]
        A list of filters to be combined.
    """

    def __init__(self, filters: Optional[List[AggregateFilter]] = None):
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

    def _filter_impl(self, compound_one: db.Compound, compound_two: Optional[db.Compound] = None) -> bool:
        ret = True
        for f in self._filters:
            ret = ret and f.filter(compound_one, compound_two)
        return ret

    def initialize_collections(self, manager: db.Manager) -> None:
        for f in self._filters:
            f.initialize_collections(manager)

    def __iter__(self):
        return (f for f in self._filters)


class AggregateFilterOrArray(AggregateFilter):
    """
    An array of logically 'or' connected filters.

    Attributes
    ----------
    filters :: List[AggregateFilter]
        A list of filters to be combined.
    """

    def __init__(self, filters: Optional[List[AggregateFilter]] = None):
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

    def _filter_impl(self, compound_one: db.Compound, compound_two: Optional[db.Compound] = None) -> bool:
        ret = False
        for f in self._filters:
            ret = ret or f.filter(compound_one, compound_two)
        return ret

    def initialize_collections(self, manager: db.Manager) -> None:
        for f in self._filters:
            f.initialize_collections(manager)

    def __iter__(self):
        return (f for f in self._filters)


class ElementCountFilter(AggregateFilter):
    """
    Filters by atom counts. All given compounds must resolve to structures
    that separately fall below the given threshold. No assumptions about atom
    counts of possible combinations/products are made in this filter.

    Attributes
    ----------
    atom_type_count :: Dict[str,int]
        A dictionary giving the number (values) of allowed occurrences of each
        atom type (atom symbol string given as keys). Atom symbols not given
        as keys are interpreted as forbidden.
    """

    def __init__(self, atom_type_count: Dict[str, int]):
        super().__init__()
        self.counts: Counter = Counter()
        for k, v in atom_type_count.items():
            self.counts.update({k.capitalize(): v})

    def _filter_impl(self, compound_one: db.Compound, compound_two: Optional[db.Compound] = None) -> bool:
        # One compound case
        if compound_two is None:
            structure = db.Structure(compound_one.get_centroid())
            return self._check_atom_counts(structure)
        # Two compounds case
        structure_one = db.Structure(compound_one.get_centroid())
        structure_two = db.Structure(compound_two.get_centroid())
        return self._check_atom_counts(structure_one) and self._check_atom_counts(structure_two)

    def _check_atom_counts(self, structure: db.Structure) -> bool:
        """
        Checks the atom counts of the given structure against the requirements
        set in the member variable.

        Attributes
        ----------
        structure :: db.Structure
            The structure of which to check the atom counts.

        Returns
        -------
        result :: bool
            True if the structure passed the check/filter, False if not.
        """
        structure.link(self._structures)
        atoms = structure.get_atoms()
        elements = [str(x) for x in atoms.elements]
        for k, v in Counter(elements).items():
            if self.counts.get(k, 0) - v < 0:
                return False
        return True


class ElementSumCountFilter(AggregateFilter):
    """
    Filters by atom counts. All given compounds must resolve to structures
    that together fall below the given threshold.

    Attributes
    ----------
    atom_type_count :: Dict[str,int]
        A dictionary giving the number (values) of allowed occurrences of each
        atom type (atom symbol string given as keys). Atom symbols not given
        as keys are interpreted as forbidden.
    """

    def __init__(self, atom_type_count: Dict[str, int]):
        super().__init__()
        self.counts: Counter = Counter()
        for k, v in atom_type_count.items():
            self.counts.update({k.capitalize(): v})

    def _filter_impl(self, compound_one: db.Compound, compound_two: Optional[db.Compound] = None) -> bool:
        # One compound case
        if compound_two is None:
            structure = db.Structure(compound_one.get_centroid())
            counter_tot = self._get_element_counter(structure)
        # Two compounds case
        else:
            structure_one = db.Structure(compound_one.get_centroid())
            counter_one = self._get_element_counter(structure_one)

            structure_two = db.Structure(compound_two.get_centroid())
            counter_two = self._get_element_counter(structure_two)

            counter_tot = counter_one + counter_two

        for k, v in counter_tot.items():
            if self.counts.get(k, 0) - v < 0:
                return False

        return True

    def _get_element_counter(self, structure: db.Structure) -> Counter:
        """
        Gets the element counter of the given structure.

        Attributes
        ----------
        structure :: db.Structure
            The structure of which to get the atom counter.

        Returns
        -------
        result :: Counter
            The Counter of the elements of the given structure.
        """

        structure.link(self._structures)
        atoms = structure.get_atoms()
        elements = [str(x) for x in atoms.elements]

        return Counter(elements)


class MolecularWeightFilter(AggregateFilter):
    """
    Filters by molecular weight. All given compounds must resolve to structures
    that separately fall below the given threshold. No assumptions about weights
    of possible combinations/products are made in this filter.

    Attributes
    ----------
    max_weight :: float
        The maximum weight to be allowed, given in unified atomic mass units (u).
        For example, dinitrogen has a weight of about 28 u.
    allow_additions_above_limit :: bool
        If true only checks if the reactants are each above the given limit.
        If false, assumes complete additions may happen and restricts all
        combinations where the sum of weights is above the given limit.
    """

    def __init__(self, max_weight: float, allow_additions_above_limit: bool = True):
        super().__init__()
        self.max_weight = max_weight
        self.allow_additions_above_limit = allow_additions_above_limit
        self._weight_cache: Dict[str, float] = {}

    def _filter_impl(self, compound_one: db.Compound, compound_two: Optional[db.Compound] = None) -> bool:
        cid_one = compound_one.get_id().string()
        # One compound case
        if compound_two is None:
            weight = self._weight_cache.get(cid_one)
            if weight is None:
                structure = db.Structure(compound_one.get_centroid())
                weight = self._calculate_weight(structure)
                self._weight_cache = {cid_one: weight}
            return weight < self.max_weight
        # Two compounds case
        cid_two = compound_two.get_id().string()
        weight_one = self._weight_cache.get(cid_one)
        if weight_one is None:
            structure_one = db.Structure(compound_one.get_centroid())
            weight_one = self._calculate_weight(structure_one)
        weight_two = self._weight_cache.get(cid_two)
        if weight_two is None:
            structure_two = db.Structure(compound_two.get_centroid())
            weight_two = self._calculate_weight(structure_two)
        self._weight_cache = {
            cid_one: weight_one,
            cid_two: weight_two
        }
        if self.allow_additions_above_limit:
            return weight_one < self.max_weight and weight_two < self.max_weight
        else:
            return (weight_one + weight_two) < self.max_weight

    def _calculate_weight(self, structure: db.Structure) -> float:
        """
        Calculates the molecular weight, given a DB structure.

        Attributes
        ----------
        structure :: db.Structure
            The structure of which to calculate the molecular weight.

        Returns
        -------
        weight :: float
            The molecular weight in a.u. .
        """
        structure.link(self._structures)
        atoms = structure.get_atoms()
        weight = 0.0
        for e in atoms.elements:
            weight += utils.ElementInfo.mass(e)
        return weight


class IdFilter(AggregateFilter):
    """
    Filters by compound id. Returns true only for the specified compounds.
    Used for testing purposes.

    Attributes
    ----------
    reactive_ids :: List[str]
        The IDs of the compounds to be considered as reactive.
    """

    def __init__(self, ids: List[str]):
        super().__init__()
        self.reactive_ids = set(ids)

    def _filter_impl(self, compound_one: db.Compound, compound_two: Optional[db.Compound] = None) -> bool:
        # Get compound ids
        compound_ids = set()
        compound_ids.add(compound_one.get_id().string())
        if compound_two is not None:
            compound_ids.add(compound_two.get_id().string())
        return compound_ids.issubset(self.reactive_ids)


class SelfReactionFilter(AggregateFilter):
    """
    Filters out bimolecular reactions of compounds with themselves.
    """

    def _filter_impl(self, compound_one: db.Compound, compound_two: Optional[db.Compound] = None) -> bool:
        if compound_two is None:
            # Do not rule out unimolecular reactions
            return True
        # Get compound ids
        compound_one_id = compound_one.get_id().string()
        compound_two_id = compound_two.get_id().string()
        return compound_one_id != compound_two_id


class TrueMinimumFilter(AggregateFilter):
    """
    Filters by checking if compound fulfills requirement to be considered a
    true minimum. This includes having frequencies and all of them are above a
    given threshold.
    All given compounds must resolve to structures that separately only have
    frequencies above the given threshold. Single atoms or ions are considered
    as minima automatically.
    Frequencies must be calculated with the ThermoGear during an exploration,
    otherwise all compounds and combinations of them are filtered out.

    Attributes
    ----------
    imaginary_frequency_threshold :: float
        The frequency in atomic units above which a structure is considered a
        minimum structure.
        For example, a molecule with one imaginary frequency of -1e-4 (-138 cm^-1)
        can be considered a minimum by setting the threshold to -1.1e-4 (-152 cm^-1)
    """

    def __init__(self, imaginary_frequency_threshold: float = 0.0):
        super().__init__()
        self.imaginary_frequency_threshold = imaginary_frequency_threshold

    def _filter_impl(self, compound_one: db.Compound, compound_two: Optional[db.Compound] = None) -> bool:
        # One compound case
        if compound_two is None:
            structure = db.Structure(compound_one.get_centroid())
            return self._frequency_check(structure)
        # Two compounds case
        structure_one = db.Structure(compound_one.get_centroid())
        structure_two = db.Structure(compound_two.get_centroid())
        # Frequency Check
        freq_check_one = self._frequency_check(structure_one)
        freq_check_two = self._frequency_check(structure_two)
        return freq_check_one and freq_check_two

    def _frequency_check(self, structure: db.Structure) -> bool:
        """
        Checks a structure for its validity as true minimum, meaning it has only
        frequencies above a set threshold.
        If no frequencies are available, it fails the check, unless it is a
        single atom.

        Attributes
        ----------
        structure :: db.Structure
            The structure of which the frequencies are checked.

        Returns
        -------
        result :: bool
            Boolean indicating if structure is true minimum.
        """
        structure.link(self._structures)
        # # # Check for single atom
        if len(structure.get_atoms()) == 1:
            return True

        # # # Check, if frequencies exist; if not, the compound will not be
        # # # considered for exploration
        if not structure.has_property("frequencies"):
            return False
        # # # Get Frequencies
        freq_id = structure.get_property("frequencies")
        freq = self._properties.get_vector_property(freq_id)
        freq.link(self._properties)

        # # # Check, if there is a frequency below the threshold
        return not np.any(freq.get_data() < self.imaginary_frequency_threshold)


class CatalystFilter(AggregateFilter):
    """
    Filters by defining an important structural motive.
    A user defined set of atoms present in a structure identify it to be the, or
    a version of the catalyst. This simple check should work great for common
    transition metal catalysts.

    Only specific reactions revolving around a catalytic cycle are then allowed:
    i) reactions that feature the catalyst and one other compound that is not
    the catalyst.
    ii) reactions that only involve a single compound (not restricted to the
    catalyst, unless specified otherwise with flag (see parameters))

    Attributes
    ----------
    atom_type_count :: Dict[str,int]
        A dictionary giving the number (values) of atoms that are expected to be
        only present in - and thus defining - the catalyst. Atom type (atom
        symbol strings) are given as keys. Atom symbols not given are considered
        to be irrelevant to the check and may be present in the catalyst. In
        order to ban atoms, set their count to zero.
    restrict_unimolecular_to_catalyst :: bool
        Whether unimolecular reactions should also be limited to the catalyst.
    """

    def __init__(self, atom_type_count: Dict[str, int], restrict_unimolecular_to_catalyst: bool = False):
        super().__init__()
        self.counts: Counter = Counter()
        for k, v in atom_type_count.items():
            self.counts.update({k.capitalize(): v})
        self._restrict_unimolecular_to_catalyst = restrict_unimolecular_to_catalyst

    def _filter_impl(self, compound_one: db.Compound, compound_two: Optional[db.Compound] = None) -> bool:
        # One compound case
        if compound_two is None:
            if self._restrict_unimolecular_to_catalyst:
                return self._check_if_catalyst(db.Structure(compound_one.get_centroid()))
            return True
        # Two compounds case
        structure_one = db.Structure(compound_one.get_centroid())
        structure_two = db.Structure(compound_two.get_centroid())
        return self._check_if_catalyst(structure_one) != self._check_if_catalyst(structure_two)

    def _check_if_catalyst(self, structure: db.Structure) -> bool:
        """
        Check if the given structure is the or a version of the catalyst.

        Attributes
        ----------
        structure :: db.Structure
            The structure of which to check.

        Returns
        -------
        check :: bool
            True is the structure is a version of the catalyst.
        """
        structure.link(self._structures)
        atoms = structure.get_atoms()
        elements = [str(x) for x in atoms.elements]
        actual = Counter(elements)
        for k, v in self.counts.items():
            if actual.get(k, 0) != v:
                return False
        return True


class AtomNumberFilter(AggregateFilter):
    """
    Filters out all compounds with a total number of atoms larger than the given
    value. For multiple compounds the total number of atoms has to be equal or
    smaller than the given value.

    Attributes
    ----------
    max_n_atoms :: int
        The maximum number of allowed atoms.
    """

    def __init__(self, max_n_atoms: int):
        super().__init__()
        self.max_n_atoms = max_n_atoms

    def _filter_impl(self, compound_one: db.Compound, compound_two: Optional[db.Compound] = None) -> bool:
        structure_one = db.Structure(compound_one.get_centroid())
        structure_one.link(self._structures)
        n_atoms = len(structure_one.get_atoms())
        if not (compound_two is None):
            structure_two = db.Structure(compound_two.get_centroid())
            structure_two.link(self._structures)
            n_atoms += len(structure_two.get_atoms())
        return n_atoms <= self.max_n_atoms


class OneAggregateIdFilter(AggregateFilter):
    """
    Filters all compounds that are not present on a given "white list" of IDs.
    In the case of multiple compounds, at least one has to be present in the
    list. Note that this is identical to the IdFilter in the case of only one
    compound.

    Attributes
    ----------
    reactive_ids :: List[str]
        The IDs of the compounds to be considered as reactive.
    """

    def __init__(self, ids: List[str]):
        super().__init__()
        self.reactive_ids = set(ids)

    def _filter_impl(self, compound_one: db.Compound, compound_two: Optional[db.Compound] = None) -> bool:
        if compound_one.get_id().string() in self.reactive_ids:
            return True
        if compound_two is not None:
            if compound_two.get_id().string() in self.reactive_ids:
                return True
        return False


class SelectedAggregateIdFilter(AggregateFilter):
    """
    Filters all compounds for which one needs to be one a given "white list"
    of reactive compounds and the other has to be either on a list of
    compounds of interest or on the list of reactive compounds.

    Attributes
    ----------
    reactive_ids :: List[str]
        The IDs of the compounds to be considered as reactive.
    selected_ids :: Optional[List[str]]
        The IDs of the compounds to be of interest.
    """

    def __init__(self, reactive_ids: List[str], selected_ids: Optional[List[str]] = None):
        super().__init__()
        self.reactive_ids = set(reactive_ids)
        if selected_ids is None:
            selected_ids = []
        self.selected_ids = set(selected_ids)

    def _filter_impl(self, compound_one: db.Compound, compound_two: Optional[db.Compound] = None) -> bool:
        # One compound case: the id has to be a member of the reactive_ids.
        # Two compound case: one of the ids has to be a member of the reactive_ids.
        #                   The other one has to be either a member of the reactive_ids or the selected_ids.
        one_is_reactive = compound_one.get_id().string() in self.reactive_ids
        # One compound case
        if compound_two is None:
            return one_is_reactive
        # Two compound case
        two_is_reactive = compound_two.get_id().string() in self.reactive_ids
        one_is_selected = compound_one.get_id().string() in self.selected_ids or one_is_reactive
        two_is_selected = compound_two.get_id().string() in self.selected_ids or two_is_reactive
        return (one_is_reactive and two_is_selected) or (two_is_reactive and one_is_selected)


class ConcentrationPropertyFilter(AggregateFilter):
    def __init__(self, property_labels: List[str], min_value: float, filter_only_pairs: bool, structures: db.Collection,
                 properties: db.Collection):
        """
        Filters all compounds that have a concentration lower than a given value. For the two compound case,
        the product of both concentrations has to be larger than this value. The filter may be set to only
        filter for the two-compound case.

        Parameters
        ----------
        property_label :: str
            The label for the concentration to filter with.
        min_value :: float
            The minimum concentration/concentration product.
        filter_only_pairs :: bool
            If true, the filter will always return True for the single compound case.
        structures :: db.Collection
            The structure collection.
        properties :: db.Collection
            The property collection.
        """
        super().__init__()
        self._property_labels = property_labels
        self._min_value = min_value
        self._properties = properties
        self._structures = structures
        self._filter_only_paris = filter_only_pairs
        self._can_cache: bool = False
        self._currently_caches: bool = False

    def _filter_impl(self, compound: db.Compound, optional_compound: Optional[db.Compound] = None) -> bool:
        if self._filter_only_paris and optional_compound is None:
            return True
        concentration = self._get_concentration(compound)
        if optional_compound is not None:
            concentration *= self._get_concentration(optional_compound)
        if concentration > self._min_value:
            return True
        return False

    def _get_concentration(self, compound: db.Compound):
        concentrations: List[float] = list()
        for label in self._property_labels:
            concentration = query_concentration_with_object(label, compound, self._properties, self._structures)
            concentrations.append(concentration)
        return max(concentrations)


class ChargeCombinationFilter(AggregateFilter):
    def __init__(self, structures: db.Collection):
        """
        Forbids the combination of compounds with the same sign, non-zero charge.

        Parameters
        ----------
        structures :: db.Collection
            The structure collection.
        """
        super().__init__()
        self._structures = structures

    def _filter_impl(self, compound: db.Compound, optional_compound: Optional[db.Compound] = None) -> bool:
        if optional_compound is None:
            return True
        charge_one = db.Structure(compound.get_centroid(), self._structures).charge
        charge_two = db.Structure(optional_compound.get_centroid(), self._structures).charge
        both_plus = charge_one > 0 and charge_two > 0
        both_minus = charge_one < 0 and charge_two < 0
        if both_minus or both_plus:
            return False
        return True


class LastKineticModelingFilter(AggregateFilter):
    def __init__(self, manager: db.Manager, kinetic_modeling_job_order: Optional[str] = None,
                 aggregate_settings_key: Optional[str] = None):
        """
        Allow only compounds that were handled in the last kinetic modeling calculation or have a non-zero start
        concentration.

        Parameters
        ----------
        manager : db.Manager
            The database manager.
        kinetic_modeling_job_order : Optional[str]
            The kinetic modeling job order. By default, "kinetx_kinetic_modeling".
        aggregate_settings_key : Optional[str]
            The key to the aggregate ids in the kinetic modeling job. By default, "aggregate_ids".
        """
        super().__init__()
        self._structures: db.Collection = manager.get_collection("structures")
        self._compounds: db.Collection = manager.get_collection("compounds")
        self._properties: db.Collection = manager.get_collection("properties")
        self._calculations: db.Collection = manager.get_collection("calculations")
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

    def _filter_impl(self, compound: db.Compound, optional_compound: Optional[db.Compound] = None) -> bool:
        if not self._start_structure:
            self._initialize_start_structure()
        self._update_aggregate_ids_in_last_job()
        if compound.id().string() not in self._aggregate_str_ids_in_last_job:
            start_concentration = query_concentration_with_object(
                "start_concentration", compound, self._properties, self._structures)
            if start_concentration <= 0.0:
                return False
        if optional_compound:
            if optional_compound.id().string() not in self._aggregate_str_ids_in_last_job:
                start_concentration = query_concentration_with_object(
                    "start_concentration", optional_compound, self._properties, self._structures)
                if start_concentration <= 0.0:
                    return False
        return True

    def _initialize_start_structure(self) -> None:
        for compound in self._compounds.iterate_all_compounds():
            compound.link(self._compounds)
            c_start = query_concentration_with_object(
                "start_concentration", compound, self._properties, self._structures)
            if c_start > 0.0:
                self._start_structure = db.Structure(compound.get_centroid(), self._structures)
                break
        if not self._start_structure:
            raise RuntimeError("LastKineticModelingFilter: No compound with a non-zero starting concentration is"
                               " given! This may prevent the exploration of further species and is not allowed.")

    def _update_aggregate_ids_in_last_job(self) -> None:
        assert self._start_structure
        if self._start_structure.has_calculations(self._kinetic_modeling_job_order):
            calc_ids = self._start_structure.get_calculations(self._kinetic_modeling_job_order)
            if len(calc_ids) == self._n_calculations_last:
                return
            for i, calc_id in enumerate(reversed(calc_ids)):
                calculation = db.Calculation(calc_id, self._calculations)
                if calculation.get_status() == db.Status.COMPLETE:
                    aggregate_str_ids = calculation.get_settings()[self._aggregate_settings_key]
                    self._aggregate_str_ids_in_last_job = aggregate_str_ids  # type: ignore
                    self._n_calculations_last = len(calc_ids) - i
                    break


class SelectedStructureIdFilter(SelectedAggregateIdFilter):
    """
    See SelectedAggregateIdFilter, but filters for compounds that include the given Structure IDs.
    """

    def _filter_impl(self, compound_one: db.Compound, compound_two: Optional[db.Compound] = None) -> bool:
        compound_one_structures = [str(sid) for sid in compound_one.get_structures()]
        one_is_reactive = any(sid in self.reactive_ids for sid in compound_one_structures)
        # One compound case
        if compound_two is None:
            return one_is_reactive
        compound_two_structures = [str(sid) for sid in compound_two.get_structures()]
        two_is_reactive = any(sid in self.reactive_ids for sid in compound_two_structures)
        one_is_selected = any(sid in self.selected_ids for sid in compound_one_structures)
        two_is_selected = any(sid in self.selected_ids for sid in compound_two_structures)
        return (one_is_reactive and two_is_selected) or (two_is_reactive and one_is_selected)


class CompoundCostPropertyFilter(AggregateFilter):
    """
    Filters by compound cost. Any given compound must have a compound cost
    below the given threshold.
    For any pair of compounds, the sum of their compound costs must be below
    the threshold as well.

    Attributes
    ----------
    max_compound_cost : float
            The threshold for the allowed overall compound costs of one or the
            sum of two compounds.

    Notes
    -----
    Always the last compound cost entry is considered.
    If a compound has no compound cost assigned, it is considered to have a
    compound cost of 1e12. This corresponds to +inf in the Pathfinder logic.
    """

    def __init__(self, max_compound_cost: float):
        super().__init__()
        self._max_compound_cost = max_compound_cost

    def _filter_impl(self, compound_one: db.Compound, compound_two: Optional[db.Compound] = None) -> bool:
        cc_one = self._get_compound_cost(compound_one)
        # Unimolecular case
        if compound_two is None and cc_one < self._max_compound_cost:
            return True
        # Bimolecular case
        if compound_two is not None:
            cc_two = self._get_compound_cost(compound_two)
            if cc_one + cc_two < self._max_compound_cost:
                return True
        return False

    def _get_compound_cost(self, compound: db.Compound):
        centroid = db.Structure(compound.get_centroid(), self._structures)
        property_list = centroid.get_properties("compound_cost")
        # Return +inf for compound cost has no compound_cost yet
        if len(property_list) < 1:
            return 1e12
        # # # Pick last entry of list
        prop = db.NumberProperty(property_list[-1], self._properties)
        return prop.get_data()
