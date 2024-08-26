#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

# Third party imports
import numpy as np
import scine_database as db
import scine_utilities as utils

from scine_chemoton.gears import HoldsCollections, HasName


class _AbstractFilter(ABC):

    @abstractmethod
    def _filter_impl(self, structure_one: db.Structure,
                     structure_two: Union[db.Structure, None] = None) -> bool:
        pass


class StructureFilter(HoldsCollections, HasName, _AbstractFilter):
    """
    Structure filter provide a logic of filtering for specific structures within a larger set. For instance,
    these filters can be used to restrict computationally expensive calculations to some structures.
    """

    def __init__(self) -> None:
        super().__init__()
        self._remove_chemoton_from_name()
        self._required_collections = ["structures", "properties"]
        self._can_cache: bool = True
        self._currently_caches: bool = True
        self._cache: Dict[int, bool] = {}

    def __and__(self, o):
        if not isinstance(o, StructureFilter):
            raise TypeError("StructureFilter expects StructureFilter "
                            "(or derived class) to chain with.")
        return StructureFilterAndArray([self, o])

    def __or__(self, o):
        if not isinstance(o, StructureFilter):
            raise TypeError("StructureFilter expects StructureFilter "
                            "(or derived class) to chain with.")
        return StructureFilterOrArray([self, o])

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
    def _hash_structures(structure: db.Structure, optional_structure: Optional[db.Structure] = None) -> int:
        if optional_structure is not None:
            return hash(';'.join(sorted([structure.get_id().string(), optional_structure.get_id().string()])))
        else:
            return hash(structure.get_id().string())

    def filter(self, structure: db.Structure, optional_structure: Optional[db.Structure] = None) -> bool:
        """
        The filter function, checking if both of the Structures
        are allowed to be used in an exploration (logical and). If only one
        Structure is given a check using only that one Structure is performed.
        This default StructureFilter accepts all Structures.

        Parameters
        ----------
        structure : db.Structure
            The structure to be checked.
        optional_structure : db.Structure
            The other structure to be checked in the case of bimolecular operations.

        Returns
        -------
        result : bool
            True if the structure passed the check/filter, False if the structure
            did not pass the check and should not be used.
        """
        if self._can_cache and self._currently_caches:
            key = self._hash_structures(structure, optional_structure)
            result = self._cache.get(key)
            if result is None:
                result = self._filter_impl(structure, optional_structure)
                self._cache[key] = result
            return result
        else:
            return self._filter_impl(structure, optional_structure)

    def _filter_impl(self, _: db.Structure, __: Optional[db.Structure] = None) -> bool:
        """
        The blueprint for a filter implementation function.
        See public function for detailed description.
        This default StructureFilter accepts all Structures.

        Parameters
        ----------
        _ : db.Structure
            The Structure to be checked.
        __ : db.Structure
            The other Structure to be checked in the case of bimolecular reactions.

        Returns
        -------
        result : bool
            True if the Structure passed the check/filter, False if the Structure
            did not pass the check and should not be used.
        """
        return True


class StructureFilterAndArray(StructureFilter):
    """
    An array of logically 'and' connected filters.

    Parameters
    ----------
    filters : Optional[List[StructureFilter]]
        A list of filters to be combined.
    """

    def __init__(self, filters: Optional[List[StructureFilter]] = None) -> None:
        super().__init__()
        if filters is None:
            filters = []
        self._filters = filters
        for f in filters:
            if not isinstance(f, StructureFilter):
                raise TypeError("StructureFilterAndArray expects StructureFilter "
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

    def _filter_impl(self, structure_one: db.Structure, structure_two: Optional[db.Structure] = None) -> bool:
        return all(f.filter(structure_one, structure_two) for f in self._filters)

    def initialize_collections(self, manager: db.Manager) -> None:
        for f in self._filters:
            f.initialize_collections(manager)

    def __iter__(self):
        return (f for f in self._filters)

    def __setitem__(self, key, value):
        self._filters[key] = value


class StructureFilterOrArray(StructureFilter):
    """
    An array of logically 'or' connected filters.

    Parameters
    ----------
    filters : Optional[List[StructureFilter]]
        A list of filters to be combined.
    """

    def __init__(self, filters: Optional[List[StructureFilter]] = None) -> None:
        super().__init__()
        if filters is None:
            filters = []
        self._filters = filters
        for f in filters:
            if not isinstance(f, StructureFilter):
                raise TypeError("StructureFilterOrArray expects StructureFilter "
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

    def _filter_impl(self, structure_one: db.Structure, structure_two: Optional[db.Structure] = None) -> bool:
        return any(f.filter(structure_one, structure_two) for f in self._filters)

    def initialize_collections(self, manager: db.Manager) -> None:
        for f in self._filters:
            f.initialize_collections(manager)

    def __iter__(self):
        return (f for f in self._filters)

    def __setitem__(self, key, value):
        self._filters[key] = value


class StructureLabelFilter(StructureFilter):
    """
    Filter structures by their label.
    """

    def __init__(self, labels: List[db.Label]) -> None:
        super().__init__()
        self._can_cache: bool = False
        self._allowed_labels: List[db.Label] = []
        for label in labels:
            if isinstance(label, str):
                # todo replace with a classmethod implemented in scine_database
                compare_label = label.lower()
                for potential_label, value in db.Label.__members__.items():
                    if potential_label.lower() == compare_label:
                        self._allowed_labels.append(value)
                        break
                else:
                    raise ValueError(f"Unknown label '{label}'")
            elif isinstance(label, db.Label):
                self._allowed_labels.append(label)
            else:
                raise TypeError(f"Expected str or Label, got {type(label)}")

    def _filter_impl(self, structure_one: db.Structure, structure_two: Optional[db.Structure] = None) -> bool:
        one_condition = structure_one.get_label() in self._allowed_labels
        if structure_two is not None:
            return one_condition and structure_two.get_label() in self._allowed_labels
        return one_condition


class ModelFilter(StructureFilter):
    """
    Filter structures by their model.
    """

    def __init__(self, model: db.Model) -> None:
        super().__init__()
        self._can_cache: bool = False
        self._model = model

    def _filter_impl(self, structure_one: db.Structure, structure_two: Optional[db.Structure] = None) -> bool:
        one_condition = structure_one.get_model() == self._model
        if structure_two is not None:
            return one_condition and structure_two.get_model() == self._model
        return one_condition


class ElementCountFilter(StructureFilter):
    """
    Filters by atom counts. All given structures must resolve to structures
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
        self.counts: Counter = Counter()
        self.allow_above_limit = allow_above_limit
        for k, v in atom_type_count.items():
            self.counts.update({utils.ElementInfo.element_from_symbol(k.capitalize()): v})
        # remembers the last compound_one to save time in bimolecular loop
        self._partial_cache: Tuple[db.ID, bool] = (db.ID(), False)

    def _filter_impl(self, structure_one: db.Structure,
                     structure_two: Optional[db.Structure] = None) -> bool:
        # One structure case
        if structure_two is None:
            return self._check_atom_counts(structure_one, False)
        # Two structures case
        return self._check_atom_counts(structure_one, True) and self._check_atom_counts(structure_two, False)

    def _check_atom_counts(self, structure: db.Structure, write_to_cache: bool) -> bool:
        """
        Checks the atom counts of the given structure against the requirements
        set in the member variable.

        Attributes
        ----------
        structure : db.Structure
            The structure of which to check the atom counts.
        write_to_cache : bool
            If true, the result of the check will be written to the cache.

        Returns
        -------
        result : bool
            True if the structure passed the check/filter, False if not.
        """
        if structure.id() == self._partial_cache[0]:
            return self._partial_cache[1]

        def evaluate() -> bool:
            this_count = Counter(structure.get_atoms().elements)
            if self.allow_above_limit and any(k not in this_count for k in self.counts.keys()):
                return False
            for k, v in this_count.items():
                count = self.counts.get(k, 0)
                if count - v < 0:
                    if not self.allow_above_limit:
                        return False
                elif count - v == 0:
                    pass
                elif self.allow_above_limit:
                    return False
            return True

        ret = evaluate()
        if write_to_cache:
            self._partial_cache = (structure.id(), ret)
        return ret


class ElementSumCountFilter(StructureFilter):
    """
    Filters by atom counts. All given structures must resolve to structures
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
        self.counts: Counter = Counter()
        self.allow_above_limit = allow_above_limit
        for k, v in atom_type_count.items():
            self.counts.update({utils.ElementInfo.element_from_symbol(k.capitalize()): v})
        # remembers the last compound_one to save time in bimolecular loop
        self._partial_cache: Tuple[db.ID, Counter] = (db.ID(), Counter())

    def _filter_impl(self, structure_one: db.Structure,
                     structure_two: Optional[db.Structure] = None) -> bool:
        # One structure case
        if structure_two is None:
            counter_tot = Counter(structure_one.get_atoms().elements)
        # Two structures case
        else:
            # first counter
            if structure_one.id() == self._partial_cache[0]:
                counter_one = self._partial_cache[1]
            else:
                counter_one = Counter(structure_one.get_atoms().elements)
                self._partial_cache = (structure_one.id(), counter_one)

            # second counter
            counter_two = Counter(structure_two.get_atoms().elements)
            counter_tot = counter_one + counter_two

        if self.allow_above_limit and any(k not in counter_tot for k in self.counts.keys()):
            return False

        for k, v in counter_tot.items():
            count = self.counts.get(k, 0)
            if count - v < 0:
                if not self.allow_above_limit:
                    return False
            elif count - v == 0:
                pass
            elif self.allow_above_limit:
                return False
        return True


class MolecularWeightFilter(StructureFilter):
    """
    Filters by molecular weight. All given structures must resolve to structures
    that separately fall below the given threshold. No assumptions about weights
    of possible combinations/products are made in this filter.
    """

    def __init__(self, max_weight: float, allow_additions_above_limit: bool = True) -> None:
        """
        Construct the filter with all options.

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
        self.max_weight = max_weight
        self.allow_additions_above_limit = allow_additions_above_limit
        self._weight_cache: Dict[str, float] = {}

    def _filter_impl(self, structure_one: db.Structure,
                     structure_two: Optional[db.Structure] = None) -> bool:
        cid_one = structure_one.get_id().string()
        # One structure case
        if structure_two is None:
            weight = self._weight_cache.get(cid_one)
            if weight is None:
                weight = self._calculate_weight(structure_one)
                self._weight_cache = {cid_one: weight}
            return weight < self.max_weight
        # Two structures case
        cid_two = structure_two.get_id().string()
        weight_one = self._weight_cache.get(cid_one)
        if weight_one is None:
            weight_one = self._calculate_weight(structure_one)
        weight_two = self._weight_cache.get(cid_two)
        if weight_two is None:
            weight_two = self._calculate_weight(structure_two)
        self._weight_cache = {
            cid_one: weight_one,
            cid_two: weight_two
        }
        if self.allow_additions_above_limit:
            return weight_one < self.max_weight and weight_two < self.max_weight
        else:
            return (weight_one + weight_two) < self.max_weight

    @staticmethod
    def _calculate_weight(structure: db.Structure) -> float:
        """
        Calculates the molecular weight, given a DB structure.

        Attributes
        ----------
        structure : db.Structure
            The structure of which to calculate the molecular weight.

        Returns
        -------
        weight : float
            The molecular weight in a.u. .
        """
        atoms = structure.get_atoms()
        weight = 0.0
        for e in atoms.elements:
            weight += utils.ElementInfo.mass(e)
        return weight


class TrueMinimumFilter(StructureFilter):
    """
    Filters by checking if structure fulfills requirement to be considered a
    true minimum. This includes having frequencies and all of them are above a
    given threshold.
    All given structures must have frequencies above the given threshold.
    Single atoms or ions are considered as minima automatically.
    Frequencies must be calculated with the ThermoGear during an exploration,
    otherwise all structures and combinations of them are filtered out.
    """

    property_name: str = "frequencies"

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
        self.imaginary_frequency_threshold = imaginary_frequency_threshold

    def _filter_impl(self, structure_one: db.Structure,
                     structure_two: Optional[db.Structure] = None) -> bool:
        # One structure case
        if structure_two is None:
            return self._frequency_check(structure_one)
        # Two structures case
        return self._frequency_check(structure_one) and self._frequency_check(structure_two)

    def _frequency_check(self, structure: db.Structure) -> bool:
        """
        Checks a structure for its validity as true minimum, meaning it has only
        frequencies above a set threshold.
        If no frequencies are available, it fails the check, unless it is a
        single atom.

        Attributes
        ----------
        structure : db.Structure
            The structure of which the frequencies are checked.

        Returns
        -------
        result : bool
            Boolean indicating if structure is true minimum.
        """
        # # # Check for single atom
        if len(structure.get_atoms()) == 1:
            return True

        # # # Check, if frequencies exist; if not, the structure will not be
        # # # considered for exploration
        if not structure.has_property(self.property_name):
            return False
        # # # Get Frequencies
        freq_id = structure.get_property(self.property_name)
        freq = self._properties.get_vector_property(freq_id)
        freq.link(self._properties)

        # # # Check, if there is a frequency below the threshold
        return not np.any(freq.get_data() < self.imaginary_frequency_threshold)


class CatalystFilter(StructureFilter):
    """
    Filters by defining an important structural motive.
    A user defined set of atoms present in a structure identify it to be the, or
    a version of the catalyst. This simple check should work great for common
    transition metal catalysts.

    Only specific reactions revolving around a catalytic cycle are then allowed:
    i) reactions that feature the catalyst and one other structure that is not
    the catalyst.
    ii) reactions that only involve a single structure (not restricted to the
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
        self.counts: Counter = Counter()
        for k, v in atom_type_count.items():
            self.counts.update({utils.ElementInfo.element_from_symbol(k.capitalize()): v})
        # remembers the last compound_one to save time in bimolecular loop
        self._partial_cache: Tuple[db.ID, bool] = (db.ID(), False)
        self._restrict_unimolecular_to_catalyst = restrict_unimolecular_to_catalyst

    def _filter_impl(self, structure_one: db.Structure,
                     structure_two: Optional[db.Structure] = None) -> bool:
        # One structure case
        if structure_two is None:
            if self._restrict_unimolecular_to_catalyst:
                return self._check_if_catalyst(structure_one, False)
            return True
        # Two structures case
        return self._check_if_catalyst(structure_one, True) != self._check_if_catalyst(structure_two, False)

    def _check_if_catalyst(self, structure: db.Structure, write_to_cache: bool) -> bool:
        """
        Check if the given structure is the or a version of the catalyst.

        Parameters
        ----------
        structure : db.Structure
            The structure of which to check.
        write_to_cache : bool
            If the structure id should be written to the partial cache.

        Returns
        -------
        check : bool
            True is the structure is a version of the catalyst.
        """
        if structure.id() == self._partial_cache[0]:
            return self._partial_cache[1]

        def evaluate() -> bool:
            actual = Counter(structure.get_atoms().elements)
            for k, v in self.counts.items():
                if actual.get(k, 0) != v:
                    return False
            return True

        ret = evaluate()
        if write_to_cache:
            self._partial_cache = (structure.id(), ret)
        return ret


class ChargeCombinationFilter(StructureFilter):
    """
    Avoid combination of two structures that both have negative charges or both have positive charges.
    """

    def _filter_impl(self, structure_one: db.Structure,
                     structure_two: Optional[db.Structure] = None) -> bool:
        if structure_two is None:
            return True
        charge_one = structure_one.charge
        charge_two = structure_two.charge
        both_plus = charge_one > 0 and charge_two > 0
        both_minus = charge_one < 0 and charge_two < 0
        if both_minus or both_plus:
            return False
        return True


class SpecificChargeFilter(StructureFilter):
    """
    Either allow or exclude structures with the specified charge(s).
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
        self._charges = charges
        self._allow = allow
        self._both_charges_must_be_valid = both_charges_must_be_valid

    def _filter_impl(self, structure_one: db.Structure,
                     structure_two: Optional[db.Structure] = None) -> bool:
        charge_one = structure_one.charge
        one_included = charge_one in self._charges
        if structure_two is None:
            return one_included if self._allow else not one_included
        charge_two = structure_two.charge
        two_included = charge_two in self._charges
        if (one_included and two_included) or (not self._both_charges_must_be_valid and (one_included or two_included)):
            return self._allow
        return not self._allow


class AtomNumberFilter(StructureFilter):
    """
    Filters out all structures with a total number of atoms larger than the given
    value. For multiple structures the total number of atoms has to be equal or
    smaller than the given value.
    The optional minimum limit is also defined as equal or smaller, i.e., 1 allows structures with 1 atom.
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
        self.max_n_atoms = max_n_atoms
        self.min_n_atoms = min_n_atoms
        # remembers the last structure_one to save time in bimolecular loop
        self._partial_cache: Tuple[db.ID, int] = (db.ID(), 0)

    def _filter_impl(self, structure_one: db.Structure,
                     structure_two: Optional[db.Structure] = None) -> bool:
        sid = structure_one.id()
        if sid == self._partial_cache[0]:
            n_atoms = self._partial_cache[1]
        else:
            n_atoms = len(structure_one.get_atoms())
            self._partial_cache = (sid, n_atoms)
        if structure_two is not None:
            n_atoms += len(structure_two.get_atoms())
        return self.min_n_atoms <= n_atoms <= self.max_n_atoms
