#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import Dict, List, Optional
from collections import Counter
import numpy as np

# Third party imports
import scine_database as db
import scine_utilities as utils

from scine_chemoton.gears import HoldsCollections


class CompoundFilter(HoldsCollections):
    """
    The base and default class for all filters. The default is to allow/pass all
    given checks.

    CompoundFilters are optional barriers in Chemoton that allow the user to cut down
    the exponential growth of the combinatorial explosion. The different
    subclasses of this main CompoundFilter allow for a tailored choice of which
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
        self._required_collections = ["calculations", "compounds", "elementary_steps",
                                      "properties", "structures", "reactions"]

    def __and__(self, o):
        if not isinstance(o, CompoundFilter):
            raise TypeError("CompoundFilter expects CompoundFilter "
                            "(or derived class) to chain with.")
        return CompoundFilterAndArray([self, o])

    def __or__(self, o):
        if not isinstance(o, CompoundFilter):
            raise TypeError("CompoundFilter expects CompoundFilter "
                            "(or derived class) to chain with.")
        return CompoundFilterOrArray([self, o])

    def filter(self, _: db.Compound, __: db.Compound = None) -> bool:
        """
        The blueprint for a filter function, checking if both of the compounds
        are allowed to be used in an exploration (logical and). If only one
        compound is given a check using only that one compound is performed.
        This default CompoundFilter accepts all compounds.

        Attributes
        ----------
        _ :: db.Compound
            The compound to be checked.
        _ :: db.Compound
            The other compound to be checked in the case of bimolecular reactions.

        Returns
        -------
        result :: bool
            True if the compound passed the check/filter, False if the compound
            did not pass the check and should not be used.
        """
        return True


class CompoundFilterAndArray(CompoundFilter):
    """
    An array of logically 'and' connected filters.

    Attributes
    ----------
    filters :: List[CompoundFilter]
        A list of filters to be combined.
    """

    def __init__(self, filters: Optional[List[CompoundFilter]] = None):
        super().__init__()
        if filters is None:
            filters = []
        self._filters = filters
        for f in filters:
            if not isinstance(f, CompoundFilter):
                raise TypeError("CompoundFilterAndArray expects CompoundFilter "
                                "(or derived class) to chain with.")

    def filter(self, compound_one: db.Compound, compound_two: db.Compound = None) -> bool:
        ret = True
        for f in self._filters:
            ret = ret and f.filter(compound_one, compound_two)
        return ret


class CompoundFilterOrArray(CompoundFilter):
    """
    An array of logically 'or' connected filters.

    Attributes
    ----------
    filters :: List[CompoundFilter]
        A list of filters to be combined.
    """

    def __init__(self, filters: List[CompoundFilter] = None):
        super().__init__()
        if filters is None:
            filters = []
        self._filters = filters
        for f in filters:
            if not isinstance(f, CompoundFilter):
                raise TypeError("CompoundFilterOrArray expects CompoundFilter "
                                "(or derived class) to chain with.")

    def filter(self, compound_one: db.Compound, compound_two: db.Compound = None) -> bool:
        ret = False
        for f in self._filters:
            ret = ret or f.filter(compound_one, compound_two)
        return ret


class ElementCountFilter(CompoundFilter):
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
    structures :: db.Collection
        The collection of structures to be used for the counting of elements.
    """

    def __init__(self, atom_type_count: Dict[str, int], structures: db.Collection):
        super().__init__()
        self.counts: Counter = Counter()
        for k, v in atom_type_count.items():
            self.counts.update({k.capitalize(): v})
        self._structures = structures

    def filter(self, compound_one: db.Compound, compound_two: db.Compound = None) -> bool:
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


class ElementSumCountFilter(CompoundFilter):
    """
    Filters by atom counts. All given compounds must resolve to structures
    that together fall below the given threshold.

    Attributes
    ----------
    atom_type_count :: Dict[str,int]
        A dictionary giving the number (values) of allowed occurrences of each
        atom type (atom symbol string given as keys). Atom symbols not given
        as keys are interpreted as forbidden.
    structures :: db.Collection
        The collection of structures to be used for the counting of
        elements.
    """

    def __init__(self, atom_type_count: Dict[str, int], structures: db.Collection):
        super().__init__()
        self.counts: Counter = Counter()
        for k, v in atom_type_count.items():
            self.counts.update({k.capitalize(): v})
        self._structures = structures

    def filter(self, compound_one: db.Compound, compound_two: Optional[db.Compound] = None) -> bool:
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


class MolecularWeightFilter(CompoundFilter):
    """
    Filters by molecular weight. All given compounds must resolve to structures
    that separately fall below the given threshold. No assumptions about weights
    of possible combinations/products are made in this filter.

    Attributes
    ----------
    max_weight :: float
        The maximum weight to be allowed, given in unified atomic mass units (u).
        For example, dinitrogen has a weight of about 28 u.
    structures :: db.Collection
        The collection of structures to be used for the molecular weight
        calculations.
    """

    def __init__(self, max_weight: float, structures: db.Collection):
        super().__init__()
        self.max_weight = max_weight
        self._structures = structures

    def filter(self, compound_one: db.Compound, compound_two: db.Compound = None) -> bool:
        # One compound case
        if compound_two is None:
            structure = db.Structure(compound_one.get_centroid())
            return self._calculate_weight(structure) < self.max_weight
        # Two compounds case
        structure_one = db.Structure(compound_one.get_centroid())
        structure_two = db.Structure(compound_two.get_centroid())
        weight_one = self._calculate_weight(structure_one)
        weight_two = self._calculate_weight(structure_two)
        return weight_one < self.max_weight and weight_two < self.max_weight

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


class IDFilter(CompoundFilter):
    """
    Filters by compound id. Returns true only for the specified compounds.
    Used for testing purposes.

    Attributes
    ----------
    reactive_ids :: List[db.ID]
        The IDs of the compounds to be considered as reactive.
    """

    def __init__(self, ids: List[db.ID]):
        super().__init__()
        self.reactive_ids = set(id.string() for id in ids)

    def filter(self, compound_one: db.Compound, compound_two: db.Compound = None) -> bool:
        # Get compound ids
        compound_ids = set()
        compound_ids.add(compound_one.get_id().string())
        if compound_two is not None:
            compound_ids.add(compound_two.get_id().string())
        return compound_ids.issubset(self.reactive_ids)


class SelfReactionFilter(CompoundFilter):
    """
    Filters out bimolecular reactions of compounds with themselves.
    """

    def filter(self, compound_one: db.Compound, compound_two: db.Compound = None) -> bool:
        if compound_two is None:
            # Do not rule out unimolecular reactions
            return True
        # Get compound ids
        compound_one_id = compound_one.get_id().string()
        compound_two_id = compound_two.get_id().string()
        return compound_one_id != compound_two_id


class TrueMinimumFilter(CompoundFilter):
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
    structures :: db.Collection
        The collection of structures to be used to be checked for being a true
        minimum.
    properties :: db.Collection
        The collection of properties to be used to look up frequencies of
        structures.
    imaginary_frequency_threshold :: float
        The frequency in atomic units above which a structure is considered a
        minimum structure.
        For example, a molecule with one imaginary frequency of -1e-4 (-138 cm^-1)
        can be considered a minimum by setting the threshold to -1.1e-4 (-152 cm^-1)
    """

    def __init__(
        self,
        structures: db.Collection,
        properties: db.Collection,
        imaginary_frequency_threshold: float = 0.0,
    ):
        super().__init__()
        self._structures = structures
        self._properties = properties
        self.imaginary_frequency_threshold = imaginary_frequency_threshold

    def filter(self, compound_one: db.Compound, compound_two: db.Compound = None) -> bool:
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


class CatalystFilter(CompoundFilter):
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
    structures :: db.Collection
        The collection of structures to be used for the counting of elements.
    restrict_unimolecular_to_catalyst :: bool
        Whether unimolecular reactions should also be limited to the catalyst.
    """

    def __init__(self, atom_type_count: Dict[str, int], structures: db.Collection,
                 restrict_unimolecular_to_catalyst: bool = False):
        super().__init__()
        self.counts: Counter = Counter()
        for k, v in atom_type_count.items():
            self.counts.update({k.capitalize(): v})
        self._structures = structures
        self._restrict_unimolecular_to_catalyst = restrict_unimolecular_to_catalyst

    def filter(self, compound_one: db.Compound, compound_two: db.Compound = None) -> bool:
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


class AtomNumberFilter(CompoundFilter):
    """
    Filters out all compounds with a total number of atoms larger than the given
    value. For multiple compounds the total number of atoms has to be equal or
    smaller than the given value.

    Attributes
    ----------
    max_n_atoms :: int
        The maximum number of allowed atoms.
    structures :: db.Collection
        The collection of structures to be used for the counting of elements.
    """

    def __init__(self, max_n_atoms: int, structures: db.Collection):
        super().__init__()
        self.max_n_atoms = max_n_atoms
        self._structures = structures

    def filter(self, compound_one: db.Compound, compound_two: db.Compound = None) -> bool:
        structure_one = db.Structure(compound_one.get_centroid())
        structure_one.link(self._structures)
        n_atoms = len(structure_one.get_atoms())
        if not (compound_two is None):
            structure_two = db.Structure(compound_two.get_centroid())
            structure_two.link(self._structures)
            n_atoms += len(structure_two.get_atoms())
        return n_atoms <= self.max_n_atoms


class OneCompoundIDFilter(CompoundFilter):
    """
    Filters all compounds that are not present on a given "white list" of IDs.
    In the case of multiple compounds, at least one has to be present in the
    list. Note that this is identical to the IDFilter in the case of only one
    compound.

    Attributes
    ----------
    reactive_ids :: List[db.ID]
        The IDs of the compounds to be considered as reactive.
    """

    def __init__(self, ids: List[db.ID]):
        super().__init__()
        self.reactive_ids = set(id.string() for id in ids)

    def filter(self, compound_one: db.Compound, compound_two: db.Compound = None) -> bool:
        if compound_one.get_id().string() in self.reactive_ids:
            return True
        if compound_two is not None:
            if compound_two.get_id().string() in self.reactive_ids:
                return True
        return False


class SelectedCompoundIDFilter(CompoundFilter):
    """
    Filters all compounds for which one needs to be one a given "white list"
    of reactive compounds and the other has to be either on a list of
    compounds of interest or on the list of reactive compounds.

    Attributes
    ----------
    reactive_ids :: List[db.ID]
        The IDs of the compounds to be considered as reactive.
    selected_ids :: List[db.ID]
        The IDs of the compounds to be of interest.
    """

    def __init__(self, reactive_ids: List[db.ID], selected_ids: List[db.ID]):
        super().__init__()
        self.reactive_ids = set(id.string() for id in reactive_ids)
        self.selected_ids = set(id.string() for id in selected_ids)

    def filter(self, compound_one: db.Compound, compound_two: db.Compound = None) -> bool:
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
