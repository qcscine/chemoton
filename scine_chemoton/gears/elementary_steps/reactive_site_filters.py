#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import ast
from typing import List, Optional, Tuple, Union

# Third party imports
import scine_database as db

# Local application imports
from .reaction_rules.distance_rules import DistanceRuleSet
from .reaction_rules.element_rules import ElementRuleSet, SimpleElementCombinationRule
from .reaction_rules.polarization_rules import PolarizationRuleSet
from scine_chemoton.gears import HoldsCollections, HasName
from scine_chemoton.utilities.masm import deserialize_molecules, distinct_atoms, get_atom_pairs


class ReactiveSiteFilter(HoldsCollections, HasName):
    """
    The base and default class for all reactive site filters. The default is to
    allow/pass all given checks.

    ReactiveSiteFilter are optional barriers in Chemoton that allow the user to
    cut down the exponential growth of the combinatorial explosion. The
    different subclasses of this main ReactiveSiteFilter allow for a tailored
    choice of which parts of a molecule to deem reactive.

    There are some predefined filters that will be given in Chemoton, however,
    it should be simple to extend as needed even on a per-project basis.
    The key element when sub-classing this interface is to override the `filter`
    functions as defined here. When sub-classing please be aware that these
    filters are expected to be called often. Having each call do loops over
    entire collections is not wise.

    For the latter reason, user defined subclasses are intended to be more
    complex, allowing for non-database stored/cached data across a run.
    This can be a significant speed-up and allow for more intricate filtering.

    The different filter methods are applied in a subsequent matter, i.e.,
    only the atoms that pass the atom filter will be used to construct atom pairs,
    and only those of the atom pairs that pass the pair filter will be used to
    construct trial reaction coordinates in the TrialGenerators.

    NOTE: Although there is the possibility to apply a filter to many sites /
    trial coordinates simultaneously, there is no guarantee that all possible
    sites / coordinates with the specified settings are passed at the same time.
    """

    def __init__(self):
        super().__init__()
        self._remove_chemoton_from_name()
        self._required_collections = ["calculations", "compounds", "elementary_steps", "flasks",
                                      "properties", "structures", "reactions"]

    def __and__(self, o):
        if not isinstance(o, ReactiveSiteFilter):
            raise TypeError("ReactiveSiteFilter expects ReactiveSiteFilter "
                            "(or derived class) to chain with.")
        return ReactiveSiteFilterAndArray([self, o])

    def __or__(self, o):
        if not isinstance(o, ReactiveSiteFilter):
            raise TypeError("ReactiveSiteFilter expects ReactiveSiteFilter "
                            "(or derived class) to chain with.")
        return ReactiveSiteFilterOrArray([self, o])

    def filter_atoms(self, _: List[db.Structure], atom_indices: List[int]) -> List[int]:
        """
        The blueprint for a filter function, checking  a list of atoms
        regarding their reactivity as defined by the filter.

        Parameters
        ----------
        _ : List[db.Structure]
            The structures to be checked. Unused in this function.
        atom_indices : [List[int]]
            The list of atoms to consider. If several structures are listed
            atom indices are expected to refer to the entity of all structure
            in the order they are given in the structure list.
            For example, the first atom of the second structure has the index
            equalling the number of atoms of the first structure.

        Returns
        -------
        result : List[int]
            The list of all relevant atom indices after applying the filter.
        """
        return atom_indices

    def filter_atom_pairs(
            self, _: List[db.Structure], pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        The blueprint for a filter function, checking a list of atom pairs
        regarding their reactivity as defined by the filter.

        Parameters
        ----------
        _ : List[db.Structure]
            The structures to be checked. Unused in this implementation.
        pairs : List[Tuple[int, int]]
            The list of atom pairs to consider. If several structures are listed
            atom indices are expected to refer to the entity of all structure in
            the order they are given in the structure list.
            For example, the first atom of the second structure has the index
            equalling the number of atoms of the first structure.

        Returns
        -------
        result :: List[Tuple[int, int]]
            The list of all relevant reactive atom pairs (given as atom index
            pairs) after applying the filter.
        """
        return pairs

    def filter_reaction_coordinates(
            self, _: List[db.Structure], coordinates: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[int, int]]]:
        """
        The blueprint for a filter function, checking  a list of trial reaction
        coordinates each given as a tuple of reactive atom pairs for their
        reactivity as defined by the filter.

        Parameters
        ----------
        _ : List[db.Structure]
            The structures to be checked. Unused in this implementation.
        coordinates : List[List[Tuple[int, int]]]
            The list of trial reaction coordinates to consider.
            If several structures are listed atom indices are expected to refer
            to the entity of all structure in the order they are given in the
            structure list.
            For example, the first atom of the second structure has the index
            equalling the number of atoms of the first structure.

        Returns
        -------
        result :: List[List[Tuple[int, int]]]
            The list of all relevant reaction coordinates given as tuples of
            reactive atom pairs after applying the filter.
        """
        return coordinates


class ReactiveSiteFilterAndArray(ReactiveSiteFilter):
    """
    An array of logically 'and' connected filters.

    Attributes
    ----------
    filters : List[ReactiveSiteFilter]
        A list of filters to be combined.
    """

    def __init__(self, filters: Optional[List[ReactiveSiteFilter]] = None, **kwargs):
        super().__init__(**kwargs)  # required for multiple inheritance
        if filters is None:
            filters = []
        self._filters = filters
        for f in filters:
            if not isinstance(f, ReactiveSiteFilter):
                raise TypeError("ReactiveSiteFilterAndArray expects ReactiveSiteFilter "
                                "(or derived class) to chain with.")
        self._join_names(self._filters)

    def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
        distinct = atom_indices
        for filter in self._filters:
            distinct = filter.filter_atoms(structure_list, distinct)

        return distinct

    def filter_atom_pairs(
            self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        distinct = pairs
        for filter in self._filters:
            distinct = filter.filter_atom_pairs(structure_list, pairs=distinct)
        return distinct

    def filter_reaction_coordinates(
            self, structure_list: List[db.Structure], coordinates: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[int, int]]]:
        distinct = coordinates
        for filter in self._filters:
            distinct = filter.filter_reaction_coordinates(structure_list, coordinates=distinct)
        return distinct

    def initialize_collections(self, manager: db.Manager) -> None:
        for f in self._filters:
            f.initialize_collections(manager)

    def __iter__(self):
        return (f for f in self._filters)


class ReactiveSiteFilterOrArray(ReactiveSiteFilter):
    """
    An array of logically 'or' connected filters.

    Attributes
    ----------
    filters : List[ReactiveSiteFilter]
        A list of filters to be combined.
    """

    def __init__(self, filters: Optional[List[ReactiveSiteFilter]] = None):
        super().__init__()
        if filters is None:
            filters = []
        self._filters = filters
        for f in filters:
            if not isinstance(f, ReactiveSiteFilter):
                raise TypeError("ReactiveSiteFilterOrArray expects ReactiveSiteFilter "
                                "(or derived class) to chain with.")
        self._join_names(self._filters)

    def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
        ret: List[int] = []
        for filter in self._filters:
            distinct = filter.filter_atoms(structure_list, atom_indices)
            ret = list(set(ret) | set(distinct))
        return ret

    def filter_atom_pairs(
            self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        ret: List[Tuple[int, int]] = []
        for filter in self._filters:
            distinct = filter.filter_atom_pairs(structure_list, pairs=pairs)
            ret = list(set(ret) | set(distinct))
        return ret

    def filter_reaction_coordinates(
            self, structure_list: List[db.Structure], coordinates: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[int, int]]]:
        ret: List[List[Tuple[int, int]]] = []
        for filter in self._filters:
            distinct = filter.filter_reaction_coordinates(structure_list, coordinates=coordinates)
            for d in distinct:
                if d not in ret:
                    ret.append(d)
        return ret

    def initialize_collections(self, manager: db.Manager) -> None:
        for f in self._filters:
            f.initialize_collections(manager)

    def __iter__(self):
        return (f for f in self._filters)


class MasmChemicalRankingFilter(ReactiveSiteFilter):
    """
    Filters atoms by chemical ranking as determined by the Molassembler graph
    representation.
    NOTE: Only atoms that pass the atom filtering are considered for the
    generation of reactive pairs and trial reaction coordinates.
    Pairs/trial coordinates built from different atoms of the same molecule that
    are regarded as alike based on the Molassembler graph representation, can,
    however, be distinct/valid. However, these will not be generated when this
    filter is applied. Use with care!
    NOTE: This filter does not have any specific filter methods for pairs and
    reaction coordinates. The methods of the `ReactiveSiteFilter` base class
    will be applied.

    Attributes
    ----------
    prune : str
        Whether to prune by molassembler's ranking distinct atoms descriptor.
        Allowed values: `'None'`, `'Hydrogen'`, `'All'`
    """

    def __init__(self, prune: str = "None"):
        super().__init__()
        self.prune = prune
        valid_option_values = ["None", "Hydrogen", "All"]
        if prune not in valid_option_values:
            raise RuntimeError(f"Option for masm atom pruning invalid: {prune}")

    def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:

        if self.prune == "None":
            return atom_indices
        distinct = []
        idx_shift = 0  # To account for the shifting of indices in structures further down the structure_list
        for structure in structure_list:
            atom_map = ast.literal_eval(structure.get_graph("masm_idx_map"))
            mols = deserialize_molecules(structure)
            for c, m in enumerate(mols):
                mol_distinct = distinct_atoms(m, self.prune == "Hydrogen")
                atom_distinct = [atom_map.index((c, i)) + idx_shift for i in mol_distinct]
                distinct.extend(atom_distinct)
            idx_shift += structure.get_atoms().size()

        return list(set(atom_indices) & set(distinct))


class SimpleRankingFilter(ReactiveSiteFilter):
    """
    Filters atoms and bonds by a simple ranking algorithm.
    All atom types are assigned a basic rank/importance, with ``H`` being rank
    0, ``C`` being rank 1, ``['N', 'O', 'S', 'P', 'Si']`` being rank 2 and all
    other elements being rank 3. Based on these initial rankings an importance
    of atoms and bonds is calculated.

    For atoms the base rank of them self and all bonded atoms is added to give
    the final importance. Hence, a carbon in CH_4 would rank as 1 and one in
    CH3OH would rank as 3. The protons in this example would rank 1 and 1, 2.

    For atom pairs and coordinates, the ranking of the atoms in the bond is
    simply added.

    Attributes
    ----------
    atom_threshold : int
        The threshold for the importance of atoms. All atoms ranking above the
        threshold will be considered for reactions.
    pair_threshold : int
        The threshold for the importance of atom pairs, All pairs ranking above
        the threshold will be considered for reactions.
    coordinate_threshold : int
        The threshold for the importance of trial reaction coordinates, All
        reaction coordinates ranking above the threshold will be considered
        for reactions.
    """

    def __init__(self, atom_threshold: int = 0, pair_threshold: int = 0, coordinate_threshold: int = 0):
        super().__init__()
        self.atom_threshold = atom_threshold
        self.pair_threshold = pair_threshold
        self.coordinate_threshold = coordinate_threshold

    def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
        idx_shift = 0  # To account for the shifting of indices in structures further down the structure_list
        above_threshold = []
        for structure in structure_list:
            neighbors = get_atom_pairs(structure, (1, 1), "None")
            initial_ranks = self._rank_atoms(structure)
            ranks = [r for r in initial_ranks]
            for i, j in neighbors:
                ranks[i] += initial_ranks[j]
                ranks[j] += initial_ranks[i]
            for i, rank in enumerate(ranks):
                if rank > self.atom_threshold:
                    above_threshold.append(i + idx_shift)
            idx_shift += structure.get_atoms().size()

        return list(set(atom_indices) & set(above_threshold))

    def filter_atom_pairs(
            self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        all_atom_ranks = []  # Concatenated atom ranks in the order of the structures
        for structure in structure_list:
            neighbors = get_atom_pairs(structure, (1, 1), "None")
            initial_ranks = self._rank_atoms(structure)
            atom_ranks = [r for r in initial_ranks]
            for i, j in neighbors:
                atom_ranks[i] += initial_ranks[j]
                atom_ranks[j] += initial_ranks[i]

            all_atom_ranks += atom_ranks

        above_threshold = []
        for i, j in pairs:
            if (all_atom_ranks[i] + all_atom_ranks[j]) > self.pair_threshold:
                above_threshold.append((i, j))

        return above_threshold

    def filter_reaction_coordinates(
            self, structure_list: List[db.Structure], coordinates: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[int, int]]]:
        all_atom_ranks = []  # Concatenated atom ranks in the order of the structures
        for structure in structure_list:
            neighbors = get_atom_pairs(structure, (1, 1), "None")
            initial_ranks = self._rank_atoms(structure)
            atom_ranks = [r for r in initial_ranks]
            for i, j in neighbors:
                atom_ranks[i] += initial_ranks[j]
                atom_ranks[j] += initial_ranks[i]

            all_atom_ranks += atom_ranks

        above_threshold = []
        for coord in coordinates:
            if sum((all_atom_ranks[pair[0]] + all_atom_ranks[pair[1]]) for pair in coord) > self.coordinate_threshold:
                above_threshold.append(coord)

        return above_threshold

    def _rank_atoms(self, structure: db.Structure):
        atoms = structure.get_atoms()
        elements = [str(x) for x in atoms.elements]
        ranks = []
        for e in elements:
            if e == "H":
                ranks.append(0)
            elif e == "C":
                ranks.append(1)
            elif e in ["N", "O", "S", "P", "Si"]:
                ranks.append(2)
            else:
                ranks.append(3)
        return ranks


class AtomRuleBasedFilter(ReactiveSiteFilter):
    """
    A filter that only classifies atoms as reactive if they correspond to a given rule.
    The rules are given for each element. They may be an exact, required distance to another
    atom/element, or they may label an atom as generally reactive or not reactive. If no rules
    are given for an element, all atoms of this type will be considered as non-reactive independent of exclude_mode\n

    Example rules::

      reactive_site_rules = {
        'H': DistanceRuleOrArray([SimpleDistanceRule('O', 3), SimpleDistanceRule('N', 1)]),
        'C': DistanceRuleOrArray([SimpleDistanceRule('O', 2), SimpleDistanceRule('N', 2)]),
        'O': True,
        'N': True
      }

    Allow reactions for all O and N, for all H that are exactly in a distance of 3 bonds to O or one bond
    to N, and for all C that are at most two bonds away from O or N.
    """

    def __init__(self, rules: Union[DistanceRuleSet, dict], exclude_mode=False):
        """
        Parameters
        ----------
        rules :: Union[DistanceRuleSet, dict]
            The dictionary containing the rules (vide supra). The given dictionary is checked for correct typing.
        exclude_mode :: bool
            If true, all atoms are excluded that correspond to the given rules.
        """
        super().__init__()
        if not isinstance(rules, DistanceRuleSet):
            self._rules = DistanceRuleSet(rules)
        else:
            self._rules = rules
        self._return_value_upon_hit = not exclude_mode

    def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
        reactive_atom_indices = []
        idx_shift = 0
        for structure in structure_list:
            atoms = structure.get_atoms()
            elements = [str(x) for x in atoms.elements]
            molecules = deserialize_molecules(structure)
            idx_map = ast.literal_eval(structure.get_graph("masm_idx_map"))
            for i, e in enumerate(elements):
                # If no rule is specified, set the atom to be not reactive.
                if e not in self._rules:
                    continue
                rule = self._rules[e]
                if rule.filter_by_rule(molecules, idx_map, elements, i) == self._return_value_upon_hit:
                    reactive_atom_indices.append(i + idx_shift)
            idx_shift += structure.get_atoms().size()
        return list(set(atom_indices) & set(reactive_atom_indices))


class ElementWiseReactionCoordinateFilter(ReactiveSiteFilter):
    """
    A filter that can restrict the combination of atoms with a specific element.
    The filter can be operated in two modes: Allow all reaction coordinates for the
    element combinations encoded in the rules or forbid all of them.\n
    Example rules::

      reaction_coordinate_rules = {
        'H': ['H', 'C']
      }

    In the default "forbid-mode" these rules mean that no reaction coordinates between two H-atoms and H and C are
    allowed.
    """

    def __init__(self, rules: Union[ElementRuleSet, dict], reactive_if_rules_apply: bool = False):
        """
        Parameters
        ----------
        rules : Union[ElementRuleSet, dict]
            The dictionary containing the rules. The given dictionary is checked for correct typing.
        reactive_if_rules_apply : bool
            The mode to operate the filter in. If true, only reaction coordinates in the given rule set pass.
            If false, no reaction coordinate in the given rule set pass. By default, false.
        """
        super().__init__()
        if not isinstance(rules, ElementRuleSet):
            self._rules = ElementRuleSet(rules, rule_type=SimpleElementCombinationRule)
        else:
            self._rules = rules
        self._reactive_if_rules_apply = reactive_if_rules_apply

    def filter_atom_pairs(
            self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        all_element_symbols = []
        for structure in structure_list:
            atoms = structure.get_atoms()
            elements = [str(x) for x in atoms.elements]
            all_element_symbols += elements

        valid_pairs = []
        for pair in pairs:
            rule_applies = self._any_rule_applies(all_element_symbols[pair[0]], all_element_symbols[pair[1]])
            if self._reactive_if_rules_apply == rule_applies:
                valid_pairs.append(pair)
        return valid_pairs

    def _any_rule_applies(self, element_1: str, element_2: str):
        if element_1 in self._rules:
            if self._rules[element_1].rule_applies(element_1, element_2):
                return True
        return element_2 in self._rules and self._rules[element_2].rule_applies(element_1, element_2)


class HeuristicPolarizationReactionCoordinateFilter(ReactiveSiteFilter):
    """
    Reaction coordinates are only allowed that combine polarizations '+' and '-', or '+-' with either '+' or '-'.
    The polarizations are defined by the given PolarizationRuleSet

    Example rules::

      rules = {
        'H': [PolarizationFunctionalGroupRule("+", 2, 'C', (3, 3), {'N': 1, 'O': 1})],
        'C': PaulingElectronegativityRule(),
        'N': PaulingElectronegativityRule(),
        'O': PaulingElectronegativityRule()
      }

    """

    def __init__(self, rules: Union[PolarizationRuleSet, dict]):
        """
        Parameters
        ----------
        rules : Union[PolarizationRuleSet, dict]
            The dictionary containing the rules. The given dictionary is checked for correct typing.
        """
        super().__init__()
        if not isinstance(rules, PolarizationRuleSet):
            self._rules = PolarizationRuleSet(rules)
        else:
            self._rules = rules

    def filter_atom_pairs(
            self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        all_polarization_characters = []
        for structure in structure_list:
            all_polarization_characters += self._get_characters_for_structure(structure)

        valid_pairs = []
        for pair in pairs:
            character_0 = all_polarization_characters[pair[0]]
            character_1 = all_polarization_characters[pair[1]]
            plus_minus = '+' in character_0 and '-' in character_1
            minus_plus = '-' in character_0 and '+' in character_1
            # Valid atom pairs must combine + and - signs.
            if plus_minus or minus_plus:
                valid_pairs.append(pair)
        return valid_pairs

    def _get_characters_for_structure(self, structure: db.Structure):
        atoms = structure.get_atoms()
        elements = [str(x) for x in atoms.elements]
        molecules = deserialize_molecules(structure)
        idx_map = ast.literal_eval(structure.get_graph("masm_idx_map"))
        string_list = []
        for atom_index, element in enumerate(elements):
            if element not in self._rules:
                string_list.append('')
            else:
                rule = self._rules[element]
                string_list.append(rule.string_from_rule(molecules, idx_map, elements, atom_index))
        return string_list
