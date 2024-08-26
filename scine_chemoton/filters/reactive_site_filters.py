#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import ast
import os
from abc import ABC, abstractmethod
from json import load
from typing import List, Optional, Tuple, Union
from itertools import combinations

# Third party imports
import scine_database as db
import scine_utilities as utils
from scine_art.io import load_file
from scine_art.molecules import maximum_matching_fragments

# Local application imports
from scine_chemoton.reaction_rules.distance_rules import DistanceRuleSet
from scine_chemoton.reaction_rules.element_rules import (
    ElementRuleSet,
    SimpleElementCombinationRule
)
from scine_chemoton.reaction_rules.polarization_rules import PolarizationRuleSet
from scine_chemoton.gears import HoldsCollections, HasName
from scine_chemoton.utilities.masm import deserialize_molecules, distinct_atoms, get_atom_pairs


class _AbstractSiteFilter(ABC):

    @abstractmethod
    def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
        pass

    @abstractmethod
    def filter_atom_pairs(
            self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def filter_reaction_coordinates(
            self, structure_list: List[db.Structure], coordinates: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[int, int]]]:
        pass

    @abstractmethod
    def supports_flasks(self) -> bool:
        pass


class ReactiveSiteFilter(HoldsCollections, HasName, _AbstractSiteFilter):
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

    def __init__(self) -> None:
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

    def supports_flasks(self) -> bool:
        return True

    def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
        """
        The blueprint for a filter function, checking a list of atoms
        regarding their reactivity as defined by the filter.

        Parameters
        ----------
        structure_list : List[db.Structure]
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
            self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        The blueprint for a filter function, checking a list of atom pairs
        regarding their reactivity as defined by the filter.

        Parameters
        ----------
        structure_list : List[db.Structure]
            The structures to be checked. Unused in this implementation.
        pairs : List[Tuple[int, int]]
            The list of atom pairs to consider. If several structures are listed
            atom indices are expected to refer to the entity of all structure in
            the order they are given in the structure list.
            For example, the first atom of the second structure has the index
            equalling the number of atoms of the first structure.

        Returns
        -------
        result : List[Tuple[int, int]]
            The list of all relevant reactive atom pairs (given as atom index
            pairs) after applying the filter.
        """
        return pairs

    def filter_reaction_coordinates(
            self, structure_list: List[db.Structure], coordinates: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[int, int]]]:
        """
        The blueprint for a filter function, checking  a list of trial reaction
        coordinates each given as a tuple of reactive atom pairs for their
        reactivity as defined by the filter.

        Parameters
        ----------
        structure_list : List[db.Structure]
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
        result : List[List[Tuple[int, int]]]
            The list of all relevant reaction coordinates given as tuples of
            reactive atom pairs after applying the filter.
        """
        return coordinates


class ReactiveSiteFilterAndArray(ReactiveSiteFilter):
    """
    An array of logically 'and' connected filters.
    """

    def __init__(self, filters: Optional[List[ReactiveSiteFilter]] = None, **kwargs) -> None:
        """
        Parameters
        ----------
        filters : Optional[List[ReactiveSiteFilter]]
            A list of filters to be combined.
        """
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

    def __setitem__(self, key, value):
        self._filters[key] = value

    def supports_flasks(self) -> bool:
        return all(f.supports_flasks() for f in self._filters)


class ReactiveSiteFilterOrArray(ReactiveSiteFilter):
    """
    An array of logically 'or' connected filters.
    """

    def __init__(self, filters: Optional[List[ReactiveSiteFilter]] = None) -> None:
        """
        Parameters
        ----------
        filters : Optional[List[ReactiveSiteFilter]]
            A list of filters to be combined.
        """
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

    def __setitem__(self, key, value):
        self._filters[key] = value

    def supports_flasks(self) -> bool:
        return all(f.supports_flasks() for f in self._filters)


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
    """

    def __init__(self, prune: str = "None") -> None:
        """
        Construct filter with pruning option.

        Parameters
        ----------
        prune : str
            Whether to prune by molassembler's ranking distinct atoms descriptor.
            Allowed values: `'None'`, `'Hydrogen'`, `'All'`
        """
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
    """

    def __init__(self, atom_threshold: int = 0, pair_threshold: int = 0, coordinate_threshold: int = 0) -> None:
        """
        Construct filter with all options.

        Parameters
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

    def __init__(self, rules: Union[DistanceRuleSet, dict], exclude_mode=False) -> None:
        """
        Parameters
        ----------
        rules : Union[DistanceRuleSet, dict]
            The dictionary containing the rules (vide supra). The given dictionary is checked for correct typing.
        exclude_mode : bool
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

    def __init__(self, rules: Union[ElementRuleSet, dict], reactive_if_rules_apply: bool = False) -> None:
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

    def __init__(self, rules: Union[PolarizationRuleSet, dict]) -> None:
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


class CentralSiteFilter(ReactiveSiteFilter):
    """
    This filter activates sites only on/around a central element, e.g. a metal center in an organometallic
    catalyst. The behavior is different for uni- and bimolecular reactions, e.g. if "C" is the central atom::

      Unimolecular:
      A-B-C-D
      C is reactive, B and D depends on set option, A is never reactive.
      E-F
      No atom is reactive.

      Bimolecular:
      A-B-C-D  +  E-F
      C is reactive, B and D depends on set option, A is never reactive, E and F are reactive.
      G-H  +  E-F
      No atom is reactive.

    Notes
    -----
    For efficiency, this filter by default works in combination of filter_atoms + filter_atom_pairs, the pair and
    coordinate filtering alone are insufficient, unless activated by the optional flag
    """

    def __init__(self,
                 central_atom: str,
                 ligand_without_central_atom_reactive: bool = False,
                 reevaluate_on_all_levels: bool = False) -> None:
        """
        Parameters
        ----------
        central_atom : str
            The central element, such as "Ru"
        ligand_without_central_atom_reactive : bool
            Whether atoms, that are directly bonded to the central element, should also be reactive. Default is False
        reevaluate_on_all_levels : bool
            Whether each filter stage (filter_atoms, filter_atom_pairs, filter_reaction_coordinates) should
            evaluate all levels below again. Default is False
        """
        super().__init__()
        self.central_atom = utils.ElementInfo.element_from_symbol(central_atom)
        self.ligand_without_central_atom_reactive = ligand_without_central_atom_reactive
        self.reevaluate_on_all_levels = reevaluate_on_all_levels

    def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
        idx_shift = 0  # To account for the shifting of indices in structures further down the structure_list
        reactives = []
        atoms_list = [structure.get_atoms() for structure in structure_list]
        structure_contains_central = [self.central_atom in atoms.elements for atoms in atoms_list]
        # if no structure contains element, we don't have reactive atoms
        if not any(structure_contains_central):
            return []
        for has_central, atoms, structure in zip(structure_contains_central, atoms_list, structure_list):
            if not has_central:
                # for bimolecular reactions we leave all atoms of the structures
                # that don't have the central element as reactive
                reactives += [idx + idx_shift for idx in range(len(atoms))]
            else:
                # structure has central atom -> limit to central + ligands
                neighbors = get_atom_pairs(structure, (1, 1))
                elements = atoms.elements
                for pair in neighbors:
                    if any(elements[p] == self.central_atom for p in pair):
                        shifted_pair = [p + idx_shift for p in pair]
                        reactives += [sp for sp in shifted_pair]
            idx_shift += len(atoms)

        return list(set(atom_indices) & set(reactives))

    def filter_atom_pairs(self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]) \
            -> List[Tuple[int, int]]:
        atoms_list = [structure.get_atoms() for structure in structure_list]
        all_elements = [atoms.elements for atoms in atoms_list]
        flat_all_elements = [item for sublist in all_elements for item in sublist]
        index_to_structure_index = self._structure_assignment_map(atoms_list)
        structure_contains_central = [self.central_atom in atoms.elements for atoms in atoms_list]

        if not self.reevaluate_on_all_levels:
            evaluate_pairs = pairs
        else:
            n = sum(len(atoms) for atoms in atoms_list)
            indices = list(range(n))
            reactive_atoms = self.filter_atoms(structure_list, indices)
            possible_pairs = list(combinations(reactive_atoms, 2))
            evaluate_pairs = [pair for pair in possible_pairs if pair in pairs]

        reactive_pairs = []
        for p0, p1 in evaluate_pairs:
            structure_indices = (index_to_structure_index[p0], index_to_structure_index[p1])
            if not any(structure_contains_central[idx] for idx in structure_indices):
                continue
            pair_elements = (flat_all_elements[p0], flat_all_elements[p1])
            # when one is a central atom reactive
            # if ligands alone are reactive, also reactive relying on the fact that filter_atoms() sorts out all
            # non-ligands for structures that contain a central atom
            if any(ele == self.central_atom for ele in pair_elements) or self.ligand_without_central_atom_reactive:
                reactive_pairs.append((p0, p1))
        return reactive_pairs

    def filter_reaction_coordinates(
            self, structure_list: List[db.Structure], coordinates: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[int, int]]]:
        if not self.reevaluate_on_all_levels:
            return coordinates
        reactive_coordinates = []
        for coordinate in coordinates:
            reactive_pairs = self.filter_atom_pairs(structure_list, coordinate)
            if len(reactive_pairs) == len(coordinate):
                reactive_coordinates.append(coordinate)
        return reactive_coordinates

    @staticmethod
    def _structure_assignment_map(atoms_list: List[utils.AtomCollection]) -> List[int]:
        lengths = [len(atoms) for atoms in atoms_list]
        result = []
        for i, length in enumerate(lengths):
            result += [i] * length
        return result


class AtomPairFunctionalGroupFilter(ReactiveSiteFilter):
    """
    A filter that classifies atom pairs as reactive if they belong to a
    user-specified pair of functional groups. Different lists of such pairs can
    be specified for associations and dissociations.
    The order of the two rules in a given pair does not matter.

    Example rules::

      any_O = {
        'O': AlwaysReactive()
      }
      O_bound_H = {
        'H': FunctionalGroupRule(1, 'O', (1, 3))
      }
      f = AtomPairFunctionalGroupFilter(association_rules=[(any_O, O_bound_H)], dissociation_rules=[])

    This allows associations between any O atom and a H atom bound to an O.

    """

    def __init__(self,
                 association_rules: List[Tuple[Union[dict, DistanceRuleSet], Union[dict, DistanceRuleSet]]],
                 dissociation_rules: List[Tuple[Union[dict, DistanceRuleSet], Union[dict, DistanceRuleSet]]]
                 ) -> None:
        """
        Parameters
        ----------
        association_rules : List[Tuple[Union[dict, DistanceRuleSet], Union[dict, DistanceRuleSet]]]
            A list of pairs of (functional group) rules. A given associative pair of atoms is reactive
            if it satisfies at least one of these pairs of rules.
        dissociation_rules : List[Tuple[Union[dict, DistanceRuleSet], Union[dict, DistanceRuleSet]]]
            A list of pairs of (functional group) rules. A given dissociative pair of atoms is reactive
            if it satisfies at least one of these pairs of rules.
        """
        super().__init__()
        self._association_rules: List[Tuple[DistanceRuleSet, DistanceRuleSet]] = []
        self._dissociation_rules: List[Tuple[DistanceRuleSet, DistanceRuleSet]] = []
        # for the moment, we do not care about duplicates
        self._atomrules: List[DistanceRuleSet] = []
        for tup in association_rules:
            corrected_tup = (tup[0] if isinstance(tup[0], DistanceRuleSet) else DistanceRuleSet(tup[0]),
                             tup[1] if isinstance(tup[1], DistanceRuleSet) else DistanceRuleSet(tup[1])
                             )
            self._association_rules.append(corrected_tup)
            self._atomrules.extend([corrected_tup[0], corrected_tup[1]])
        for tup in dissociation_rules:
            corrected_tup = (tup[0] if isinstance(tup[0], DistanceRuleSet) else DistanceRuleSet(tup[0]),
                             tup[1] if isinstance(tup[1], DistanceRuleSet) else DistanceRuleSet(tup[1])
                             )
            self._dissociation_rules.append(corrected_tup)
            self._atomrules.extend([corrected_tup[0], corrected_tup[1]])

    def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
        reactive_atom_indices = []
        idx_shift = 0  # this is an offset for different structures in structure_list
        for structure in structure_list:
            atoms = structure.get_atoms()
            elements = [str(x) for x in atoms.elements]
            molecules = deserialize_molecules(structure)
            # For each atom index (within a whole structure), idx_map gives
            # a pair of a molecule index and an index within the molecule.
            idx_map = ast.literal_eval(structure.get_graph("masm_idx_map"))
            for i, e in enumerate(elements):
                for ruledict in self._atomrules:
                    # atom is turned reactive if it fulfills at least one of the ruledicts
                    if e in ruledict:
                        rule = ruledict[e]
                        if rule.filter_by_rule(molecules, idx_map, elements, i):
                            reactive_atom_indices.append(i + idx_shift)
                            break
            idx_shift += structure.get_atoms().size()
        # only return reactive atom indices that are also contained in the input atom indices (set intersection).
        return list(set(atom_indices) & set(reactive_atom_indices))

    def filter_atom_pairs(self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]) \
            -> List[Tuple[int, int]]:
        from scine_chemoton.gears.elementary_steps.trial_generator.connectivity_analyzer import ConnectivityAnalyzer
        molecules = []
        idx_map = []
        elements = []
        totalindex_to_structureindex: List[int] = []
        totalindex_to_localindex: List[int] = []
        adjacency_matrices = []
        for i, structure in enumerate(structure_list):
            connectivity_analyzer = ConnectivityAnalyzer(structure)
            adjacency_matrices.append(connectivity_analyzer.get_adjacency_matrix())
            molecules.append(deserialize_molecules(structure))
            idx_map.append(ast.literal_eval(structure.get_graph("masm_idx_map")))
            atoms = structure.get_atoms()
            elements.append([str(x) for x in atoms.elements])
            totalindex_to_structureindex += atoms.size() * [i]
            totalindex_to_localindex += list(range(atoms.size()))

        # all_elements_symbols contains the element symbol for each atom
        # in the whole structure_list
        all_element_symbols = [item for sublist in elements for item in sublist]

        reactive_pairs = []
        for i, j in pairs:
            # Check if the pair is associative or dissociative
            # The way this is done is inspired by _get_filtered_intraform_and_diss in bond_based.py
            strucindex_i = totalindex_to_structureindex[i]
            strucindex_j = totalindex_to_structureindex[j]
            locindex_i = totalindex_to_localindex[i]
            locindex_j = totalindex_to_localindex[j]
            if strucindex_i == strucindex_j:
                is_associative_pair = not adjacency_matrices[strucindex_i][locindex_i, locindex_j]
            else:
                is_associative_pair = True
            if is_associative_pair:
                rules_to_follow = self._association_rules
            else:
                rules_to_follow = self._dissociation_rules
            if self.checkrules(i, j, rules_to_follow, all_element_symbols, totalindex_to_structureindex,
                               totalindex_to_localindex, molecules, idx_map, elements):
                reactive_pairs.append((i, j))

        return reactive_pairs

    @staticmethod
    def checkrules(i, j, rules_to_follow, all_element_symbols, totalindex_to_structureindex,
                   totalindex_to_localindex, molecules, idx_map, elements) -> bool:
        """
        this method checks whether the pair i, j fulfills the rules specified by rules_to_follow
        """
        e_i = all_element_symbols[i]
        strucindex_i = totalindex_to_structureindex[i]
        locindex_i = totalindex_to_localindex[i]
        e_j = all_element_symbols[j]
        strucindex_j = totalindex_to_structureindex[j]
        locindex_j = totalindex_to_localindex[j]
        for ruledict1, ruledict2 in rules_to_follow:
            ruledicts = []  # we need to allow for permutation of the two dictionaries
            if (e_i in ruledict1) and (e_j in ruledict2):
                ruledicts.append((ruledict1, ruledict2))
            if (e_i in ruledict2) and (e_j in ruledict1):
                ruledicts.append((ruledict2, ruledict1))
            for ruledict_i, ruledict_j in ruledicts:
                rule_i = ruledict_i[e_i]
                rule_j = ruledict_j[e_j]
                if (rule_i.filter_by_rule(molecules[strucindex_i], idx_map[strucindex_i], elements[strucindex_i],
                                          locindex_i) and
                        rule_j.filter_by_rule(molecules[strucindex_j], idx_map[strucindex_j],
                                              elements[strucindex_j], locindex_j)):
                    return True
        return False


class SubStructureFilter(ReactiveSiteFilter):
    """
    A filter that matches given substructures with reactants structures.
    Substructures are given as .xyz or .mol files and can in total or in
    parts be used as active and disallowed list.

    Note: The substructure matching does not consider local shapes of the
          coordination spheres around atoms. As an example: a benzene
          substructure will match cyclohexane blocks.

    As an example: giving a CH3-CH2- substructure and using it in a disallowed
    list (exclude_mode=True) will allow only the OH group of ethanol to be
    allowed to react.

    For each substructure a list of indices can be provided in an additional
    .json file. If this list is provided the entire substructure will be matched
    but only the subset of atoms will be applied. Indices start at index 0 for
    the first atom in the corresponding substructure .xyz/.mol file.

    As an example: giving a -CH2-OH substructure and an additional index list
    containing only the atom indices of the -OH atoms will allow only the
    OH-group of ethanol to react if used in an allowed-list fashion
    (exclude_mode=False). The same input would not allow the OH-group in
    (CH3)2-CH-OH to react.

    Multiple substructures are added in an 'or' fashion, meaning that matching
    any of the given substructures will trigger the filter.

    Notes
    -----
    This filter is not suitable for flasks
    """

    def __init__(self, library_folder: str, exclude_mode: bool = False) -> None:
        """
        Parameters
        ----------
        library_folder : str
            Path to a folder containing xyz or mol files of substructures
            to be used for filtering. For each xyz/mol file a json file of the
            same name can be supplied to mark only specific atoms as relevant
            for the final filtering. The json file should contain a single list
            of atom indices of active atoms matching the corresponding xyz/mol
            file. If not such json file is provided, all atoms in the xyz/mol file
            are recognized as active for the filter.
        exclude_mode : bool
            If false, only (active) atoms of given substructures are allowed to react.
            If true, all (active) atoms of the given substructures are disallowed from
            reactions trials, but all others are allowed.
        """
        super().__init__()
        assert library_folder
        self.library_folder = library_folder
        assert os.path.isdir(self.library_folder)
        self.exclude_mode = exclude_mode

    def supports_flasks(self) -> bool:
        return False

    def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
        gathered_atom_indices = set()
        offsets = [0]  # Filled while parsing
        for structure_idx, structure in enumerate(structure_list):
            deserialized_molecules = deserialize_molecules(structure)
            atom_map = ast.literal_eval(structure.get_graph("masm_idx_map"))
            assert len(deserialized_molecules) == 1
            reactant_molecule = deserialized_molecules[0]
            offsets.append(sum(offsets) + reactant_molecule.graph.V)
            unique_atom_idxs = set()
            for path, _, files in os.walk(self.library_folder):
                for file in files:
                    if not (file.endswith('.xyz') or file.endswith('.mol')):
                        continue
                    loaded_file = load_file(os.path.join(path, file))
                    assert len(loaded_file) == 1
                    substructure = loaded_file[0]
                    if substructure.graph.V > reactant_molecule.graph.V:
                        continue

                    selection_file_name = os.path.splitext(file)[0] + '.json'
                    if os.path.isfile(os.path.join(path, selection_file_name)):
                        with open(os.path.join(path, selection_file_name), 'r') as f:
                            selection = load(f)
                    else:
                        selection = [i for i in range(substructure.graph.V)]

                    match = maximum_matching_fragments(substructure, reactant_molecule, substructure.graph.V)

                    if not match[2]:
                        continue
                    for atm_idxs in match[2][0]:
                        tmp = []
                        for allowed_idx in selection:
                            tmp.append(atm_idxs[allowed_idx])
                        unique_atom_idxs.update(tmp)
            gathered_atom_indices.update([atom_map.index((0, i)) + offsets[structure_idx] for i in unique_atom_idxs])

        if self.exclude_mode:
            return sorted(list(set(atom_indices).difference(set(gathered_atom_indices))))
        else:
            return sorted(list(gathered_atom_indices.intersection(set(atom_indices))))
