#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import List, Optional, Tuple

# Third party imports
import scine_database as db
import scine_molassembler as masm

# Local application imports
from ...utilities.masm import deserialize_molecules, distinct_atoms, get_atom_pairs
from scine_chemoton.gears import HoldsCollections


class ReactiveSiteFilter(HoldsCollections):
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

    def __init__(self, filters: Optional[List[ReactiveSiteFilter]] = None):
        super().__init__()
        if filters is None:
            filters = []
        self.filters = filters
        for f in filters:
            if not isinstance(f, ReactiveSiteFilter):
                raise TypeError("ReactiveSiteFilterAndArray expects ReactiveSiteFilter "
                                "(or derived class) to chain with.")

    def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
        distinct = atom_indices
        for filter in self.filters:
            distinct = filter.filter_atoms(structure_list, distinct)

        return distinct

    def filter_atom_pairs(
            self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        distinct = pairs
        for filter in self.filters:
            distinct = filter.filter_atom_pairs(structure_list, pairs=distinct)
        return distinct

    def filter_reaction_coordinates(
            self, structure_list: List[db.Structure], coordinates: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[int, int]]]:
        distinct = coordinates
        for filter in self.filters:
            distinct = filter.filter_reaction_coordinates(structure_list, coordinates=distinct)
        return distinct


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
        self.filters = filters
        for f in filters:
            if not isinstance(f, ReactiveSiteFilter):
                raise TypeError("ReactiveSiteFilterOrArray expects ReactiveSiteFilter "
                                "(or derived class) to chain with.")

    def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
        ret: List[int] = []
        for filter in self.filters:
            distinct = filter.filter_atoms(structure_list, atom_indices)
            ret = list(set(ret) | set(distinct))
        return ret

    def filter_atom_pairs(
            self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        ret: List[Tuple[int, int]] = []
        for filter in self.filters:
            distinct = filter.filter_atom_pairs(structure_list, pairs=pairs)
            ret = list(set(ret) | set(distinct))
        return ret

    def filter_reaction_coordinates(
            self, structure_list: List[db.Structure], coordinates: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[int, int]]]:
        ret: List[List[Tuple[int, int]]] = []
        for filter in self.filters:
            distinct = filter.filter_reaction_coordinates(structure_list, coordinates=coordinates)
            ret = list(set(ret) | set(distinct))
        return ret


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
            msg = "Option for masm atom pruning invalid: {}"
            raise RuntimeError(msg.format(prune))

    def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
        import ast

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
    the final importance. Hence a carbon in CH_4 would rank as 1 and one in
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
    The rules are given for each element. They may be a maximum required distance to another
    atom/element or they may label an atom as generally reactive or not reactive. If not rules
    are given for an element. All atoms of this type will be considered as non-reactive.\n
    Example rules:
    reactive_site_rules = {
    'H': [ReactiveRuleFilterOrArray([('O', 3), ('N', 1)])],
    'C': [ReactiveRuleFilterOrArray([('O', 2), ('N', 2)])],
    'O': True,
    'N': True
    }
    Allow reactions for all O and N, for all H that are at most in a distance of 3 bonds to O or one bond
    to N, and for all C that are at most two bonds away from O or N.

    Attributes
    ----------
    rules : dict
        The dictionary containing the rules (vide supra).
    """

    def __init__(self, rules: dict):
        super().__init__()
        self._rules = rules

    def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
        import ast
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
                rule_set = self._rules[e]
                # resolve lists of logical and/or arrays.
                if isinstance(rule_set, List):
                    for rule_array in rule_set:
                        if rule_array.filter_by_rule(molecules, idx_map, elements, i):
                            reactive_atom_indices.append(i + idx_shift)
                            break
                    continue
                # Simple boolean rule, e.g., 'Si' : False
                if rule_set:
                    reactive_atom_indices.append(i + idx_shift)
            idx_shift += structure.get_atoms().size()
        return list(set(atom_indices) & set(reactive_atom_indices))

    class ReactiveRuleFilterAndArray:
        """
        An array of logically 'and' connected rules.

        Attributes
        ----------
        primitive_rules : List[Tuple[str, int]]
            A list of bond distance based rules that have all to be fulfilled.
            Syntax example: ('O', 2) -> An oxygen atom has to be within two bonds distance of the current atom for it
            to be allowed to react.
        """

        def __init__(self, primitive_rules: Optional[List[Tuple[str, int]]] = None):
            if primitive_rules is None:
                primitive_rules = []
            self._primitive_rules = primitive_rules

        def filter_by_rule(self, molecules: List[masm.Molecule], idx_map: List[Tuple[int, int]], elements: List[str],
                           atom_index: int) -> bool:
            mol_idx, i = idx_map[atom_index]
            distances = masm.distance(i, molecules[mol_idx].graph)
            for rule in self._primitive_rules:
                max_distance = rule[1]
                element_key = rule[0]
                rule_fulfilled = False
                for j, e in enumerate(elements):
                    if e == element_key and distances[idx_map[j][1]] == max_distance:
                        rule_fulfilled = True
                        break
                # All rules have to be fulfilled. Stop if one is not given.
                if not rule_fulfilled:
                    return False
            return True

    class ReactiveRuleFilterOrArray:
        """
        An array of logically 'or' connected rules.

        Attributes
        ----------
        primitive_rules : List[Tuple[str, int]]
            A list of bond distance based rules of which at least one has to be fulfilled.
            Syntax example: ('O', 2) -> An oxygen atom has to be at a distance of two bonds of the current atom for it
            to be allowed to react.
        """

        def __init__(self, primitive_rules: Optional[List[Tuple[str, int]]] = None):
            if primitive_rules is None:
                primitive_rules = []
            self._primitive_rules = primitive_rules

        def filter_by_rule(self, molecules: List[masm.Molecule], idx_map: List[Tuple[int, int]], elements: List[str],
                           atom_index: int) -> bool:
            mol_idx, i = idx_map[atom_index]
            distances = masm.distance(i, molecules[mol_idx].graph)
            for rule in self._primitive_rules:
                max_distance = rule[1]
                element_key = rule[0]
                for j, e in enumerate(elements):
                    if e == element_key and distances[idx_map[j][1]] == max_distance:
                        return True
            return False

    class FunctionalGroupRule:
        """
        A rule that encodes a distance criterion to a very general functional group. The functional group is
        encoded in terms of a central atom type, a list of hetero atoms, the number of bonds to the central
        atom and the number of hetero atoms bonded to the central atom.\n
        carbonyle_group_d2 = FunctionalGroupRule(2, ['O'], 'C', 3, 1)
        imine_group_d0 = FunctionalGroupRule(0, ['N'], 'C', 3, 1)
        acetal_group_d1 =  FunctionalGroupRule(0, ['O'], 'C', 4, 2)
        acetal_like_group_d1   =  FunctionalGroupRule(0, ['O, N'], 'C', 4, 2)

        Attributes
        ----------
        distance : int
            The bond distance to the functional group that must be matched.
        hetero_atoms : List[str]
            The list of atoms that are considered hetero atoms for this group.
        central_atom : str
            The central atom element symbol (default 'C')
        n_bonds : int
            The number of bonds to the central atom.
        n_hetero_atoms : int
            The number of hetero atoms that must bond to the central atom to constitute the group.

        """

        def __init__(self, distance: int, hetero_atoms: List[str], central_atom: str = 'C', n_bonds: int = 3,
                     n_hetero_atoms: int = 1):
            self._distance = distance
            self._hetero_atoms = hetero_atoms
            self._central_atom = central_atom
            self._n_bonds = n_bonds
            self._n_hetero_atoms = n_hetero_atoms

        def filter_by_rule(self, molecules: List[masm.Molecule], idx_map: List[Tuple[int, int]], elements: List[str],
                           atom_index: int) -> bool:
            mol_idx, i = idx_map[atom_index]
            distances_i = masm.distance(i, molecules[mol_idx].graph)
            # Loop all C within the distance
            for atom_j, e in enumerate(elements):
                mol_jdx, j = idx_map[atom_j]
                if e == self._central_atom and distances_i[j] == self._distance:
                    distances_j = masm.distance(j, molecules[mol_jdx].graph)
                    # For carbonyle or imine groups, there must be at least one hetero atom in distance of 1 to j and
                    # exactly three atoms with distance of 1 to j
                    n_close_to_j = 0
                    n_hetero_atoms_found = 0
                    for atom_k, e_k in enumerate(elements):
                        _, k = idx_map[atom_k]
                        if distances_j[k] == 1:
                            n_close_to_j += 1
                            if e_k in self._hetero_atoms:
                                n_hetero_atoms_found += 1
                            if n_close_to_j > self._n_bonds:
                                break
                    if n_hetero_atoms_found == self._n_hetero_atoms and n_close_to_j == self._n_bonds:
                        return True
            return False


class ElementWiseReactionCoordinateFilter(ReactiveSiteFilter):
    """
    A filter that can restrict the combination of atoms with a specific element.
    The filter can be operated in two modes: Allow all reaction coordinates for the
    element combinations encoded in the rules or forbid all of them.\n
    Example rules:
    reaction_coordinate_rules = {
    'H': ['H', 'C']
    }
    In the default "forbid-mode" these rules mean that no reaction coordinates between two H-atoms and H and C are
    allowed.

    Attributes
    ----------
    rules : dict
        The dictionary containing the rules (vide supra).
    reactive_if_rules_apply : bool
        The mode to operate the filter in. If true, only reaction coordinates in the given rule set pass.
        If false, no reaction coordinate in the given rule set pass. By default, false.
    """

    def __init__(self, rules: dict, reactive_if_rules_apply: bool = False):
        super().__init__()
        self._rules = rules
        self._reactive_if_rules_apply = reactive_if_rules_apply

    def filter_reaction_coordinates(
            self, structure_list: List[db.Structure], coordinates: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[int, int]]]:
        all_element_symbols = []
        for structure in structure_list:
            atoms = structure.get_atoms()
            elements = [str(x) for x in atoms.elements]
            all_element_symbols += elements

        valid_coordinates = []
        for coord in coordinates:
            skip = False
            for pair in coord:
                element_0 = all_element_symbols[pair[0]]
                element_1 = all_element_symbols[pair[1]]
                e_0_in_rule = False
                e_1_in_rule = False
                if element_0 in self._rules:
                    e_0_in_rule = element_1 in self._rules[element_0]
                if element_1 in self._rules:
                    e_1_in_rule = element_0 in self._rules[element_1]
                # If the rules are meant to exclude reaction coordinates, either e_0_in_rule or e_1_in_rule
                # have to be true to skip the coordinate.
                # If the rules are meant to include reaction coordinates, both of e_0_in_rule and e_1_in_rule have to
                # be false to skip it.
                if (not self._reactive_if_rules_apply and (e_0_in_rule or e_1_in_rule)) \
                        or (self._reactive_if_rules_apply and not (e_0_in_rule or e_1_in_rule)):
                    skip = True
                    break
            if not skip:
                valid_coordinates.append(coord)
        return valid_coordinates


class HeuristicPolarizationReactionCoordinateFilter(ReactiveSiteFilter):
    """
    A filter that assigns polarizations (+, -, or +-) to each atom. Reaction coordinates are only allowed
    that combine + and - or +- with either + or -.
    Example rules:
    rules = {
    'H': [PaulingElectronegativityRule(), FunctionalGroupRule("+", 2, ['N', 'O'], 'C', 3, 1)],
    'C': [PaulingElectronegativityRule()],
    'N': [PaulingElectronegativityRule()],
    'O': [PaulingElectronegativityRule()]
    }

    Attributes
    ----------
    rules : dict
        The dictionary containing the rules. The rule object must implement a function called string_from_rule(...).
    """

    def __init__(self, rules: dict):
        super().__init__()
        self._rules = rules

    class PaulingElectronegativityRule:
        """
        Polarization rule for the Pauli electronegativity scalar.

        Attributes
        ----------
        min_difference : dict
            The minimum difference in electronegativities to assign a polarization.
        """

        def __init__(self, min_difference: float = 0.4):
            super().__init__()
            self._min_difference = min_difference

        def string_from_rule(self, molecules: List[masm.Molecule], idx_map: List[Tuple[int, int]], elements: List[str],
                             atom_index: int) -> str:
            """
            Return '+' if the atom is electron poor, '-' if it is electron rich, some combination thereof if the
            atom is neighbouring elements with significantly higher and lower electronegativity, and '' if it is
            neighbouring neither.
            """
            from scine_utilities import ElementInfo
            mol_idx, i = idx_map[atom_index]
            distances_i = masm.distance(i, molecules[mol_idx].graph)
            return_str = ''
            for j, distance in enumerate(distances_i):
                if distance == 1:
                    atom_j = idx_map.index((mol_idx, j))
                    element_i = ElementInfo.element_from_symbol(elements[atom_index])
                    element_j = ElementInfo.element_from_symbol(elements[atom_j])
                    electronegativity_i = ElementInfo.pauling_electronegativity(element_i)
                    electronegativity_j = ElementInfo.pauling_electronegativity(element_j)
                    difference = electronegativity_i - electronegativity_j
                    if abs(difference) < self._min_difference:
                        continue
                    if difference <= 0.0:
                        return_str += '+'
                    else:
                        # i > j
                        return_str += '-'
            return return_str

    class FunctionalGroupRule:
        """
        Polarization rule for functional groups.
        The functional group is encoded in terms of a central atom type, a
        list of hetero atoms, the number of bonds to the central atom and the
        number of hetero atoms bonded to the central atom.\n
        carbonyle_group_d2 = FunctionalGroupRule('+' ,2, ['O'], 'C', 3, 1)
        imine_group_d0 = FunctionalGroupRule('+' ,0, ['N'], 'C', 3, 1)
        acetal_group_d1 =  FunctionalGroupRule('+', 0, ['O'], 'C', 4, 2)
        acetal_like_group_d1   =  FunctionalGroupRule('+', 0, ['O, N'], 'C', 4, 2)

        Attributes
        ----------
        character : str
            The polarization character (+ or -) to assign if the rule applies.
        distance : int
            The bond distance to the functional group that must be matched.
        hetero_atoms : List[str]
            The list of atoms that are considered hetero atoms for this group.
        central_atom : str
            The central atom element symbol (default 'C')
        n_bonds : int
            The number of bonds to the central atom.
        n_hetero_atoms : int
            The number of hetero atoms that must bond to the central atom to constitute the group.
        """

        def __init__(self, character: str, distance: int, hetero_atoms: List[str], central_atom: str = 'C',
                     n_bonds: int = 3, n_hetero_atoms: int = 1):
            self._character = character
            self._rule = AtomRuleBasedFilter.FunctionalGroupRule(
                distance, hetero_atoms, central_atom, n_bonds, n_hetero_atoms)

        def string_from_rule(self, molecules: List[masm.Molecule], idx_map: List[Tuple[int, int]], elements: List[str],
                             atom_index: int) -> str:
            if self._rule.filter_by_rule(molecules, idx_map, elements, atom_index):
                return self._character
            else:
                return ''

    def filter_reaction_coordinates(
            self, structure_list: List[db.Structure], coordinates: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[int, int]]]:
        all_polarization_characters = []
        for structure in structure_list:
            all_polarization_characters += self._get_characters_for_structure(structure)

        valid_coordinates = []
        for coord in coordinates:
            skip = False
            for pair in coord:
                character_0 = all_polarization_characters[pair[0]]
                character_1 = all_polarization_characters[pair[1]]
                plus_minus = '+' in character_0 and '-' in character_1
                minus_plus = '-' in character_0 and '+' in character_1
                # All atom pairs must combine + and - signs.
                if not (plus_minus or minus_plus):
                    skip = True
                    break
            if not skip:
                valid_coordinates.append(coord)
        return valid_coordinates

    def _get_characters_for_structure(self, structure: db.Structure):
        import ast
        atoms = structure.get_atoms()
        n_atoms = len(atoms)
        elements = [str(x) for x in atoms.elements]
        molecules = deserialize_molecules(structure)
        idx_map = ast.literal_eval(structure.get_graph("masm_idx_map"))
        string_list = ['' for _ in range(n_atoms)]
        for atom_index in range(n_atoms):
            for rule in self._rules[elements[atom_index]]:
                string_list[atom_index] += rule.string_from_rule(molecules, idx_map, elements, atom_index)
        return string_list
