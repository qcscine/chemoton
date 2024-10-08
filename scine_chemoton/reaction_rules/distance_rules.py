#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from abc import abstractmethod
from typing import List, Optional, Tuple, Dict

# Third party imports
import scine_molassembler as masm

# Local application imports
from . import valid_element, RuleSet, BaseRule


class DistanceBaseRule(BaseRule):
    """
    A rule that decides for each atom index if it is reactive or not according to the defined rule.
    The rules are all based on graph distances.
    """

    @abstractmethod
    def filter_by_rule(self, molecules: List[masm.Molecule], idx_map: List[Tuple[int, int]], elements: List[str],
                       atom_index: int) -> bool:
        """
        Method to be implemented for each rule, giving back whether atom_index is reactive or not.

        Parameters
        ----------
        molecules : List[masm.Molecule]
            A list of molassembler molecules representing the total system
        idx_map : List[Tuple[int, int]]
            An index map generated by molassembler that allows to map the atom_index to the atom in the molecules
        elements : List[str]
            A list of all elements
        atom_index : int
            The index we rule out or not

        Returns
        -------
        reactive : bool
        """

    def __and__(self, o):
        if not isinstance(o, DistanceBaseRule):
            raise TypeError(f"{self.__class__.__name__} expects DistanceBaseRule "
                            f"(or derived class) to chain with.")
        return DistanceRuleAndArray([self, o])

    def __or__(self, o):
        if not isinstance(o, DistanceBaseRule):
            raise TypeError(f"{self.__class__.__name__} expects DistanceBaseRule "
                            f"(or derived class) to chain with.")
        return DistanceRuleOrArray([self, o])


class DistanceRuleSet(RuleSet):
    """
    A dictionary holding elements as keys and a respective DistanceBasedRule for it.
    All keys are checked for valid element types.
    Lists as values are automatically transformed to a DistanceRuleAndArray.
    Booleans as values are automatically transformed to `AlwaysTrue` or `AlwaysFalse`
    """

    def __init__(self, kwargs: Dict[str, DistanceBaseRule]) -> None:
        super().__init__(kwargs)
        for k, v in kwargs.items():
            if not isinstance(k, str):
                raise TypeError(f"{self.__class__.__name__} expects strings as keys")
            if not valid_element(k):
                raise TypeError(f"{k} is not a valid element symbol")
            if not isinstance(v, DistanceBaseRule):
                if isinstance(v, bool):
                    rule = AlwaysTrue() if v else AlwaysFalse()
                    self.data[k] = rule
                elif isinstance(v, list):
                    self.data[k] = DistanceRuleAndArray(v)
                else:
                    raise TypeError(f"{self.__class__.__name__} expects distance based rules as values")


class DistanceRuleAndArray(DistanceBaseRule):
    """
    An array of logically 'and' connected rules.
    """

    def __init__(self, rules: Optional[List[DistanceBaseRule]] = None) -> None:
        """
        Parameters
        ----------
        rules : Optional[List[DistanceBaseRule]]
            A list of bond distance based rules that have all to be fulfilled.
        """
        super().__init__()
        if rules is None:
            rules = []
        self._rules = rules
        self._join_names(self._rules)

    def filter_by_rule(self, *args, **kwargs) -> bool:
        return all(rule.filter_by_rule(*args, **kwargs) for rule in self._rules)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._rules)})"


class DistanceRuleOrArray(DistanceBaseRule):
    """
    An array of logically 'or' connected rules.
    """

    def __init__(self, rules: Optional[List[DistanceBaseRule]] = None) -> None:
        """
        Parameters
        ----------
        rules : Optional[List[DistanceBaseRule]]
            A list of bond distance based rules of which at least one has to be fulfilled.
        """
        super().__init__()
        if rules is None:
            rules = []
        self._rules = rules
        self._join_names(self._rules)

    def filter_by_rule(self, *args, **kwargs) -> bool:
        return any(rule.filter_by_rule(*args, **kwargs) for rule in self._rules)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._rules)})"


class AlwaysTrue(DistanceBaseRule):
    """
    Makes each given index reactive.
    """

    def filter_by_rule(self, *args, **kwargs) -> bool:
        return True


class AlwaysFalse(DistanceBaseRule):
    """
    Makes each given index unreactive.
    """

    def filter_by_rule(self, *args, **kwargs) -> bool:
        return False


class SimpleDistanceRule(DistanceBaseRule):
    """
    Allows to define a rule simply based on a distance to another specified element.
    SimpleDistanceRule('O', 1) defines each atom that is directly bonded to an oxygen as reactive.
    """

    def __init__(self, element: str, distance: int) -> None:
        """
        Parameters
        ----------
        element : str
            Another element defining if the given index is reactive or not
        distance : int
            The distance to the other element that defines the rule
        """
        super().__init__()
        if not valid_element(element):
            raise ValueError(f"{element} is not a valid element symbol")
        self._element_key = element
        self._distance = distance

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self._element_key}', {self._distance})"

    def filter_by_rule(self, molecules: List[masm.Molecule], idx_map: List[Tuple[int, int]], elements: List[str],
                       atom_index: int):
        mol_idx, i = idx_map[atom_index]
        distances = masm.distance(i, molecules[mol_idx].graph)
        for j, e in enumerate(elements):
            other_mol_idx, shifted_index = idx_map[j]
            if mol_idx != other_mol_idx:
                continue
            if e == self._element_key and distances[shifted_index] == self._distance:
                return True
        return False


class FunctionalGroupRule(DistanceBaseRule):
    """
    The functional group is encoded in terms of a central atom type, a
    dictionary of bonding atoms to the central atom and how often they are bound to the central atom, and
    the minimum and maximum number of bonds to the central atom.\n
    An atom is then termed reactive, if the specified distance in this rule, is the distance between the atom and the
    central atom.
    E.g. a hydroxyl group can be specified with an O as a central atom with exactly 2 bonds, one to H and one to C.
    Based on the distance then different atoms in relation to any found hydroxyl group are termed reactive.\n
    0 makes the O of all hydroxyl groups reactive, 1 makes the C and H of all hydroxyl groups reactive and 2 makes
    the bonding partners of the C and H of all hydroxyl groups reactive etc.

    hydroxyl_d0 = FunctionalGroupRule(0, 'O', (2, 2), {'H': 1, 'C': 1}, True)
    carbonyl_group_d2 = FunctionalGroupRule(2, 'C', (3, 3), {'O': 1}, True)
    imine_group_d0 = FunctionalGroupRule(0, 'C', (3, 3), {'N': 1}, True)
    acetal_group_d1 = FunctionalGroupRule(0, 'C', (4, 4), {'O': 2, 'H': 1, 'C': 1}, True)
    acetal_like_group_d1 = FunctionalGroupRule(0, 'C', (4, 4), {'O': 1, 'N': 1, 'C': 1, 'H': 1}, True)
    four_or_five_bonded_fe = FunctionalGroupRule(0, 'Fe', (4, 5))
    acetonitrile_d1 = FunctionalGroupRule(1, 'C', (2, 2), {'N': 1, 'C': 1}, True)

    chaining definitions with `or` is also possible for general bonding pattern with different bonding atoms::

      general_acetal_like_d1 = DistanceRuleOrArray([
        FunctionalGroupRule(1, 'C', (4, 4), {'O': 2, 'H': 1, 'C': 1}, True),
        FunctionalGroupRule(1, 'C', (4, 4), {'O': 1, 'N': 1, 'H': 1, 'C': 1}, True),
        FunctionalGroupRule(1, 'C', (4, 4), {'N': 2, 'H': 1, 'C': 1}, True),
        FunctionalGroupRule(1, 'C', (4, 4), {'S': 1, 'N': 1, 'H': 1, 'C': 1}, True),
        FunctionalGroupRule(1, 'C', (4, 4), {'S': 2, 'H': 1, 'C': 1}, True),
      ])

    """

    def __init__(self, distance: int, central_atom: str, n_bonds: Tuple[int, int],
                 specified_bond_partners: Optional[Dict[str, int]] = None, strict_counts: bool = False) -> None:
        """
        Parameters
        ----------
        distance : int
            The bond distance to the functional group that must be matched.
        central_atom : str
            The central atom element symbol.
        n_bonds : Tuple[int, int]
            The minimum and maximum number of bonds to the central atom.
        specified_bond_partners : Dict[str, int]
            An optional dictionary specifying the elements that further specify the reactive group by being directly
            bonded to the central atom together with how often they are bonded to the central atom.
            E.g. a carbonyl can be specified with the central atom O that has exactly 1 bond partner being C,
            while an ether group can be specified with the central atom O that has exactly 2 bond partner being C.
            In general, the choice of the central atom is arbitrary, i.e. both groups could also be specified with C
            being the central atom, see class description.
        strict_counts : bool
            A flag specifying whether the specified bond partner counts must hold strictly (True)
            or constitute a minimum number (False). Default: False.
        """
        super().__init__()
        if not valid_element(central_atom):
            raise ValueError(f"{central_atom} is not a valid element symbol")
        if specified_bond_partners is None:
            specified_bond_partners = {}
        if any(not valid_element(e) for e in specified_bond_partners.keys()):
            raise ValueError(f"{specified_bond_partners} contains invalid element symbols")
        self._distance = distance
        self._central_atom = central_atom
        self._n_bonds_min = n_bonds[0]
        self._n_bonds_max = n_bonds[1]
        self._specified_bond_partners = specified_bond_partners
        self._strict_counts = strict_counts

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._distance}, '{self._central_atom}', " \
               f"({self._n_bonds_min}, {self._n_bonds_max}), {self._specified_bond_partners}, {self._strict_counts})"

    def filter_by_rule(self, molecules: List[masm.Molecule], idx_map: List[Tuple[int, int]], elements: List[str],
                       atom_index: int) -> bool:
        mol_idx, i = idx_map[atom_index]  # index i is the atom on which we apply the rule
        distances_i = masm.distance(i, molecules[mol_idx].graph)
        for atom_j, e_j in enumerate(elements):  # loop over possible central atoms
            mol_jdx, j = idx_map[atom_j]
            if mol_jdx != mol_idx:
                # atom j is part of another molecule, there cannot be a valid graph distance
                continue
            if e_j == self._central_atom and distances_i[j] == self._distance:
                # distances_j are all distances from any atoms in the molecule to atom j,
                # which is the central atom of the functional group:
                distances_j = masm.distance(j, molecules[mol_jdx].graph)
                n_bonds_found = 0
                bond_partners_found = {elem: 0 for elem in self._specified_bond_partners}
                for atom_k, e_k in enumerate(elements):  # loop over possible atoms bound to the central atom
                    mol_kdx, k = idx_map[atom_k]
                    if mol_kdx != mol_jdx:
                        # atom k is part of another molecule, there cannot be a valid graph distance
                        continue
                    if distances_j[k] == 1:
                        n_bonds_found += 1
                        if e_k in self._specified_bond_partners:
                            bond_partners_found[e_k] += 1
                        if n_bonds_found > self._n_bonds_max:
                            break
                bond_partners_fulfilled = all(defined_count == bond_partners_found[e] if self._strict_counts
                                              else defined_count <= bond_partners_found[e]
                                              for e, defined_count in self._specified_bond_partners.items())
                n_bonds_fulfilled = (self._n_bonds_min <= n_bonds_found) and (n_bonds_found <= self._n_bonds_max)
                if bond_partners_fulfilled and n_bonds_fulfilled:
                    return True
        return False
