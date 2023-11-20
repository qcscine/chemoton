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
from scine_utilities import ElementInfo

# Local application imports
from . import valid_element, RuleSet, BaseRule
from .distance_rules import DistanceBaseRule


class PolarizationBaseRule(BaseRule):
    """
    A rule that assigns polarizations (+, -, or +-) to each atom based on given molecules
    """

    @abstractmethod
    def string_from_rule(self, molecules: List[masm.Molecule], idx_map: List[Tuple[int, int]], elements: List[str],
                         atom_index: int) -> str:
        """
        Method to be implemented for each rule, giving back a polarization character for atom_index in
        multiple molecules

        Parameters
        ----------
        molecules :: List[masm.Molecule]
            A list of molassembler molecules representing the total system
        idx_map :: List[Tuple[int, int]]
            An index map generated by molassembler that allows to map the atom_index to the atom in the molecules
        elements :: List[str]
            A list of all elements
        atom_index :: int
            The index we want the polarization of

        Returns
        -------
        polarization_char :: str
            Either '+', '-', or '+-'
        """
        pass  # pylint: disable=unnecessary-pass

    def __and__(self, o):
        if not isinstance(o, PolarizationBaseRule):
            raise TypeError(f"{self.__class__.__name__} expects PolarizationBaseRule "
                            f"(or derived class) to chain with.")
        return PolarizationRuleAndArray([self, o])


class PolarizationRuleSet(RuleSet):
    """
    A dictionary holding elements as keys and a respective PolarizationRule for it.
    All keys are checked for valid element types.
    Lists as values are automatically transformed to an PolarizationRuleAndArray.
    """

    def __init__(self, kwargs: Dict[str, PolarizationBaseRule]):
        super().__init__(kwargs)
        for k, v in kwargs.items():
            if not isinstance(k, str):
                raise TypeError(f"{self.__class__.__name__} expects strings as keys")
            if not valid_element(k):
                raise TypeError(f"{k} is not a valid element symbol")
            if not isinstance(v, PolarizationBaseRule):
                if isinstance(v, list):
                    self.data[k] = PolarizationRuleAndArray(v)
                elif isinstance(v, str) and v.lower().strip() == "pauling":
                    self.data[k] = PaulingElectronegativityRule()
                else:
                    raise TypeError(f"{self.__class__.__name__} expects polarization rules as values")


class PolarizationRuleAndArray(PolarizationBaseRule):
    """
    An array of logically 'and' connected rules.
    """

    def __init__(self, rules: Optional[List[PolarizationBaseRule]] = None):
        """
        Parameters
        ----------
        rules : Optional[List[PolarizationBaseRule]]
            A list of Polarization rules that all may add a character.
        """
        super().__init__()
        if rules is None:
            rules = []
        self._rules = rules
        self._join_names(self._rules)

    def string_from_rule(self, *args, **kwargs) -> str:
        res = ""
        for rule in self._rules:
            res += rule.string_from_rule(*args, **kwargs)
        return res

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._rules)})"


class PaulingElectronegativityRule(PolarizationBaseRule):
    """
    Polarization rule for the Pauli electronegativity scale.
    """

    def __init__(self, min_difference: float = 0.4):
        """
        Attributes
        ----------
        min_difference : float
            The minimum difference in electronegativities to assign a polarization.
            Default: 0.4 (avoids polarization of C--H bonds)
        """
        super().__init__()
        self._min_difference = min_difference

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._min_difference})"

    def string_from_rule(self, molecules: List[masm.Molecule], idx_map: List[Tuple[int, int]], elements: List[str],
                         atom_index: int) -> str:
        """
        Return '+' if the atom is electron poor, '-' if it is electron rich, some combination thereof if the
        atom is neighbouring elements with significantly higher and lower electronegativity, and '' if it is
        neighbouring neither.
        """
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


class PolarizationFunctionalGroupRule(PolarizationBaseRule):

    def __init__(self, polarization_char: str, distance_rule: DistanceBaseRule):
        super().__init__()
        if not isinstance(polarization_char, str) or polarization_char not in ['+', '-', '+-']:
            raise TypeError(f"{self.__class__.__name__} received invalid polarization_char {polarization_char}, "
                            f"expected '+' or '-'")
        self._character = polarization_char
        self._rule = distance_rule

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self._character}', {repr(self._rule)})"

    def string_from_rule(self, molecules: List[masm.Molecule], idx_map: List[Tuple[int, int]], elements: List[str],
                         atom_index: int) -> str:
        if self._rule.filter_by_rule(molecules, idx_map, elements, atom_index):
            return self._character
        return ''
