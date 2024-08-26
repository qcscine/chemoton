#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from abc import abstractmethod
from typing import List, Union, Dict

# Local application imports
from . import valid_element, RuleSet, BaseRule


class ElementBaseRule(BaseRule):
    """
    A rule that defines reactivity simply based on element combinations
    """

    def __init__(self, base_element: str, other_elements: Union[List[str], str]) -> None:
        """
        Parameters
        ----------
        base_element : str
            The element that is the base of the rule
        other_elements : Union[List[str], str]
            The other elements that are allowed to be combined with the base element
        """
        super().__init__()
        if not valid_element(base_element):
            raise ValueError(f"{base_element} is not a valid element")
        self.base_element = base_element
        if isinstance(other_elements, str):
            self._other_elements = [other_elements]
        else:
            self._other_elements = other_elements
        for element in self._other_elements:
            if not valid_element(element):
                raise ValueError(f"{element} is not a valid element")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.base_element}', {repr(self._other_elements)})"

    @abstractmethod
    def rule_applies(self, element_1: str, element_2: str) -> bool:
        """
        If the rule applies for the two given elements
        """
        pass  # pylint: disable=unnecessary-pass


class SimpleElementCombinationRule(ElementBaseRule):
    """
    A rule that defines possible combinations of elements.
    """

    def rule_applies(self, element_1: str, element_2: str) -> bool:
        if element_1 == self.base_element and element_2 in self._other_elements:
            return True
        return element_2 == self.base_element and element_1 in self._other_elements


class ElementRuleSet(RuleSet):
    """
    A dictionary holding elements as keys and either multiple or a single element as values.
    All keys and values are checked for valid element types.
    """

    def __init__(self, kwargs: Dict[str, Union[ElementBaseRule, str, List[str]]],
                 rule_type: type = SimpleElementCombinationRule) -> None:
        super().__init__(kwargs)
        self._rule_type = rule_type
        for k, v in kwargs.items():
            if not isinstance(k, str):
                raise TypeError(f"{self.__class__.__name__} expects strings as keys")
            if not valid_element(k):
                raise TypeError(f"{k} is not a valid element symbol")
            if not isinstance(v, ElementBaseRule):
                rule = rule_type(k, v)
                self.data[k] = rule
        assert all(isinstance(v, ElementBaseRule) for v in self.data.values())
        if any(k != v.base_element for k, v in self.data.items()):
            raise ValueError("All keys must be the same as the base element of their rule")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.data)}, {self._rule_type.__name__})"
