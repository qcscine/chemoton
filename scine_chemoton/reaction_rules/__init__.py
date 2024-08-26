#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""
from abc import ABC
from collections import UserDict

import scine_utilities as utils

from scine_chemoton.gears import HasName


class RuleSet(UserDict):
    """
    Mainly exists for typing purposes, and default representation.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.data)})"


class BaseRule(HasName, ABC):
    """
    Mainly exists for typing purposes, and default representation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._remove_chemoton_from_name()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def valid_element(element: str) -> bool:
    try:
        _ = utils.ElementInfo.element_from_symbol(element)
    except RuntimeError:
        return False
    return True
