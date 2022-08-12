#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""
# Standard library imports
from typing import List
from abc import ABCMeta, abstractmethod

# Third party imports
import scine_database as db
from scine_chemoton.gears import HoldsCollections


class TrialGenerator(HoldsCollections, metaclass=ABCMeta):
    """
    Base class for elementary step trial generators
    """

    def __init__(self):
        super().__init__()
        self._required_collections = ["calculations", "structures"]

    @abstractmethod
    def unimolecular_reactions(self, structure: db.Structure) -> None:
        raise NotImplementedError

    @abstractmethod
    def bimolecular_reactions(self, structure_list: List[db.Structure]) -> None:
        raise NotImplementedError
