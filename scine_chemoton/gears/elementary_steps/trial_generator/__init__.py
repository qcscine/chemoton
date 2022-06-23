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


class TrialGenerator(metaclass=ABCMeta):
    """
    Base class for elementary step trial generators
    """

    def __init__(self):
        pass

    def initialize_collections(self, manager: db.Manager) -> None:
        # Get required collections
        if hasattr(self, "_calculations"):
            if self._calculations:
                self._calculations = manager.get_collection("calculations")
        if hasattr(self, "_compounds"):
            if self._compounds:
                self._compounds = manager.get_collection("compounds")
        if hasattr(self, "_reactions"):
            if self._reactions:
                self._reactions = manager.get_collection("reactions")
        if hasattr(self, "_elementary_steps"):
            if self._elementary_steps:
                self._elementary_steps = manager.get_collection("elementary_steps")
        if hasattr(self, "_structures"):
            if self._structures:
                self._structures = manager.get_collection("structures")
        if hasattr(self, "_properties"):
            if self._properties:
                self._properties = manager.get_collection("properties")

    @abstractmethod
    def unimolecular_reactions(self, structure: db.Structure) -> None:
        raise NotImplementedError

    @abstractmethod
    def bimolecular_reactions(self, structure_list: List[db.Structure]) -> None:
        raise NotImplementedError
