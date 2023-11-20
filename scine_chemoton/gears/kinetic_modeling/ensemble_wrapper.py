#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Union

import scine_database as db
from .thermodynamic_properties import ThermodynamicPropertiesCache


class Ensemble:
    """
    Base class of the wrapper for an ensemble of structures that is derived from either a database compound, flask, or
    reaction.
    """

    def __init__(self, db_id: db.ID, manager: db.Manager, electronic_model: db.Model, hessian_model: db.Model,
                 only_electronic: bool = False):
        self._electronic_model = electronic_model
        self._hessian_model = hessian_model
        self._only_electronic = only_electronic
        self._structures = manager.get_collection("structures")
        self._properties = manager.get_collection("properties")
        self._db_id = db_id
        self._db_object = self._initialize_db_object(manager)
        assert self._db_object.exists()
        self._structure_thermodynamics = ThermodynamicPropertiesCache(self._structures, self._properties,
                                                                      self._electronic_model, self._hessian_model,
                                                                      self._only_electronic)

    def _initialize_db_object(self, manager: db.Manager) -> Union[db.Compound, db.Flask, db.Reaction]:
        raise NotImplementedError()

    def get_db_id(self) -> db.ID:
        return self._db_id

    def __int__(self):
        return int(self.get_db_id().string(), 16)

    def __eq__(self, other) -> bool:
        return self._db_id == other.get_db_id()

    def __hash__(self):
        return hash(self.__int__())

    def get_db_object(self):
        assert self._db_object
        return self._db_object
