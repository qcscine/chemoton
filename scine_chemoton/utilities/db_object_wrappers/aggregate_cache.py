#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Dict

import scine_database as db

from .aggregate_wrapper import Aggregate


class AggregateCache:
    """
    A cache for aggregate objects.

    Parameters
    ----------
    manager
        The database manager.
    electronic_model
        The electronic structure model from which the electronic energy is taken.
    hessian_model
        The electronic structure model with which the geometrical hessian was calculated.
    only_electronic
        If true, only the electronic energies are used to determine the thermodynamics.
    """

    def __init__(self, manager: db.Manager, electronic_model: db.Model, hessian_model: db.Model,
                 only_electronic: bool = False) -> None:
        self._manager = manager
        self._electronic_model = electronic_model
        self._hessian_model = hessian_model
        self._only_electronic = only_electronic
        self._aggregates: Dict[int, Aggregate] = {}

    def get_or_produce(self, aggregate_id: db.ID) -> Aggregate:
        """
        Get an instance of the aggregate wrapper corresponding the reaction ID. The instance may be newly constructed or
        retrieved from the cache.

        Parameters
        ----------
        aggregate_id : db.ID
            The aggregate database ID.

        Returns
        -------
        Aggregate
            The aggregate wrapper instance.
        """
        int_id = int(aggregate_id.string(), 16)
        if int_id not in self._aggregates:
            self._aggregates[int_id] = Aggregate(aggregate_id, self._manager, self._electronic_model,
                                                 self._hessian_model, self._only_electronic)
        return self._aggregates[int_id]

    def get_electronic_model(self):
        return self._electronic_model
