#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Optional, Dict

import scine_database as db

from .aggregate_cache import AggregateCache
from .reaction_wrapper import Reaction


class ReactionCache:
    """
    A cache for reaction objects.

    Parameters
    ----------
    manager
        The database manager.
    electronic_model
        The electronic structure model from which the electronic energy is taken.
    hessian_model
        The electronic structure model with which the geometrical hessian was calculated.
    aggregate_cache : Optional[AggregateCache]
        An optional aggregate cache with which the aggregates taking part in the reaction are constructed.
    only_electronic
        If true, only the electronic energies are used to determine the thermodynamics.
    """

    def __init__(self, manager: db.Manager, electronic_model: db.Model, hessian_model: db.Model,
                 aggregate_cache: Optional[AggregateCache] = None,
                 only_electronic: bool = False) -> None:
        self._manager = manager
        self._electronic_model = electronic_model
        self._hessian_model = hessian_model
        self._only_electronic = only_electronic
        self._reactions: Dict[int, Reaction] = {}
        if aggregate_cache is None:
            aggregate_cache = AggregateCache(manager, electronic_model, hessian_model, only_electronic)
        self._aggregate_cache = aggregate_cache

    def get_or_produce(self, reaction_id: db.ID) -> Reaction:
        """
        Get an instance of the reaction wrapper corresponding the reaction ID. The instance may be newly constructed or
        retrieved from the cache.

        Parameters
        ----------
        reaction_id : db.ID
            The reaction database ID.

        Returns
        -------
        Reaction
            The reaction wrapper instance.
        """
        int_id = int(reaction_id.string(), 16)
        if int_id not in self._reactions:
            self._reactions[int_id] = Reaction(reaction_id, self._manager, self._electronic_model, self._hessian_model,
                                               self._aggregate_cache, self._only_electronic)
        return self._reactions[int_id]
