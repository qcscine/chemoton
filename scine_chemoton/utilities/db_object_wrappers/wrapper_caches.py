#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, Optional, Union

import scine_database as db

from .aggregate_cache import AggregateCache
from .reaction_cache import ReactionCache
from ..model_combinations import ModelCombination
from scine_chemoton.utilities.db_object_wrappers.aggregate_wrapper import Aggregate
from scine_chemoton.utilities.db_object_wrappers.reaction_wrapper import Reaction


class MultiModelCacheFactory(object):
    """
    Singleton to create multi model aggregate/reaction caches.
    """
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(MultiModelCacheFactory, cls).__new__(cls)
            cls.instance.only_electronic_caches_aggregates: List[AggregateCache] = []
            cls.instance.caches_aggregates: List[AggregateCache] = []
            cls.instance.only_electronic_caches_reactions: List[ReactionCache] = []
            cls.instance.caches_reactions: List[ReactionCache] = []
            cls.instance.model_combinations: List[ModelCombination] = []
        return cls.instance

    def clear(self) -> None:
        """
        Clear caches.
        """
        # pylint: disable=attribute-defined-outside-init
        self.model_combinations: List[ModelCombination] = []
        self.caches_reactions: List[ReactionCache] = []
        self.caches_aggregates: List[AggregateCache] = []
        self.only_electronic_caches_reactions: List[ReactionCache] = []
        self.only_electronic_caches_aggregates: List[AggregateCache] = []
        # pylint: enable=attribute-defined-outside-init

    def _get_cache_index(self, model_combination: ModelCombination, manager: db.Manager) -> int:
        # pylint: disable=no-member
        for i, combi in enumerate(self.model_combinations):  # type: ignore
            if combi == model_combination:
                return i
        self._add_caches(model_combination, manager)
        return len(self.model_combinations) - 1  # type: ignore
        # pylint: enable=no-member

    def _add_caches(self, model_combination: ModelCombination, manager: db.Manager):
        # pylint: disable=no-member
        h_model = model_combination.hessian_model
        e_model = model_combination.electronic_model
        self.model_combinations.append(model_combination)  # type: ignore
        self.only_electronic_caches_aggregates.append(AggregateCache(manager, e_model, h_model, True))  # type: ignore
        self.caches_aggregates.append(AggregateCache(manager, e_model, h_model, False))  # type: ignore
        self.only_electronic_caches_reactions.append(  # type: ignore
            ReactionCache(manager, e_model, h_model, self.only_electronic_caches_aggregates[-1], True))  # type: ignore
        self.caches_reactions.append(  # type: ignore
            ReactionCache(manager, e_model, h_model, self.caches_aggregates[-1], False))  # type: ignore
        # pylint: enable=no-member

    def get_aggregates_cache(self, only_electronic: bool, model_combination, manager: db.Manager) -> AggregateCache:
        """
        Get an aggregate cache corresponding to the given parameters

        Parameters
        ----------
        only_electronic : bool
            If true, only the electronic energy is used to approximate free energies.
        model_combination : ModelCombination
            The model combination with which aggregate wrappers will be constructed.
        manager : db.Manager
            The database manager.
        """
        # pylint: disable=no-member
        ind = self._get_cache_index(model_combination, manager)
        if only_electronic:
            return self.only_electronic_caches_aggregates[ind]  # type: ignore
        return self.caches_aggregates[ind]  # type: ignore
        # pylint: enable=no-member

    def get_reaction_cache(self, only_electronic: bool, model_combination, manager: db.Manager) -> ReactionCache:
        """
        Get a reaction cache corresponding to the given parameters

        Parameters
        ----------
        only_electronic : bool
            If true, only the electronic energy is used to approximate free energies.
        model_combination : ModelCombination
            The model combination with which reaction wrappers will be constructed.
        manager : db.Manager
            The database manager.
        """
        # pylint: disable=no-member
        ind = self._get_cache_index(model_combination, manager)
        if only_electronic:
            return self.only_electronic_caches_reactions[ind]  # type: ignore
        return self.caches_reactions[ind]  # type: ignore
        # pylint: enable=no-member


class MultipleModelCache:
    """
    Base class for hierarchical ordered models, aggregate and reaction sets, i.e., the model order is relevant.
    It is assumed that the model importance/accuracy decreases with its index in the given model combination list.
    """

    def __init__(self) -> None:
        self._caches: List[Union[AggregateCache, ReactionCache]] = []
        self._model_combinations: List[ModelCombination] = []

    def get_or_produce(self, db_id: db.ID) -> Union[Aggregate, Reaction]:
        """
        Get an aggregate or reaction from the caches.
        """
        for cache in self._caches:
            aggregate = cache.get_or_produce(db_id)
            if aggregate.complete():
                return aggregate
        assert self._caches
        return self._caches[0].get_or_produce(db_id)

    def get_cache(self, model_combination: ModelCombination) -> Optional[Union[AggregateCache, ReactionCache]]:
        """
        Getter for a cache corresponding to the given model combination.

        Parameters
        ----------
        model_combination : ModelCombination
            The model combination.

        Returns
        -------
        Optional[Union[AggregateCache, ReactionCache]]
            The cache.
        """
        if len(self._caches) != len(self._model_combinations):
            raise RuntimeError("MultipleModelCache was initialized with different model combinations and caches.")
        for cache, combination in zip(self._caches, self._model_combinations):
            if combination == model_combination:
                return cache
        return None

    def has_cache(self, model_combination: ModelCombination) -> bool:
        """
        Checks if there is a cache with the given model combination.

        Parameters
        ----------
        model_combination : ModelCombination
            The model combination.

        Returns
        -------
        Returns true if there is a cache with the given model combination, otherwise false.
        """
        return self.get_cache(model_combination) is not None

    def get_caches(self) -> List[Union[AggregateCache, ReactionCache]]:
        """
        Getter for all caches.
        """
        return self._caches

    def get_model_combinations(self):
        """
        Getter for all model combinations associated to the caches.
        """
        return self._model_combinations


class MultiModelAggregateCache(MultipleModelCache):
    """
    Multi model cache for aggregate wrappers.

    Parameters
    ----------
    manager : Manager
        The database manager.
    model_combinations : List[ModelCombination]
        The model combinations.
    only_electronic : bool
        If true, free energies are approximated by their electronic energy contribution only.
    """

    def __init__(self, manager: db.Manager, model_combinations: List[ModelCombination],
                 only_electronic: bool = False) -> None:
        super().__init__()
        self._caches = []
        self._only_electronic = only_electronic
        self._model_combinations = []
        factory = MultiModelCacheFactory()
        for combi in model_combinations:
            if not self.has_cache(combi):
                self._model_combinations.append(combi)
                self._caches.append(factory.get_aggregates_cache(self._only_electronic, combi, manager))


class MultiModelReactionCache(MultipleModelCache):
    """
    Multi model cache for reaction wrappers.

    Parameters
    ----------
    manager : Manager
        The database manager.
    model_combinations : List[ModelCombination]
        The model combinations.
    only_electronic : bool
        If true, free energies are approximated by their electronic energy contribution only.
    """

    def __init__(self, manager: db.Manager, model_combinations: List[ModelCombination],
                 only_electronic: bool = False) -> None:
        super().__init__()
        self._caches = []
        self._only_electronic = only_electronic
        self._model_combinations: List[ModelCombination] = []
        factory = MultiModelCacheFactory()
        for combi in model_combinations:
            if not self.has_cache(combi):
                self._model_combinations.append(combi)
                self._caches.append(factory.get_reaction_cache(self._only_electronic, combi, manager))
