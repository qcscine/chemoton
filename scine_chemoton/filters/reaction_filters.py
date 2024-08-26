#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Optional, List, Set
from json import dumps
from abc import ABC, abstractmethod

from scine_chemoton.gears import HoldsCollections, HasName
from scine_chemoton.utilities.db_object_wrappers.reaction_cache import ReactionCache
from scine_chemoton.utilities.db_object_wrappers.wrapper_caches import MultiModelCacheFactory
from scine_chemoton.utilities.model_combinations import ModelCombination
from scine_chemoton.utilities.db_object_wrappers.thermodynamic_properties import ReferenceState
from scine_database.concentration_query_functions import query_reaction_flux_with_model
from scine_database.queries import model_query
from scine_chemoton.filters.elementary_step_filters import (
    StructureModelFilter,
    ConsistentEnergyModelFilter,
    ElementaryStepFilter
)

# Third party imports
import scine_database as db


class ReactionFilter(HoldsCollections, HasName):
    """
    This class provides the interface for filtering reactions.
    """

    def __init__(self) -> None:
        super().__init__()
        self._required_collections = ["manager", "calculations", "compounds", "flasks", "properties", "structures"]
        self._remove_chemoton_from_name()

    def __and__(self, o):
        if not isinstance(o, ReactionFilter):
            raise TypeError("ReactionFilter expects ReactionFilter "
                            "(or derived class) to chain with.")
        return ReactionFilterAndArray([self, o])

    def __or__(self, o):
        if not isinstance(o, ReactionFilter):
            raise TypeError("ReactionFilter expects ReactionFilter "
                            "(or derived class) to chain with.")
        return ReactionFilterOrArray([self, o])

    def filter(self, _: db.Reaction) -> bool:
        return True


class PlaceHolderReactionFilter(ReactionFilter):
    def __init__(self) -> None:
        """
        Place-holder reaction filter that should be replaced by other ReactionFilters. This avoids using the
        optional/None type arguments in options or member variables.
        """
        super().__init__()
        self._required_collections = []
        self._remove_chemoton_from_name()

    def filter(self, _: db.Reaction) -> bool:
        raise NotImplementedError


class ReactionFilterAndArray(ReactionFilter):
    """
    Logical and combination between reaction filters.
    """

    def __init__(self, filters: Optional[List[ReactionFilter]] = None) -> None:
        super().__init__()
        if filters is None:
            filters = []
        self._filters = filters
        for f in filters:
            if not isinstance(f, ReactionFilter):
                raise TypeError("ReactionFilterAndArray expects ReactionFilter "
                                "(or derived class) to chain with.")
        self._join_names(self._filters)

    def filter(self, reaction: db.Reaction) -> bool:
        return all(f.filter(reaction) for f in self._filters)

    def initialize_collections(self, manager: db.Manager) -> None:
        for f in self._filters:
            f.initialize_collections(manager)

    def __iter__(self):
        return (f for f in self._filters)

    def __setitem__(self, key, value):
        self._filters[key] = value


class ReactionFilterOrArray(ReactionFilter):
    """
    Logical or combination for reaction filters.
    """

    def __init__(self, filters: Optional[List[ReactionFilter]] = None) -> None:
        super().__init__()
        if filters is None:
            filters = []
        self._filters = filters
        for f in filters:
            if not isinstance(f, ReactionFilter):
                raise TypeError("ReactionFilterOrArray expects ReactionFilter "
                                "(or derived class) to chain with.")
        self._join_names(self._filters)

    def filter(self, reaction: db.Reaction) -> bool:
        return any(f.filter(reaction) for f in self._filters)

    def initialize_collections(self, manager: db.Manager) -> None:
        for f in self._filters:
            f.initialize_collections(manager)

    def __iter__(self):
        return (f for f in self._filters)

    def __setitem__(self, key, value):
        self._filters[key] = value


class ReactionCachedBasedFilter(ReactionFilter, ABC):
    """
    A filter that uses a reaction cache in the background.

    Parameters
    ----------
    model_combination : ModelCombination
        The model combination for the reaction cache.
    only_electronic_energies : bool
        If true, only electronic energies are evaluated by the reaction cache if newly constructed.
        By default, False.
    thermodynamic_reference_state : ReferenceState, optional
        The reference state for the reaction cache.
    """

    def __init__(self, model_combination: ModelCombination, only_electronic_energies: bool = False,
                 thermodynamic_reference_state: Optional[ReferenceState] = None) -> None:
        super().__init__()
        self._reaction_cache: Optional[ReactionCache] = None
        self._model_combination = model_combination
        self._reference_state = thermodynamic_reference_state
        self._only_electronic_energies = only_electronic_energies
        assert isinstance(model_combination, ModelCombination)
        self._init_reference_state()

    @abstractmethod
    def filter(self, reaction: db.Reaction) -> bool:
        raise NotImplementedError

    def set_cache(self, reaction_cache: ReactionCache) -> None:
        self._reaction_cache = reaction_cache

    def initialize_collections(self, manager: db.Manager):
        super().initialize_collections(manager)
        if self._reaction_cache is None:
            self._reaction_cache = MultiModelCacheFactory().get_reaction_cache(self._only_electronic_energies,
                                                                               self._model_combination, self._manager)
        assert isinstance(self._reaction_cache, ReactionCache)

    def _init_reference_state(self) -> None:
        if self._reference_state is None:
            electronic_model = self._model_combination.electronic_model
            self._reference_state = ReferenceState(float(electronic_model.temperature),
                                                   float(electronic_model.pressure))
        assert isinstance(self._reference_state, ReferenceState)

    def get_cache(self) -> ReactionCache:
        assert self._reaction_cache
        return self._reaction_cache


class ReactionBarrierFilter(ReactionCachedBasedFilter):
    """
    Filters all reactions that are not accessible or have a barrier larger than max_barrier from the reachable
    reaction side. A reaction side is considered accessible if all aggregates on the side are enabled for
    exploration. The free energy barrier is calculated as the free energy difference between reactants and
    transition state if available. Note that this does not track individual elementary steps, but only considers
    the aggregate/transition state ensembles. For more information see gears/kinetic_modeling/reaction_wrapper.py

    Parameters
    ----------
    max_barrier : float
        Maximum barrier cutoff in kJ/mol.
    model_combination : ModelCombination
        The model combination with which to calculate the free energies.
    only_electronic_energies : bool
        If true, only the electronic energies are used as a cutoff criterion. By default, False.
    thermodynamic_reference_state : Optional[ReferenceState]
        Optional thermodynamic reference state (temperature, pressure). If none is given, the reference state
        is constructed from the given electronic structure model.
    """

    def __init__(self, max_barrier: float, model_combination: ModelCombination, only_electronic_energies: bool = False,
                 thermodynamic_reference_state: Optional[ReferenceState] = None) -> None:
        super().__init__(model_combination, only_electronic_energies, thermodynamic_reference_state)
        self._max_barrier = max_barrier * 1e+3  # in J/mol -> convenient for the comparison.

    def filter(self, reaction: db.Reaction) -> bool:
        assert self._reaction_cache
        assert self._reference_state
        rxn = self._reaction_cache.get_or_produce(reaction.id())
        accessible_lhs = all([a.get_db_object().explore() for a in rxn.get_lhs_aggregates()])
        accessible_rhs = all([a.get_db_object().explore() for a in rxn.get_rhs_aggregates()])
        # Early skip to avoid calculating the reaction barriers.
        if not accessible_rhs and not accessible_lhs:
            return False
        lhs_barrier, rhs_barrier = rxn.get_free_energy_of_activation(self._reference_state, in_j_per_mol=True)
        lhs_passed = accessible_lhs and False if lhs_barrier is None else lhs_barrier < self._max_barrier
        rhs_passed = accessible_rhs and False if rhs_barrier is None else rhs_barrier < self._max_barrier
        return lhs_passed or rhs_passed


class BarrierlessReactionFilter(ReactionCachedBasedFilter):
    """
    Filter all barrierless reactions. If "exclude_barrierless" is True, the barrierless reactions are excluded.
    If "exclude_barrierless" is False, only the barrierless reactions are accepted. This filter is based on
    reaction wrapper caches in the background to determine which elementary steps are accepted for the reaction
    check.

    Parameters
    ----------
        model_combination : ModelCombination
            The electronic structure model combinations used to construct the reaction wrapper objects.
        exclude_barrierless : bool
            If true, barrierless reactions are excluded. If false, only barrierless reactions are accepted.
        only_electronic_energies : bool, optional
            If true, only electronic energies are considered in the reaction cache. By default, False.
        thermodynamic_reference_state : ReferenceState, optional
            The thermodynamic reference state of the reaction cache. By default, None.
    """

    def __init__(self, model_combination: ModelCombination, exclude_barrierless: bool = True,
                 only_electronic_energies: bool = False,
                 thermodynamic_reference_state: Optional[ReferenceState] = None) -> None:
        super().__init__(model_combination, only_electronic_energies, thermodynamic_reference_state)
        self._exclude_barrierless = exclude_barrierless

    def filter(self, reaction: db.Reaction) -> bool:
        assert self._reaction_cache
        assert self._reference_state  # only needed for mypy
        barrierless = self._reaction_cache.get_or_produce(reaction.id()).barrierless(self._reference_state)
        if self._exclude_barrierless:
            return not barrierless
        return barrierless


class ReactionNumberPropertyFilter(ReactionFilter):
    """
    Filters reaction based on the value of a number property attached to the reaction.

    Parameters
    ----------
    property_label : str
        The label of the property.
    threshold : float
        The threshold value that must be exceeded by the property.
    model : db.Model
        The electronic structure model attached to the property.
    threshold_must_be_exceeded : bool
        If true, the given threshold must be exceeded to return true by the filter. If false, the property must be
        lower or equal to the given the threshold. By default, True.
    """

    def __init__(self, property_label: str, threshold: float, model: db.Model,
                 threshold_must_be_exceeded: bool = True) -> None:
        super().__init__()
        self._property_label = property_label
        self._threshold = threshold
        self._threshold_must_be_exceeded = threshold_must_be_exceeded
        self._model = model

    def filter(self, reaction: db.Reaction) -> bool:
        value = query_reaction_flux_with_model(self._property_label, reaction, self._compounds, self._flasks,
                                               self._structures, self._properties, self._model)
        if self._threshold_must_be_exceeded:
            return value > self._threshold
        return value <= self._threshold


class StopDuringExploration(ReactionFilter):
    """
    This filter returns True as long as exploration or single point calculations are pending, new, or on hold.

    Parameters
    ----------
    orders_to_wait_for : List[str], optional
        List of jobs that need to be done before allowing the filter to pass. By default,
        ["scine_react_complex_nt2", "scine_single_point"]
    model : db.Model, optional
        The electronic structure model of the calculations to wait for. If none is provided, any calculation with
        a job order in the list orders_to_wait_for will ensure that the filter returns False.
    """

    def __init__(self, orders_to_wait_for: Optional[List[str]] = None, model: Optional[db.Model] = None) -> None:
        super().__init__()
        if orders_to_wait_for is None:
            orders_to_wait_for = ["scine_react_complex_nt2", "scine_single_point", "scine_hessian"]
        self._orders_to_wait_for = orders_to_wait_for
        self._model = model

    def filter(self, _: db.Reaction) -> bool:
        selection = {
            "$and": [
                {"status": {"$in": ["hold", "new", "pending"]}},
                {"job.order": {"$in": self._orders_to_wait_for}}
            ]
        }
        if self._model is not None:
            selection["$and"] += model_query(self._model)
        return self._calculations.get_one_calculation(dumps(selection)) is None


class ReactionIDFilter(ReactionFilter):
    """
    Filter reactions based on their database id.

    Parameters
    ----------
    id_white_list : List[db.ID]
        List of reaction ids that are accepted by the filter.
    """

    def __init__(self, id_white_list: List[db.ID]) -> None:
        super().__init__()
        self._id_white_set: Set[int] = set([int(r_id.string(), 16) for r_id in id_white_list])

    def filter(self, reaction: db.Reaction) -> bool:
        return int(reaction.id().string(), 16) in self._id_white_set


class ReactionHasStepWithModel(ReactionFilter):
    """
    This filter returns true, if the reaction has an elementary step with the given electronic structure model.

    Parameters
    ----------
    model : db.Model
        The electronic structure model.
    check_only_energies : bool, optional
        If true, only electronic energies must be present for every structure on the elementary step. Otherwise,
        also the structures must have the given model. By default, False.
    """

    def __init__(self, model: db.Model, check_only_energies: bool = False) -> None:
        super().__init__()
        self._required_collections = ["elementary_steps"]
        if check_only_energies:
            self._step_filter: ElementaryStepFilter = ConsistentEnergyModelFilter(model)
        else:
            self._step_filter = StructureModelFilter(model)

    def initialize_collections(self, manager: db.Manager):
        super().initialize_collections(manager)
        self._step_filter.initialize_collections(manager)

    def filter(self, reaction: db.Reaction) -> bool:
        return any(self._step_filter.filter(db.ElementaryStep(s_id, self._elementary_steps))
                   for s_id in reaction.get_elementary_steps())


class MaximumTransitionStateEnergyFilter(ReactionCachedBasedFilter):
    """
    This filter returns true if the free energy approximation for the transition state of a given reaction is lower
    than the given threshold.

    Application case: Exploration on a single potential energy surface.

    Parameters
    ----------
    model_combination: ModelCombination
        The model combination to evaluate the free energies with.
    max_energy: float
        The energy threshold in Hartree.
    only_electronic_energies: bool
        If true, free energies are approximated by their electronic energy alone.
    thermodynamic_reference_state: Optional[ReferenceState]
        The thermodynamic reference state (pressure and temperature).
    """

    def __init__(self, model_combination: ModelCombination,
                 max_energy: float,
                 only_electronic_energies: bool = False,
                 thermodynamic_reference_state: Optional[ReferenceState] = None) -> None:
        super().__init__(model_combination, only_electronic_energies, thermodynamic_reference_state)
        self.__max_energy = max_energy

    def filter(self, reaction: db.Reaction) -> bool:
        assert self._reaction_cache  # only for mypy
        assert self._reference_state  # only for mypy
        reaction_wrapper = self._reaction_cache.get_or_produce(reaction.id())
        ts_free_energy = reaction_wrapper.get_transition_state_free_energy(self._reference_state)
        if ts_free_energy is None:
            return False
        return ts_free_energy <= self.__max_energy
