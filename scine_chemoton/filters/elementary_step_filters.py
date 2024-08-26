#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Optional, List
from json import dumps

from scine_chemoton.gears import HoldsCollections, HasName
from scine_chemoton.utilities.model_combinations import ModelCombination
from scine_database.energy_query_functions import get_barriers_for_elementary_step_by_type, get_energy_for_structure

# Third party imports
import scine_database as db


class ElementaryStepFilter(HoldsCollections, HasName):
    """
    This class defines a generic filter for elementary steps.
    This base class does not filter any elementary step.
    """

    def __init__(self) -> None:
        super().__init__()
        self._required_collections = ["manager", "calculations", "compounds", "flasks", "properties", "structures"]
        self._remove_chemoton_from_name()

    def __and__(self, o):
        if not isinstance(o, ElementaryStepFilter):
            raise TypeError("ElementaryStepFilter expects ReactionFilter "
                            "(or derived class) to chain with.")
        return ElementaryStepFilterAndArray([self, o])

    def __or__(self, o):
        if not isinstance(o, ElementaryStepFilter):
            raise TypeError("ElementaryStepFilter expects ReactionFilter "
                            "(or derived class) to chain with.")
        return ElementaryStepFilterOrArray([self, o])

    def filter(self, _: db.ElementaryStep) -> bool:
        return True


class PlaceHolderElementaryStepFilter(ElementaryStepFilter):
    """
    Place-holder for an elementary step filter that can be used instead of a None default argument.
    """

    def __init__(self) -> None:
        super().__init__()
        self._required_collections = []
        self._remove_chemoton_from_name()

    def filter(self, _: db.ElementaryStep) -> bool:
        raise NotImplementedError


class ElementaryStepFilterAndArray(ElementaryStepFilter):
    """
    Logical and combination for elementary step filters.
    """

    def __init__(self, filters: Optional[List[ElementaryStepFilter]] = None) -> None:
        super().__init__()
        if filters is None:
            filters = []
        self._filters = filters
        for f in filters:
            if not isinstance(f, ElementaryStepFilter):
                raise TypeError("ElementaryStepFilterAndArray expects ReactionFilter "
                                "(or derived class) to chain with.")
        self._join_names(self._filters)

    def filter(self, step: db.ElementaryStep) -> bool:
        return all(f.filter(step) for f in self._filters)

    def initialize_collections(self, manager: db.Manager) -> None:
        for f in self._filters:
            f.initialize_collections(manager)

    def __iter__(self):
        return (f for f in self._filters)

    def __setitem__(self, key, value):
        self._filters[key] = value


class ElementaryStepFilterOrArray(ElementaryStepFilter):
    """
    Logical or combination for elementary step filters.
    """

    def __init__(self, filters: Optional[List[ElementaryStepFilter]] = None) -> None:
        super().__init__()
        if filters is None:
            filters = []
        self._filters = filters
        for f in filters:
            if not isinstance(f, ElementaryStepFilter):
                raise TypeError("ElementaryStepFilterOrArray expects ElementaryStepFilter "
                                "(or derived class) to chain with.")
        self._join_names(self._filters)

    def filter(self, step: db.ElementaryStep) -> bool:
        return any(f.filter(step) for f in self._filters)

    def initialize_collections(self, manager: db.Manager) -> None:
        for f in self._filters:
            f.initialize_collections(manager)

    def __iter__(self):
        return (f for f in self._filters)

    def __setitem__(self, key, value):
        self._filters[key] = value


class ElementaryStepBarrierFilter(ElementaryStepFilter):
    """
    Maximum barrier filter for elementary steps.

    Parameters
    ----------
    max_barrier : float
        The maximum barrier in kJ/mol.
    model_combination : ModelCombination
        The electronic structure model combination with which the barriers are evaluated.
    only_electronic_energies : bool, optional
        If true, only electronic energies are checked. By default, False.
    accessible_side : db.Side, optional
        The side of the elementary step considered accessible. If both sides are accessible only one barrier
        (lhs/rhs) must be lower than the max barrier. Otherwise, the respective side of the step is taken into
        account.
    """

    def __init__(self, max_barrier: float, model_combination: ModelCombination, only_electronic_energies: bool = False,
                 accessible_side: db.Side = db.Side.BOTH) -> None:
        super().__init__()
        self._model_combination = model_combination
        self._only_electronic_energies = only_electronic_energies
        self._max_barrier = max_barrier
        self._electronic_label = "electronic_energy"
        self._free_energy_correction_label = "gibbs_energy_correction"
        self._accessible_side = accessible_side

    def filter(self, step: db.ElementaryStep):
        lhs_barrier, rhs_barrier = get_barriers_for_elementary_step_by_type(step, self._electronic_label,
                                                                            self._model_combination.electronic_model,
                                                                            self._structures, self._properties)
        if lhs_barrier is None or rhs_barrier is None:
            return False
        if not self._only_electronic_energies:
            lhs_g, rhs_g = get_barriers_for_elementary_step_by_type(step, self._free_energy_correction_label,
                                                                    self._model_combination.hessian_model,
                                                                    self._structures, self._properties)
            if lhs_g is None or rhs_g is None:
                return False
            lhs_barrier += lhs_g
            rhs_barrier += rhs_g

        valid_lhs = lhs_barrier < self._max_barrier
        valid_rhs = rhs_barrier < self._max_barrier

        if valid_lhs and self._accessible_side in [db.Side.LHS, db.Side.BOTH]:
            return True
        if valid_rhs and self._accessible_side in [db.Side.RHS, db.Side.BOTH]:
            return True
        return False


class BarrierlessElementaryStepFilter(ElementaryStepFilter):
    """
    Filter for all barrier-less elementary steps.

    Parameters
    ----------
    exclude_barrierless : bool, optional
        If true, barrier-less elementary are excluded. If false, only barrier-less elementary steps are accepted.
    """

    def __init__(self, exclude_barrierless: bool = True) -> None:
        super().__init__()
        self._exclude_barrierless_steps = exclude_barrierless

    def filter(self, step: db.ElementaryStep) -> bool:
        barrierless = step.get_type() == db.ElementaryStepType.BARRIERLESS
        if self._exclude_barrierless_steps:
            return not barrierless
        return barrierless


class StopDuringExploration(ElementaryStepFilter):
    """
    Do nothing while jobs with the given orders are still on pending, new, or hold.

    Parameters
    ----------
    orders_to_wait_for : List[str], optional
        The job orders to wait for. By default, ["scine_react_complex_nt2", "scine_single_point"].
    """

    def __init__(self, orders_to_wait_for: Optional[List[str]] = None) -> None:
        super().__init__()
        if orders_to_wait_for is None:
            orders_to_wait_for = ["scine_react_complex_nt2", "scine_single_point"]
        self._orders_to_wait_for = orders_to_wait_for

    def filter(self, _: db.ElementaryStep) -> bool:
        selection = {
            "$and": [
                {"status": {"$in": ["hold", "new", "pending"]}},
                {"job.order": {"$in": self._orders_to_wait_for}}
            ]
        }
        return self._calculations.get_one_calculation(dumps(selection)) is None


class ConsistentEnergyModelFilter(ElementaryStepFilter):
    """
    Filter for elementary steps with electronic energies for all structures on the step of the given model.

    Parameters
    ----------
    model : db.Model
        The electronic structure model.
    """

    def __init__(self, model: db.Model) -> None:
        super().__init__()
        self._model = model

    def filter(self, step: db.ElementaryStep) -> bool:
        reactants = step.get_reactants(db.Side.BOTH)
        s_ids = reactants[0] + reactants[1]
        if step.has_transition_state():
            s_ids.append(step.get_transition_state())
        return all(get_energy_for_structure(db.Structure(s_id), "electronic_energy", self._model, self._structures,
                                            self._properties) is not None for s_id in s_ids)


class StructureModelFilter(ElementaryStepFilter):
    """
    Return true if the electronic structure model of the elementary step matches the given model. For completeness,
    all structures of the step are checked.

    Parameters
    ----------
    model : db.Model
        The electronic structure model.
    """

    def __init__(self, model: db.Model) -> None:
        super().__init__()
        self._model = model

    def filter(self, step: db.ElementaryStep) -> bool:
        reactants = step.get_reactants(db.Side.BOTH)
        s_ids = reactants[0] + reactants[1]
        if step.has_transition_state():
            s_ids.append(step.get_transition_state())
        return all(db.Structure(s_id, self._structures).get_model() == self._model for s_id in s_ids)
