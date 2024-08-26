#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Tuple, List, Dict, Optional
from abc import ABC, abstractmethod
import numpy as np

import scine_database as db

from .model_combinations import ModelCombination
from .db_object_wrappers.ensemble_wrapper import Ensemble
from .db_object_wrappers.aggregate_wrapper import Aggregate
from .db_object_wrappers.reaction_wrapper import Reaction
from .db_object_wrappers.reaction_cache import ReactionCache
from .db_object_wrappers.wrapper_caches import MultiModelCacheFactory
from .db_object_wrappers.thermodynamic_properties import ReferenceState, PlaceHolderReferenceState


class Uncertainty(ABC):

    @abstractmethod
    def lower(self, _: Ensemble) -> float:
        raise NotImplementedError

    @abstractmethod
    def upper(self, _: Ensemble) -> float:
        raise NotImplementedError


class ConstantUncertainty(Uncertainty):
    def __init__(self, uncertainty_bounds: Tuple[float, float]) -> None:
        """
        Constant uncertainty within the given bound.

        Parameters:
        -----------
            uncertainty_bounds : Tuple[float, float]
                Lower and upper bound for the uncertainty in J/mol.
        """
        self._bounds = uncertainty_bounds
        assert self._bounds[0] is not None
        assert self._bounds[1] is not None
        if self._bounds[0] < 0.0 or self._bounds[1] < 0:
            raise RuntimeError("Error: The uncertainty bounds must be positive.")

    def get_uncertainty_bounds(self) -> Tuple[float, float]:
        """
        Getter for the uncertainty bounds.
        """
        return self._bounds

    def lower(self, _: Ensemble) -> float:
        """
        Lower parameter bound.

        Parameters
        ----------
        ensemble : Ensemble
            The ensemble to calculate the uncertainty bound for.
        """
        return self._bounds[0]  # in J/mol

    def upper(self, _: Ensemble) -> float:
        """
        Upper parameter bound.

        Parameters
        ----------
        ensemble : Ensemble
            The ensemble to calculate the uncertainty bound for.
        """
        return self._bounds[1]  # in J/mol


class AtomWiseUncertainty(Uncertainty):
    def __init__(self, atom_wise_uncertainties: Dict[str, Tuple[float, float]]) -> None:
        """
        Uncertainty given as the sum over constant atom contributions.

        Parameters:
        ----------
        atom_wise_uncertainties :  Dict[str, Tuple[float, float]]
            For each element encoded by its element string, the lower and upper uncertainty.
        """
        self._atom_wise_uncertainties = atom_wise_uncertainties

    def lower(self, ensemble: Ensemble) -> float:
        counts = ensemble.get_element_count()
        uncertainty = 0.0
        for element, count in counts.items():
            uncertainty += self._atom_wise_uncertainties[element][0] * count
        return uncertainty  # in J/mol

    def upper(self, ensemble: Ensemble) -> float:
        counts = ensemble.get_element_count()
        uncertainty = 0.0
        for element, count in counts.items():
            uncertainty += self._atom_wise_uncertainties[element][1] * count
        return uncertainty  # in J/mol


class StandardDeviationUncertainty(Uncertainty):
    def __init__(self, model_combinations: List[ModelCombination],
                 fall_back_uncertainty: float,
                 only_electronic: bool = False,
                 reference_state: ReferenceState = PlaceHolderReferenceState(),
                 minimum_uncertainty: float = 0.0) -> None:
        """
        Take the standard deviation of the free energy of activation calculated with multiple electronic structure model
        combinations as the uncertainty.

        Parameters
        ----------
        model_combinations : List[ModelCombination]
            A list of model combinations.
        fall_back_uncertainty : float
            If only one value is available for the free energy of activation this value is returned as its uncertainty
            instead.
        only_electronic : bool
            If true, only the electronic energy is when approximating the free energy of activation.
        reference_state : ReferenceState
            The reference state at which the free energy of activation is calculated.
        minimum_uncertainty : float
            A minimum uncertainty value.
        """
        super().__init__()
        self._model_combinations: List[ModelCombination] = model_combinations
        self._fall_back = fall_back_uncertainty
        self._alternative_wrappers: Dict[int, List[Reaction]] = {}
        self._only_electronic = only_electronic
        self._caches: Optional[List[ReactionCache]] = None
        self._minimum = minimum_uncertainty
        if reference_state is None or isinstance(reference_state, PlaceHolderReferenceState):
            assert model_combinations
            reference_state = ReferenceState(float(model_combinations[0].electronic_model.temperature),
                                             float(model_combinations[0].electronic_model.pressure))
        self._reference_state: ReferenceState = reference_state

    def _get_caches(self, manager: db.Manager) -> List[ReactionCache]:
        if self._caches is None:
            fac = MultiModelCacheFactory()
            self._caches = [fac.get_reaction_cache(self._only_electronic, comb, manager)
                            for comb in self._model_combinations]
        return self._caches

    def _get_alternative_wrappers(self, reaction: Reaction) -> List[Reaction]:
        int_id = int(reaction)
        if int_id not in self._alternative_wrappers:
            self._alternative_wrappers[int_id] = [c.get_or_produce(reaction.get_db_id()) for c in self._get_caches(
                reaction.get_manager())]
        return self._alternative_wrappers[int_id]

    def _get_standard_deviation(self, reaction: Reaction) -> float:
        """
        Calculates the standard deviation of the free energy of activation.
        """
        reactions = self._get_alternative_wrappers(reaction)
        barriers = [r.get_free_energy_of_activation(self._reference_state, True)[0] for r in reactions if r.complete()]
        if len(barriers) < 2:
            return self._fall_back
        return np.std(np.asarray(barriers))

    def lower(self, ensemble: Ensemble) -> float:
        if not isinstance(ensemble, Reaction):
            raise RuntimeError("Error: Model spread uncertainty is only implemented for reactions.")
        standard_deviation = self._get_standard_deviation(ensemble)
        return max(standard_deviation, self._minimum)

    def upper(self, ensemble: Ensemble) -> float:
        return self.lower(ensemble)


class UncertaintyEstimator(ABC):
    """
    This is the base class for all uncertainty estimators.
    """
    @abstractmethod
    def get_uncertainty(self, _: Ensemble) -> Uncertainty:
        """
        Returns the uncertainty for a given ensemble (Reaction/Aggregate).
        """
        raise NotImplementedError


class ZeroUncertainty(UncertaintyEstimator):
    """
    Always return zero as the uncertainty.
    """

    def get_uncertainty(self, _: Ensemble) -> Uncertainty:
        return ConstantUncertainty((0.0, 0.0))


class ModelCombinationBasedUncertaintyEstimator(UncertaintyEstimator):
    def __init__(self, uncertainties_for_models: List[Tuple[ModelCombination, Uncertainty, Uncertainty]]) -> None:
        """
        This uncertainty estimator provides uncertainties for Reactions and Aggregates based on their model combination.

        Parameters
        ----------
        uncertainties_for_models : List[Tuple[ModelCombination, Uncertainty, Uncertainty]]
            For each model combination we have one uncertainty for Aggregates and one for Reactions.
        """
        super().__init__()
        self._uncertainties_for_models = uncertainties_for_models

    def get_model_uncertainties(self) -> List[Tuple[ModelCombination, Uncertainty, Uncertainty]]:
        """
        Getter for the tuples of model combinations and uncertainties. The first uncertainty is applied to the
        free energies, the second to activation energies.
        """
        return self._uncertainties_for_models

    def get_uncertainty(self, ensemble: Ensemble) -> Uncertainty:
        model_combination = ensemble.get_model_combination()
        for combination, aggregate_uncertainty, reaction_uncertainty in self._uncertainties_for_models:
            if combination == model_combination:
                return aggregate_uncertainty if isinstance(ensemble, Aggregate) else reaction_uncertainty
        raise RuntimeError("Error: There is no uncertainty available for the given model combination.")
