#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Optional, Dict, Tuple, List

import scine_database as db
import scine_utilities as utils
from scine_database.energy_query_functions import rate_constant_from_barrier, get_energy_for_structure

from .aggregate_wrapper import Aggregate, AggregateCache
from .thermodynamic_properties import ReferenceState
from .ensemble_wrapper import Ensemble


class ReactionCache:
    """
    A cache for reaction objects.
    """

    def __init__(self, manager: db.Manager, electronic_model: db.Model, hessian_model: db.Model,
                 aggregate_cache: Optional[AggregateCache] = None,
                 only_electronic: bool = False):
        self._manager = manager
        self._electronic_model = electronic_model
        self._hessian_model = hessian_model
        self._only_electronic = only_electronic
        self._reactions: Dict[int, Reaction] = {}
        if aggregate_cache is None:
            aggregate_cache = AggregateCache(manager, electronic_model, hessian_model, only_electronic)
        self._aggregate_cache = aggregate_cache

    def get_or_produce(self, reaction_id: db.ID):
        int_id = int(reaction_id.string(), 16)
        if int_id not in self._reactions:
            self._reactions[int_id] = Reaction(reaction_id, self._manager, self._electronic_model, self._hessian_model,
                                               self._aggregate_cache, self._only_electronic)
        return self._reactions[int_id]


class Reaction(Ensemble):
    """
    A class that wraps around db.Reaction to provide easy access to all thermodynamic contributions (barriers, reaction
    energies etc.). Free energies are calculated with the harmonic oscillator, rigid rotor, particle in a box model
    according to the given thermodynamic reference state (caching + on the fly if necessary).
    """

    def __init__(self, reaction_id: db.ID, manager: db.Manager, electronic_model: db.Model, hessian_model: db.Model,
                 aggregate_cache: Optional[AggregateCache] = None, only_electronic: bool = False):
        """
        Parameters
        ----------
        reaction_id : db.ID
            The database ID.
        manager
            The database manager.
        electronic_model
            The electronic structure model from which the electronic energy is taken.
        hessian_model
            The electronic structure model with which the geometrical hessian was calculated.
        aggregate_cache
            A cache of already existing aggregates (optional).
        only_electronic
            If true, only the electronic energies are used to determine the thermodynamics (optional).
        """
        super().__init__(reaction_id, manager, electronic_model, hessian_model, only_electronic)
        if aggregate_cache is None:
            aggregate_cache = AggregateCache(manager, electronic_model, hessian_model, only_electronic)
        self._elementary_steps = manager.get_collection("elementary_steps")
        reactants = self.get_db_object().get_reactants(db.Side.BOTH)
        self._lhs = [aggregate_cache.get_or_produce(a_id) for a_id in reactants[0]]
        self._rhs = [aggregate_cache.get_or_produce(a_id) for a_id in reactants[1]]
        self._has_barrierless_elementary_step = False  # Barrierless reactions are treated differently (vide infra).
        self._n_steps_last = 0  # Keep track of the number of elementary steps in each reaction.

    def _initialize_db_object(self, manager: db.Manager) -> db.Reaction:
        return db.Reaction(self.get_db_id(), manager.get_collection("reactions"))

    def _update_transition_state_thermodynamics(self):
        """
        Update the thermodynamic property container for the transition states and search for barrier-less elementary
        steps.
        """
        # We may want to remove this early return if we change the way we handle barrier-less reactions
        if self._has_barrierless_elementary_step:
            return
        elementary_step_ids = self.get_db_object().get_elementary_steps()
        assert elementary_step_ids
        barrierless_types = [db.ElementaryStepType.BARRIERLESS, db.ElementaryStepType.MODEL_TRANSFORMATION]
        for step_id in elementary_step_ids:
            step = db.ElementaryStep(step_id, self._elementary_steps)
            if step.get_type() == db.ElementaryStepType.REGULAR:
                _ = self._structure_thermodynamics.get_or_produce(step.get_transition_state())
            elif not self._has_barrierless_elementary_step and step.get_type() in barrierless_types:
                if not self._is_valid_barrierless_step(step):
                    continue
                self._has_barrierless_elementary_step = True
                self._structure_thermodynamics.clear()
                break  # We may want to remove this early break if we change the way we handle barrier-less reactions
        self._n_steps_last = len(elementary_step_ids)

    def _is_valid_barrierless_step(self, step: db.ElementaryStep) -> bool:
        """
        Return true if the elementary step was calculated with the electronic structure model or that at least
        the required electronic energies are available.
        """
        reactant_structures = step.get_reactants(db.Side.BOTH)
        lhs_energies = [get_energy_for_structure(db.Structure(s_id), "electronic_energy",
                                                 self._electronic_model, self._structures, self._properties)
                        for s_id in reactant_structures[0]]
        if None in lhs_energies:
            return False
        rhs_energies = [get_energy_for_structure(db.Structure(s_id), "electronic_energy",
                                                 self._electronic_model, self._structures, self._properties)
                        for s_id in reactant_structures[1]]
        if None in rhs_energies:
            return False
        return True

    def get_transition_state_free_energy(self, reference_state: ReferenceState) -> Optional[float]:
        # We may want to remove this early return if we change the way we handle barrier-less reactions
        if self._has_barrierless_elementary_step:
            return None
        elementary_step_ids = self.get_db_object().get_elementary_steps()
        # Update only if the reference state or the number of elementary steps changed.
        if self._structure_thermodynamics.minimum_values_need_update(reference_state)\
                or self._n_steps_last != len(elementary_step_ids):
            self._update_transition_state_thermodynamics()
            if self._structure_thermodynamics.get_n_cached() == 0:
                return None
        return self._structure_thermodynamics.get_ensemble_gibbs_free_energy(
            reference_state, self._structure_thermodynamics.get_n_cached())

    def get_free_energy_of_activation(self, reference_state: ReferenceState, in_j_per_mol: bool = False)\
            -> Tuple[Optional[float], Optional[float]]:
        """
        Getter for the free energy of activation/barriers as a tuple for lhs and rhs.
        Parameters
        ----------
        reference_state
            The reference state (temperature, and pressure)
        in_j_per_mol
            If true, the barriers are returned in J/mol (NOT kJ/mol)
        Return
        ------
        A tuple for the lhs and rhs barriers. Returns None if the energies are incomplete. For barrier-less reactions,
        one barrier will be the reaction energy, the other zero.
        """
        lhs_energies = [a.get_free_energy(reference_state) for a in self._lhs]
        if None in lhs_energies:
            return None, None
        rhs_energies = [a.get_free_energy(reference_state) for a in self._rhs]
        if None in rhs_energies:
            return None, None
        e_lhs = sum(lhs_energies)  # type: ignore
        e_rhs = sum(rhs_energies)  # type: ignore
        if self.barrierless(reference_state):
            e_ts = max(e_lhs, e_rhs)  # type: ignore
        else:
            e_ts = self.get_transition_state_free_energy(reference_state)  # type: ignore
        if e_ts is None:
            return None, None
        ts_energy: float = max(e_lhs, e_rhs, e_ts)
        lhs_diff = ts_energy - e_lhs
        rhs_diff = ts_energy - e_rhs
        if in_j_per_mol:
            lhs_diff *= utils.KJPERMOL_PER_HARTREE * 1e+3
            rhs_diff *= utils.KJPERMOL_PER_HARTREE * 1e+3
        assert abs(lhs_diff - lhs_diff) < 1e-9  # inf/nan checks
        assert abs(rhs_diff - rhs_diff) < 1e-9  # inf/nan checks
        return lhs_diff, rhs_diff

    @staticmethod
    def get_arrhenius_prefactor(reference_state) -> float:
        return utils.BOLTZMANN_CONSTANT * reference_state.temperature / utils.PLANCK_CONSTANT

    def get_ts_theory_rate_constants(self, reference_state: ReferenceState) -> Tuple[Optional[float], Optional[float]]:
        """
        Getter for the transition state theory based reaction rate constants.
        Parameters
        ----------
        reference_state
            The reference state (temperature, and pressure)
        Returns
        -------
        The transition state theory based reaction rate constants as a tuple for lhs and rhs. May return
        Tuple[None, None] if the energies are incomplete.
        """
        barriers = self.get_free_energy_of_activation(reference_state, False)
        if barriers[0] is None or barriers[1] is None:
            return None, None
        lhs_b = barriers[0] * utils.KJPERMOL_PER_HARTREE
        rhs_b = barriers[1] * utils.KJPERMOL_PER_HARTREE
        t = reference_state.temperature
        return rate_constant_from_barrier(lhs_b, t), rate_constant_from_barrier(rhs_b, t)

    def get_reaction_free_energy(self, reference_state: ReferenceState, in_j_per_mol: bool = False) -> Optional[float]:
        """
        Getter for the reaction free energy.
        Parameters
        ----------
        reference_state
            The reference state (temperature, and pressure)
        in_j_per_mol
            If true, the barriers are returned in J/mol (NOT kJ/mol)
        Returns
        -------
        The reaction free energy. May return None, if the energies are incomplete.
        """
        lhs_energies = [a.get_free_energy(reference_state) for a in self._lhs]
        if None in lhs_energies:
            return None
        rhs_energies = [a.get_free_energy(reference_state) for a in self._rhs]
        if None in rhs_energies:
            return None
        diff = sum(rhs_energies) - sum(lhs_energies)  # type: ignore
        if in_j_per_mol:
            diff *= utils.KJPERMOL_PER_HARTREE * 1e+3  # type: ignore
        return diff

    def barrierless(self, reference_state: ReferenceState) -> bool:
        if self._has_barrierless_elementary_step:
            return True
        self.get_transition_state_free_energy(reference_state)
        return self._has_barrierless_elementary_step

    def get_lhs_aggregates(self) -> List[Aggregate]:
        return self._lhs

    def get_rhs_aggregates(self) -> List[Aggregate]:
        return self._rhs
