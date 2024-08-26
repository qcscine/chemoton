#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Optional, Tuple, List, Dict

import scine_database as db
import scine_utilities as utils
from scine_database.energy_query_functions import rate_constant_from_barrier, get_energy_for_structure

from .aggregate_wrapper import Aggregate
from .aggregate_cache import AggregateCache
from .thermodynamic_properties import ReferenceState
from .ensemble_wrapper import Ensemble


class Reaction(Ensemble):
    """
    A class that wraps around db.Reaction to provide easy access to all thermodynamic contributions (barriers, reaction
    energies etc.). Free energies are calculated with the harmonic oscillator, rigid rotor, particle in a box model
    according to the given thermodynamic reference state (caching + on the fly if necessary).

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

    def __init__(self, reaction_id: db.ID, manager: db.Manager, electronic_model: db.Model, hessian_model: db.Model,
                 aggregate_cache: Optional[AggregateCache] = None, only_electronic: bool = False) -> None:
        super().__init__(reaction_id, manager, electronic_model, hessian_model, only_electronic)
        if aggregate_cache is None:
            aggregate_cache = AggregateCache(manager, electronic_model, hessian_model, only_electronic)
        self._elementary_steps = manager.get_collection("elementary_steps")
        reactants = self.get_db_object().get_reactants(db.Side.BOTH)
        self._lhs = [aggregate_cache.get_or_produce(a_id) for a_id in reactants[0]]
        self._rhs = [aggregate_cache.get_or_produce(a_id) for a_id in reactants[1]]
        self._is_circle_reaction = reactants[0] == reactants[1]
        self._has_barrierless_elementary_step = False  # Barrierless reactions are treated differently (vide infra).
        self._n_steps_last = 0  # Keep track of the number of elementary steps in each reaction.
        self._ts_id_to_step_map: Dict[int, db.ID] = {}

    def circle_reaction(self):
        return self._is_circle_reaction

    def _initialize_db_object(self, manager: db.Manager) -> db.Reaction:
        return db.Reaction(self.get_db_id(), manager.get_collection("reactions"))

    def _update_thermodynamics(self):
        """
        Update the thermodynamic property container for the transition states and search for barrier-less elementary
        steps.
        """
        # We may want to remove this early return if we change the way we handle barrier-less reactions
        if self._has_barrierless_elementary_step:
            return
        elementary_step_ids = self.get_db_object().get_elementary_steps()
        assert elementary_step_ids
        for step_id in elementary_step_ids:
            step = db.ElementaryStep(step_id, self._elementary_steps)
            if not step.analyze() or not step.explore():
                continue
            # update the thermodynamics for the transition state by calling the get_or_produce function once.
            if step.get_type() == db.ElementaryStepType.REGULAR:
                _ = self._structure_thermodynamics.get_or_produce(step.get_transition_state())
                self._ts_id_to_step_map[int(step.get_transition_state().string(), 16)] = step_id
            elif not self._has_barrierless_elementary_step and step.get_type() == db.ElementaryStepType.BARRIERLESS:
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
        reactants = step.get_reactants(db.Side.BOTH)
        return all([get_energy_for_structure(db.Structure(s_id), "electronic_energy", self._electronic_model,
                                             self._structures, self._properties) is not None
                    for s_id in reactants[0] + reactants[1]])

    def get_lhs_free_energy(self, reference_state: ReferenceState) -> Optional[float]:
        """
        Getter for the total free energy of the LHS.

        Parameters
        ----------
        reference_state : ReferenceState
            The reference state.

        Returns
        -------
        Optional[float]
            The total free energy if available. Otherwise, None.
        """
        lhs_energies = [a.get_free_energy(reference_state) for a in self._lhs]
        if None in lhs_energies:
            return None
        return sum(lhs_energies)  # type: ignore

    def get_rhs_free_energy(self, reference_state: ReferenceState) -> Optional[float]:
        """
        Getter for the total free energy of the RHS.

        Parameters
        ----------
        reference_state : ReferenceState
            The reference state.

        Returns
        -------
        Optional[float]
            The total free energy if available. Otherwise, None.
        """
        rhs_energies = [a.get_free_energy(reference_state) for a in self._rhs]
        if None in rhs_energies:
            return None
        return sum(rhs_energies)  # type: ignore

    def get_transition_state_free_energy(self, reference_state: ReferenceState) -> Optional[float]:
        """
        Getter fo the free energy of the transition state ensemble in Hartree.

        Parameters
        ----------
        reference_state
            The reference state (temperature, and pressure)

        Returns
        -------
        The free energy of the transition in Hartree.
        """
        if self._has_barrierless_elementary_step:
            e_lhs = self.get_lhs_free_energy(reference_state)
            if e_lhs is None:
                return None
            e_rhs = self.get_rhs_free_energy(reference_state)
            if e_rhs is None:
                return None
            return max(e_lhs, e_rhs)
        # Update only if the reference state or the number of elementary steps changed.
        self._update_thermodynamics()
        if self._structure_thermodynamics.get_n_cached() == 0:
            return None
        return self._structure_thermodynamics.get_ensemble_gibbs_free_energy(reference_state)

    def get_free_energy_of_activation(self, reference_state: ReferenceState, in_j_per_mol: bool = False)\
            -> Tuple[Optional[float], Optional[float]]:
        """
        Getter for the free energy of activation/barriers as a tuple for lhs and rhs.

        Parameters
        ----------
        reference_state : ReferenceState
            The reference state (temperature, and pressure)
        in_j_per_mol : bool, optional
            If true, the barriers are returned in J/mol (NOT kJ/mol), by default False

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            A tuple for the lhs and rhs barriers. Returns None if the energies are incomplete.
            For barrier-less reactions, one barrier will be the reaction energy, the other zero.
        """
        e_lhs = self.get_lhs_free_energy(reference_state)
        if e_lhs is None:
            return None, None
        e_rhs = self.get_rhs_free_energy(reference_state)
        if e_rhs is None:
            return None, None
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
    def get_arrhenius_prefactor(reference_state: ReferenceState) -> float:
        """
        Getter for the factor kBT/h from Eyring's transition state theory.

        Parameters
        ----------
        reference_state : ReferenceState
            The reference state for the temperature.

        Returns
        -------
        float
            The factor kBT/h.
        """
        return utils.BOLTZMANN_CONSTANT * reference_state.temperature / utils.PLANCK_CONSTANT

    def get_ts_theory_rate_constants(self, reference_state: ReferenceState) -> Tuple[Optional[float], Optional[float]]:
        """
        Getter for the transition state theory based reaction rate constants.

        Parameters
        ----------
        reference_state : ReferenceState
            The reference state (temperature, and pressure)

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
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
        reference_state : ReferenceState
            The reference state (temperature, and pressure)
        in_j_per_mol : bool, optional
            If true, the barriers are returned in J/mol (NOT kJ/mol), by default False

        Returns
        -------
        Optional[float]
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
        """
        Checks if the reaction has a valid barrier-less elementary step.

        Parameters
        ----------
        reference_state : ReferenceState
            The reference state.

        Returns
        -------
        bool
            True if the reaction has a barrier-less elementary step.
        """
        if self._has_barrierless_elementary_step:
            return True
        self.get_transition_state_free_energy(reference_state)
        return self._has_barrierless_elementary_step

    def get_lhs_aggregates(self) -> List[Aggregate]:
        """
        Getter for the aggregate wrappers of the reaction's LHS.

        Returns
        -------
        List[Aggregate]
            The LHS aggregates.
        """
        return self._lhs

    def get_rhs_aggregates(self) -> List[Aggregate]:
        """
        Getter for the aggregate wrappers of the reaction's RHS.

        Returns
        -------
        List[Aggregate]
            The RHS aggregates.
        """
        return self._rhs

    def complete(self) -> bool:
        ts_complete = super(Reaction, self).complete() or self._has_barrierless_elementary_step
        return ts_complete and all([a.complete() for a in self._lhs]) and all([a.complete() for a in self._rhs])

    def analyze(self) -> bool:
        return super().analyze() and all([a.analyze() for a in self.get_lhs_aggregates() + self.get_rhs_aggregates()])

    def get_element_count(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for aggregate in self._lhs:
            aggregate_wise_count = aggregate.get_element_count()
            for key, count in aggregate_wise_count.items():
                if key in counts:
                    counts[key] += count
                else:
                    counts[key] = count
        return counts

    def get_sorted_structure_list(self, reference_state: ReferenceState) -> List[Tuple[db.ID, float]]:
        if self.barrierless(reference_state):
            return []
        # getter will update class members if necessary
        self.get_free_energy_of_activation(reference_state)
        return self._structure_thermodynamics.get_sorted_structure_list(reference_state)

    def get_sorted_structure_step_list(self, reference_state: ReferenceState) -> List[Tuple[db.ID, db.ID, float]]:
        struc_energies = self.get_sorted_structure_list(reference_state)
        return [(s, self._ts_id_to_step_map[int(s.string(), 16)], e) for s, e in struc_energies]

    def get_lowest_n_steps(self, n: int, energy_cut_off: float, reference_state: ReferenceState) -> List[db.ID]:
        """
        Getter for the n elementary steps with the lowest energy transition state.

        Parameters
        ----------
        n : int
            The number of elementary steps.
        energy_cut_off : float
            An energy cutoff after which no additional elementary steps are considered for the list.
        reference_state : ReferenceState
            The thermodynamic reference state used to calculate the free energies.

        Returns
        -------
        List[db.ID]
            The n elementary step IDs with the lowest energy transition state.
        """
        lowest_n_strucs = self.get_lowest_n_structures(n, energy_cut_off, reference_state)
        return [self._ts_id_to_step_map[int(s.string(), 16)] for s in lowest_n_strucs]
