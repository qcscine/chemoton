#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import math
from typing import Optional, Tuple, List, Dict

import numpy as np
import scine_database as db
import scine_utilities as utils
from scine_database.energy_query_functions import rate_constant_from_barrier, get_energy_for_structure

from .aggregate_wrapper import Aggregate
from .aggregate_cache import AggregateCache
from .thermodynamic_properties import ReferenceState
from .ensemble_wrapper import Ensemble
from scine_chemoton.utilities.kinetics.eckart_tunneling import EckartTunneling


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
        self.__transition_state_wavenumber: Optional[float] = None

    def set_test_transition_state_wavenumber(self, test_transition_state_wavenumber: float) -> None:
        """
        Set the wavenumber for the transition state. This can be used for unit tests.
        """
        self.__transition_state_wavenumber = test_transition_state_wavenumber

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

    def get_wigner_transmission_coefficients(self, reference_state: ReferenceState) -> Optional[float]:
        """
        Getter for the wigner transmission coefficient.

        Parameters
        ----------
        reference_state : ReferenceState
            The reference state is required to select a representative structure from the transition state ensemble.

        Returns
        -------
        The wigner transmission coefficient.
        """
        harmonic_frequency = self.get_transition_state_wavenumber(reference_state)
        if harmonic_frequency is None or harmonic_frequency >= 0.0:
            return None
        harmonic_frequency *= utils.SPEED_OF_LIGHT * 100.0  # conversion to Hz
        kbT = reference_state.temperature * utils.BOLTZMANN_CONSTANT
        return 1 + 1/24 * (utils.PLANCK_CONSTANT * abs(harmonic_frequency) / kbT) ** 2  # type: ignore

    def get_transition_state_wavenumber(self, reference_state: ReferenceState) -> Optional[float]:
        """
        Getter for the wavenumber of the transition state in cm^-1.

        Parameters
        ----------
        reference_state : ReferenceState
            The reference state. This is only used to select a transition state structure from the ensemble.

        Returns
        -------
            If available, it returns the wavenumber of the transition state in cm^-1.
        """
        if self.__transition_state_wavenumber is not None:
            return self.__transition_state_wavenumber
        if self.barrierless(reference_state):
            return None
        molecular_degrees_of_freedom = self.get_molecular_degrees_of_freedom(reference_state)
        if molecular_degrees_of_freedom is None:
            return None
        normal_modes = molecular_degrees_of_freedom.get_normal_modes_container()
        harmonic_frequency = normal_modes.get_wave_numbers()[0]
        return harmonic_frequency

    def get_eckart_tunneling_penetration(self, reference_state: ReferenceState, delta_e: float = 1e-5,
                                         access_energy_in_rt: float = 40.0) -> Optional[float]:
        """
        Getter for the transmission coefficient from the Eckart tunneling model.

        Parameters
        ----------
        reference_state : ReferenceState
            The reference state is required to select representative structures for the ensemble.
        delta_e: float, optional
            This parameter controls how dense the numerical integration grid is.
        access_energy_in_rt: float, optional
            This parameter controls the upper limit to which the integration is done. By default, it is integrated
            up to 40 RT above the transition state.

        Returns
        -------
        If available, it returns the Eckart tunneling/transmission coefficient Gamma = k_quantum/k_classical.
        """
        if self.barrierless(reference_state):
            return None
        lhs_barrier, rhs_barrier = self.get_free_energy_of_activation(reference_state, only_electronic_energies=True)
        if lhs_barrier is None or rhs_barrier is None:
            return None
        rt_in_au = utils.MOLAR_GAS_CONSTANT * reference_state.temperature * 1e-3 * utils.HARTREE_PER_KJPERMOL
        delta_V_1: float = min(lhs_barrier, rhs_barrier)
        delta_e = min(delta_e, delta_V_1 / 10)
        energy_grid = np.arange(0.0, delta_V_1 + access_energy_in_rt * rt_in_au, delta_e)
        kappa = self.get_eckart_tunneling_function(energy_grid, reference_state)
        if kappa is None:
            return None
        gamma = math.exp(delta_V_1/rt_in_au) * np.sum(np.exp(- energy_grid / rt_in_au) * kappa) * delta_e / rt_in_au
        return gamma

    def get_eckart_tunneling_function(self, energies: np.ndarray,
                                      reference_state: ReferenceState) -> Optional[np.ndarray]:
        """
        Calculate the eckart tunneling function for the given energy range.

        Parameters
        ----------
        energies : np.ndarray
            The energy range.
        reference_state : ReferenceState
            The reference state is required to select representative structures for the ensemble.

        Returns
        -------
        The transmission probability for the given energies.
        """
        eckart_tunneling = self.get_eckart_tunneling_object(reference_state)
        if eckart_tunneling is None:
            return None
        return np.asarray([eckart_tunneling.calculate_tunneling_function(energy) for energy in energies])

    def get_eckart_tunneling_object(self, reference_state: ReferenceState) -> Optional[EckartTunneling]:
        """
        Getter for the Eckart tunneling calculator object.

        Parameters
        ----------
        reference_state : ReferenceState
            The reference state is required to select representative structures for the ensemble.

        Returns
        -------
        The Eckart tunneling calculator object.
        """
        wavenumber = self.get_transition_state_wavenumber(reference_state)
        if wavenumber is None:
            return None
        lhs_barrier, rhs_barrier = self.get_free_energy_of_activation(reference_state)
        if lhs_barrier is None or rhs_barrier is None or lhs_barrier < 1e-6 or rhs_barrier < 1e-6:
            return None
        return EckartTunneling(wavenumber, lhs_barrier, rhs_barrier)

    def get_lhs_free_energy(self, reference_state: ReferenceState, only_electronic: bool = False) -> Optional[float]:
        """
        Getter for the total free energy of the LHS.

        Parameters
        ----------
        reference_state : ReferenceState
            The reference state.
        only_electronic : bool
            If true, only the electronic energies are summed.

        Returns
        -------
        Optional[float]
            The total free energy if available. Otherwise, None.
        """
        lhs_energies = [a.get_free_energy(reference_state) for a in self._lhs] if not only_electronic\
            else [a.get_electronic_energy(reference_state) for a in self._lhs]
        if None in lhs_energies:
            return None
        return sum(lhs_energies)  # type: ignore

    def get_rhs_free_energy(self, reference_state: ReferenceState, only_electronic: bool = False) -> Optional[float]:
        """
        Getter for the total free energy of the RHS.

        Parameters
        ----------
        reference_state : ReferenceState
            The reference state.
        only_electronic : bool
            If true, only the electronic energies are summed.

        Returns
        -------
        Optional[float]
            The total free energy if available. Otherwise, None.
        """
        rhs_energies = [a.get_free_energy(reference_state) for a in self._rhs] if not only_electronic\
            else [a.get_electronic_energy(reference_state) for a in self._rhs]
        if None in rhs_energies:
            return None
        return sum(rhs_energies)  # type: ignore

    def unimolecular(self):
        """
        Check if the reaction is unimolecular.

        Returns
        -------
            Returns true if the reaction is unimolecular. False, otherwise.
        """
        return len(self.get_lhs_aggregates()) == 1 and len(self.get_rhs_aggregates()) == 1

    def one_sided_unimolecular(self):
        """
        Returns true if at least one side of the reaction is unimolecular.

        Returns
        -------
            Returns true if at least one side of the reaction is unimolecular. False, otherwise.
        """
        return len(self.get_lhs_aggregates()) == 1 or len(self.get_rhs_aggregates()) == 1

    def get_rrkm_rate_constants(self, energies: np.ndarray,
                                active_rotors: bool = False,
                                reference_state: ReferenceState = utils.vacuum_zero_kelvin()) \
            -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Getter for the (unimolecular, microcanonical) RRKM rate constants.

        Parameters
        ----------
        energies : np.ndarray
            The energy range over which the rate constants will be calculated.
        active_rotors : bool
            If true, the rotational degrees of freedom are considered active, i.e., energy may be distributed to them.
        reference_state : ReferenceState
            The reference state which is used to pick the structure representative to the molecular ensemble.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The forward and backward rate constants over the energy range. Returns None if data for the reaction
            is missing. Furthermore, None is returned for each reaction side that is not unimolecular.
        """
        ts_df = self.get_molecular_degrees_of_freedom(reference_state)
        r_df = self.get_lhs_aggregates()[0].get_molecular_degrees_of_freedom(reference_state)
        p_df = self.get_rhs_aggregates()[0].get_molecular_degrees_of_freedom(reference_state)
        if ts_df is None or r_df is None or p_df is None:
            return None, None
        r_is_uni = len(self.get_lhs_aggregates()) == 1
        p_is_uni = len(self.get_rhs_aggregates()) == 1
        if not r_is_uni and not p_is_uni:
            return None, None

        if r_is_uni and p_is_uni:
            rrkm = utils.RRKMRateConstantCalculator(r_df, ts_df, p_df, active_rotor=active_rotors)
            kf, kb = rrkm.get_rate_constants(energies)
        elif r_is_uni:
            rrkm = utils.RRKMRateConstantCalculator(r_df, ts_df, active_rotor=active_rotors)
            kf, kb = rrkm.get_rate_constants(energies)
        else:
            rrkm = utils.RRKMRateConstantCalculator(p_df, ts_df, active_rotor=active_rotors)
            kb, kf = rrkm.get_rate_constants(energies)
        kf_return: Optional[np.ndarray] = kf if r_is_uni else None  # just to satisfy mypy
        kb_return: Optional[np.ndarray] = kb if p_is_uni else None
        return kf_return, kb_return

    def get_transition_state_free_energy(self, reference_state: ReferenceState,
                                         only_electronic_energies: bool = False) -> Optional[float]:
        """
        Getter fo the free energy of the transition state ensemble in Hartree.

        Parameters
        ----------
        reference_state
            The reference state (temperature, and pressure)
        only_electronic_energies : bool, optional
            If true, only the electronic energy is returned. Default is False.

        Returns
        -------
        The free energy of the transition in Hartree.
        """
        e_lhs = self.get_lhs_free_energy(reference_state, only_electronic=only_electronic_energies)
        if e_lhs is None:
            return None
        e_rhs = self.get_rhs_free_energy(reference_state, only_electronic=only_electronic_energies)
        if e_rhs is None:
            return None
        if self._has_barrierless_elementary_step:
            return max(e_lhs, e_rhs)
        # Update only if the reference state or the number of elementary steps changed.
        self._update_thermodynamics()
        if self._structure_thermodynamics.get_n_cached() == 0:
            return None
        e_ts = self._structure_thermodynamics.get_ensemble_gibbs_free_energy(reference_state) \
            if not only_electronic_energies else self.get_electronic_energy(reference_state)
        if e_ts is None:
            return None
        return max(e_lhs, e_rhs, e_ts)

    def get_free_energy_of_activation(self, reference_state: ReferenceState, in_j_per_mol: bool = False,
                                      only_electronic_energies: bool = False) \
            -> Tuple[Optional[float], Optional[float]]:
        """
        Getter for the free energy of activation/barriers as a tuple for lhs and rhs.

        Parameters
        ----------
        reference_state : ReferenceState
            The reference state (temperature, and pressure)
        in_j_per_mol : bool, optional
            If true, the barriers are returned in J/mol (NOT kJ/mol), by default False
        only_electronic_energies : bool, optional
            If true, only electronic energies are used to calculate the barriers. By default, False.

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            A tuple for the lhs and rhs barriers. Returns None if the energies are incomplete.
            For barrier-less reactions, one barrier will be the reaction energy, the other zero.
        """
        e_lhs = self.get_lhs_free_energy(reference_state, only_electronic=only_electronic_energies)
        if e_lhs is None:
            return None, None
        e_rhs = self.get_rhs_free_energy(reference_state, only_electronic=only_electronic_energies)
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
