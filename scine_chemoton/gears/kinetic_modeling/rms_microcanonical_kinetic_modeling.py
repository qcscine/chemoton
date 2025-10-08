#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, Tuple, Set, Dict, Optional
import numpy as np
import math

import scine_database as db
import scine_utilities as utils

from .rms_kinetic_modeling import RMSKineticModelingJobFactory
from .atomization import MultiModelEnergyReferences
from ...utilities.model_combinations import ModelCombination
from ...utilities.uncertainties import ZeroUncertainty, UncertaintyEstimator
from ...utilities.db_object_wrappers.reaction_wrapper import Reaction
from ...utilities.db_object_wrappers.aggregate_wrapper import Aggregate


class RMSMicrocanonicalKineticModelingJobFactory(RMSKineticModelingJobFactory):
    """
    A class that creates RMS kinetic modeling jobs for microcanonical kinetic models.
    """

    def __init__(self, model_combinations: List[ModelCombination], model_combinations_reactions: List[ModelCombination],
                 manager: db.Manager,
                 energy_references: MultiModelEnergyReferences,
                 available_energy: float,
                 prescreening_tolerance: float,
                 active_rotors: bool = False,
                 eckart_tunneling: bool = True,
                 uncertainty_estimator: UncertaintyEstimator = ZeroUncertainty()) -> None:
        super().__init__(model_combinations, model_combinations_reactions, manager, energy_references,
                         uncertainty_estimator)
        self.__reference_state = utils.vacuum_zero_kelvin()
        """
        reference_state : utils.ThermodynamicReferenceState
            Dummy reference state used to select representative structures for each aggregate/reaction to evaluate
            the rate constants with. The structure with the lowest ZPE-corrected energy is used by default for each
            reaction(TS)/aggregate
        """
        self.available_energy: float = available_energy
        """
        available_energy : float
            The available energy.
        """
        self.prescreening_tolerance = prescreening_tolerance
        """
        prescreening_tolerance : float
            Reactions are considered accessible to the microkinetics if the TS energy is lower than
            the available energy plus this tolerance (in Hartree).
        """
        self.active_rotors: bool = active_rotors
        """
        active_rotors : bool
            If true, the energy is not only distributed to classical harmonic oscillators but also
            to the rigid rotor degrees of freedom.
        """
        self.eckart_tunneling: bool = eckart_tunneling
        """
        eckart_tunneling : bool
            If true, use the eckart tunneling model.
        """
        self.__pseudo_barrier: float = 20e+3
        """
        __pseudo_barrier : float
            Dummy base barrier that is used to encode the rate constants in an Arrhenius fashion, i.e.,
            each rate constant is written as k = A exp[B/(RT)]. This encoding is necessary for RMS and
            facilitates sensitivity analysis of the rate constants by changing the pseudo activation energy.
        """
        self.__pseudo_temperature: float = 298.15
        """
        __pseudo_temperature : float
            Dummy temperature for the Arrhenius encoding.
        """
        self.__barrierless_rate: float = 1e+5
        """
        __barrierless_rate : float
            RRKM theory does not provide rate constants for barrierless processes. This is a dummy rate for
            the time being.
        """

    def get_rate_constant(self, reaction: Reaction) -> Tuple[Optional[float], bool]:
        """
        Getter for the rate constant of the reaction. Also determines whether the reaction should be inverted
        because one side is unimolecular (RRKM constant available) but the other side is bimolecular. We will
        always try to write the reaction such that we can use the RRKM rate constant.

        Parameters
        ----------
        reaction : Reaction
            The reaction wrapper object.

        Returns
        -------
        Tuple[Optional[float], bool]
            The rate constant if available and true if the reaction should be inverted.
        """
        if reaction.barrierless(self.__reference_state) and reaction.one_sided_unimolecular():
            invert_reaction = len(reaction.get_lhs_aggregates()) != 1
            return self.__barrierless_rate, invert_reaction
        ts_energy = reaction.get_transition_state_free_energy(self.__reference_state)
        if ts_energy is None or ts_energy >= self.available_energy:
            return None, False
        energy = np.asarray([self.available_energy])
        kf, kb = reaction.get_rrkm_rate_constants(energy, self.active_rotors,
                                                  self.__reference_state)
        invert_reaction = False
        if kf is None:
            kf = kb
            invert_reaction = True
        if kf is None:
            return None, True

        if self.eckart_tunneling:
            g_lhs = reaction.get_lhs_free_energy(self.__reference_state)
            g_rhs = reaction.get_rhs_free_energy(self.__reference_state)
            assert None not in [g_lhs, g_rhs]
            zero_energy = max(g_lhs, g_rhs)  # type: ignore
            gamma = reaction.get_eckart_tunneling_function(energy - zero_energy, self.__reference_state)
            if gamma is not None:
                kf *= gamma[0]
        return float(kf[0]), invert_reaction

    def create_job_input(self, settings: utils.ValueCollection)\
            -> Tuple[Optional[utils.ValueCollection], Optional[Dict[int, Aggregate]], Optional[List[Reaction]]]:
        """
        Create the main job input of the calculation (settings and reaction/aggregate lists).

        Parameters
        ----------
        settings : utils.ValueCollection
            The base settings.

        Returns
        -------
        Tuple[Optional[utils.ValueCollection], Optional[Dict[int, Aggregate]], Optional[List[Reaction]]]
            The updated settings, a dictionary mapping the aggregate ids to the aggregate wrapper objects and
            a list of reactions. May return None if no aggregates or reactions are available.
        """
        if self.__reference_state != utils.vacuum_zero_kelvin():
            raise RuntimeError("The reference state should be zero kelvin and vacuum for microcanonical"
                               "kinetic modeling.")
        reaction_set, aggregates = self._setup_general_settings(settings)
        hartree_to_J_per_mol = utils.KJPERMOL_PER_HARTREE * 1e+3
        if reaction_set is None:
            return settings, None, None
        # We can only handle exit/entry channels that are bimolecular. Everything else must be unimolecular.
        reactions = [r for r in reaction_set if r.one_sided_unimolecular()]

        if aggregates is None or reactions is None:
            return settings, None, None
        # allow only one model combination for enthalpies at the moment!
        k_and_inversion = [self.get_rate_constant(r) for r in reactions]
        reactions = [r for r, (k, inv) in zip(reactions, k_and_inversion) if k is not None]
        ea = [self.__pseudo_barrier for r in reactions]
        aggregates = {}
        for r in reactions:
            for a in r.get_lhs_aggregates() + r.get_rhs_aggregates():
                aggregates[int(a.get_db_id().string(), 16)] = a
        a_values = aggregates.values()
        entropies = [0.0 for _ in a_values]
        enthalpies = [a.get_microcanonical_entropy(self.available_energy, rrkm=True, active_rotors=self.active_rotors)
                      for a in a_values]
        default_enthalpy = 0.0
        enthalpies = [e * hartree_to_J_per_mol if e is not None else default_enthalpy for e in enthalpies]
        inversion = [inversion for ea, inversion in k_and_inversion if ea is not None]
        a_uncertainties = [self._uncertainty_estimator.get_uncertainty(a) for a in a_values]
        r_uncertainties = [self._uncertainty_estimator.get_uncertainty(r) for r in reactions]
        assert None not in enthalpies
        assert None not in entropies
        assert None not in ea
        settings["reaction_ids"] = [r.get_db_id().string() for r in reactions]
        settings["enthalpies"] = [round(e, 4) for e in enthalpies]  # type: ignore
        settings["entropies"] = [round(e, 4) for e in entropies]  # type: ignore
        settings["ea"] = [round(e, 4) for e in ea]  # type: ignore
        settings["inverted_reactions"] = inversion  # type: ignore
        settings["reversible_reactions"] = [r.unimolecular() for r in reactions]
        exp_ea_rt = math.exp(+ self.__pseudo_barrier/(self.__pseudo_temperature * utils.MOLAR_GAS_CONSTANT))
        settings["arrhenius_prefactors"] = [k * exp_ea_rt for k, _ in k_and_inversion if k is not None]
        settings["arrhenius_temperature_exponents"] = [0 for _ in reactions]
        assert len(settings["arrhenius_prefactors"]) == len(settings["reversible_reactions"])  # type: ignore
        settings["enthalpy_lower_uncertainty"] = [round(u_a.lower(a), 2) for u_a, a in zip(a_uncertainties, a_values)]
        settings["enthalpy_upper_uncertainty"] = [round(u_a.upper(a), 2) for u_a, a in zip(a_uncertainties, a_values)]
        settings["ea_lower_uncertainty"] = [round(u_r.upper(r), 2) for u_r, r in zip(r_uncertainties, reactions)]
        settings["ea_upper_uncertainty"] = [round(u_r.upper(r), 2) for u_r, r in zip(r_uncertainties, reactions)]
        settings["reactor_temperature"] = self.__pseudo_temperature

        return settings, aggregates, reactions

    def create_kinetic_modeling_job(self, settings: utils.ValueCollection) -> bool:
        """
        Create the microcanonical kinetic modeling job.

        Parameters
        ----------
        settings : utils.ValueCollection
            The base settings.

        Returns
        -------
        bool
            Returns true, if the job was created and false otherwise.
        """
        actual_settings, aggregates, _ = self.create_job_input(settings)
        if actual_settings is None or aggregates is None:
            return False
        all_structure_ids = [a.get_db_object().get_centroid() for a in aggregates.values()]
        if self._calc_already_set_up(all_structure_ids, actual_settings):
            return False

        return self._finalize_calculation(actual_settings,
                                          [a.get_db_object().get_centroid() for a in aggregates.values()])

    def _reaction_is_accessible(self, reaction: Reaction, accessible_aggregate_ids: Set[int]) -> bool:
        """
        This function checks the total TS energy instead of checking the barrier height.

        Parameters
        ----------
        reaction : Reaction
            The reaction to check.
        accessible_aggregate_ids : Set[int]
            The aggregate ids that are already accessible.

        Returns
        -------
        bool
            True, if the reaction is accessible, false otherwise.
        """
        lhs_rhs_ids = reaction.get_db_object().get_reactants(db.Side.BOTH)
        if not reaction.analyze() or not reaction.one_sided_unimolecular():
            return False
        all_lhs = all(int(lhs_id.string(), 16) in accessible_aggregate_ids for lhs_id in lhs_rhs_ids[0])
        all_rhs = all(int(rhs_id.string(), 16) in accessible_aggregate_ids for rhs_id in lhs_rhs_ids[1])
        if not all_lhs and not all_rhs:
            return False
        ts_energy = reaction.get_transition_state_free_energy(self.__reference_state)
        if ts_energy is None or not reaction.complete():
            return False
        return ts_energy < self.available_energy + self.prescreening_tolerance

    @staticmethod
    def get_job():
        """
        Returns
        -------
        db.Job
            The job object.
        """
        return db.Job('rms_kinetic_modeling')
