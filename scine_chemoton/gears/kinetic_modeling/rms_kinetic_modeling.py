#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, Dict
from copy import deepcopy
import numpy as np

import scine_database as db
import scine_utilities as utils

from scine_chemoton.gears import HoldsCollections
from .prepare_kinetic_modeling_job import KineticModelingJobFactory
from .atomization import MultiModelEnergyReferences
from ...utilities.model_combinations import ModelCombination
from ...utilities.uncertainties import ZeroUncertainty, UncertaintyEstimator


class RMSKineticModelingJobFactory(KineticModelingJobFactory):
    """
    A class that creates RMS kinetic modeling jobs.
    """

    def __init__(self, model_combinations: List[ModelCombination], model_combinations_reactions: List[ModelCombination],
                 manager: db.Manager,
                 energy_references: MultiModelEnergyReferences,
                 uncertainty_estimator: UncertaintyEstimator = ZeroUncertainty(),
                 only_electronic: bool = False) -> None:
        super().__init__(model_combinations=model_combinations,
                         model_combinations_reactions=model_combinations_reactions,
                         manager=manager, only_electronic=only_electronic)
        self.energy_references = energy_references
        self._uncertainty_estimator = uncertainty_estimator
        if isinstance(self._uncertainty_estimator, HoldsCollections):
            self._uncertainty_estimator.initialize_collections(manager)

    def create_kinetic_modeling_job(self, settings: utils.ValueCollection) -> bool:
        reactions, aggregates = self._setup_general_settings(settings)
        if aggregates is None or reactions is None:
            return False
        hartree_to_J_per_mol = utils.KJPERMOL_PER_HARTREE * 1e+3
        a_values = aggregates.values()
        # allow only one model combination for enthalpies at the moment!
        main_energy_model = self._model_combinations[0].electronic_model
        assert all(a.get_model_combination().electronic_model == main_energy_model for a in a_values)
        # Apply the energy reference to the enthalpy. The reference may differ between electronic structure models
        # and hopefully allows the combination of multiple different models.
        enthalpies = [a.get_enthalpy(self.reference_state) for a in a_values]
        entropies = [a.get_entropy(self.reference_state) for a in a_values]
        ea = [r.get_free_energy_of_activation(self.reference_state)[0] for r in reactions]
        a_uncertainties = [self._uncertainty_estimator.get_uncertainty(a) for a in a_values]
        r_uncertainties = [self._uncertainty_estimator.get_uncertainty(r) for r in reactions]
        assert None not in enthalpies
        enthalpies = [e - self.energy_references.get_value(a) for a, e in zip(a_values, enthalpies)]  # type: ignore
        assert None not in entropies
        assert None not in ea
        ea = self.assert_non_negative_barriers(enthalpies, entropies, ea, settings["reaction_ids"],  # type: ignore
                                               settings["aggregate_ids"],  # type: ignore
                                               self.reference_state.temperature)  # type: ignore
        settings["enthalpies"] = [round(e * hartree_to_J_per_mol, 4) for e in enthalpies]  # type: ignore
        settings["entropies"] = [round(e * hartree_to_J_per_mol, 4) for e in entropies]  # type: ignore
        settings["ea"] = [round(e * hartree_to_J_per_mol, 4) for e in ea]  # type: ignore
        settings["arrhenius_prefactors"] = [r.get_arrhenius_prefactor(self.reference_state) for r in reactions]
        settings["arrhenius_temperature_exponents"] = [0 for _ in reactions]
        settings["enthalpy_lower_uncertainty"] = [round(u_a.lower(a), 2) for u_a, a in zip(a_uncertainties, a_values)]
        settings["enthalpy_upper_uncertainty"] = [round(u_a.upper(a), 2) for u_a, a in zip(a_uncertainties, a_values)]
        settings["ea_lower_uncertainty"] = [round(u_r.upper(r), 2) for u_r, r in zip(r_uncertainties, reactions)]
        settings["ea_upper_uncertainty"] = [round(u_r.upper(r), 2) for u_r, r in zip(r_uncertainties, reactions)]

        all_structure_ids = [a.get_db_object().get_centroid() for a in aggregates.values()]
        if self._calc_already_set_up(all_structure_ids, settings):
            return False

        return self._finalize_calculation(settings, [a.get_db_object().get_centroid() for a in aggregates.values()])

    def _identical_model_definition(self, settings: utils.ValueCollection, other_settings: Dict) -> bool:
        sorted_parameters = self._sorted_model_parameters(settings.as_dict())
        other_sorted_parameters = self._sorted_model_parameters(other_settings)
        for values, other_values in zip(sorted_parameters, other_sorted_parameters):
            if len(values) != len(other_values):
                return False
            delta = np.argmax(np.abs(np.asarray(values) - np.asarray(other_values)))
            if delta > 1e+1:  # values in J/mol
                return False
        return True

    @staticmethod
    def order_dependent_setting_keys() -> List[str]:
        return ["ea", "ea_lower_uncertainty", "ea_upper_uncertainty", "enthalpies", "entropies",
                "enthalpy_lower_uncertainty", "enthalpy_upper_uncertainty"]

    @staticmethod
    def _sorted_model_parameters(settings: Dict):
        for key in RMSKineticModelingJobFactory.order_dependent_setting_keys():
            if key not in settings:
                raise RuntimeError("Missing key in settings definition. Setting sorting is impossible.")
        r_ids = deepcopy(settings["reaction_ids"])
        a_ids = deepcopy(settings["aggregate_ids"])
        ea = deepcopy(settings["ea"])
        ea_uq_l = deepcopy(settings["ea_lower_uncertainty"])
        ea_uq_u = deepcopy(settings["ea_upper_uncertainty"])
        h = deepcopy(settings["enthalpies"])
        s = deepcopy(settings["entropies"])
        h_uq_l = deepcopy(settings["enthalpy_lower_uncertainty"])
        h_uq_u = deepcopy(settings["enthalpy_upper_uncertainty"])

        r_ids, ea, ea_uq_l, ea_uq_u = (list(start_val) for start_val in zip(*sorted(zip(r_ids, ea, ea_uq_l, ea_uq_u))))
        a_ids, h, s, h_uq_l, h_uq_u = (list(start_val) for start_val in zip(*sorted(zip(a_ids, h, s, h_uq_l, h_uq_u))))

        return [ea, ea_uq_l, ea_uq_u, h, s, h_uq_l, h_uq_u]

    def assert_non_negative_barriers(self, enthalpies: List[float], entropies: List[float], ea: List[float],
                                     reaction_str_ids: List[str], aggregate_str_ids: List[str], temperature: float):
        """
        Assert that no forward or backward reaction barrier is negative.
        """
        updated_ea = deepcopy(ea)
        for j, (r_str_id, e_a) in enumerate(zip(reaction_str_ids, ea)):
            if e_a < 0.0:
                raise RuntimeError("Error: Negative reaction barrier in kinetic modeling. This is not allowed.")
            reaction = db.Reaction(db.ID(r_str_id), self._reactions)
            reactants = reaction.get_reactants(db.Side.BOTH)
            lhs_indices = [aggregate_str_ids.index(a_id.string()) for a_id in reactants[0]]
            rhs_indices = [aggregate_str_ids.index(a_id.string()) for a_id in reactants[1]]
            g_lhs = sum(enthalpies[i] - temperature * entropies[i] for i in lhs_indices)
            g_rhs = sum(enthalpies[i] - temperature * entropies[i] for i in rhs_indices)
            r_ea = g_lhs + e_a - g_rhs
            if r_ea < 0.0:
                updated_ea[j] = g_rhs - g_lhs
                print("Warning: Negative reverse reaction barrier in kinetic modeling. This is not allowed.\n"
                      "The reaction is changed to barrier-less from the right side and the activation energy\n"
                      "is increased to the reaction energy. Reaction:\n", r_str_id, "EA", e_a, "reverse EA", r_ea,
                      "updated variant ", updated_ea[j])
        return updated_ea

    @staticmethod
    def get_job():
        return db.Job('rms_kinetic_modeling')

    @staticmethod
    def get_default_settings():
        return utils.ValueCollection({
            "solver": "CVODE_BDF",
            "phase_type": "ideal_gas",  # other: ideal_dilute_solution
            "max_time": 3600.0,  # in s
            "viscosity": "none",  # taken from tabulated values corresponding to the solvent in the model if not given
            "reactor_solvent": "none",  # taken from the electronic structure model if not given
            "diffusion_limited": False,  # Only viable for phase_type: ideal_dilute_solution. May destabilize ODE.
            "reactor_temperature": "none",  # taken from the the electronic structure model if not given
            "reactor_pressure": "none",  # taken from the the electronic structure model if not given
            "absolute_tolerance": 1e-22,  # 1e-22 should be very conservative. 1e-18 could be fine as well.
            "relative_tolerance": 1e-9,
            "solvent_aggregate_str_id": "none",  # allow the solvent concentration to change by providing the solvent id
            "solvent_concentration": 14.3,
        })
