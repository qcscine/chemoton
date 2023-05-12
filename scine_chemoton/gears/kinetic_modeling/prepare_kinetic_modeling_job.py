#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
from typing import Tuple, Union, List, Set, Dict, Any
from copy import deepcopy

# Third party imports
import scine_database as db
import scine_utilities as utils

from ...utilities.queries import (
    calculation_exists_in_structure
)
from .concentration_query_functions import query_concentration_with_object, query_reaction_fluxes_with_model
from ...utilities.energy_query_functions import get_energy_for_structure,\
    rate_constant_from_barrier, get_min_energy_for_aggregate
from ...utilities.calculation_creation_helpers import finalize_calculation
from ...utilities.compound_and_flask_creation import get_compound_or_flask


class KineticModelingJobFactory:
    """
    This class sets up kinetic modeling jobs.

    Parameters
    ----------
    model : db.Model
        The electronic structure model used for the reaction rates.
    manager : db.Manager
        The database manager.
    energy_label : str
        The property label to be used for the reaction rate determination.
    job : db.Job
        The job to set up.
    ts_energy_threshold_deduplication : float
        If two reactions are mirrors/inverses of one another all elementary steps with the same transition state
        energy are eliminated. This threshold determines the energy tolerance for the elimination.
    rate_from_lowest_conformer : bool
        Calculate the reaction rate always with respect to the energy of the
        lowest energy conformer without any reweighing.
    use_spline_barrier : bool
        If true, the reaction barrier from the spline is used.
    max_barrier : float
        The maximum allowed barrier.
    min_barrier_intermolecular : float
        The minimum allowed barrier in kJ/mol for intermolecular reactions.
    min_barrier_intramolecular : float
        The minimum allowed barrier in kJ/mol for intramolecular reactions.
    min_flux_truncation : float
        Minimum flux in a previous kinetic modeling job for a reaction to be included in the next kinetic modeling.
    use_max_flux : bool
        If true, the maximum entry of all concentration fluxes for a given reaction is used for the flux based
        truncation.
    """

    def __init__(self, model: db.Model, manager: db.Manager, energy_label: str, job: db.Job,
                 ts_energy_threshold_deduplication: float, rate_from_lowest_conformer: bool, use_spline_barrier: bool,
                 max_barrier: float, min_barrier_intermolecular: float, min_barrier_intramolecular: float,
                 min_flux_truncation: float, concentration_label_postfix: str, diffusion_controlled_barrierless: bool,
                 use_max_flux: bool):
        self._model = model
        self._kinetic_modeling_job = job
        self._properties = manager.get_collection("properties")
        self._elementary_steps = manager.get_collection("elementary_steps")
        self._compounds = manager.get_collection("compounds")
        self._reactions = manager.get_collection("reactions")
        self._structures = manager.get_collection("structures")
        self._calculations = manager.get_collection("calculations")
        self._flasks = manager.get_collection("flasks")
        self._energy_label = energy_label
        self._min_energy_per_aggregate: Dict[str, Union[float, Any]] = dict()
        self._ts_energy_threshold_deduplication = ts_energy_threshold_deduplication
        self._rate_from_lowest_conformer = rate_from_lowest_conformer
        self._use_spline_barrier = use_spline_barrier
        self._min_rate_constant = rate_constant_from_barrier(max_barrier, float(model.temperature))
        self._max_rate_constant_inter: float = rate_constant_from_barrier(
            min_barrier_intermolecular, float(model.temperature))
        self._max_rate_constant_intra: float = rate_constant_from_barrier(
            min_barrier_intramolecular, float(model.temperature))
        self._min_flux_truncation = min_flux_truncation
        self._concentration_label_postfix = concentration_label_postfix
        self._diffusion_controlled_barrierless = diffusion_controlled_barrierless
        self._use_max_flux = use_max_flux
        self.edge_flux_label = "_reaction_edge_flux"
        self.vertex_flux_label = "concentration_flux"
        self.access_sides: List[db.Side] = [db.Side.LHS, db.Side.BOTH]
        """
        access_sides : List[db.Side]
            The sides which must be accessible for a reaction in order to consider it in the kinetic modeling.
            Accessibility is given if one of the sides can be reached from a starting species without ever crossing
            a barrier higher than allowed by the settings.
        """

    def create_local_barrier_analysis_jobs(self, settings: utils.ValueCollection):
        a_id_list, a_type_list, lhs_rates_per_reaction, rhs_rates_per_reaction, reaction_ids, all_structure_ids \
            = self._collect_all_data()
        concentrations = self._get_starting_concentrations(a_id_list, a_type_list)
        if sum(concentrations) == 0.0:
            print("No starting concentrations are available!")
            return False
        # Take the maximum rate found for an elementary step.
        lhs_rates = [max(rates) for rates in lhs_rates_per_reaction]
        rhs_rates = [max(rates) for rates in rhs_rates_per_reaction]
        settings["aggregate_ids"] = [c_id.string() for c_id in a_id_list]
        settings["aggregate_types"] = a_type_list  # type: ignore
        settings["energy_model_program"] = self._model.program
        settings["start_concentrations"] = concentrations
        self._model.program = "any"

        for i_reaction in range(len(reaction_ids)):
            settings_copy = deepcopy(settings)
            # copy the reaction ids and rates. Then remove the i-th element from the copies.
            reduced_reaction_ids = reaction_ids.copy()
            reduced_lhs_rates = lhs_rates.copy()
            reduced_rhs_rates = rhs_rates.copy()
            reduced_reaction_ids.pop(i_reaction)
            reduced_lhs_rates.pop(i_reaction)
            reduced_rhs_rates.pop(i_reaction)
            # complete the settings
            settings_copy["lhs_rates"] = reduced_lhs_rates
            settings_copy["rhs_rates"] = reduced_rhs_rates
            settings_copy["reaction_ids"] = [r_id.string() for r_id in reduced_reaction_ids]
            settings_copy["concentration_label_postfix"] =\
                self._concentration_label_postfix + "_local_barrier_" + str(i_reaction)
            if not self._finalize_calculation_set_up(settings_copy, all_structure_ids):
                print("Unable to set up local barrier analysis job. This was probably already done before!")

    def create_kinetic_modeling_job(self, settings: utils.ValueCollection) -> bool:
        """
        Create the kinetic modeling job.

        Parameters
        ----------
        settings : utils.ValueCollection
            The job settings.

        Return
        ------
        bool
            True if the calculation was set up. False, otherwise.
        """
        a_id_list, a_type_list, lhs_rates_per_reaction, rhs_rates_per_reaction, reaction_ids, all_structure_ids\
            = self._collect_all_data()
        concentrations = self._get_starting_concentrations(a_id_list, a_type_list)
        if sum(concentrations) == 0.0:
            print("No starting concentrations are available!")
            return False
        # Take the maximum rate found for an elementary step.
        lhs_rates = [max(rates) for rates in lhs_rates_per_reaction]
        rhs_rates = [max(rates) for rates in rhs_rates_per_reaction]
        settings["lhs_rates"] = lhs_rates
        settings["rhs_rates"] = rhs_rates
        settings["start_concentrations"] = concentrations
        settings["reaction_ids"] = [r_id.string() for r_id in reaction_ids]
        settings["aggregate_ids"] = [c_id.string() for c_id in a_id_list]
        settings["aggregate_types"] = a_type_list  # type: ignore
        settings["energy_model_program"] = self._model.program
        settings["concentration_label_postfix"] = self._concentration_label_postfix
        self._model.program = "any"
        return self._finalize_calculation_set_up(settings, all_structure_ids)

    def _finalize_calculation_set_up(self, settings: utils.ValueCollection, all_structure_ids: List[db.ID]) -> bool:
        if self._calc_already_set_up(all_structure_ids, settings):
            return False

        calc = db.Calculation(db.ID())
        calc.link(self._calculations)
        calc.create(self._model, self._kinetic_modeling_job, [])
        calc.set_settings(settings)
        calc.set_structures(all_structure_ids)
        finalize_calculation(calc, self._structures, all_structure_ids)
        return True

    def _calc_already_set_up(self, structure_ids: List[db.ID], settings: utils.ValueCollection) -> bool:
        if calculation_exists_in_structure(self._kinetic_modeling_job.order, structure_ids, self._model,
                                           self._structures, self._calculations, settings.as_dict()):
            print("Kinetic modeling calculation already set up! Exploration converged!")
            return True
        return False

    def _get_min_energy_in_structure_ensemble(self, aggregate_id: db.ID, aggregate_type: db.CompoundOrFlask):
        key = aggregate_id.string()
        if key not in self._min_energy_per_aggregate:
            aggregate = get_compound_or_flask(aggregate_id, aggregate_type, self._compounds, self._flasks)
            self._min_energy_per_aggregate[key] = get_min_energy_for_aggregate(
                aggregate, self._model, self._energy_label, self._structures, self._properties)
        return self._min_energy_per_aggregate[key]

    def _get_reaction_str_ids_of_aggregates(self, a_ids: List[db.ID], a_types: List[db.CompoundOrFlask]) -> Set[str]:
        reaction_str_id_set: Set[str] = set()
        for a_id, a_type in zip(a_ids, a_types):
            aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
            reaction_str_id_set.update([r_id.string() for r_id in aggregate.get_reactions()])
        return reaction_str_id_set

    def _get_accessible_reactions(self) -> Tuple[List[db.ID], List[db.ID], List[db.CompoundOrFlask], List[List[float]],
                                                 List[List[float]]]:
        accessible_reactions: List[db.ID] = list()
        accessible_aggregates: List[db.ID] = self._get_starting_compounds()
        accessible_aggregate_types: List[db.CompoundOrFlask] = [db.CompoundOrFlask.COMPOUND
                                                                for _ in range(len(accessible_aggregates))]
        starting_flasks = self._get_starting_flasks()
        accessible_aggregates += starting_flasks
        accessible_aggregate_types += [db.CompoundOrFlask.FLASK for _ in range(len(starting_flasks))]
        lhs_rates_per_reaction: List[List[float]] = list()
        rhs_rates_per_reaction: List[List[float]] = list()
        # exclude reactions that were unimportant previously.
        old_zero_flux_reaction_ids = self._get_old_zero_flux_reactions(accessible_aggregates,
                                                                       accessible_aggregate_types)
        sets_changed = True
        rxn_string_ids_zero_flux = set([r_id.string() for r_id in old_zero_flux_reaction_ids])
        reactions_str_ids_to_iterate: Set[str] = self._get_reaction_str_ids_of_aggregates(accessible_aggregates,
                                                                                          accessible_aggregate_types)
        reaction_str_ids_not_to_iterate = rxn_string_ids_zero_flux

        while sets_changed:
            sets_changed = False

            # iterate over all reactions of all currently accessible compounds minus the reactions already considered
            # accessible, zero flux, or inaccessible because of a too high barrier/low reaction rate constant.
            reactions_str_ids_to_iterate = reactions_str_ids_to_iterate.difference(reaction_str_ids_not_to_iterate)
            newly_added_aggregates: List[db.ID] = list()
            newly_added_aggregate_types: List[db.CompoundOrFlask] = list()
            for r_str_id in reactions_str_ids_to_iterate:
                reaction = db.Reaction(db.ID(r_str_id), self._reactions)
                access = self._get_accessible_reaction_side(reaction, accessible_aggregates)
                lhs_rates_per_elementary_step: List[float] = list()
                rhs_rates_per_elementary_step: List[float] = list()
                qualified_elementary_steps: List[db.ID] = list()
                if access in self.access_sides:
                    for elementary_step_id in reaction.get_elementary_steps():
                        lhs_rate, rhs_rate = self._get_reaction_rates_according_to_model(
                            self._model, elementary_step_id, reaction)
                        if lhs_rate is not None and rhs_rate is not None:
                            lhs_quali = access in [db.Side.LHS, db.Side.BOTH] and lhs_rate > self._min_rate_constant
                            rhs_quali = access in [db.Side.RHS, db.Side.BOTH] and rhs_rate > self._min_rate_constant
                            if lhs_quali or rhs_quali:
                                lhs_rates_per_elementary_step.append(lhs_rate)
                                rhs_rates_per_elementary_step.append(rhs_rate)
                                qualified_elementary_steps.append(elementary_step_id)
                                elementary_step = db.ElementaryStep(elementary_step_id, self._elementary_steps)
                                if elementary_step.get_type() == db.ElementaryStepType.BARRIERLESS\
                                        and self._diffusion_controlled_barrierless:
                                    break
                # Stop if no elementary steps qualify.
                if len(qualified_elementary_steps) < 1:
                    continue

                lhs_rates_per_reaction.append(lhs_rates_per_elementary_step)
                rhs_rates_per_reaction.append(rhs_rates_per_elementary_step)

                reactants = reaction.get_reactants(db.Side.BOTH)
                reactant_types = reaction.get_reactant_types(db.Side.BOTH)
                for o_id, o_type in zip(reactants[0] + reactants[1], reactant_types[0] + reactant_types[1]):
                    if o_id not in accessible_aggregates:
                        accessible_aggregates.append(o_id)
                        accessible_aggregate_types.append(o_type)
                        newly_added_aggregates.append(o_id)
                        newly_added_aggregate_types.append(o_type)
                accessible_reactions.append(reaction.id())
                reaction_str_ids_not_to_iterate.add(r_str_id)
                sets_changed = True
            reactions_str_ids_to_iterate = self._get_reaction_str_ids_of_aggregates(newly_added_aggregates,
                                                                                    newly_added_aggregate_types)
        return accessible_reactions, accessible_aggregates, accessible_aggregate_types, lhs_rates_per_reaction,\
            rhs_rates_per_reaction

    def _get_starting_compounds(self) -> List[db.ID]:
        start_compound_ids: List[db.ID] = list()
        for compound in self._compounds.iterate_all_compounds():
            compound.link(self._compounds)
            start_concentration = query_concentration_with_object("start_concentration", compound, self._properties,
                                                                  self._structures)
            if start_concentration > 0:
                start_compound_ids.append(compound.id())
        return start_compound_ids

    def _get_starting_flasks(self) -> List[db.ID]:
        start_flask_ids: List[db.ID] = list()
        for flask in self._flasks.iterate_all_compounds():
            flask.link(self._flasks)
            start_concentration = query_concentration_with_object("start_concentration", flask, self._properties,
                                                                  self._structures)
            if start_concentration > 0:
                start_flask_ids.append(flask.id())
        return start_flask_ids

    @staticmethod
    def _get_accessible_reaction_side(
            reaction: db.Reaction, accessible_aggregates: List[db.ID]) -> Union[None, db.Side]:
        lhs_rhs_ids = reaction.get_reactants(db.Side.BOTH)
        all_lhs = True
        for lhs_id in lhs_rhs_ids[0]:
            if lhs_id not in accessible_aggregates:
                all_lhs = False
                break
        all_rhs = True
        for rhs_id in lhs_rhs_ids[1]:
            if rhs_id not in accessible_aggregates:
                all_rhs = False
                break
        if all_rhs and all_lhs:
            return db.Side.BOTH
        if all_rhs:
            return db.Side.RHS
        if all_lhs:
            return db.Side.LHS
        return None

    def _collect_all_data(self) -> Tuple[List[db.ID], List[db.CompoundOrFlask], List[List[float]], List[List[float]],
                                         List[db.ID], List[db.ID]]:
        r_ids, a_ids, a_types, lhs_rates_per_reaction, rhs_rates_per_reaction = self._get_accessible_reactions()
        all_structure_ids = list()
        for o_id, o_type in zip(a_ids, a_types):
            aggregate = get_compound_or_flask(o_id, o_type, self._compounds, self._flasks)
            all_structure_ids.append(aggregate.get_centroid())
        assert len(lhs_rates_per_reaction) == len(rhs_rates_per_reaction)
        assert len(r_ids) == len(rhs_rates_per_reaction)
        return a_ids, a_types, lhs_rates_per_reaction, rhs_rates_per_reaction, r_ids, all_structure_ids

    def _get_starting_concentrations(self, aggregate_ids: List[db.ID], aggregate_types: List[db.CompoundOrFlask])\
            -> List[float]:
        concentrations = []
        for o_id, o_type in zip(aggregate_ids, aggregate_types):
            aggregate = get_compound_or_flask(o_id, o_type, self._compounds, self._flasks)
            start_concentration = query_concentration_with_object("start_concentration", aggregate, self._properties,
                                                                  self._structures)
            concentrations.append(start_concentration)
        return concentrations

    def _get_reaction_rates_according_to_model(self, model, elementary_step_id, reaction)\
            -> Union[Tuple[float, float], Tuple[None, None]]:
        temperature = float(model.temperature)
        if temperature is None:
            return None, None
        elementary_step = db.ElementaryStep(elementary_step_id, self._elementary_steps)
        elementary_step_type = elementary_step.get_type()
        reactants = reaction.get_reactants(db.Side.BOTH)
        reactant_types = reaction.get_reactant_types(db.Side.BOTH)

        lhs_energy = 0.0
        rhs_energy = 0.0
        for a_id, a_type in zip(reactants[0], reactant_types[0]):
            energy = self._get_min_energy_in_structure_ensemble(a_id, a_type)
            if energy is None:
                return None, None
            lhs_energy += energy
        for a_id, a_type in zip(reactants[1], reactant_types[1]):
            energy = self._get_min_energy_in_structure_ensemble(a_id, a_type)
            if energy is None:
                return None, None
            rhs_energy += energy
        if elementary_step_type == db.ElementaryStepType.BARRIERLESS:
            # TODO Include some sophisticated model for barrierless reactions.
            lhs_barrier = 0.0
            rhs_barrier = 0.0
            if not self._diffusion_controlled_barrierless:
                energy_diff = (rhs_energy - lhs_energy) * utils.KJPERMOL_PER_HARTREE
                lhs_barrier = max(energy_diff, 0.0)
                rhs_barrier = max(-energy_diff, 0.0)
        else:
            if self._use_spline_barrier:
                lhs_barrier, rhs_barrier = elementary_step.get_barrier_from_spline()
            else:
                ts = db.Structure(elementary_step.get_transition_state())
                ts_energy = get_energy_for_structure(ts, self._energy_label, model, self._structures, self._properties)
                if ts_energy is None:
                    return None, None
                lhs_barrier = (ts_energy - lhs_energy) * utils.KJPERMOL_PER_HARTREE
                rhs_barrier = (ts_energy - rhs_energy) * utils.KJPERMOL_PER_HARTREE

        if lhs_barrier is None or rhs_barrier is None:
            return None, None
        rhs_rate_constant = rate_constant_from_barrier(rhs_barrier, temperature)
        lhs_rate_constant = rate_constant_from_barrier(lhs_barrier, temperature)
        if lhs_rate_constant < self._min_rate_constant and rhs_rate_constant < self._min_rate_constant:
            return None, None
        lhs_is_intra_molecular = len(reactant_types[0]) == 1
        rhs_is_intra_molecular = len(reactant_types[1]) == 1
        lhs_rate_constant = min(self._max_rate_constant_intra, lhs_rate_constant) \
            if lhs_is_intra_molecular else min(self._max_rate_constant_inter, lhs_rate_constant)
        rhs_rate_constant = min(self._max_rate_constant_intra, rhs_rate_constant) \
            if rhs_is_intra_molecular else min(self._max_rate_constant_inter, rhs_rate_constant)
        return lhs_rate_constant, rhs_rate_constant

    def _get_old_zero_flux_reactions(self, accessible_aggregates: List[db.ID],
                                     accessible_aggregate_types: List[db.CompoundOrFlask]) -> List[db.ID]:
        reaction_ids = self._get_reactions_in_last_kinetic_modeling_jobs(accessible_aggregates,
                                                                         accessible_aggregate_types)
        zero_flux_reactions = list()
        for rxn_id in reaction_ids:
            if self._use_max_flux:
                zero_flux = self._reaction_has_zero_edge_flux(rxn_id)
            else:
                zero_flux = self._reaction_has_zero_flux(rxn_id)
            if zero_flux:
                zero_flux_reactions.append(rxn_id)
        return zero_flux_reactions

    def _get_n_queuing_calculations(self) -> int:
        selection = {
            "$and": [
                {"$or": [{"status": "new"}, {"status": "hold"}, {"status": "pending"}]},
            ]
        }
        return self._calculations.count(dumps(selection))

    def _get_reactions_in_last_kinetic_modeling_jobs(self, accessible_aggregates: List[db.ID],
                                                     accessible_aggregate_types: List[db.CompoundOrFlask]
                                                     ) -> List[db.ID]:
        reaction_str_id_set: Set[str] = set()
        for a_id, a_type in zip(accessible_aggregates, accessible_aggregate_types):
            aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
            centroid = db.Structure(aggregate.get_centroid(), self._structures)
            calc_ids = centroid.get_calculations(self._kinetic_modeling_job.order)
            for c_id in calc_ids:
                calculation = db.Calculation(c_id, self._calculations)
                if calculation.status == db.Status.COMPLETE:
                    settings = calculation.get_settings()
                    if "concentration_label_postfix" in settings:
                        if settings["concentration_label_postfix"] != "":
                            continue
                    str_ids: List[str] = calculation.get_settings()["reaction_ids"]  # type: ignore
                    reaction_str_id_set.update(str_ids)
        return [db.ID(str_id) for str_id in reaction_str_id_set]

    def _reaction_has_zero_edge_flux(self, reaction_id: db.ID):
        reaction = db.Reaction(reaction_id, self._reactions)
        fluxes = query_reaction_fluxes_with_model(self.edge_flux_label, reaction, self._compounds, self._flasks,
                                                  self._structures, self._properties, self._model)
        if len(fluxes) == 0:
            return True
        return max(fluxes) < self._min_flux_truncation

    def _reaction_has_zero_flux(self, reaction_id: db.ID) -> bool:
        reaction = db.Reaction(reaction_id, self._reactions)
        reactants = reaction.get_reactants(db.Side.BOTH)
        reactant_types = reaction.get_reactant_types(db.Side.BOTH)
        for a_id, a_type in zip(reactants[0] + reactants[1], reactant_types[0] + reactant_types[1]):
            aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
            flux = query_concentration_with_object(self.vertex_flux_label, aggregate, self._properties,
                                                   self._structures)
            if flux < self._min_flux_truncation:
                return True
        return False
