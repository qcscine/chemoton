#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
from typing import Tuple, Union, List, Optional, Dict, Set

# Third party imports
import scine_database as db
import scine_utilities as utils

from ...utilities.queries import (
    stop_on_timeout,
    model_query
)
from .concentration_query_functions import query_concentration_with_object
from ...utilities.energy_query_functions import get_barriers_for_elementary_step_by_type, get_energy_for_structure,\
    rate_constant_from_barrier
from ...utilities.compound_and_flask_creation import get_compound_or_flask, get_aggregate_type


class KineticModelingJobFactory:
    """
    This class sets up kinetic modeling jobs.

    Attributes
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
    only_active_aggregates : bool
        If true, reactions are only included in the kinetic modeling if the lhs or rhs consists purely of
        aggregates that are considered explorable.
    _structure_weights : dict
        The Boltzmann populations for each structure of each compound.
        Constructed on the fly when create_kinetic_modeling_job is called.
    """

    def __init__(self, model: db.Model, manager: db.Manager, energy_label: str, job: db.Job,
                 ts_energy_threshold_deduplication, rate_from_lowest_conformer, use_spline_barrier,
                 max_barrier, min_barrier_intermolecular, min_barrier_intramolecular,
                 only_active_aggregates):
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
        self._structure_weights: Dict[str, Dict[str, float]] = dict()
        self._ts_energy_threshold_deduplication = ts_energy_threshold_deduplication
        self._rate_from_lowest_conformer = rate_from_lowest_conformer
        self._use_spline_barrier = use_spline_barrier
        self._min_rate_constant = rate_constant_from_barrier(max_barrier, float(model.temperature))
        self._max_rate_constant_inter = rate_constant_from_barrier(min_barrier_intermolecular, float(model.temperature))
        self._max_rate_constant_intra = rate_constant_from_barrier(min_barrier_intramolecular, float(model.temperature))
        self._only_active_aggregates = only_active_aggregates
        self._min_flux_truncation = 1e-6

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
        a_id_list, a_type_list, lhs_rates_per_reaction, rhs_rates_per_reaction, reaction_ids = self._collect_all_data()
        all_structure_ids = [db.ID(s_id_str) for c_id_str in self._structure_weights for s_id_str in
                             self._structure_weights[c_id_str]]
        concentrations = self._get_starting_concentrations(a_id_list, a_type_list)
        if sum(concentrations) == 0.0:
            print("No starting concentrations are available!")
            return False
        # Sum up the ordinary differential equations.
        lhs_rates = [sum(rates) for rates in lhs_rates_per_reaction]
        rhs_rates = [sum(rates) for rates in rhs_rates_per_reaction]
        settings["lhs_rates"] = lhs_rates
        settings["rhs_rates"] = rhs_rates
        settings["start_concentrations"] = concentrations
        settings["reaction_ids"] = [r_id.string() for r_id in reaction_ids]
        settings["aggregate_ids"] = [c_id.string() for c_id in a_id_list]
        settings["aggregate_types"] = a_type_list  # type: ignore
        settings["energy_model_program"] = self._model.program
        self._model.program = "any"
        if self._calc_already_set_up(all_structure_ids, settings):
            return False
        calc = db.Calculation(db.ID())
        calc.link(self._calculations)
        calc.create(self._model, self._kinetic_modeling_job, [])
        calc.set_settings(settings)
        calc.set_structures(all_structure_ids)
        calc.set_status(db.Status.NEW)
        return True

    def _calc_already_set_up(self, structure_ids: List[db.ID], settings: utils.ValueCollection) -> bool:
        # This query is terribly slow. But it should only be rarely necessary to do it.
        structures_string_ids = [{"$oid": str(s_id)} for s_id in structure_ids]
        selection = {
            "$and": [
                {"job.order": {"$eq": self._kinetic_modeling_job.order}},
                {"structures": {"$all": structures_string_ids, "$size": len(structures_string_ids)}},
                {"settings.solver": settings["solver"]},
            ]
            + model_query(self._model)  # type: ignore
        }
        # (direct setting comparison in query is dependent on order in dict and string-double comparison has problems)
        for calculation in stop_on_timeout(iter(self._calculations.query_calculations(dumps(selection)))):
            calculation.link(self._calculations)
            if calculation.get_settings() == settings:
                return True
        return False

    def _get_structure_weights(self, aggregate_id: db.ID, aggregate_type: db.CompoundOrFlask) -> dict:
        string_id = aggregate_id.string()
        if string_id not in self._structure_weights:
            self._structure_weights[string_id] = self._calculate_structure_weights(aggregate_id, aggregate_type)
        return self._structure_weights[string_id]

    def _calculate_structure_weights(self, aggregate_id: db.ID, aggregate_type: db.CompoundOrFlask) -> dict:
        import math
        temperature = float(self._model.temperature)
        beta_per_hartree = 1.0 / (utils.BOLTZMANN_CONSTANT * utils.HARTREE_PER_JOULE * temperature)
        aggregate = get_compound_or_flask(aggregate_id, aggregate_type, self._compounds, self._flasks)
        structures_ids = aggregate.get_structures()
        weight_dict = dict()
        total_weight = 0.0
        energies: List[Optional[float]] = []
        reference_energy = 0.0
        for s_id in structures_ids:
            structure = db.Structure(s_id, self._structures)
            structure_properties = structure.query_properties(self._energy_label, self._model, self._properties)
            if not structure_properties:
                energies.append(None)
                continue
            prop = db.NumberProperty(structure_properties[-1], self._properties)
            energy = prop.get_data()
            energies.append(energy)
            if energy < reference_energy:
                reference_energy = energy
        for i, opt_energy in enumerate(energies):
            if opt_energy is None:
                continue
            s_id = structures_ids[i]
            weight = math.exp(- beta_per_hartree * (opt_energy - reference_energy))
            weight_dict[s_id.string()] = weight
            total_weight += weight
        if not self._rate_from_lowest_conformer:
            for key in weight_dict:
                weight_dict[key] /= total_weight
        return weight_dict

    def _collect_all_data(self) -> Tuple[List[db.ID], List[db.CompoundOrFlask], List[List[float]], List[List[float]],
                                         List[db.ID]]:
        old_zero_flux_reaction_ids = self._get_old_zero_flux_reactions()
        lhs_rates_per_reaction = []
        rhs_rates_per_reaction = []
        c_id_list = list()
        c_type_list = list()
        # Search for duplicated reactions and eliminate all non-unique transition states.
        reaction_step_tuples = list()
        # exclude reactions that were unimportant previously.
        rxn_string_ids = [{"$oid": str(r_id)} for r_id in old_zero_flux_reaction_ids]
        selection = {
            "_id": {"$nin": rxn_string_ids}
        }
        for reaction in self._reactions.iterate_reactions(dumps(selection)):
            reaction.link(self._reactions)
            # check if the lhs or rhs side of the reaction consists purely of explorable compounds
            reactant_ids = reaction.get_reactants(db.Side.BOTH)
            reactant_types = reaction.get_reactant_types(db.Side.BOTH)
            # collect the elementary steps.
            elementary_steps = reaction.get_elementary_steps()
            lhs_rates_per_elementary_step = []
            rhs_rates_per_elementary_step = []
            qualified_elementary_steps = []
            for elementary_step_id in elementary_steps:
                elementary_step = db.ElementaryStep(elementary_step_id, self._elementary_steps)
                if not elementary_step.analyze():
                    continue
                lhs_rate, rhs_rate = self._get_reaction_rates_according_to_model(self._model, elementary_step_id,
                                                                                 reaction)
                if lhs_rate is None or rhs_rate is None:
                    continue
                lhs_rates_per_elementary_step.append(lhs_rate)
                rhs_rates_per_elementary_step.append(rhs_rate)
                qualified_elementary_steps.append(elementary_step.id())

            n_elementary_steps = len(qualified_elementary_steps)
            # Stop if no elementary steps qualify.
            if n_elementary_steps < 1:
                continue

            all_ids = reactant_ids[0] + reactant_ids[1]
            all_types = reactant_types[0] + reactant_types[1]
            for o_id, o_type in zip(all_ids, all_types):
                if o_id not in c_id_list:
                    c_id_list.append(o_id)
                    c_type_list.append(o_type)

            # Weight the elementary step according to the Boltzmann populations of their structures.
            lhs_new_rate_constants = list()
            rhs_new_rate_constants = list()
            new_qualified_steps = list()
            for i, elementary_step_id in enumerate(qualified_elementary_steps):
                step = db.ElementaryStep(elementary_step_id, self._elementary_steps)
                # Barrier-less reactions will always be assigned the dummy diffusion rate constant.
                if step.get_type == db.ElementaryStepType.BARRIERLESS:
                    lhs_rate_constant = self._max_rate_constant_inter
                    rhs_rate_constant = self._max_rate_constant_inter
                else:
                    lhs_weight, rhs_weight = self._get_structure_weights_for_elementary_step(elementary_step_id)
                    assert lhs_weight <= 1.0
                    assert rhs_weight <= 1.0
                    lhs_rate_constant = lhs_rates_per_elementary_step[i] * lhs_weight
                    rhs_rate_constant = rhs_rates_per_elementary_step[i] * rhs_weight
                    if lhs_rate_constant < self._min_rate_constant and rhs_rate_constant < self._min_rate_constant:
                        continue
                    reactant_types = reaction.get_reactant_types(db.Side.BOTH)
                    lhs_is_intra_molecular = len(reactant_types[0]) == 1
                    rhs_is_intra_molecular = len(reactant_types[1]) == 1
                    lhs_rate_constant = min(self._max_rate_constant_intra, lhs_rate_constant)\
                        if lhs_is_intra_molecular else min(self._max_rate_constant_inter, lhs_rate_constant)
                    rhs_rate_constant = min(self._max_rate_constant_intra, rhs_rate_constant)\
                        if rhs_is_intra_molecular else min(self._max_rate_constant_inter, rhs_rate_constant)
                lhs_new_rate_constants.append(lhs_rate_constant)
                rhs_new_rate_constants.append(rhs_rate_constant)
                new_qualified_steps.append(elementary_step_id)
            if len(lhs_new_rate_constants) < 1:
                continue
            lhs_rates_per_reaction.append(lhs_new_rate_constants)
            rhs_rates_per_reaction.append(rhs_new_rate_constants)
            reaction_step_tuples.append(tuple((reaction, new_qualified_steps)))
        reaction_ids = [reaction.id() for reaction, _ in reaction_step_tuples]
        assert len(lhs_rates_per_reaction) == len(rhs_rates_per_reaction)
        assert len(reaction_ids) == len(rhs_rates_per_reaction)
        return c_id_list, c_type_list, lhs_rates_per_reaction, rhs_rates_per_reaction, reaction_ids

    def _get_starting_concentrations(self, aggregate_ids: List[db.ID], aggregate_types: List[db.CompoundOrFlask])\
            -> List[float]:
        concentrations = []
        for o_id, o_type in zip(aggregate_ids, aggregate_types):
            aggregate = get_compound_or_flask(o_id, o_type, self._compounds, self._flasks)
            start_concentration = query_concentration_with_object("start_concentration", aggregate, self._properties,
                                                                  self._structures)
            concentrations.append(start_concentration)
        return concentrations

    def _get_structure_weights_for_elementary_step(self, elementary_step_id: db.ID) -> Tuple[float, float]:
        elementary_step = db.ElementaryStep(elementary_step_id, self._elementary_steps)
        structure_ids = elementary_step.get_reactants(db.Side.BOTH)
        lhs_weight = 1.0
        for s_id in structure_ids[0]:
            structure = db.Structure(s_id, self._structures)
            aggregate_type = get_aggregate_type(structure)
            c_id = db.Structure(s_id, self._structures).get_aggregate()
            lhs_weight *= self._get_structure_weights(c_id, aggregate_type)[s_id.string()]
        rhs_weight = 1.0
        for s_id in structure_ids[1]:
            structure = db.Structure(s_id, self._structures)
            aggregate_type = get_aggregate_type(structure)
            c_id = db.Structure(s_id, self._structures).get_aggregate()
            rhs_weight *= self._get_structure_weights(c_id, aggregate_type)[s_id.string()]
        return lhs_weight, rhs_weight

    def _is_parallel_to_reaction(self, step: db.ElementaryStep, reaction: db.Reaction):
        rxn_reactants = reaction.get_reactants(db.Side.BOTH)
        step_reactant_structure_ids = step.get_reactants(db.Side.BOTH)
        if len(rxn_reactants[0]) != len(step_reactant_structure_ids[0]):
            assert len(rxn_reactants[0]) == len(step_reactant_structure_ids[1])
            return False
        for s_id in step_reactant_structure_ids[0]:
            aggregate_id = db.Structure(s_id, self._structures).get_aggregate()
            if aggregate_id not in rxn_reactants[0]:
                assert aggregate_id in rxn_reactants[1]
                return False
        for s_id in step_reactant_structure_ids[1]:
            aggregate_id = db.Structure(s_id, self._structures).get_aggregate()
            if aggregate_id not in rxn_reactants[1]:
                assert aggregate_id in rxn_reactants[0]
                return False
        return True

    def _get_reaction_rates_according_to_model(self, model, elementary_step_id, reaction)\
            -> Union[Tuple[float, float], Tuple[None, None]]:
        temperature = float(model.temperature)
        if temperature is None:
            return None, None
        elementary_step = db.ElementaryStep(elementary_step_id, self._elementary_steps)
        elementary_step_type = elementary_step.get_type()

        lhs_barrier = None
        rhs_barrier = None
        if elementary_step_type == db.ElementaryStepType.BARRIERLESS:
            reactants = elementary_step.get_reactants(db.Side.BOTH)
            # TODO Include some sophisticated model for barrierless reactions.
            for s_id in reactants[0] + reactants[1]:
                energy = get_energy_for_structure(db.Structure(s_id), self._energy_label, model, self._structures,
                                                  self._properties)
                if energy is None:
                    return None, None
            lhs_barrier = 0.0
            rhs_barrier = 0.0
        else:
            if self._use_spline_barrier:
                lhs_barrier, rhs_barrier = elementary_step.get_barrier_from_spline()
            else:
                lhs_barrier, rhs_barrier = get_barriers_for_elementary_step_by_type(elementary_step, self._energy_label,
                                                                                    model, self._structures,
                                                                                    self._properties)
        if lhs_barrier is None or rhs_barrier is None:
            return None, None
        rhs_rate = rate_constant_from_barrier(rhs_barrier, temperature)
        lhs_rate = rate_constant_from_barrier(lhs_barrier, temperature)
        is_parallel = self._is_parallel_to_reaction(elementary_step, reaction)
        if is_parallel:
            return lhs_rate, rhs_rate
        else:
            return rhs_rate, lhs_rate

    def _get_old_zero_flux_reactions(self) -> List[db.ID]:
        reaction_ids = self._get_reactions_in_last_kinetic_modeling_jobs()
        zero_flux_reactions = list()
        for rxn_id in reaction_ids:
            if self._reaction_has_zero_flux(rxn_id):
                zero_flux_reactions.append(rxn_id)
        return zero_flux_reactions

    def _get_reactions_in_last_kinetic_modeling_jobs(self) -> List[db.ID]:
        selection = {"$and": [
            {"status": "complete"},
            {"job.order": self._kinetic_modeling_job.order}
        ]
        }
        reaction_str_id_set: Set[str] = set()
        for calculation in self._calculations.iterate_calculations(dumps(selection)):
            calculation.link(self._calculations)
            str_ids = calculation.get_settings()["reaction_ids"]
            reaction_str_id_set = reaction_str_id_set.union(str_ids)
        return [db.ID(str_id) for str_id in reaction_str_id_set]

    def _reaction_has_zero_flux(self, reaction_id: db.ID) -> bool:
        reaction = db.Reaction(reaction_id, self._reactions)
        reactants = reaction.get_reactants(db.Side.BOTH)
        reactant_types = reaction.get_reactant_types(db.Side.BOTH)
        for a_id, a_type in zip(reactants[0] + reactants[1], reactant_types[0] + reactant_types[1]):
            aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
            flux = query_concentration_with_object("concentration_flux", aggregate, self._properties, self._structures)
            if flux < self._min_flux_truncation:
                return True
        return False
