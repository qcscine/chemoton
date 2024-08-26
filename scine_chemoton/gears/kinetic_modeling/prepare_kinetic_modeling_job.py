#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import Tuple, List, Set, Dict, Optional
from copy import deepcopy
from abc import ABC
import math

# Third party imports
import scine_database as db
import scine_utilities as utils
from scine_database.queries import get_common_calculation_ids
from scine_database.concentration_query_functions import query_concentration_with_object
from scine_database.compound_and_flask_creation import get_compound_or_flask
from ...utilities.calculation_creation_helpers import finalize_calculation
from ...utilities.db_object_wrappers.thermodynamic_properties import ReferenceState
from ...utilities.db_object_wrappers.reaction_wrapper import Reaction
from ...utilities.db_object_wrappers.wrapper_caches import (
    MultiModelReactionCache,
    MultiModelAggregateCache,
)
from ...utilities.db_object_wrappers.aggregate_wrapper import Aggregate
from ...utilities.model_combinations import ModelCombination


class KineticModelingJobFactory(ABC):
    """
    Base class for classes that creates kinetic modeling jobs. See rms_kinetic_modeling.py and
    kinetx_kinetic_modeling.py.
    """

    def __init__(self, model_combinations: List[ModelCombination], model_combinations_reactions: List[ModelCombination],
                 manager: db.Manager,
                 only_electronic: bool = False) -> None:
        self._model_combinations = model_combinations
        assert self._model_combinations
        self._model_combinations_reactions = model_combinations_reactions
        assert self._model_combinations_reactions
        self._manager = manager
        self._properties = manager.get_collection("properties")
        self._elementary_steps = manager.get_collection("elementary_steps")
        self._compounds = manager.get_collection("compounds")
        self._reactions = manager.get_collection("reactions")
        self._structures = manager.get_collection("structures")
        self._calculations = manager.get_collection("calculations")
        self._flasks = manager.get_collection("flasks")
        self.min_flux_truncation = 1e-5
        self.vertex_flux_label = "concentration_flux"
        self.flux_variance_label: Optional[str] = None
        self.reference_state = ReferenceState(298.15, 1e+5)
        self.max_barrier = 100.0  # kJ/mol
        self._calculation_model = deepcopy(self._model_combinations[0].electronic_model)
        self._electronic_structure_program = self._calculation_model.program
        self._calculation_model.program = "any"
        self._only_electronic = only_electronic
        self._aggregate_cache = MultiModelAggregateCache(self._manager, self._model_combinations, self._only_electronic)
        self._reaction_cache = MultiModelReactionCache(self._manager, self._model_combinations_reactions,
                                                       self._only_electronic)
        self.use_zero_flux_truncation = True

    def _setup_general_settings(self, settings: utils.ValueCollection)\
            -> Tuple[Optional[Set[Reaction]], Optional[Dict[int, Aggregate]]]:
        reactions, aggregates = self.get_reactions()
        start_concentrations = [a.get_starting_concentration() for a in aggregates.values()]
        if sum(start_concentrations) == 0:
            print("No starting concentrations are available!")
            return None, None
        if not reactions or not aggregates:
            print("No reactive species or feasible reactions!")
            return None, None
        settings["reaction_ids"] = [r.get_db_id().string() for r in reactions]
        settings["aggregate_ids"] = [a.get_db_id().string() for a in aggregates.values()]
        settings["energy_model_program"] = self._electronic_structure_program
        settings["start_concentrations"] = start_concentrations
        # Set this after checking if the calculation already exists since the order of the aggregates may differ.
        # Hence, the order in the aggregate types may differ, too.
        settings["aggregate_types"] = [int(a.get_aggregate_type()) for a in aggregates.values()]
        return reactions, aggregates

    def create_kinetic_modeling_job(self, settings: utils.ValueCollection) -> bool:
        """
        Create the kinetic modeling job.

        Parameters
        ----------
        settings : utils.ValueCollection
            The job settings.

        Returns
        -------
        bool
            True if the calculation was set up. False, otherwise.
        """
        raise NotImplementedError()

    @staticmethod
    def get_job():
        """
        Getter for the db.Job object.

        Returns
        -------
        db.Job
            The object.
        """
        raise NotImplementedError()

    @staticmethod
    def get_default_settings():
        """
        Getter for the job's default settings.

        Returns
        -------
        Dict
            The settings dictionary.
        """
        raise NotImplementedError()

    def _finalize_calculation(self, settings: utils.ValueCollection, all_structure_ids: List[db.ID]) -> bool:
        calc = db.Calculation(db.ID())
        calc.link(self._calculations)
        calc.create(self._calculation_model, self.get_job(), [])
        calc.set_settings(settings)
        calc.set_structures(all_structure_ids)
        finalize_calculation(calc, self._structures, all_structure_ids)
        return True

    def _calc_already_set_up(self, structure_ids: List[db.ID], settings: utils.ValueCollection) -> bool:
        common_calc_ids: Set[str] = get_common_calculation_ids(self.get_job().order, structure_ids,
                                                               self._calculation_model, self._structures,
                                                               self._calculations)
        """
        We do a manual loop over all calculations that may be the same to check some of the settings. We do not
        check all settings to avoid floating point comparisons.
        """
        for str_id in common_calc_ids:
            calculation = db.Calculation(db.ID(str_id), self._calculations)
            if not calculation.exists():
                continue
            calc_settings = calculation.get_settings().as_dict()
            same_settings = True
            for key in settings.keys():
                if key in self.order_dependent_setting_keys() or key in ["aggregate_types"]:
                    continue
                if key not in calc_settings or settings[key] != calc_settings[key]:
                    # We do not care about the order of the reaction/aggregate ids.
                    if key in ["reaction_ids", "aggregate_ids"]\
                            and sorted(settings[key]) == sorted(calc_settings[key]):  # type: ignore
                        continue
                    same_settings = False
                    break
            if same_settings:  # and self._identical_model_definition(settings, calc_settings):
                print("Kinetic modeling calculation already set up! Exploration converged!",)
                return True
            if self._get_n_queuing_calculations():
                print("Stopping calculation submission since there are still calculations queueing.")
                return True
        return False

    def _identical_model_definition(self, _: utils.ValueCollection, __: Dict) -> bool:
        raise NotImplementedError

    @staticmethod
    def order_dependent_setting_keys() -> List[str]:
        raise NotImplementedError

    def _get_n_queuing_calculations(self) -> int:
        from json import dumps
        stati = ["new", "hold", "pending"]
        selection = {
            "$and": [
                {"status": {"$in": stati}},
            ]
        }
        return self._calculations.count(dumps(selection))

    def get_reactions(self) -> Tuple[Set[Reaction], Dict[int, Aggregate]]:
        """
        The reactions are added iteratively starting from all compounds with a non-zero starting concentration.
        Reactions are added subject to the following conditions:
        1. The barrier does not exceed max_barrier.
        2. The reaction was not found to be negligible in a previous kinetic modeling run.
        3. One site of the reaction can be reached through already added reactions or consists solely of starting
        aggregates.
        """
        accessible_reactions: Set[Reaction] = set()
        accessible_aggregates: Dict[int, Aggregate] = self._get_starting_aggregates()
        # exclude reactions that were unimportant previously.
        old_zero_flux_reaction_ids = self._get_old_zero_flux_reactions(list(accessible_aggregates.values()))
        sets_changed = True
        rxn_string_ids_zero_flux = set([r_id.string() for r_id in old_zero_flux_reaction_ids])
        reactions_str_ids_to_iterate: Set[str] = self._get_reaction_str_ids_of_aggregates(
            set(accessible_aggregates.values()))
        reaction_str_ids_not_to_iterate = rxn_string_ids_zero_flux
        print("Iterative kinetic model construction from starting reactants")
        print('#Iter  N-Reactions N-Aggregates')
        while sets_changed:
            loop_iteration = 1
            sets_changed = False
            # iterate over all reactions of all currently accessible compounds minus the reactions already considered
            # accessible, zero flux, or inaccessible because of a too high barrier/low reaction rate constant.
            reactions_str_ids_to_iterate = reactions_str_ids_to_iterate.difference(reaction_str_ids_not_to_iterate)
            newly_added_aggregates: Set[Aggregate] = set()
            for r_str_id in reactions_str_ids_to_iterate:
                db_reaction = db.Reaction(db.ID(r_str_id), self._reactions)
                if not db_reaction.analyze() or not db_reaction.explore():
                    continue
                reaction = self._reaction_cache.get_or_produce(db.ID(r_str_id))
                # Otherwise, mypy is not able to differentiate between Aggregates and Reactions
                assert isinstance(reaction, Reaction)
                if reaction.circle_reaction() or not reaction.explore() or not reaction.analyze():
                    continue
                if self._reaction_is_accessible(reaction, set(accessible_aggregates.keys())):
                    accessible_reactions.add(reaction)
                    reaction_str_ids_not_to_iterate.add(r_str_id)
                    reactants = reaction.get_lhs_aggregates() + reaction.get_rhs_aggregates()
                    sets_changed = True
                    for aggregate in reactants:
                        int_id = int(aggregate)
                        if int_id not in accessible_aggregates:
                            # Call the cache here again to make sure that the aggregate is constructed with the correct
                            # electronic structure model. This becomes necessary if the reaction barriers are calculated
                            # with different models than the aggregate's energies of formation.
                            # Note that the singleton factory makes sure that there is no duplicated aggregates
                            # construction in the case of identical models for reactions and aggregates.
                            new_aggregate = self._aggregate_cache.get_or_produce(aggregate.get_db_id())
                            assert isinstance(new_aggregate, Aggregate)
                            newly_added_aggregates.add(new_aggregate)
                            accessible_aggregates[int_id] = new_aggregate
            reactions_str_ids_to_iterate = self._get_reaction_str_ids_of_aggregates(newly_added_aggregates)
            n_reactions = len(accessible_reactions)
            n_aggregates = len(accessible_aggregates.keys())
            print(f'#{loop_iteration:3}   {n_reactions:11} {n_aggregates:12}')
            loop_iteration += 1
        print("Converged!")
        return accessible_reactions, accessible_aggregates

    @staticmethod
    def _get_reaction_str_ids_of_aggregates(aggregates: Set[Aggregate]) -> Set[str]:
        reaction_str_id_set: Set[str] = set()
        for aggregate in aggregates:
            reaction_str_id_set.update([r_id.string() for r_id in aggregate.get_db_object().get_reactions()])
        return reaction_str_id_set

    def _get_starting_aggregates(self) -> Dict[int, Aggregate]:
        start_compound_ids: List[db.ID] = list()
        for compound in self._compounds.iterate_all_compounds():
            compound.link(self._compounds)
            start_concentration = query_concentration_with_object("start_concentration", compound, self._properties,
                                                                  self._structures)
            if start_concentration > 0:
                start_compound_ids.append(compound.id())
        aggregates = [self._aggregate_cache.get_or_produce(a_id) for a_id in start_compound_ids]
        return {int(a): a for a in aggregates if a.get_free_energy(self.reference_state) is not None}  # type: ignore

    def _reaction_is_accessible(self, reaction: Reaction, accessible_aggregate_ids: Set[int]) -> bool:
        lhs_rhs_ids = reaction.get_db_object().get_reactants(db.Side.BOTH)
        lhs_rhs_types = reaction.get_db_object().get_reactant_types(db.Side.BOTH)
        for a_id, a_type in zip(lhs_rhs_ids[0] + lhs_rhs_ids[1], lhs_rhs_types[0] + lhs_rhs_types[1]):
            db_aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
            if not db_aggregate.analyze():
                return False
        lhs_barrier, rhs_barrier = reaction.get_free_energy_of_activation(self.reference_state, in_j_per_mol=True)
        if lhs_barrier is None or rhs_barrier is None:
            return False
        all_lhs = True if lhs_barrier * 1e-3 < self.max_barrier else False  # Barrier in J/mol, threshold in kJ/mol
        all_rhs = True if rhs_barrier * 1e-3 < self.max_barrier else False

        if all_lhs:
            for lhs_id in lhs_rhs_ids[0]:
                if int(lhs_id.string(), 16) not in accessible_aggregate_ids:
                    all_lhs = False
                    break
        if all_rhs:
            for rhs_id in lhs_rhs_ids[1]:
                if int(rhs_id.string(), 16) not in accessible_aggregate_ids:
                    all_rhs = False
                    break
        return all_rhs or all_lhs

    def _get_old_zero_flux_reactions(self, aggregates: List[Aggregate]) -> List[db.ID]:
        if not self.use_zero_flux_truncation:
            return []
        reaction_ids = self._get_reactions_in_last_kinetic_modeling_jobs(aggregates)
        zero_flux_reactions: List[db.ID] = []
        for rxn_id in reaction_ids:
            if self._reaction_has_zero_flux(rxn_id):
                zero_flux_reactions.append(rxn_id)
        return zero_flux_reactions

    def _get_reactions_in_last_kinetic_modeling_jobs(self, aggregates: List[Aggregate]) -> List[db.ID]:
        reaction_str_id_set: Set[str] = set()
        for aggregate in aggregates:
            centroid = db.Structure(aggregate.get_db_object().get_centroid(), self._structures)
            calc_ids = centroid.get_calculations(self.get_job().order)
            for c_id in calc_ids:
                calculation = db.Calculation(c_id, self._calculations)
                if not calculation.exists():
                    continue
                if calculation.status == db.Status.COMPLETE:
                    settings = calculation.get_settings()
                    if "concentration_label_postfix" in settings:
                        if settings["concentration_label_postfix"] != "":
                            continue
                    str_ids: List[str] = settings["reaction_ids"]  # type: ignore
                    reaction_str_id_set.update(str_ids)
        return [db.ID(str_id) for str_id in reaction_str_id_set]

    def _reaction_has_zero_flux(self, reaction_id: db.ID) -> bool:
        reaction = db.Reaction(reaction_id, self._reactions)
        reactants = reaction.get_reactants(db.Side.BOTH)
        reactant_types = reaction.get_reactant_types(db.Side.BOTH)
        for a_id, a_type in zip(reactants[0] + reactants[1], reactant_types[0] + reactant_types[1]):
            aggregate = get_compound_or_flask(a_id, a_type, self._compounds, self._flasks)
            flux = query_concentration_with_object(self.vertex_flux_label, aggregate, self._properties,
                                                   self._structures)
            if self.flux_variance_label:
                flux += math.sqrt(query_concentration_with_object(self.flux_variance_label, aggregate, self._properties,
                                                                  self._structures))
            if flux < self.min_flux_truncation:
                return True
        return False
