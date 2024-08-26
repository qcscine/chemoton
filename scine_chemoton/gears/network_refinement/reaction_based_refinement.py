#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, Optional, Set, Dict
from json import dumps
import os
import pickle

from .refinement import NetworkRefinement

# Third party imports
import scine_database as db

from scine_database.queries import (
    model_query,
    stop_on_timeout,
)
from scine_chemoton.utilities.db_object_wrappers.wrapper_caches import MultiModelCacheFactory
from scine_chemoton.utilities.db_object_wrappers.reaction_wrapper import Reaction
from scine_chemoton.gears.network_refinement.enabling import PlaceHolderReactionEnabling


class ReactionBasedRefinement(NetworkRefinement):
    """
    This class allows the refinement of elementary steps (or the associated energies and structures). The elementary
    steps are selected based on their reaction. For instance one can calculate the single point energies of the 10
    lowest transition states and structures on the lhs/rhs of the reaction with a different electronic structure method.

    The reactions and steps may be filtered through elementary step and reaction filters.
    """

    def __init__(self) -> None:
        super().__init__()
        self._elementary_step_to_calculation_map: Dict[str, str] = {}

    def _loop(self, job_label: str):
        """
        Create refinement calculations for only a selection of the elementary
        steps for each reaction (e.g. the most favorable one).

        Parameters
        ----------
        job_label: The label for the refinement to be executed.
        """
        # Create an elementary step id to calculation id map if necessary.
        self._load_elementary_step_index()
        if not self._elementary_step_to_calculation_map:
            self._create_elementary_step_index()
        for reaction in stop_on_timeout(self._reactions.iterate_all_reactions()):
            if self.have_to_stop_at_next_break_point():
                return
            reaction.link(self._reactions)
            if self.reaction_filter.filter(reaction):
                self._refine_reaction(reaction, job_label)
        self._save_elementary_step_index()

    def _refine_reaction(self, reaction: db.Reaction, job_label: str) -> None:
        # If a reaction enabling policy is given, we skip the refinement and just enable the reaction we would like
        # to refine.
        if not isinstance(self.reaction_enabling, PlaceHolderReactionEnabling):
            self.reaction_enabling.process(reaction)
            # Check if the reaction has an enabled step with the correct model now.
            if self.reaction_validation.filter(reaction):
                return
        assert self.model_combination
        wrapper = MultiModelCacheFactory().get_reaction_cache(
            self.options.only_electronic_energies, self.model_combination, self._manager).get_or_produce(reaction.id())
        eligible_steps = self._select_elementary_steps(wrapper)
        setup_tracker: List[bool] = []
        if job_label in ["refine_single_points"]:
            self._refine_single_points_on_reaction(wrapper, eligible_steps)
            self.reaction_disabling.process(reaction, job_label)
            return
        for step_id in eligible_steps:
            if job_label in ["refine_structures_and_irc", "refine_single_ended_search", "double_ended_refinement"]:
                calc_id = self._get_calculation_id_for_step(step_id)
                if calc_id:
                    calculation = db.Calculation(calc_id, self._calculations)
                    setup_tracker.append(self._set_up_calculation(job_label, calculation, step_id))
                else:
                    raise Warning(
                        "Warning: An elementary step was encountered that was not created by a calculation"
                        "in the database. It will be skipped during refinement. Step-id: " + step_id.string())
            else:
                setup_tracker.append(self._set_up_calculation(job_label, None, step_id))
        if all(setup_tracker) and setup_tracker:
            self.reaction_disabling.process(reaction, job_label)

    def _refine_single_points_on_reaction(self, reaction: Reaction, eligible_step_ids: List[db.ID]) -> None:
        all_structures: Set[str] = set([db.ElementaryStep(step_id, self._elementary_steps)
                                       .get_transition_state().string() for step_id in eligible_step_ids])
        aggregates = reaction.get_lhs_aggregates() + reaction.get_rhs_aggregates()
        for a in aggregates:
            all_structures.update([s_id.string() for s_id in a.get_lowest_n_structures(
                self.options.refine_n_per_reaction, self.options.aggregate_energy_window,
                self.options.reference_state)])
        self._refine_single_points([db.ID(str_id) for str_id in all_structures])

    def _select_elementary_steps(self, reaction: Reaction) -> List[db.ID]:
        """
        Select the most favorable (the lowest energy transition state) elementary step(s) for a given reaction if they
        fulfill the given elementary step filter.

        Parameters
        ----------
        reaction : Reaction
            The reaction wrapper.
        Returns
        -------
        List[db.ID]
            The list of the selected elementary steps.
        """
        all_eligible_steps = [step_id for step_id in reaction.get_lowest_n_steps(
            len(reaction.get_db_object().get_elementary_steps()),
            self.options.transition_state_energy_window,
            self.options.reference_state) if self.elementary_step_filter.filter(
            db.ElementaryStep(step_id, self._elementary_steps))]
        select_n = min(len(all_eligible_steps), self.options.refine_n_per_reaction)
        return all_eligible_steps[:select_n]

    def _create_elementary_step_index(self) -> None:
        selection = {
            "$and": [
                {"status": "complete"},
                {"results.elementary_steps.0": {"$exists": True}},
            ]
            + model_query(self.options.calculation_model)
        }
        self._elementary_step_to_calculation_map = dict()
        for calculation in stop_on_timeout(self._calculations.iterate_calculations(dumps(selection))):
            calculation.link(self._calculations)
            self._update_elementary_step_index(calculation)
        self._save_elementary_step_index()

    def _save_elementary_step_index(self):
        # save dictionary to pickle file
        with open(self.options.elementary_step_index_file_name, 'wb') as file:
            pickle.dump(self._elementary_step_to_calculation_map, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_elementary_step_index(self) -> None:
        if os.path.exists(self.options.elementary_step_index_file_name) \
                and os.path.getsize(self.options.elementary_step_index_file_name) > 0:
            with open(self.options.elementary_step_index_file_name, "rb") as file:
                load_cache = pickle.load(file)
                if load_cache:
                    self._elementary_step_to_calculation_map.update(load_cache)

    def _update_elementary_step_index(self, calculation: db.Calculation) -> None:
        elementary_step_ids = calculation.get_results().elementary_step_ids
        for step_id in elementary_step_ids:
            self._elementary_step_to_calculation_map[step_id.string()] = calculation.id().string()

    def _get_calculation_id_for_step(self, elementary_step_id: db.ID) -> Optional[db.ID]:
        if elementary_step_id.string() in self._elementary_step_to_calculation_map:
            return db.ID(self._elementary_step_to_calculation_map[elementary_step_id.string()])
        # TODO: This will be slow and could time out. But I do not see a better way to do it at the moment. We need the
        #  calculation id for the original calculation's settings.
        selection = {
            "$and": [
                {"results.elementary_steps": {"$oid": elementary_step_id.string()}}
            ]
        }
        calc = self._calculations.get_one_calculation(dumps(selection))
        if calc is None:
            print(f"Failed to find calculation that resulted in elementary step "
                  f"{str(elementary_step_id)}")
            return None
        calc.link(self._calculations)
        self._update_elementary_step_index(calc)
        return calc.id()

    def _all_structures_match_model(self, structure_ids: List[db.ID], model: db.Model) -> bool:
        for s_id in structure_ids:
            structure = db.Structure(s_id, self._structures)
            structure_model = structure.get_model()
            if structure_model != model:
                return False
        return True
