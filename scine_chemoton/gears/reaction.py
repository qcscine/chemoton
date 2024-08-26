#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
from typing import List, Union, Optional, Tuple
from warnings import warn
import numpy as np
from copy import copy

# Third party imports
import scine_database as db
from scine_database.queries import stop_on_timeout
from scine_database.energy_query_functions import get_energy_for_structure
from scine_database.compound_and_flask_creation import get_compound_or_flask
import scine_utilities as utils

# Local application imports
from . import Gear


class BasicReactionHousekeeping(Gear):
    """
    This Gear updates all Elementary Steps stored in the database such that they
    are added to an existing Reaction or that a new Reaction is created if the
    Step does not fit an existing one.

    Attributes
    ----------
    options : BasicReactionHousekeeping.Options
        The options for the BasicReactionHousekeeping Gear.

    Notes
    -----
    Checks for all Elementary Steps without a 'reaction'.
    """

    def __init__(self) -> None:
        super().__init__()
        self.options = self.Options()
        self._required_collections = ["compounds", "elementary_steps", "flasks", "reactions", "properties",
                                      "structures"]
        self._reaction_cache: Optional[dict] = None
        self._model_is_required = False

    class Options(Gear.Options):
        """
        The options for the BasicReactionHousekeeping Gear.
        """

        __slots__ = ("use_structure_deduplication",
                     "use_energy_deduplication",
                     "use_rmsd_deduplication",
                     "energy_tolerance",
                     "rmsd_tolerance")

        def __init__(self) -> None:
            super().__init__()
            self.use_structure_deduplication = True
            """
            bool
                If true duplicated elementary steps are not sorted into reactions.
                The criterion is that all structures on the lhs of the elementary step
                have to be identical. This criterion can be combined with other selection criteria.
                Steps are only eliminated if all selection criteria signal identical steps.
            """
            self.use_energy_deduplication = True
            """
            bool
                If true duplicated elementary steps are not sorted into reactions.
                The criterion is that the total electronic energy of the transition state
                (if available) has to be within the threshold limits of `energy_tolerance`.
                This criterion can be combined with other selection criteria. Steps are only eliminated
                if all selection criteria signal identical steps.
            """
            self.use_rmsd_deduplication = True
            """
            bool
                If true duplicated elementary steps are not sorted into reactions.
                The criterion is that the RMSD of the transition state coordinates
                (if available) have to be within the threshold limits of `rmsd_tolerance`.
                This criterion can be combined with other selection criteria. Steps are only eliminated
                if all selection criteria signal identical steps. Note that this deduplication
                strategy may fail to eliminate elementary steps if the atom ordering the transition
                state differs.
            """
            self.energy_tolerance = 1e-5
            """
            float
                The energy tolerance for the transition state energy deduplication.
            """
            self.rmsd_tolerance = 1e-3
            """
            float
                The RMSD tolerance for the transition state RMSD deduplication.
            """

    options: Options

    def _loop_impl(self):
        # Setup query for elementary steps without reactions
        selection = {"$and": [
            {"reaction": ""},
            {"analysis_disabled": {"$ne": True}}
        ]}
        uses_elementary_step_deduplication =\
            self.options.use_structure_deduplication or self.options.use_energy_deduplication or \
            self.options.use_rmsd_deduplication
        # Loop over all results
        for step in stop_on_timeout(self._elementary_steps.iterate_elementary_steps(dumps(selection))):
            step.link(self._elementary_steps)
            reactants = step.get_reactants(db.Side.BOTH)
            if self.have_to_stop_at_next_break_point():
                return

            # Determine structure types
            types = [[], []]
            aggregates = [[], []]
            all_structures_have_aggregates = True
            for i, side in enumerate(reactants):
                if not all_structures_have_aggregates:
                    # we are breaking here to avoid rhs evaluation if lhs is already incomplete
                    break
                for structure_id in side:
                    structure = db.Structure(structure_id, self._structures)
                    # check aggregate
                    if not structure.has_aggregate():
                        all_structures_have_aggregates = False
                        break
                    if not structure.has_graph("masm_cbor_graph"):
                        raise RuntimeError(f"The Structure {str(structure.id())} has an aggregate, but no graph")
                    graph = structure.get_graph("masm_cbor_graph")
                    agg_type = db.CompoundOrFlask.FLASK if ";" in graph else db.CompoundOrFlask.COMPOUND
                    types[i].append(agg_type)
                    aggregates[i].append(structure.get_aggregate())
            if not all_structures_have_aggregates:
                continue

            self._replace_duplicate_structures(step)
            # Check for a reactions with the same structures/compounds
            true_hit, is_parallel = self._check_cached_reactions(aggregates[0], aggregates[1])
            if true_hit is not None:
                if not is_parallel:
                    self._invert_elementary_step(step)
                # Add elementary step to reaction if it is not duplicated.
                if uses_elementary_step_deduplication and self._is_duplicate(step, true_hit):
                    step.disable_analysis()
                    step.disable_exploration()
                reaction = true_hit
                # check for regular vs barrierless
                self._disable_barrierless_if_mixed_types_in_reaction(reaction, step)
                reaction.add_elementary_step(step.id())
                step.set_reaction(reaction.id())
                reaction.enable_exploration()
                reaction.enable_analysis()
                continue
            # Generate new reaction
            reaction = db.Reaction(db.ID(), self._reactions)
            reaction.create(aggregates[0], aggregates[1], types[0], types[1])
            reaction.add_elementary_step(step.id())
            step.set_reaction(reaction.id())
            self._add_reaction_to_aggregate(aggregates[0] + aggregates[1], types[0] + types[1], reaction.id())
            self._add_to_cache(reaction, aggregates[0], aggregates[1])

    def _check_cached_reactions(
            self, lhs_aggregates: List[db.ID], rhs_aggregates: List[db.ID]
    ) -> Tuple[Optional[db.Reaction], bool]:
        self._rebuild_cache()
        assert self._reaction_cache is not None
        key = ','.join(sorted([x.string() for x in lhs_aggregates]))
        key += '<->'
        key += ','.join(sorted([x.string() for x in rhs_aggregates]))
        if key in self._reaction_cache:
            reaction = db.Reaction(db.ID(self._reaction_cache[key]))
            reaction.link(self._reactions)
            return reaction, True
        key = ','.join(sorted([x.string() for x in rhs_aggregates]))
        key += '<->'
        key += ','.join(sorted([x.string() for x in lhs_aggregates]))
        if key in self._reaction_cache:
            reaction = db.Reaction(db.ID(self._reaction_cache[key]))
            reaction.link(self._reactions)
            return reaction, False
        return None, True

    def _rebuild_cache(self) -> None:
        if self._reaction_cache is None:
            self._reaction_cache = {}
            for r in self._reactions.iterate_reactions('{}'):
                r.link(self._reactions)
                reactants = r.get_reactants(db.Side.BOTH)
                key = ','.join(sorted([x.string() for x in reactants[0]]))
                key += '<->'
                key += ','.join(sorted([x.string() for x in reactants[1]]))
                self._reaction_cache[key] = r.get_id().string()

    def _add_to_cache(
        self, reaction: db.Reaction, lhs_aggregates: List[db.ID], rhs_aggregates: List[db.ID]
    ) -> None:
        assert self._reaction_cache is not None
        key = ','.join(sorted([x.string() for x in lhs_aggregates]))
        key += '<->'
        key += ','.join(sorted([x.string() for x in rhs_aggregates]))
        assert key not in self._reaction_cache
        self._reaction_cache[key] = reaction.get_id().string()

    def _replace_duplicate_structures(self, elementary_step: db.ElementaryStep):
        reactants = elementary_step.get_reactants(db.Side.BOTH)
        unique_lhs = self._make_unique_structure_id_list(reactants[0])
        unique_rhs = self._make_unique_structure_id_list(reactants[1])
        elementary_step.set_reactants(unique_lhs, db.Side.LHS)
        elementary_step.set_reactants(unique_rhs, db.Side.RHS)

    def _make_unique_structure_id_list(self, id_list: List[db.ID]) -> List[db.ID]:
        unique_list: List[db.ID] = list()
        for s_id in id_list:
            structure = db.Structure(s_id, self._structures)
            if structure.get_label() == db.Label.DUPLICATE:
                unique_list.append(structure.get_original())
            else:
                unique_list.append(s_id)
        return unique_list

    @staticmethod
    def _invert_elementary_step(elementary_step: db.ElementaryStep):
        reactants = copy(elementary_step.get_reactants(db.Side.BOTH))
        elementary_step.set_reactants(reactants[1], db.Side.LHS)
        elementary_step.set_reactants(reactants[0], db.Side.RHS)
        if elementary_step.has_spline():
            spline = elementary_step.get_spline()
            knots = np.flipud(np.asarray([1 - x for x in spline.knots]))
            trajectory = np.flipud(spline.data)
            ts_position = 1 - spline.ts_position
            elements = spline.elements
            inverted_spline = utils.bsplines.TrajectorySpline(elements, knots, trajectory, ts_position)
            elementary_step.set_spline(inverted_spline)

    def _disable_barrierless_if_mixed_types_in_reaction(self, reaction: db.Reaction, new_step: db.ElementaryStep):
        """
        If we have a regular elementary step and a barrierless elementary step for an identical reaction,
        we are for now distrusting the barrierless steps and disable them, but keep them in the reaction.

        NOTE: This will not work with old databases that have not worked with this method before out of the box.
        To ensure backwards applicability, the content of the reaction collection has to be deleted first.
        NOTE: This method assumes there are only TWO types of elementary steps.
        """
        # we have two possible procedures here:
        # 1) check new step type:
        #    - if regular, loop over all steps of reaction and disable all barrierless steps
        #    - if barrierless, loop over steps of reaction, if we find a regular step, break and disable new
        # 2) check type of first enabled step in reaction:
        #    - if different to new step type, check new type
        #        - if regular, loop over all steps of reaction and disable barrierless
        #        - if barrierless, only disable new
        #    - if identical do nothing
        # case 2 seems to be more efficient, works on the assumption that newly added entry has always been checked,
        # hence added note in doc string
        step_type = None
        steps = reaction.get_elementary_steps()
        # find step type of first enabled step in reaction
        for step_id in steps:
            reaction_step = db.ElementaryStep(step_id, self._elementary_steps)
            if reaction_step.explore() and reaction_step.analyze():
                step_type = reaction_step.get_type()
                break
        # we have a mismatch between the first activate step in reaction and the new step
        if step_type is not None and step_type != new_step.get_type():
            warn(f"Found barrierless and regular elementary step for identical reaction {str(reaction.id())}. "
                 f"We are now disabling all barrierless steps in this reaction.")
            new_type = new_step.get_type()
            barrierless_types = [db.ElementaryStepType.BARRIERLESS]
            if new_type in barrierless_types:
                # new is barrierless, first enabled was regular, hence we should not need to check all steps of reaction
                new_step.disable_exploration()
                new_step.disable_analysis()
            elif new_type == db.ElementaryStepType.REGULAR:
                # new step is regular, and first enabled was barrierless, disable all barrierless in reaction
                for step_id in steps:
                    reaction_step = db.ElementaryStep(step_id, self._elementary_steps)
                    if reaction_step.get_type() in barrierless_types:
                        reaction_step.disable_exploration()
                        reaction_step.disable_analysis()
            else:
                raise TypeError(f"Unsupported elementary step type for elementary step {str(new_step.id())}")

    def _add_reaction_to_aggregate(self, aggregates_to_change: List[db.ID], aggregate_types: List[db.CompoundOrFlask],
                                   reaction_id: db.ID):
        for aggregate_id, aggregate_type in zip(aggregates_to_change, aggregate_types):
            flask_or_compound = get_compound_or_flask(aggregate_id, aggregate_type, self._compounds, self._flasks)
            flask_or_compound.add_reaction(reaction_id)

    def _same_energy(self, ts_new_energy: Union[float, None], ts_old: db.Structure, model: db.Model) -> bool:
        ts_old_energy = get_energy_for_structure(ts_old, "electronic_energy", model, self._structures,
                                                 self._properties)
        # Check transition state energy.
        if ts_old_energy is None or ts_new_energy is None:
            return False
        abs_difference = abs(ts_old_energy - ts_new_energy)
        return abs_difference <= self.options.energy_tolerance

    def _same_coordinates(self, ts_positions_new: np.ndarray, ts_elements: List[utils.ElementType],
                          ts_old: db.Structure) -> bool:
        fit = utils.QuaternionFit(ts_positions_new, ts_old.get_atoms().positions, ts_elements)
        rmsd = fit.get_weighted_rmsd()
        # Check structure RMSD
        # TODO It may be possible to reorder the atoms of both transition states to be in identical
        # order before checking the RMSD using atom-index maps.
        return rmsd <= self.options.rmsd_tolerance

    def _same_starting_structures(self, lhs_new_step: List[db.ID], old_step: db.ElementaryStep) -> bool:
        lhs_old_step = old_step.get_reactants(db.Side.LHS)[0]
        if len(lhs_old_step) != len(lhs_new_step):
            return False
        return sorted(lhs_old_step) == sorted(lhs_new_step)

    def _is_duplicate(self, elementary_step: db.ElementaryStep, reaction: db.Reaction) -> bool:
        lhs_new_step = elementary_step.get_reactants(db.Side.LHS)[0]
        if not elementary_step.has_transition_state():
            return False
        ts_new = db.Structure(elementary_step.get_transition_state(), self._structures)
        model = ts_new.model
        ts_new_energy = get_energy_for_structure(
            ts_new, "electronic_energy", model, self._structures, self._properties)
        ts_positions_new = ts_new.get_atoms().positions
        ts_elements = ts_new.get_atoms().elements
        for old_step_id in reaction.get_elementary_steps():
            old_step = db.ElementaryStep(old_step_id, self._elementary_steps)
            if self.options.use_structure_deduplication:
                if not self._same_starting_structures(lhs_new_step, old_step):
                    continue

            if not old_step.has_transition_state():
                continue
            ts_old = db.Structure(old_step.get_transition_state(), self._structures)

            if self.options.use_energy_deduplication:
                if not self._same_energy(ts_new_energy, ts_old, model):
                    continue
            if self.options.use_rmsd_deduplication:
                if not self._same_coordinates(ts_positions_new, ts_elements, ts_old):
                    continue

            # If all checks signal identical structures. This is a duplicate!
            return True
        return False
