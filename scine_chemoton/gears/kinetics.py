#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from abc import ABC
from json import dumps
from typing import Dict, List, Tuple, Union

# Third party imports
import scine_database as db

# Local application imports
from . import Gear
from ..utilities.queries import stop_on_timeout
from .kinetic_modeling.concentration_query_functions import query_concentration_with_model_object
from ..utilities.energy_query_functions import get_barriers_for_elementary_step_by_type
from ..utilities.compound_and_flask_creation import get_compound_or_flask


class KineticsBase(Gear, ABC):
    """
    Base class for kinetics gears.
    """

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self._required_collections = ["compounds", "elementary_steps", "flasks",
                                      "properties", "reactions", "structures"]

    class Options:
        """
        The options for the MinimalConnectivityKinetics Gear.
        """
        __slots__ = ("cycle_time", "restart")

        def __init__(self):
            self.cycle_time = 10
            """
            int
                The minimum number of seconds between two cycles of the Gear.
                Cycles are finished independent of this option, thus if a cycle
                takes longer than the cycle_time will effectively lead to longer
                cycle times and not cause multiple cycles of the same Gear.
            """
            self.restart = False
            """
            bool
                Option to restart the filtering of the network, by disabling
                all aggregates again. Set this to ``True`` if you want to
                reevaluate a network with different settings.
            """

    def _loop_impl(self):
        if self.options.restart:
            self._disable_all_aggregates()
            self.options.restart = False
        # Loop over all deactivated aggregates
        selection = {"exploration_disabled": True}
        for compound in stop_on_timeout(self._compounds.iterate_compounds(dumps(selection))):
            compound.link(self._compounds)
            if self._aggregate_was_inserted_by_user(compound):
                compound.enable_exploration()
            elif self._aggregate_accessible_by_reaction(compound) \
                    and self._filter(compound):
                compound.enable_exploration()
        for flask in stop_on_timeout(self._flasks.iterate_flasks(dumps(selection))):
            flask.link(self._flasks)
            if self._aggregate_was_inserted_by_user(flask):
                flask.enable_exploration()
            elif self._aggregate_accessible_by_reaction(flask) \
                    and self._filter(flask):
                flask.enable_exploration()

    def _disable_all_aggregates(self):
        for compound in self._compounds.iterate_all_compounds():
            compound.link(self._compounds)
            compound.disable_exploration()
        for flask in self._flasks.iterate_all_flasks():
            flask.link(self._flasks)
            flask.disable_exploration()

    def _aggregate_was_inserted_by_user(self, aggregate: Union[db.Compound, db.Flask]) -> bool:
        """
        Any structure of aggregates has a user label

        Parameters
        ----------
        aggregate : Union[scine_database.Compound, scine_database.Flask]
            The aggregate that may have been inserted by user

        Returns
        -------
        bool
            aggregate was inserted
        """
        user_labels = [db.Label.USER_OPTIMIZED, db.Label.USER_GUESS]
        for s_id in aggregate.get_structures():
            structure = db.Structure(s_id)
            structure.link(self._structures)
            if structure.get_label() in user_labels:
                return True
        return False

    def _aggregate_accessible_by_reaction(self, aggregate: Union[db.Compound, db.Flask]) -> bool:
        """
        Aggregate is on RHS of a reaction that requires only explorable
        aggregates and has not been deactivated.

        Parameters
        ----------
        aggregate : Union[scine_database.Compound, scine_database.Flask]
            The aggregate to check to be accessible

        Returns
        -------
        bool
            aggregate is accessible
        """
        selection = {"$and": [
            {"rhs": {"$elemMatch": {"id": {"$oid": aggregate.get_id().string()}}}},
            {"exploration_disabled": {"$ne": True}}
        ]}
        for hit in self._reactions.iterate_reactions(dumps(selection)):
            hit.link(self._reactions)
            accessible = True
            for reactant_id, reactant_type in zip(hit.get_reactants(db.Side.LHS)[0],
                                                  hit.get_reactant_types(db.Side.LHS)[0]):
                reactant = get_compound_or_flask(reactant_id, reactant_type, self._compounds, self._flasks)
                if not reactant.explore():
                    accessible = False
            if accessible:
                return True
        return False

    def _filter(self, _: Union[db.Compound, db.Flask]) -> bool:
        return True


class MinimalConnectivityKinetics(KineticsBase):
    """
    This Gear enables the exploration of aggregates (Compounds and Flasks)
    if they were inserted by the user or created by a reaction
    that requires only explorable aggregates.
    This should be the case for any aggregates after a sufficient number of
    iterations and simply drive the exploration.

    Attributes
    ----------
    options :: MinimalConnectivityKinetics.Options
        The options for the MinimalConnectivityKinetics Gear.
    """

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self._required_collections = ["compounds", "elementary_steps", "flasks",
                                      "properties", "reactions", "structures"]

    class Options:
        """
        The options for the MinimalConnectivityKinetics Gear.
        """
        __slots__ = ("cycle_time", "restart", "user_input_only")

        def __init__(self):
            self.cycle_time = 10
            """
            int
                The minimum number of seconds between two cycles of the Gear.
                Cycles are finished independent of this option, thus if a cycle
                takes longer than the cycle_time will effectively lead to longer
                cycle times and not cause multiple cycles of the same Gear.
            """
            self.restart = False
            """
            bool
                Option to restart the filtering of the network, by disabling
                all aggregates again. Set this to ``True`` if you want to
                reevaluate a network with different settings.
            """
            self.user_input_only = False
            """
            bool
                Option to only ever allow compounds that contain structures
                added to the database by a user. If enabled, no other compounds
                will ever be enabled irrespective of any connectivity.
            """

    def _filter(self, _: Union[db.Compound, db.Flask]) -> bool:
        return not self.options.user_input_only


class BasicBarrierHeightKinetics(KineticsBase):
    """
    This Gear enables the exploration of aggregates if they were inserted by
    the user or created by a reaction that requires only explorable aggregates
    and has a forward reaction barrier below a given threshold.

    Attributes
    ----------
    options :: BasicBarrierHeightKinetics.Options
        The options for the BasicBarrierHeightKinetics Gear.

    Notes
    -----
    Checks for all aggregates that are accessed via a 'reaction'. Manually
    inserted Compounds/Flasks are always activated by this gear.
    """

    def __init__(self):
        super().__init__()
        self.options = self.Options()

    class Options:
        """
        The options for the BasicBarrierHeightKinetics Gear.
        """

        __slots__ = ("cycle_time", "max_allowed_barrier", "model", "enforce_free_energies", "restart")

        def __init__(self):
            self.cycle_time = 60
            """
            int
                The minimum number of seconds between two cycles of the Gear.
                Cycles are finished independent of this option, thus if a cycle
                takes longer than the cycle_time will effectively lead to longer
                cycle times and not cause multiple cycles of the same Gear.
            """
            self.max_allowed_barrier = 1000.0  # kJ/mol
            """
                float
                The maximum barrier height of the reaction resulting in the
                aggregate in kJ/mol to allow the aggregate to be further explored.
            """
            self.model: db.Model = db.Model("PM6", "PM6", "")
            """
            db.Model (Scine::Database::Model)
                The Model determining the energies for the barrier determination.
            """
            self.enforce_free_energies = False
            """
            bool
                Whether the gear should only enable aggregates based on free
                energy barriers or can also enable based on electronic energies
                alone. Make sure to run a Thermo gear if you set this to ``True``.
            """
            self.restart = False
            """
            bool
                Option to restart the filtering of the network, by disabling
                each aggregate again. Set this to True if you want to reevaluate
                a network with different settings
            """

    def _filter(self, aggregate: Union[db.Compound, db.Flask]):
        selection = {"$and": [
            {"rhs": {"$elemMatch": {"id": {"$oid": aggregate.get_id().string()}}}},
            {"exploration_disabled": {"$ne": True}}
        ]}
        for reaction in self._reactions.iterate_reactions(dumps(selection)):
            reaction.link(self._reactions)
            if not self._reaction_barrier_too_high(reaction):
                return True
        return False

    def _reaction_barrier_too_high(self, reaction: db.Reaction) -> bool:
        """
        Whether the reaction barrier of the reaction is too high

        Parameters
        ----------
        reaction : scine_database.Reaction (Scine::Database::Reaction)
            The reaction
        """
        barrier_heights = self._barrier_height(reaction)
        if barrier_heights[0] is None:
            if self.options.enforce_free_energies or barrier_heights[1] is None:
                # skip if free energy barrier not available yet and only free energies allowed or electronic energy also
                # not available (e.g., because of model refinement)
                return True
            return barrier_heights[1] > self.options.max_allowed_barrier
        return barrier_heights[0] > self.options.max_allowed_barrier

    def _barrier_height(self, reaction: db.Reaction) -> Tuple[Union[float, None], Union[float, None]]:
        """
        Gives the lowest barrier height of the forward reaction (left to right) in kJ/mol out of all the elementary
        steps grouped into this reaction. Barrier height are given as Tuple with the first being gibbs free energy
        and second one the electronic energy. Returns None for not available energies

        Parameters
        ----------
        reaction : scine_database.Reaction (Scine::Database::Reaction)
            The reaction we want the barrier height from

        Returns
        -------
        Tuple[Union[float, None], Union[float, None]]
            barrier height in kJ/mol
        """
        barriers: Dict[str, List[float]] = {"gibbs_free_energy": [], "electronic_energy": []}
        for step_id in reaction.get_elementary_steps():
            step = db.ElementaryStep(step_id)
            step.link(self._elementary_steps)
            for energy_type, values in barriers.items():
                barrier, _ = get_barriers_for_elementary_step_by_type(step, energy_type, self.options.model,
                                                                      self._structures, self._properties)
                if barrier is not None:
                    values.append(barrier)
        gibbs = None if not barriers["gibbs_free_energy"] else min(barriers["gibbs_free_energy"])
        electronic = None if not barriers["electronic_energy"] else min(barriers["electronic_energy"])
        return gibbs, electronic


class MaximumFluxKinetics(BasicBarrierHeightKinetics):
    """
    This Gear enables the exploration of compounds if they were inserted by the user, created by a reaction
    that requires only compounds with a concentration flux larger than a given threshold, has a forward reaction barrier
    below a given maximum, and has reached a minimum concentration larger than a given threshold during kinetic
    modeling.

    Attributes
    ----------
    options :: MaximumFluxKinetics.Options
        The options for the MaximumFluxKinetics Gear.

    Notes
    -----
    Checks for all compounds that are accessed via a 'reaction'. Manually inserted Compounds
    are always activated by this gear.
    """

    def __init__(self):
        super().__init__()
        self.options = self.Options()

    class Options:
        """
        The options for the MaximumPopulationKinetics Gear.
        """

        __slots__ = ("cycle_time", "max_allowed_barrier", "model", "enforce_free_energies", "restart",
                     "min_allowed_concentration", "property_label", "min_concentration_flux", "flux_property_label")

        def __init__(self):
            self.cycle_time = 60
            """
            int
                The minimum number of seconds between two cycles of the Gear.
                Cycles are finished independent of this option, thus if a cycle
                takes longer than the cycle_time will effectively lead to longer
                cycle times and not cause multiple cycles of the same Gear.
            """
            self.max_allowed_barrier = 1000.0
            """
            float
                The maximum barrier height of the reaction resulting in the compound
                in kJ/mol to allow the compound to be further explored.
            """
            self.model: db.Model = db.Model("PM6", "", "")
            """
            db.Model (Scine::Database::Model)
                The Model determining the energies for the barrier determination.
            """
            self.enforce_free_energies = False
            """
            bool
                Whether the gear should only enable compounds based on free energy barriers
                or can also enable based on electronic energies alone.
                Make sure to run a Thermo gear if you set this to True
            """
            self.restart = False
            """
            bool
                Option to restart the filtering of the network, by disabling each compound again.
                Set this to True if you want to reevaluate a network with different settings
            """
            self.min_allowed_concentration = 1e-4
            """
            float
                The minimum allowed concentration flux to be considered for further exploration.
            """
            self.property_label = "max_concentration"
            """
            str
                The label of the concentration property that is used to determine explorable compounds.
            """
            self.min_concentration_flux = 1e-4
            """
            float
                The minimum concentration flux that is required to consider the compound as accessible.
            """
            self.flux_property_label = "concentration_flux"
            """
            str
                The property label for the concentration flux.
            """

    def _filter(self, aggregate: Union[db.Compound, db.Flask]):
        selection = {
            "$and": [
                {"rhs.id": {"$oid": aggregate.get_id().string()}},
                {"exploration_disabled": {"$ne": True}}
            ]
        }
        barrier_too_high = True
        for reaction in self._reactions.iterate_reactions(dumps(selection)):
            reaction.link(self._reactions)
            if not self._reaction_barrier_too_high(reaction):
                barrier_too_high = False
                break
        if barrier_too_high:
            return False
        max_concentration = query_concentration_with_model_object(self.options.property_label,
                                                                  aggregate,
                                                                  self._properties,
                                                                  self._structures,
                                                                  self.options.model)
        if self.options.min_allowed_concentration > max_concentration:
            return False
        return True

    def _compound_accessible_by_reaction(self, _: Union[db.Compound, db.Flask]) -> bool:
        """
        Assume every compound to be accessible. Compound elimination happens via the concentration.

        Parameters
        ----------
        compound : scine_database.Compound (Scine::Database::Compound)
            The compound to check to be accessible

        Returns
        -------
        bool
            compound is accessible
        """
        return True
