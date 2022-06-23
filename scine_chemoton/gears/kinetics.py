#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
from typing import Tuple, Union

# Third party imports
import scine_database as db

# Local application imports
from . import Gear
from ..utilities.queries import stop_on_timeout
from ..gears.energy_query_functions import get_single_barrier_for_elementary_step_by_type


class MinimalConnectivityKinetics(Gear):
    """
    This Gear enables the exploration of compounds if they were inserted by the user or created by a reaction
    that requires only explorable compounds.
    This should be the case for any compound after a sufficient number of iterations and simply drive the exploration.

    Attributes
    ----------
    options :: MinimalConnectivityKinetics.Options
        The options for the MinimalConnectivityKinetics Gear.
    """

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self._elementary_steps = "required"
        self._structures = "required"
        self._reactions = "required"
        self._compounds = "required"

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
                Option to restart the filtering of the network, by disabling each compound again.
                Set this to True if you want to reevaluate a network with different settings
            """

    def _loop_impl(self):
        if self.options.restart:
            self._disable_all_compounds()
            self.options.restart = False
        # Loop over all deactivated compounds
        selection = {"exploration_disabled": {"$eq": True}}
        for compound in stop_on_timeout(self._compounds.iterate_compounds(dumps(selection))):
            compound.link(self._compounds)
            if self._compound_was_inserted_by_user(compound) or (
                self._compound_accessible_by_reaction(compound) and self._filter(compound)
            ):
                compound.enable_exploration()

    def _disable_all_compounds(self):
        for compound in self._compounds.iterate_all_compounds():
            compound.link(self._compounds)
            compound.disable_exploration()

    def _compound_was_inserted_by_user(self, compound: db.Compound) -> bool:
        """
        Any structure of compound has a user label

        Parameters
        ----------
        compound : scine_database.Compound (Scine::Database::Compound)
            The compound that may have been inserted by user

        Returns
        -------
        bool
            compound was inserted
        """
        user_labels = [db.Label.USER_OPTIMIZED, db.Label.USER_GUESS]
        for s_id in compound.get_structures():
            structure = db.Structure(s_id)
            structure.link(self._structures)
            if structure.get_label() in user_labels:
                return True
        return False

    def _compound_accessible_by_reaction(self, compound: db.Compound) -> bool:
        """
        Compound is on RHS of a reaction that requires only explorable compounds and has not been deactivated.

        Parameters
        ----------
        compound : scine_database.Compound (Scine::Database::Compound)
            The compound to check to be accessible

        Returns
        -------
        bool
            compound is accessible
        """
        selection = {"$and": [{"rhs": {"$oid": compound.get_id().string()}}, {"exploration_disabled": {"$ne": True}}]}
        for hit in self._reactions.iterate_reactions(dumps(selection)):
            hit.link(self._reactions)
            accessible = True
            for reactant_id in hit.get_reactants(db.Side.LHS)[0]:
                reactant = db.Compound(reactant_id)
                reactant.link(self._compounds)
                if not reactant.explore():
                    accessible = False
            if accessible:
                return True
        return False

    def _filter(self, _: db.Compound) -> bool:
        return True


class BasicBarrierHeightKinetics(MinimalConnectivityKinetics):
    """
    This Gear enables the exploration of compounds if they were inserted by the user or created by a reaction
    that requires only explorable compounds and has a forward reaction barrier below a given threshold.

    Attributes
    ----------
    options :: BasicBarrierHeightKinetics.Options
        The options for the BasicBarrierHeightKinetics Gear.

    Notes
    -----
    Checks for all compounds that are accessed via a 'reaction'. Manually inserted Compounds
    are always activated by this gear.
    """

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self._elementary_steps = "required"
        self._structures = "required"
        self._properties = "required"
        self._reactions = "required"
        self._compounds = "required"

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

    def _filter(self, compound: db.Compound):
        selection = {"$and": [{"rhs": {"$oid": str(compound.id())}}, {"exploration_disabled": {"$ne": True}}]}
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
        barriers = {"gibbs_free_energy": [], "electronic_energy": []}
        for step_id in reaction.get_elementary_steps():
            step = db.ElementaryStep(step_id)
            step.link(self._elementary_steps)
            for energy_type, values in barriers.items():
                barrier = get_single_barrier_for_elementary_step_by_type(step, energy_type, self.options.model,
                                                                         self._structures, self._properties)
                if barrier is not None:
                    values.append(barrier)
        gibbs = None if not barriers["gibbs_free_energy"] else min(barriers["gibbs_free_energy"])
        electronic = None if not barriers["electronic_energy"] else min(barriers["electronic_energy"])
        return gibbs, electronic
