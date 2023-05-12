#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from abc import ABC, abstractmethod
from warnings import warn
import os
import datetime
from json import dumps
from typing import Optional, Dict, List, Tuple, Union

# Third party imports
import scine_database as db

# Local application imports
from . import Gear
from .pathfinder import Pathfinder as pf
from .elementary_steps.aggregate_filters import AggregateFilter
from .kinetic_modeling.concentration_query_functions import query_concentration_with_model_object,\
    query_concentration_with_object
from ..utilities.energy_query_functions import get_barriers_for_elementary_step_by_type
from ..utilities.queries import stop_on_timeout, model_query
from ..utilities.compound_and_flask_creation import get_compound_or_flask


class KineticsBase(Gear, ABC):
    """
    Base class for kinetics gears.

    Attributes
    ----------
    options :: KineticsBase.Options
        The options for the Kinetics Gear.
    aggregate_filter :: AggregateFilter
        An optional filter to limit the activated aggregates, by default none are filtered
    """

    class Options(Gear.Options):
        """
        The options for the KineticsBase Gear.
        """
        __slots__ = ("cycle_time", "restart")

        def __init__(self):
            super().__init__()
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

    options: Options

    def __init__(self):
        super().__init__()
        self.aggregate_filter = AggregateFilter()
        self._filtered_cache: Dict[str, bool] = {}
        self._required_collections = ["compounds", "elementary_steps", "flasks",
                                      "properties", "reactions", "structures"]

    def _propagate_db_manager(self, manager: db.Manager):
        self._sanity_check_configuration()
        self.aggregate_filter.initialize_collections(manager)

    def _sanity_check_configuration(self):
        if not isinstance(self.aggregate_filter, AggregateFilter):
            raise TypeError(f"Expected a AggregateFilter (or a class derived "
                            f"from it) in {self.name}.aggregate_filter.")

    def _loop_impl(self):
        if self.options.restart:
            self._disable_all_aggregates()
            self.options.restart = False
        # Loop over all deactivated aggregates
        self._aggregate_loop(self._compounds, db.CompoundOrFlask.COMPOUND)
        self._aggregate_loop(self._flasks, db.CompoundOrFlask.FLASK)

    def _aggregate_loop(self, collection: db.Collection, agg_type: db.CompoundOrFlask):
        selection = {"exploration_disabled": True}
        if agg_type == db.CompoundOrFlask.COMPOUND:
            iterator = collection.iterate_compounds(dumps(selection))
        elif agg_type == db.CompoundOrFlask.FLASK:
            iterator = collection.iterate_flasks(dumps(selection))
        else:
            raise RuntimeError(f"Unknown aggregate type {agg_type}")
        for aggregate in stop_on_timeout(iterator):
            aggregate.link(collection)
            if self.stop_at_next_break_point:
                return
            str_id = str(aggregate.id())
            try:
                # check if already filtered out once
                if not self._filtered_cache[str_id]:
                    continue
            except KeyError:
                # KeyError means filter has not been evaluated for this aggregate
                valid_aggregate = self.aggregate_filter.filter(aggregate)
                self._filtered_cache[str_id] = valid_aggregate
                if not valid_aggregate:
                    continue
            if self._aggregate_was_inserted_by_user(aggregate):
                aggregate.enable_exploration()
            elif self._filter(aggregate, self._aggregate_accessible_by_reaction(aggregate)):
                aggregate.enable_exploration()

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

    def _aggregate_accessible_by_reaction(self, aggregate: Union[db.Compound, db.Flask]) -> List[db.Reaction]:
        """
        Aggregate is on RHS of a reaction that requires only explorable
        aggregates and has not been deactivated.

        Parameters
        ----------
        aggregate : Union[scine_database.Compound, scine_database.Flask]
            The aggregate to check to be accessible

        Returns
        -------
        List[db.Reaction]
            List of all viable reactions giving access
        """
        selection = {"$and": [
            {"rhs": {"$elemMatch": {"id": {"$oid": aggregate.get_id().string()}}}},
            {"exploration_disabled": {"$ne": True}}
        ]}
        hits = []
        for hit in self._reactions.iterate_reactions(dumps(selection)):
            hit.link(self._reactions)
            accessible = True
            for reactant_id, reactant_type in zip(hit.get_reactants(db.Side.LHS)[0],
                                                  hit.get_reactant_types(db.Side.LHS)[0]):
                reactant = get_compound_or_flask(reactant_id, reactant_type, self._compounds, self._flasks)
                if not reactant.explore():
                    accessible = False
                    break
            if accessible:
                hits.append(hit)
        return hits

    @abstractmethod
    def _filter(self, _: Union[db.Compound, db.Flask], __: List[db.Reaction]) -> bool:
        raise NotImplementedError


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
    aggregate_filter :: AggregateFilter
        An optional filter to limit the activated aggregates, by default none are filtered
    """

    class Options(KineticsBase.Options):
        """
        The options for the MinimalConnectivityKinetics Gear.
        """
        __slots__ = "user_input_only"

        def __init__(self):
            super().__init__()
            self.user_input_only = False
            """
            bool
                Option to only ever allow compounds that contain structures
                added to the database by a user. If enabled, no other compounds
                will ever be enabled irrespective of any connectivity.
            """

    options: Options

    def _filter(self, _: Union[db.Compound, db.Flask], access_reactions: List[db.Reaction]):
        if self.options.user_input_only:
            # relies on the fact that user inputs are activated in loop regardless
            return False
        return len(access_reactions) > 0


class BasicBarrierHeightKinetics(KineticsBase):
    """
    This Gear enables the exploration of aggregates if they were inserted by
    the user or created by a reaction that requires only explorable aggregates
    and has a forward reaction barrier below a given threshold.

    Attributes
    ----------
    options :: BasicBarrierHeightKinetics.Options
        The options for the BasicBarrierHeightKinetics Gear.
    aggregate_filter :: AggregateFilter
        An optional filter to limit the activated aggregates, by default none are filtered

    Notes
    -----
    Checks for all aggregates that are accessed via a 'reaction'. Manually
    inserted Compounds/Flasks are always activated by this gear.
    """

    class Options(KineticsBase.Options):
        """
        The options for the BasicBarrierHeightKinetics Gear.
        """

        __slots__ = ("max_allowed_barrier", "max_allowed_energy", "enforce_free_energies")

        def __init__(self):
            super().__init__()
            self.max_allowed_barrier = 1000.0  # kJ/mol
            """
                float
                The maximum barrier height of the reaction resulting in the
                aggregate in kJ/mol to allow the aggregate to be further explored.
            """
            self.max_allowed_energy = None
            """
                float
                The maximum energy threshold for the reaction energies allowed
                in the aggregate in kJ/mol to allow the aggregate to be further
                explored.
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

    options: Options

    def __init__(self):
        super().__init__()
        self.__cache: Dict[str, dict] = {}

    def clear_cache(self) -> None:
        self.__cache = {}

    def _filter(self, aggregate: Union[db.Compound, db.Flask], access_reactions: List[db.Reaction]):
        for reaction in access_reactions:
            if aggregate.get_id() not in reaction.get_reactants(db.Side.RHS)[1]:
                continue
            if not self._reaction_barrier_too_high(reaction):
                if self.options.max_allowed_energy is not None:
                    if not self._reaction_energy_too_high(reaction):
                        return True
                else:
                    return True
        return False

    def _reaction_energy_too_high(self, reaction: db.Reaction) -> bool:
        """
        Whether the reaction barrier of the reaction is too high

        Parameters
        ----------
        reaction : scine_database.Reaction (Scine::Database::Reaction)
            The reaction
        """
        reaction_id = reaction.get_id().string()
        es_list = [i.string() for i in reaction.get_elementary_steps()].sort()
        if reaction_id in self.__cache:
            if self.__cache[reaction_id]['es_ids'] == es_list:
                if self.options.enforce_free_energies and self.__cache[reaction_id]['dG'] is not None:
                    return self.__cache[reaction_id]['dG'] > self.options.max_allowed_energy
                if not self.options.enforce_free_energies and self.__cache[reaction_id]['dE'] is not None:
                    return self.__cache[reaction_id]['dE'] > self.options.max_allowed_energy
            return True
        else:
            # expect data in cache,
            # expect call of _reaction_barrier_too_high before _reaction_energy_too_high
            raise RuntimeError("Internal error in BasicBarrierHeightKinetics Gear.")

    def _reaction_barrier_too_high(self, reaction: db.Reaction) -> bool:
        """
        Whether the reaction barrier of the reaction is too high

        Parameters
        ----------
        reaction : scine_database.Reaction (Scine::Database::Reaction)
            The reaction
        """
        reaction_id = reaction.get_id().string()
        es_list = [i.string() for i in reaction.get_elementary_steps()].sort()
        if reaction_id in self.__cache:
            if self.__cache[reaction_id]['es_ids'] == es_list:
                if self.options.enforce_free_energies and self.__cache[reaction_id]['ddG'] is not None:
                    return self.__cache[reaction_id]['ddG'] > self.options.max_allowed_barrier
                if not self.options.enforce_free_energies and self.__cache[reaction_id]['ddE'] is not None:
                    return self.__cache[reaction_id]['ddE'] > self.options.max_allowed_barrier
        barrier_heights = self._barrier_height(reaction)
        self.__cache[reaction_id] = {
            'es_ids': es_list,
            'ddE': barrier_heights[1][0],
            'ddG': barrier_heights[0][0],
            'dE': barrier_heights[1][1],
            'dG': barrier_heights[0][1],
        }
        if barrier_heights[0][0] is None:
            if self.options.enforce_free_energies or barrier_heights[1][0] is None:
                # skip if free energy barrier not available yet and only free energies allowed or electronic energy also
                # not available (e.g., because of model refinement)
                return True
            return barrier_heights[1][0] > self.options.max_allowed_barrier
        return barrier_heights[0][0] > self.options.max_allowed_barrier

    def _barrier_height(self, reaction: db.Reaction) -> Tuple[
        Tuple[Union[float, None], Union[float, None]],
        Tuple[Union[float, None], Union[float, None]],
    ]:
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
        energies: Dict[str, List[float]] = {"gibbs_free_energy": [], "electronic_energy": []}
        for step_id in reaction.get_elementary_steps():
            step = db.ElementaryStep(step_id)
            step.link(self._elementary_steps)
            for energy_type, values in barriers.items():
                lhs, rhs = get_barriers_for_elementary_step_by_type(step, energy_type, self.options.model,
                                                                    self._structures, self._properties)
                if lhs is not None:
                    values.append(lhs)
                if lhs is not None and rhs is not None:
                    energies[energy_type].append(lhs-rhs)
        barrier_gibbs = None if not barriers["gibbs_free_energy"] else min(barriers["gibbs_free_energy"])
        barrier_electronic = None if not barriers["electronic_energy"] else min(barriers["electronic_energy"])
        energy_gibbs = None if not barriers["gibbs_free_energy"] else min(energies["gibbs_free_energy"])
        energy_electronic = None if not barriers["electronic_energy"] else min(energies["electronic_energy"])
        return (barrier_gibbs, energy_gibbs), (barrier_electronic, energy_electronic)


class MaximumFluxKinetics(KineticsBase):
    """
    This Gear enables the exploration of compounds if they were inserted by the user, created by a reaction
    that requires only compounds with a concentration flux larger than a given threshold, has a forward reaction barrier
    below a given maximum, and has reached a minimum concentration larger than a given threshold during kinetic
    modeling.

    Attributes
    ----------
    options :: MaximumFluxKinetics.Options
        The options for the MaximumFluxKinetics Gear.
    aggregate_filter :: AggregateFilter
        An optional filter to limit the activated aggregates, by default none are filtered

    Notes
    -----
    Checks for all compounds that are accessed via a 'reaction'. Manually inserted Compounds
    are always activated by this gear.
    """

    class Options(KineticsBase.Options):
        """
        The options for the MaximumPopulationKinetics Gear.
        """

        __slots__ = ("min_allowed_concentration", "property_label",
                     "min_concentration_flux", "flux_property_label")

        def __init__(self):
            super().__init__()
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

    options: Options

    def _filter(self, aggregate: Union[db.Compound, db.Flask], __: List[db.Reaction]):
        start_concentration = query_concentration_with_object("start_concentration",
                                                              aggregate,
                                                              self._properties,
                                                              self._structures)

        max_concentration = query_concentration_with_model_object(self.options.property_label,
                                                                  aggregate,
                                                                  self._properties,
                                                                  self._structures,
                                                                  self.options.model)
        if self.options.min_allowed_concentration > max_concentration and start_concentration < 1e-10:
            return False
        return True

    def _aggregate_accessible_by_reaction(self, _: Union[db.Compound, db.Flask]) -> List[db.Reaction]:
        """
        Assume every compound to be accessible. Compound elimination happens via the concentration.
        This function returns an empty list to accelerate the gear.

        Parameters
        ----------
        _ : Union[db.Compound, db.Flask]
            The compound to check to be accessible

        Returns
        -------
        Empty list.
        """
        return []


class PathfinderKinetics(KineticsBase):
    """
    This Gear enables the exploration of compounds if they were inserted by the user
    or have a compound cost lower than a given compound cost threshold.
    The compound costs are determined by running Pathfinder with the set options and the given starting conditions.
    The graph is built and the compound costs determined if a the ratio of structures over compounds considered for
    building the last graph have changed by more than a given excess ratio, by default 10%.

    Attributes
    ----------
    options :: PathfinderKinetics.Options
        The options for the PathfinderKinetics Gear.
    """

    class Options(KineticsBase.Options):
        """
        The options for the PathfinderKinetics Gear.
        """

        __slots__ = ("max_compound_cost", "restart_excess_ratio",
                     "structure_model", "start_conditions", "barrierless_weight", "filter_negative_barriers",
                     "allow_unsolved_compound_costs", "temperature", "store_pathfinder_output")

        def __init__(self):
            super().__init__()
            self.max_compound_cost = 10.0
            """
            float
                The maximal compound cost allowed for a compound to be enabled for exploration.
            """
            self.restart_excess_ratio = 0.1
            """
            float
                Percentage of ratio (structure / compounds in graph) change to trigger pathfinder analysis.
            """
            self.store_pathfinder_output = False
            """
            bool
                Bool to indicate if graph and compound costs of run should be exported to files.
            """
            self.structure_model = None
            """
            Optional[db.Model]
                Structure model for the structures to be considered.
            """
            self.start_conditions: Dict[str, float] = {}
            """
            Dict[str, float]
                Starting conditions for pathfinder with the compound IDs as string and the cost as float.
            """
            self.barrierless_weight = 1e12
            """
            float
                Weight for barrierless reactions, by default 1e12.
            """
            self.filter_negative_barriers = False
            """
            bool
                Bool to indicate if negative barriers should be considered.
            """
            self.allow_unsolved_compound_costs = False
            """
            bool
                Bool to indicate if unsolved compound costs (compounds still having a +inf cost) are allowed.
            """
            self.temperature = 298.15
            """
            float
                Temperature to calculate the rate constants from the barriers.
            """

    options: Options

    def __init__(self):
        super().__init__()
        self.finder = None
        self._old_ratio = None
        self._trigger = 0.0

    def _loop_impl(self):
        # # # Restart the gear
        if self.options.restart:
            self._disable_all_aggregates()
            self.options.restart = False
        # # # Check if finder is setup
        if self.finder is None:
            self._setup_finder()
        # # # Get current count of structures with model
        if self.options.structure_model is None:
            structure_count = self._count_structures(self.options.model)
        else:
            structure_count = self._count_structures(self.options.structure_model)

        # # # If a graph is there and compound costs set up, determine ratio to set trigger
        if self.finder.graph_handler is not None and len(self.finder.compound_costs) > 0:
            current_ratio = structure_count / len(self.finder.compound_costs)
            self._trigger = current_ratio / self._old_ratio - 1.0  # better to look at change of structures with model

        # # # Build the graph if not build yet or the current aggregate count exceeds the tolerated percentage
        if self.finder.graph_handler is None or \
           self._trigger > self.options.restart_excess_ratio:
            self.finder.build_graph()
            print("Nodes:", len(self.finder.graph_handler.graph.nodes))
            self.finder.set_start_conditions(self.options.start_conditions)
            print(self.finder.start_compounds)
            # # # Check if start compounds have compound cost in database
            if not self._start_compounds_have_compound_costs():
                print("Start Compounds have no cost in DB")
                for key in self.finder.start_compounds:
                    centroid = self._get_centroid_from_aggregate_id_string(key, db.CompoundOrFlask.COMPOUND)
                    self._write_compound_cost(centroid, self.finder.compound_costs[key])
            # # # Only calculate compound costs, if there are any nodes in the graph
            if len(self.finder.graph_handler.graph.nodes) > 0:
                # # # Safety check if start compounds are in graph
                if not set(self.finder.start_compounds).issubset(self.finder.graph_handler.graph.nodes):
                    warn("Warning: Not all start compounds in graph.")
                    # Reset graph handler
                    self.finder.graph_handler = None
                    return
                # # # Calculate the compound costs
                self.finder.calculate_compound_costs()
                # # # Write new ratio as old ratio
                self._old_ratio = structure_count / len(self.finder.compound_costs)
                # # # Write compound cost of compound to corresponding centroid
                for key in self.finder.compound_costs.keys():
                    centroid = self._get_centroid_from_aggregate_id_string(key)
                    self._write_compound_cost(centroid, self.finder.compound_costs[key])
                # # # Export graph and compound costs if requested
                if self.options.store_pathfinder_output:
                    directory_name = "./pathfinder_outputs/"
                    if not os.path.exists(directory_name):
                        os.makedirs(directory_name)
                    timestamp = datetime.datetime.now().strftime("%m%d-%H%M%S")
                    self.finder.export_graph(directory_name + timestamp + ".kinetic.graph.json")
                    self.finder.export_compound_costs(directory_name + timestamp + ".compound_costs.json")
            else:
                self.finder.graph_handler = None
        # # # If unsolved compound costs persist and they are not allowed, don't try to activate stuff
        if not self.finder.compound_costs_solved and not self.options.allow_unsolved_compound_costs:
            return

        # Loop over all deactivated aggregates
        selection = {"exploration_disabled": True}
        # # # Loop over deactivated compounds
        for compound in stop_on_timeout(self._compounds.iterate_compounds(dumps(selection))):
            compound.link(self._compounds)
            if self._aggregate_was_inserted_by_user(compound):
                compound.enable_exploration()
            elif self._filter(compound, self._aggregate_accessible_by_reaction(compound)):
                compound.enable_exploration()
        # # # Loop over deactivated flasks
        for flask in stop_on_timeout(self._flasks.iterate_flasks(dumps(selection))):
            flask.link(self._flasks)
            if self._aggregate_was_inserted_by_user(flask):
                flask.enable_exploration()
            elif self._filter(flask, self._aggregate_accessible_by_reaction(flask)):
                flask.enable_exploration()

    def _filter(self, aggregate: Union[db.Compound, db.Flask], __: List[db.Reaction]) -> bool:
        aggregate_id = aggregate.id().string()
        if self.finder.compound_costs_solved or self.options.allow_unsolved_compound_costs:
            if aggregate_id in self.finder.compound_costs.keys() and \
                    self.finder.compound_costs[aggregate_id] < self.options.max_compound_cost:
                return True
            # # #
            else:
                return False
        # # # Finder complete but compound cost not solved, raise error
        else:
            raise RuntimeError("Pathfinder can not find a solution under given starting conditions.")

    def _setup_finder(self):
        """
        Sets up the pathfinder with the barrier handler and
        sets the options to the pathfinder.options
        """
        self.finder = pf(self._manager)
        # # # Set correction options for BarrierBased Pathfinder
        self.finder.options.graph_handler = "barrier"
        self.finder.options.model = self.options.model
        self.finder.options.structure_model = self.options.structure_model
        self.finder.options.barrierless_weight = self.options.barrierless_weight
        self.finder.options.filter_negative_barriers = self.options.filter_negative_barriers

    def _count_structures(self, model: db.Model) -> int:
        """
        Counts the optimized structures with the given model.

        Parameters
        ----------
        model : db.Model
            Model for the structure query
        Returns
        -------
        int
            Number of optimized structures with the given model.
        """
        selection = {
            "$and": [
                {"label": {"$in": ["user_optimized", "minimum_optimized", "ts_optimized", "complex_optimized"]}}
            ]
            + model_query(model)
        }

        return self._structures.count(dumps(selection))

    def _start_compounds_have_compound_costs(self):
        """
        Checks if all start compounds have the property compound cost in their centroid.

        Returns
        -------
        bool
            Bool indicating if all start compounds have the property compound costs in their centroid.
        """
        centroid_list = [self._get_centroid_from_aggregate_id_string(
            key, db.CompoundOrFlask.COMPOUND) for key in self.finder.start_compounds]
        return all(centroid.has_property("compound_cost") for centroid in centroid_list)

    def _get_centroid_from_aggregate_id_string(
            self, id: str, aggregate_type: Optional[db.CompoundOrFlask] = None) -> db.Structure:
        """
        Get the centroid structure of an aggregate.

        Parameters
        ----------
        id : str
            The ID of the aggregate as string
        aggregate_type : Optional[db.CompoundOrFlask], optional
            The aggregate type (COMPOUND or FLASK) if known, by default None

        Returns
        -------
        centroid : db.Structure
            The centroid of the the given aggregate.
        """
        if aggregate_type is None:
            aggregate_type_determined = self._get_type(id)
        else:
            aggregate_type_determined = aggregate_type
        aggregate = get_compound_or_flask(db.ID(id), aggregate_type_determined, self._compounds, self._flasks)
        centroid = aggregate.get_centroid(self._manager)
        return centroid

    def _get_type(self, key: str) -> db.CompoundOrFlask:
        """
        Determine the aggregate type based on the node information stored in the graph.
        This information is encoded in each node when constructing the graph, therefore it can always be retrieved.

        Parameters
        ----------
        key : str
            The ID of the node, either a compound or flask ID as string.

        Returns
        -------
        db.CompoundOrFlask
            The aggregate type of the queried node.
        """
        type_string = self.finder.graph_handler.graph.nodes[key]["type"]
        if type_string == db.CompoundOrFlask.COMPOUND.name:
            return db.CompoundOrFlask.COMPOUND
        else:
            return db.CompoundOrFlask.FLASK

    def _write_compound_cost(self, centroid: db.Structure, compound_cost: float):
        """
        Write a compound cost as property to the database and add this property to the centroid structure.

        Parameters
        ----------
        centroid : db.Structure
            The centroid structure of a compound or flask to which the compound cost property should be appended.
        compound_cost : float
            The determined compound cost of the centroid's compound or flask.
        """
        label = "compound_cost"
        prop = db.NumberProperty.make(label, self.options.model, compound_cost, self._properties)
        centroid.add_property(label, prop.id())
        prop.set_structure(centroid.id())
