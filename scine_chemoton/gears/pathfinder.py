#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from itertools import islice
from typing import Any, Optional, Iterator, List, Tuple, Union, Dict, Callable
import copy
import json
import math
import warnings
import yaml

# Third party imports
import networkx as nx
import numpy as np
import scine_database as db
import scine_utilities as utils
from scine_database.compound_and_flask_creation import get_compound_or_flask
from scine_database.energy_query_functions import (
    get_energy_for_structure,
    get_barriers_for_elementary_step_by_type,
    get_elementary_step_with_min_ts_energy,
    rate_constant_from_barrier,
)

# Local application imports
from . import HoldsCollections
from ..utilities.get_molecular_formula import get_molecular_formula_of_aggregate
from ..utilities.options import BaseOptions
from scine_chemoton.utilities.db_object_wrappers.thermodynamic_properties import ReferenceState
from scine_chemoton.utilities.db_object_wrappers.reaction_wrapper import Reaction
from scine_chemoton.utilities.db_object_wrappers.wrapper_caches import MultiModelCacheFactory
from scine_chemoton.utilities.db_object_wrappers.reaction_cache import ReactionCache
from scine_chemoton.utilities.model_combinations import ModelCombination
from scine_chemoton.utilities.place_holder_model import construct_place_holder_model


class DifferentSubgraphsError(Exception):
    pass


class Pathfinder(HoldsCollections):
    """
    A class to represent a list of reactions as a graph and query this graph for simple paths between two nodes.
    In a simple path, every node part of the path is visited only once.

    Attributes
    ----------
    _calculations : db.Collection
        Collection of the calculations of the connected database.
    _compounds : db.Collection
        Collection of the compounds of the connected database.
    _flasks : db.Collection
        Collection of the flasks of the connected database.
    _reactions : db.Collection
        Collection of the reactions of the connected database.
    _elementary_steps : db.Collection
        Collection of the elementary steps of the connected database.
    _structures : db.Collection
        Collection of the structures of the connected database.
    _properties : db.Collection
        Collection of the properties of the connected database.
    graph_handler
        A class handling the construction of the graph. Can be adapted to one's needs.
    _use_old_iterator : bool
        Boolean to indicate if the old iterator shall be used querying for paths between a source - target pair.
    _unique_iterator_memory : Tuple[Tuple[List[str], float], Iterator]
        Memory of iterator with the corresponding path and its length as well as the iterator.
    start_compounds : List[str]
        A list containing the compounds which are present at the start.
    start_compounds_set : bool
        Boolean to indicate if start_compounds are set.
    _pseudo_inf : float
        Float for edges with infinite weight.
    compound_costs : Dict[str, float]
        A dictionary containing the cost of the compounds with the compounds as keys.
    compound_costs_solved : bool
        Boolean to indicate if all compounds have a compound cost.
    """

    def __init__(self, db_manager: db.Manager) -> None:
        super().__init__()
        self.options = self.Options()
        self._required_collections = ["manager", "calculations", "compounds", "flasks", "reactions",
                                      "elementary_steps", "structures", "properties"]
        self.initialize_collections(db_manager)

        self.graph_handler: Union[Pathfinder.BasicHandler, Pathfinder.BarrierBasedHandler,
                                  Pathfinder.MinimumReactionBarrierBasedHandler, Pathfinder.FromRMSFileBasedHandler,
                                  None] = None
        # attribute to store iterator employed in find_unique_paths; path_object, iterator
        self._use_old_iterator = False
        self._unique_iterator_memory: Union[Tuple[Tuple[List[str], float],
                                                  Iterator], None] = None

        # Compound costs
        self.start_compounds: List[str] = []
        self.start_compounds_set = False
        self._pseudo_inf = 1e12
        self.compound_costs: Dict[str, float] = {}
        self.compound_costs_solved = False
        self.graph_updated_with_compound_costs = False

    class Options(BaseOptions):
        """
        A class to vary the setup of Pathfinder.
        """
        __slots__ = {"graph_handler", "barrierless_weight", "model", "filter_negative_barriers", "use_structure_model",
                     "structure_model", "use_only_enabled_aggregates", "temperature", "barrier_limit",
                     "only_electronic_energies", "rms_file_name", "thermodynamics_model", "use_thermodynamics_model"
                     }

        def __init__(self) -> None:
            self.graph_handler: str = "basic"  # pylint: disable=no-member
            """
            str
                A string indicating which graph handler shall be used (available are : 'basic' and 'barrier').
            """
            self.barrierless_weight: float = 1e0  # 0.01
            """
            float
                The weight for barrierless reactions (basic) and rate constant (barrier), respectively.
            """
            self.model: db.Model = db.Model("any", "any", "any")
            """
            db.Model
                The model for the energies of compounds to be included in the graph.
            """
            # in kJ / mol
            self.filter_negative_barriers: bool = False
            """
            bool
                Forbid elementary steps with negative barriers or not.
            """
            self.use_structure_model: bool = False
            """
            bool
                Allow only elementary steps with a given model.
            """
            self.use_thermodynamics_model: bool = False
            """
            bool
                Composite Gibbs energies. Should only be used with structure_model and electronic energy activated.
            """
            self.use_only_enabled_aggregates: bool = False
            """
            bool
                Allow only elementary steps with a given model.
            """
            self.structure_model: db.Model = db.Model("any", "any", "any")
            """
            db.Model
                The model for the structures of compounds to be included in the graph.
            """
            self.thermodynamics_model: db.Model = db.Model("any", "any", "any")
            """
            db.Model
                The model that is used to extract the thermodynamics corrections and add them to self.model.
            """
            self.temperature: float = 298.15
            """
            float
                The temperature in Kelvin for the rate constant calculation.
            """
            self.barrier_limit: float = math.inf
            """
            float
                The maximum barrier for elementary steps to be included in the graph.
                Only valid with 'barrier' graph handler
            """
            self.only_electronic_energies: bool = False
            """
            bool
                If True, only electronic energies will be considered for the weight calculation with the
                MinimumReactionBarrierBasedHandler graph handler.
            """
            self.rms_file_name: str = ""
            """
            str
                The name of the RMS input file for the FromRMSFileBasedHandler.
            """

    @staticmethod
    def get_valid_graph_handler_options() -> List[str]:
        return ["basic", "barrier", "reaction-barrier", "from-rms-input"]

    def _construct_graph_handler(self):
        """
        Constructor for the graph handler.
        Transfers pathfinder.options to graph handler.

        Raises
        ------
        RuntimeError
            Invalid options for graph handler.
        """
        if self.options.graph_handler not in self.get_valid_graph_handler_options():
            raise RuntimeError("Invalid graph handler option.")
        if self.options.graph_handler == "basic":
            self.graph_handler = self.BasicHandler(self._manager, self.options.model, self.options.structure_model)
            self.graph_handler.barrierless_weight = self.options.barrierless_weight
            self.graph_handler.use_structure_model = self.options.use_structure_model
            self.graph_handler.use_only_enabled_aggregates = self.options.use_only_enabled_aggregates
            self.graph_handler.use_thermodynamics_model = self.options.use_thermodynamics_model
            self.graph_handler.thermodynamics_model = self.options.thermodynamics_model
        elif self.options.graph_handler == "barrier":
            self.graph_handler = self.BarrierBasedHandler(
                self._manager, self.options.model, self.options.structure_model)
            # Option copying
            self.graph_handler.barrierless_weight = self.options.barrierless_weight
            self.graph_handler.filter_negative_barriers = self.options.filter_negative_barriers
            self.graph_handler.set_temperature(self.options.temperature)
            self.graph_handler.use_structure_model = self.options.use_structure_model
            self.graph_handler.use_thermodynamics_model = self.options.use_thermodynamics_model
            self.graph_handler.thermodynamics_model = self.options.thermodynamics_model
            self.graph_handler.use_only_enabled_aggregates = self.options.use_only_enabled_aggregates
            self.graph_handler.set_barrier_limit(self.options.barrier_limit)
            # Mapping of ESs and calculating normalization
            self.graph_handler._map_elementary_steps_to_reactions()
            self.graph_handler._calculate_rate_constant_normalization()
        elif self.options.graph_handler == "reaction-barrier":
            graph_handler = self.MinimumReactionBarrierBasedHandler(
                self._manager, self.options.model, self.options.structure_model)
            graph_handler.only_electronic = self.options.only_electronic_energies
            graph_handler.use_only_enabled_aggregates = self.options.use_only_enabled_aggregates
            self.graph_handler = graph_handler
            self.graph_handler.barrierless_weight = self.options.barrierless_weight
            self.graph_handler.initialize()
        elif self.options.graph_handler == "from-rms-input":
            self.graph_handler = self.FromRMSFileBasedHandler(self._manager, self.options.rms_file_name,
                                                              self.options.temperature)
            self.graph_handler.initialize()

    def _reset_iterator_memory(self):
        """
        Reset memory for unique memory.
        """
        self._unique_iterator_memory = None

    # Build graph function from reaction list
    def build_graph(self):
        """
        Build the nx.DiGraph() from a list of filtered reactions.
        """
        self._reset_iterator_memory()
        # # # Reset booleans for compound costs
        self.compound_costs_solved = False
        self.graph_updated_with_compound_costs = False
        # Build graph
        self._construct_graph_handler()
        assert self.graph_handler
        # Fallback strategy, if there are no reactions in the database yet
        if self._reactions.count(json.dumps({})) == 0:
            warnings.warn("No reactions in database. Building graph with aggregates only.")
            for aggregate_id, aggregate_type in self.graph_handler.get_valid_aggregate_ids():
                aggregate = get_compound_or_flask(aggregate_id, aggregate_type, self._compounds, self._flasks)
                self.graph_handler.add_aggregate(aggregate, aggregate_type)
        else:
            for rxn_id in self.graph_handler.get_valid_reaction_ids():
                rxn = db.Reaction(rxn_id, self._reactions)
                self.graph_handler.add_reaction(rxn)

    def extract_connected_graph(self, included_nodes: List[str]) -> nx.DiGraph:
        """
        Extract a connected subgraph from a given graph and a given list of nodes.

        Parameters
        ----------
        included_nodes : List[str]
            A list of nodes which should be included in the graph.

        Returns
        -------
        selected_subgraph : nx.DiGraph
            The connected subgraph including the requested nodes.
        """
        assert self.graph_handler
        included_nodes_set = set(included_nodes)
        input_graph = copy.deepcopy(self.graph_handler.graph)
        undirected_graph = input_graph.to_undirected()
        for subgraph_set in nx.connected_components(undirected_graph):
            # Check starter set has to be subset
            if included_nodes_set.issubset(subgraph_set):
                # Build subgraph
                selected_subgraph = input_graph.subgraph(subgraph_set).copy()
                break
        else:
            raise DifferentSubgraphsError(f"The given nodes '{included_nodes_set}' are not connected.")
        return selected_subgraph

    def load_graph(self, graph_filename: str, compound_cost_filename: str = ""):
        """
        Initialize a basic graph handler with default settings.
        The graph is imported from the given file and set as the graph of the graph handler.
        Optionally, the compound costs are imported and set as compound_cost.
        The compound costs are considered to be solved.
        The graph is automatically updated with the compound costs.

        Parameters
        ----------
        graph_filename : str
            Name of the .json file containing the graph.
        compound_cost_filename : str
            Name of the .json file containing the compound costs.
        """
        # only for basic graph handler, we don't want to calculate any kinetic
        if self.options.graph_handler != "basic":
            print("Can only load into default basic graph handler.")
        graph = self._import_graph(graph_filename)
        # Initialize basic graph handler
        self._construct_graph_handler()
        assert self.graph_handler is not None
        self.graph_handler.graph = copy.deepcopy(graph)

        if compound_cost_filename != "":
            self.compound_costs = self._import_dictionary(compound_cost_filename)
            self.compound_costs_solved = True
            self.update_graph_compound_costs()

    def find_paths(self, source: str, target: str, n_requested_paths: int = 3,
                   n_skipped_paths: int = 0) -> List[Tuple[List[str], float]]:
        """
        Query the build graph for simple paths between a source and target node.

        Notes
        -----
        Requires a built graph

        Parameters
        ----------
        source : str
            The ID of the starting compound as string.
        target : str
            The ID of the targeted compound as string.
        n_requested_paths : int
            Number of requested paths, by default 3
        n_skipped_paths : int
            Number of skipped paths from, by default 0. For example, when four paths are found (``n_requested_paths=4``)
            and ``n_skipped_paths=2``, the third, fourth, fifth and sixth path are returned. Therefore, this allows
            setting the starting point of the query.

        Returns
        -------
        found_paths : List[Tuple[List[str] float]]
            List of paths where each item (path) consists of the list of nodes of the path and its length.
        """
        assert self.graph_handler
        found_paths = []
        for path in self._k_shortest_paths(self.graph_handler.graph, source, target, n_requested_paths,
                                           weight="weight", path_start=n_skipped_paths):
            path_length = nx.path_weight(self.graph_handler.graph, path, "weight")
            found_paths.append((path, path_length))

        return found_paths

    def find_unique_paths(self, source: str, target: str, number: int = 3,
                          custom_weight: Union[str, Callable] = "weight") -> List[Tuple[List[str], float]]:
        """
        Find a unique number of paths from a given source node to a given target node.
        Paths can have the same total length (in terms of sum over edge weights),
        but if one is solely interested in one path of paths with identical length,
        the shortest (in terms of length) longest (in terms of number of nodes) path is returned.
        This is called the unique path (shortest longest path).

        Notes
        -----
        | - Checks if a stored iterator for the given source-target pair should be used.
        | - Maximal ten paths with identical length are compared.

        Parameters
        ----------
        source : str
            The ID of the starting compound as string.
        target : str
            The ID of the targeted compound as string.
        number : int
            The number of unique paths to be returned. Per default, 3 paths are returned.

        Returns
        -------
        path_tuple_list : List[Tuple[List[str], float]]
            List of paths where each item (path) consists the list of nodes of the path and its length.
        """
        assert self.graph_handler
        counter = 0
        path_tuple_list = list()
        # # # Initialise iterator over shortest simple paths if it is either not set or source/target do not match
        if not self._use_old_iterator or \
           self._unique_iterator_memory is None or \
           self._unique_iterator_memory[0][0][0] != source or \
           self._unique_iterator_memory[0][0][-1] != target:
            path_iterator = iter(nx.shortest_simple_paths(self.graph_handler.graph, source, target,
                                                          weight=custom_weight))
            # # # Find first path and its cost
            old_path = next(path_iterator)
            old_path_cost = nx.path_weight(self.graph_handler.graph, old_path, weight="weight")
        # # # Load old iterator
        else:
            path_iterator = self._unique_iterator_memory[1]
            old_path = self._unique_iterator_memory[0][0]
            old_path_cost = self._unique_iterator_memory[0][1]

        while counter < number:
            same_cost = True
            tmp_path_list: List[List[str]] = list()
            # # # Collect all paths with same cost
            n_max_collected_paths = 10
            while same_cost and len(tmp_path_list) < n_max_collected_paths:
                # # # Append old path to tmp_path list
                tmp_path_list.append(old_path)
                # # # Get next path and its cost
                new_path = next(path_iterator, None)
                # # # Break loop if no path is returned
                if new_path is None:
                    break
                new_path_cost = nx.path_weight(self.graph_handler.graph, new_path, weight="weight")
                # # # Check if new cost different to old cost
                if abs(old_path_cost - new_path_cost) > 1e-12:
                    same_cost = False
                # # # Overwrite old path with new path
                old_path = new_path

            # # # Append path with most nodes to tuple list and its cost
            path_tuple_list.append((max(tmp_path_list, key=lambda x: len(x)),  # pylint: disable=unnecessary-lambda
                                    old_path_cost))
            # # # Break counter loop if no more paths to target are found
            if new_path is None:
                break
            old_path_cost = new_path_cost
            counter += 1
        # # # Store iterator and path info (list of nodes and length)
        if new_path is not None:
            self._unique_iterator_memory = ((new_path, new_path_cost), path_iterator)
        return path_tuple_list

    def get_elementary_step_sequence(self, path: List[str]) -> str:
        """
        Prints the sequence of elementary steps of a path with the compounds written as molecular formulas
        with multiplicity and charge as well as the final cost of the path.
        Reactant node is returned in red, product node in blue to enhance readability.

        Parameters
        ----------
        path : Tuple[List[str] float]
            Path containing a list of the traversed nodes and the cost of this path.

        Returns
        -------
        str
            A string of the elementary step sequence of a given path.
        """
        sequence_string = ""
        assert self.graph_handler
        # # # Loop over elementary steps by dissecting path
        for k in np.arange(0, len(path) - 2, 2):
            step = path[k:k + 3]
            # # # Count Reactants
            reactants = [step[0]]
            reactants += self.graph_handler.graph.edges[step[0], step[1]]['required_compounds']
            # # # Count Products
            products = [step[2]]
            products += self.graph_handler.graph.edges[step[1], step[2]]['required_compounds']

            rxn_eq = ""
            for i, side in enumerate([reactants, products]):
                for j, aggregate_id in enumerate(side):
                    # # # Identify Compound or Flask
                    aggregate_type = None
                    if self.graph_handler.graph.nodes[aggregate_id]['type'] == db.CompoundOrFlask.COMPOUND.name:
                        aggregate_type = db.CompoundOrFlask.COMPOUND
                    elif self.graph_handler.graph.nodes[aggregate_id]['type'] == db.CompoundOrFlask.FLASK.name:
                        aggregate_type = db.CompoundOrFlask.FLASK
                    else:
                        raise RuntimeError('Invalid aggregate type encountered in your graph nodes. '
                                           'Please check the graph nodes.')

                    aggregate_str = get_molecular_formula_of_aggregate(
                        db.ID(aggregate_id), aggregate_type, self._compounds, self._flasks, self._structures)
                    # # # Color reactant node
                    if j == 0 and i == 0:
                        aggregate_str = '\033[31m' + aggregate_str + '\033[0m'
                    # # # Color product node
                    elif j == 0 and i == 1:
                        aggregate_str = '\033[36m' + aggregate_str + '\033[0m'

                    rxn_eq += aggregate_str

                    if j < len(side) - 1:
                        rxn_eq += " + "
                if i == 0:
                    rxn_eq += " -> "
            sequence_string += rxn_eq + "\n"

        return sequence_string

    def get_elementary_step_sequence_as_list(self, path: List[str]) -> List[str]:
        """
        Returns the elementary step sequence as a list of strings.

        Parameters
        ----------
        path : List[str]
            The path representing the elementary step sequence.

        Returns
        -------
        List[str]
            The elementary step sequence as a list of reaction equations.

        Raises
        ------
        RuntimeError
            If an invalid aggregate type is encountered in the graph nodes.
        """
        sequence_list: List[str] = []
        assert self.graph_handler
        # # # Loop over elementary steps by dissecting path
        for k in np.arange(0, len(path) - 2, 2):
            step = path[k:k + 3]
            # # # Count Reactants
            reactants = [step[0]]
            reactants += self.graph_handler.graph.edges[step[0], step[1]]['required_compounds']
            # # # Count Products
            products = [step[2]]
            products += self.graph_handler.graph.edges[step[1], step[2]]['required_compounds']

            rxn_eq = ""
            for i, side in enumerate([reactants, products]):
                for j, aggregate_id in enumerate(side):
                    # # # Identify Compound or Flask
                    if self.graph_handler.graph.nodes[aggregate_id]['type'] == db.CompoundOrFlask.COMPOUND.name:
                        aggregate_type = db.CompoundOrFlask.COMPOUND
                    elif self.graph_handler.graph.nodes[aggregate_id]['type'] == db.CompoundOrFlask.FLASK.name:
                        aggregate_type = db.CompoundOrFlask.FLASK
                    else:
                        raise RuntimeError('Invalid aggregate type encountered in your graph nodes. '
                                           'Please check the graph nodes.')

                    aggregate_str = get_molecular_formula_of_aggregate(
                        db.ID(aggregate_id), aggregate_type, self._compounds, self._flasks, self._structures)

                    rxn_eq += aggregate_str

                    if j < len(side) - 1:
                        rxn_eq += " + "
                if i == 0:
                    rxn_eq += " -> "
            sequence_list.append(rxn_eq)

        return sequence_list

    def get_overall_reactants(self, path: List[str]) -> List[List[Tuple[str, float]]]:
        """
        Summarize the overall reactants of a given path.
        Count the appearance of compounds in a reaction, -1 for reactants and +1 for products.

        Parameters
        ----------
        path : List[str]
            Path containing a list of the traversed nodes.

        Returns
        -------
         : List[List[Tuple[str, float]]]
            A tuple containing the reactants and products of a given path as list of tuples consisting of the aggregate
            ID and its factor.
        """
        total_counter: Dict[str, float] = {}
        assert self.graph_handler
        # # # Loop over elementary steps
        for i in np.arange(0, len(path) - 2, 2):
            step = path[i:i + 3]
            # # # Count Reactants
            tmp_reactants = [step[0]]
            tmp_reactants += self.graph_handler.graph.edges[step[0], step[1]]['required_compounds']
            for key in tmp_reactants:
                if key not in total_counter.keys():
                    total_counter[key] = 0
                total_counter[key] -= 1
            # # # Count Products
            tmp_products = [step[2]]
            tmp_products += self.graph_handler.graph.edges[step[1], step[2]]['required_compounds']
            for key in tmp_products:
                if key not in total_counter.keys():
                    total_counter[key] = 0
                total_counter[key] += 1
        # # # Contract reactant and product lists
        reactants = [(cmp_id, abs(value)) for cmp_id, value in total_counter.items() if value < 0]
        products = [(cmp_id, abs(value)) for cmp_id, value in total_counter.items() if value > 0]

        return [reactants, products]

    def get_overall_reaction_equation(self, path: List[str]) -> str:
        """
        Summarize a given path to a reaction equation and return its string.
        Count the appearance of compounds in a reaction, -1 for reactants and +1 for products.
        Returns the factor and the compound as a molecular formula.

        Parameters
        ----------
        path : List[str]
            Path containing a list of the traversed nodes.

        Returns
        -------
        str
            A string of the overall reaction equation of a given path.
        """
        # # # Get overall reactants and products
        reactants, products = self.get_overall_reactants(path)
        assert self.graph_handler
        reaction_equation = ""
        try:
            for i, side in enumerate([reactants, products]):
                for j, cmp_count in enumerate(side):
                    aggregate_id = cmp_count[0]
                    # # # Identify Compound or Flask
                    aggregate_type = None
                    if self.graph_handler.graph.nodes[aggregate_id]['type'] == db.CompoundOrFlask.COMPOUND.name:
                        aggregate_type = db.CompoundOrFlask.COMPOUND
                    elif self.graph_handler.graph.nodes[aggregate_id]['type'] == db.CompoundOrFlask.FLASK.name:
                        aggregate_type = db.CompoundOrFlask.FLASK
                    else:
                        raise RuntimeError('Invalid aggregate type encountered. Please check the graph nodes.')

                    aggregate_str = get_molecular_formula_of_aggregate(
                        db.ID(aggregate_id), aggregate_type, self._compounds, self._flasks, self._structures)

                    reaction_equation += str(cmp_count[1]) + " " + aggregate_str
                    if j < len(side) - 1:
                        reaction_equation += " + "
                if i == 0:
                    reaction_equation += " = "
        except KeyError:
            reaction_equation = "Could not determine reaction equation"

        return reaction_equation

    @staticmethod
    def _k_shortest_paths(graph: nx.DiGraph, source: str, target: str, n_paths: int, weight: Union[str, None] = None,
                          path_start: int = 0) -> List[List[str]]:
        """
        Finding the k shortest paths between a source and target node in a given graph.
        The length of the returned paths increases.

        Notes
        -----
        * This procedure is based on the algorithm by Jin Y. Yen [1]. Finding the first k paths requires O(k*nodes^3)
          operations.
        * Implemented as given in the documentation of NetworkX:
          https://networkx.org/documentation/stable/reference/algorithms/generated/ \
          networkx.algorithms.simple_paths.shortest_simple_paths.html
        * [1]: Jin Y. Yen, “Finding the K Shortest Loopless Paths in a Network”, Management Science, Vol. 17, No. 11,
               Theory Series (Jul., 1971), pp. 712-716.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to be queried.
        source : str
            The ID of the starting compound as string.
        target : str
            The ID of the targeted compound as string.
        n_paths : int
            The number of paths to be returned.
        weight : Union[str, None], optional
            The key for the weight encoded in the edges to be used.
        path_start : int, optional
            An index of the first returned path, by default 0

        Returns
        -------
        List[List[str]]
            List of paths, each path consisting of a list of nodes.
        """
        return list(
            islice(nx.shortest_simple_paths(graph, source, target, weight=weight), path_start, path_start + n_paths)
        )

    def set_start_conditions(self, conditions: Dict[str, float]):
        """
        Add the IDs of the start compounds to self.start_compounds and add entries for cost in self.compound_cost.

        Parameters
        ----------
        conditions : Dict[str float]
            The IDs of the compounds as keys and its given cost as values.
        """
        # # # Reset Start conditions, if already set previously
        if self.start_compounds_set:
            # # # Reset 'weight' if compound costs were solved
            if self.graph_updated_with_compound_costs:
                self.reset_graph_compound_costs()
            # # # Reset compound costs and start compounds
            self.compound_costs = {}
            self.start_compounds = []
            self.compound_costs_solved = False
            self.graph_updated_with_compound_costs = False
        # # # Loop over conditions and set them
        for cmp_id, cmp_cost in conditions.items():
            self.compound_costs[cmp_id] = cmp_cost
            self.start_compounds.append(cmp_id)
        self.start_compounds_set = True

    def __weight_func(self, u: str, _: str, d: Dict):
        """
        Calculates the weight of the edge d connecting u with _ (directional!).
        Only consider the costs of the required compounds if the edge d is from a compound node to a reaction node.
        If the edge d connects a reaction node with a compound node, the returned weight should be 0.

        Parameters
        ----------
        u : str
            The ID of start node.
        _ : str
            The ID of end node.
        d : Dict
            The edge connecting u and _ as dictionary

        Returns
        -------
        float
            Sum over the edge weight and the costs of the required compounds.
        """
        # # # Weight of edge
        edge_wt = d.get("weight", 0)
        # # # List of required compounds
        tmp_required_compounds = d.get("required_compounds", None)
        # # # Sum over costs of required compounds.
        # # # Only for edges from compound node to rxn node
        if ';' not in u and tmp_required_compounds is not None:
            required_compound_costs = np.sum([self.compound_costs[n] for n in tmp_required_compounds])
        else:
            required_compound_costs = 0.0

        return edge_wt + required_compound_costs

    def calculate_compound_costs(self, recursive: bool = True):
        """
        Determine the cost for all compounds via determining their shortest paths from the ``start_compounds``.
        If this succeeds, set ``compound_costs_solved`` to ``True``. Otherwise it stays ``False``.

        The algorithm works as follows:
        Given the starting conditions, one loops over the individual starting compounds as long as:
        - the self._pseudo_inf entries in self.compound_costs are reduced
        - for any entry in self.compounds_cost a lower cost is found
        With each starting compound, one loops over compounds which have yet no cost assigned.
        For each start - target compound pair, the shortest path is determined employing Dijkstra's algorithm.
        The weight function checks the ``weight`` of the edges as well as the costs of the required compounds listed in
        the ``required_compounds`` of the traversed edges.
        If the path exceeds the length of self._pseudo_inf, this path is not considered for further evaluation.
        The weight of the starting compound is added to the tmp_cost.
        If the target compound has no weight assigned yet in ``compound_costs`` OR
        if the target compound has a weight assigned which is larger (in ``compound_costs`` as well as in
        ``tmp_compound_costs``) than the current ``tmp_cost`` is written to the temporary storage of
        ``tmp_compound_costs``.

        After the loop over all starting compounds completes, the collected costs for the found targets are written to
        ``compound_costs``.
        The convergence variables are updated and the while loop continues.

        Notes
        -----
        * Checks if the start compounds are set.
        * Checks if the graph contains any nodes.

        Parameters
        ----------
        recursive : bool
            All compounds are checked for shorter paths, True by default.
            If set to False, compounds for which a cost has been determined are not checked in the next loop.
        """
        assert self.graph_handler
        if not self.start_compounds_set:
            raise RuntimeError("No start conditions given.")
        if len(self.graph_handler.graph.nodes) == 0:
            raise RuntimeError("No nodes in graph.")

        # # # Set all unknown compounds to pseudo inf cost
        all_unknown_cmps = [n for n in self.graph_handler.graph.nodes if ';' not in n
                            and n not in self.start_compounds]
        for cmp_id in all_unknown_cmps:
            self.compound_costs[cmp_id] = self._pseudo_inf

        # # # Extract connected graph to reduce number of compounds to check
        try:
            subgraph = self.extract_connected_graph(self.start_compounds)
        except DifferentSubgraphsError:
            warnings.warn("Start compounds are not connected.")
            subgraph = self.graph_handler.graph
        cmps_to_check = [n for n in subgraph if ';' not in n
                         and n not in self.start_compounds]
        # # # Dictionary for current run
        tmp_compound_costs: Dict[str, float] = {}
        no_path_to_cmps = list()
        # # # Determine convergence variables for while loop
        current_inf_count = sum(value == self._pseudo_inf for value in self.compound_costs.values())
        prev_inf_count = current_inf_count
        # # # Convergence criteria
        compound_costs_change = None
        compound_costs_opt_change = 1
        converged = False
        # # # Find paths until either no costs are unknown or the None count has not changed
        while not converged:
            compound_costs_opt_change = 0
            print("Remaining Nodes:", len(cmps_to_check))
            for tmp_start_node in self.start_compounds:
                # # # Loop over all nodes to be checked starting from start nodes
                for target in cmps_to_check:
                    # # # Determine cost and path with dijkstra's algorithm
                    try:
                        tmp_cost, _ = nx.single_source_dijkstra(self.graph_handler.graph, tmp_start_node, target,
                                                                cutoff=None, weight=self.__weight_func)
                    # # # Catch nodes with no paths, remove them from compounds to check list
                    except nx.NetworkXNoPath as error:
                        print(error.__str__()[:-1] + " from " + tmp_start_node + ".")
                        # # # Remove target if no way found from other starting compounds
                        if (target not in tmp_compound_costs.keys()) and \
                           (self.compound_costs[target] == self._pseudo_inf) and \
                           (tmp_start_node == self.start_compounds[-1]):
                            print("Removing " + target + " from compounds to check.")
                            # # # Append target to be removed
                            no_path_to_cmps.append(target)
                        continue
                    # # # Check if the obtained cost is larger than infinity (pseudo_inf)
                    # # # and continue with the next target if this is the case
                    if (tmp_cost - self._pseudo_inf) > 0.0:
                        continue
                    # # # Add cost of the starting node
                    tmp_cost += self.compound_costs[tmp_start_node]
                    # # # Check for value in compound_costs dict and indicate change
                    if (self.compound_costs[target] != self._pseudo_inf and
                            10e-6 < self.compound_costs[target] - tmp_cost):
                        compound_costs_opt_change += 1
                    # # # Not already set check
                    if self.compound_costs[target] == self._pseudo_inf or (
                            self.compound_costs != self._pseudo_inf and 10e-6 < self.compound_costs[target] - tmp_cost):
                        # # # Not discovered in current run OR new tmp cost lower than stored cost
                        if target not in tmp_compound_costs.keys():
                            tmp_compound_costs[target] = tmp_cost
                        elif tmp_cost < tmp_compound_costs[target]:
                            tmp_compound_costs[target] = tmp_cost
                # # # Remove targets where no path could be found
                for cmp in no_path_to_cmps:
                    cmps_to_check.remove(cmp)
                no_path_to_cmps = list()
            # # # Write obtained compound_costs to overall dictionary
            for key, value in tmp_compound_costs.items():
                self.compound_costs[key] = value
                # # # Remove found nodes if recursive is false
                if not recursive:
                    cmps_to_check.remove(key)
            # # # Reset tmp_pr_cost
            tmp_compound_costs = {}
            # # # Convergence Check
            current_inf_count = sum(value == self._pseudo_inf for value in self.compound_costs.values())
            compound_costs_change = prev_inf_count - current_inf_count
            prev_inf_count = current_inf_count

            print(50 * '-')
            print("Current None:", current_inf_count)
            print("PR Change:", compound_costs_change)
            print("PR Opt Change:", compound_costs_opt_change)
            # # # Convergence Check
            if (compound_costs_change == 0 and compound_costs_opt_change == 0):
                converged = True

        # # # Final check if all compound costs are solved
        if current_inf_count == 0:
            self.compound_costs_solved = True

    def update_graph_compound_costs(self):
        """
        Update the 'weight' of edges from compound to reaction nodes by adding the compound costs.
        The compound costs are the sum over the costs stored in self.compound_costs of the required compounds.
        The edges of the resulting graph contain a weight including the required_compound_costs based on the starting
        conditions.
        All analysis of the graph therefore depend on the starting conditions.

        Notes
        -----
        * Checks if the compound costs have successfully been determined.
        * Checks if the graph has been updated with the compound costs.
        """

        # # # Check if all costs are available
        if not self.compound_costs_solved:
            unsolved_cmp = [key for key, value in self.compound_costs.items() if value == self._pseudo_inf]
            warnings.warn("The following compounds have no cost assigned:\n" + str(unsolved_cmp) +
                          "\nGraph will be updated anyway, but maybe reconsider the starting conditions.\n")
        # # # Check if graph has been updated
        if self.graph_updated_with_compound_costs:
            warnings.warn("The graph has been updated with compound costs previously, but will be updated anyway.")
        # # # Reset unique_iterator_list as graph changes
        self._reset_iterator_memory()
        for node in self.compound_costs.keys():
            # # # Loop over all edges of compound and manipulate weight
            for target_node, attributes in self.graph_handler.graph[node].items():
                required_compound_costs = np.asarray([self.compound_costs[k] for k in attributes['required_compounds']])
                tot_required_compound_costs = np.sum(required_compound_costs)
                # # # Set required compound costs in edge
                self.graph_handler.graph.edges[node,
                                               target_node]['required_compound_costs'] = tot_required_compound_costs
                # # # Add required compound costs to weight
                self.graph_handler.graph.edges[node, target_node]['weight'] += tot_required_compound_costs
        # # # Set bool for update
        self.graph_updated_with_compound_costs = True

    def reset_graph_compound_costs(self):
        """
        Reset the 'weight' of edges from compound to reaction nodes by subtracting the required compound costs.
        Allows to re-calculate the compound costs under different starting conditions.

         Notes
        -----
        * Checks if the compound costs have successfully been determined.
        """
        if not self.graph_updated_with_compound_costs:
            raise RuntimeError("Graph not updated with compound costs.")
        for node in self.compound_costs.keys():
            # # # Loop over all edges of compound and manipulate weight
            for target_node in self.graph_handler.graph[node].keys():
                # # # Get required compound costs from edge
                tot_required_compound_costs = self.graph_handler.graph.edges[node,
                                                                             target_node]['required_compound_costs']
                # # # Subtract required compound costs from weight
                self.graph_handler.graph.edges[node, target_node]['weight'] -= tot_required_compound_costs
                # # # Set required compound costs back to 0.0
                self.graph_handler.graph.edges[node, target_node]['required_compound_costs'] = 0.0
        # # # Reset bools for compound costs
        self.compound_costs_solved = False
        self.graph_updated_with_compound_costs = False

    def export_graph(self, filename: str = "graph.json"):
        """
        Export the graph without compound costs as dictionary to .json file.

        Parameters
        ----------
        filename : str, optional
            Name of the file to write graph into, by default "graph.json".
        """
        assert self.graph_handler
        graph_copy = copy.deepcopy(self.graph_handler.graph)
        if self.graph_updated_with_compound_costs:
            for node in self.compound_costs.keys():
                # # # Loop over all edges of compound and manipulate weight
                for target_node in graph_copy[node].keys():
                    # # # Get required compound costs from edge
                    tot_required_compound_costs = graph_copy.edges[node, target_node]['required_compound_costs']
                    # # # Subtract required compound costs from weight
                    graph_copy.edges[node, target_node]['weight'] -= tot_required_compound_costs
                    # # # Set required compound costs back to 0.0
                    graph_copy.edges[node, target_node]['required_compound_costs'] = 0.0

        graph_as_dict = nx.convert.to_dict_of_dicts(graph_copy)
        # # # Add node type to dictionary
        for node_key, prop_dict in graph_copy.nodes(data=True):
            for prop_key in prop_dict.keys():
                graph_as_dict[node_key][prop_key] = graph_copy.nodes[node_key][prop_key]
        with open(filename, 'w') as f:
            json.dump(graph_as_dict, f, indent=4)

    def export_compound_costs(self, filename: str = "compound_costs.json"):
        """
        Export the compound cost dictionary to a .json file.

        Parameters
        ----------
        filename : str, optional
            Name of the file to write compound costs into, by default "compound_costs.json"
        """
        assert len(self.compound_costs) > 0
        with open(filename, 'w') as f:
            json.dump(self.compound_costs, f, indent=4)

    @staticmethod
    def _import_graph(filename: str) -> nx.DiGraph:
        """
        Import a graph from a file.

        Parameters
        ----------
        filename : str
            The path to the file containing the graph data.

        Returns
        -------
        nx.DiGraph
            The imported graph as a NetworkX directed graph.
        """
        with open(filename, "r") as f:
            graph = json.load(f)
        # # # Extract additional node type information
        node_properties_dict: Dict[str, Any] = {}
        for node_key, node in graph.items():
            # Filter property keys
            # Property keys are not allowed to end on ';' or to have 24 characters
            # TODO: this filter is a bit doggy, no idea for alternativ though
            node_property_keys = list(filter(lambda key: len(key) != 24 and key[-1] != ";", list(node.keys())))
            for index, prop_key in enumerate(node_property_keys):
                if index == 0:
                    node_properties_dict[node_key] = {}
                node_properties_dict[node_key][prop_key] = node[prop_key]
                graph[node_key].pop(prop_key)
        # # # Load NetworkX Digraph
        nx_graph = nx.convert.from_dict_of_dicts(graph, create_using=nx.DiGraph)
        for node_key, node_properties in node_properties_dict.items():
            for prop_key, prop_value in node_properties.items():
                nx_graph.nodes[node_key][prop_key] = prop_value

        return nx_graph

    @staticmethod
    def _import_dictionary(filename: str) -> Dict[str, Any]:
        with open(filename, "r") as f:
            imported_dict = json.load(f)
        return imported_dict

    class BasicHandler(HoldsCollections):
        """
        A basic class to handle the construction of the nx.DiGraph.
        A list of reactions can be added differently, depending on the implementation of ``_get_weight`` and
        ``get_valid_reaction_ids``.

        Attributes
        ----------
        graph : nx.DiGraph
            The directed graph.
        barrierless_weight : float
            The weight to be set for barrierless reactions.
        model : db.Model
            A model for filtering the valid reactions.
            Per default ("any", "any", "any"), reactions are included regardless of the model.
        filter_negative_barriers : bool
            If True, reactions with negative barriers are filtered out.
        use_structure_model : bool
            If True, the structure model is used to filter out reactions.
        structure_model : db.Model
            A model for filtering the valid reactions.
            Per default ("any", "any", "any"), both structure and energy evaluations are based on the same model.
        use_only_enabled_aggregates : bool
            If True, only enabled aggregates are used.
        """

        def __init__(self, manager: db.Manager, model: db.Model,
                     structure_model: db.Model) -> None:
            super().__init__()
            self.graph = nx.DiGraph()
            self._required_collections = ["manager", "calculations", "compounds", "flasks", "reactions",
                                          "elementary_steps", "structures", "properties"]
            self.initialize_collections(manager)
            self.barrierless_weight = 1.0
            self.model: db.Model = model
            self.filter_negative_barriers = False
            self.use_structure_model = False
            self.use_only_enabled_aggregates = False
            self.structure_model: db.Model = structure_model
            self.use_thermodynamics_model = False
            self.thermodynamics_model = False
            self._rxn_to_es_map: Dict[str, db.ID] = {}
            self._allowed_reaction_sides: Dict[str, db.Side] = {}
            self._energy_label = "Electronic Energy"

        def add_reaction(self, reaction: db.Reaction):
            """
            Add a reaction to the graph.
            Each reaction node represents the LHS and RHS.
            Hence, every reagent of a reaction is connected to every product of a reaction via one reaction node.

            For instance:\n
            | A + B = C + D, reaction R\n
            | A -> R\\ :sub:`1` -> C
            | A -> R\\ :sub:`1` -> D
            | B -> R\\ :sub:`1` -> C
            | B -> R\\ :sub:`1` -> D
            |
            | C -> R\\ :sub:`2` -> A
            | C -> R\\ :sub:`2` -> B
            | D -> R\\ :sub:`2` -> A
            | D -> R\\ :sub:`2` -> B

            Representing this reaction in the graph yields 4 compound nodes,
            2 reaction nodes (same reaction) and 16 edges (2*2*2*2).
            The weights assigned to the edges depends on the ``_get_weight`` implementation.

            The edges from a compound node to a reaction node contain several pieces of information:
                weight:
                the weight of the edge
                1 if the reaction is not barrierless, otherwise it is set to self.barrierless_weight
                required_compounds:
                the IDs of the other reagents of this side of the reaction in a list
                required_compound_costs:
                the sum over all compound costs of the compounds in the required_compounds list
                None by default

            The edges from a reaction node to a compound node contain several information:
                weight:
                the weight of the edge, set to 0
                required_compounds:
                the IDs of the other products emerging;
                added for easier information extraction during the path analysis

            Parameters
            ----------
            reaction : db.Reaction
                The reaction to be added to the graph.
            """
            # Add two rxn nodes
            rxn_nodes: List[Optional[str]] = []
            reaction_id = reaction.id().string()
            allowed_side = self.get_allowed_reaction_sides(reaction_id)

            for i in range(0, 2):
                if allowed_side == db.Side.NONE:
                    rxn_nodes.append(None)
                    continue
                if allowed_side == db.Side.RHS and i == 0:
                    rxn_nodes.append(None)
                    continue
                if allowed_side == db.Side.LHS and i == 1:
                    rxn_nodes.append(None)
                    continue
                # Add rxn node between lhs and rhs compound
                rxn_node = ';'.join([reaction_id, str(i)])
                rxn_node += ';'
                # Construct property dict of node
                rxn_node_properties: Dict[str, Any] = {'type': 'rxn_node'}
                es_id = self._get_mapped_es_of_reaction(reaction_id)
                if es_id is not None:
                    rxn_node_properties['elementary_step_id'] = es_id
                barriers: Optional[Tuple[float, float]] = self._get_barriers(reaction_id)
                if barriers is not None:
                    rxn_node_properties['activation_energies'] = barriers
                    rxn_node_properties['energy_type'] = self._energy_label
                self.graph.add_node(rxn_node, **rxn_node_properties)
                rxn_nodes.append(rxn_node)
            # Convert to strings
            reactants = reaction.get_reactants(db.Side.BOTH)
            reactant_types = reaction.get_reactant_types(db.Side.BOTH)
            weights = self._get_weight(reaction)
            # Add lhs aggregates and connect
            for lhs_cmp, lhs_type in zip([i.string() for i in reactants[0]],
                                         [i.name for i in reactant_types[0]]):
                if lhs_cmp not in self.graph:
                    self.graph.add_node(lhs_cmp, type=lhs_type)
                required_cmps_lhs = [s.string() for s in reactants[0]]
                required_cmps_lhs.remove(lhs_cmp)
                if rxn_nodes[0] is not None:
                    self.graph.add_edge(lhs_cmp, rxn_nodes[0], weight=weights[0], required_compounds=required_cmps_lhs,
                                        required_compound_costs=None)
                if rxn_nodes[1] is not None:
                    self.graph.add_edge(rxn_nodes[1], lhs_cmp, weight=0.0, required_compounds=None)
            # Add rhs aggregates and connect
            for rhs_cmp, rhs_type in zip([i.string() for i in reactants[1]],
                                         [i.name for i in reactant_types[1]]):
                if rhs_cmp not in self.graph:
                    self.graph.add_node(rhs_cmp, type=rhs_type)
                required_cmps_rhs = [s.string() for s in reactants[1]]
                required_cmps_rhs.remove(rhs_cmp)
                if rxn_nodes[1] is not None:
                    self.graph.add_edge(rhs_cmp, rxn_nodes[1], weight=weights[1], required_compounds=required_cmps_rhs,
                                        required_compound_costs=None)
                if rxn_nodes[0] is not None:
                    self.graph.add_edge(rxn_nodes[0], rhs_cmp, weight=0.0, required_compounds=None)

            # # # Loop over reaction nodes to add required compounds info to downwards edges; might be unnecessary
            if all(node is not None for node in rxn_nodes):
                node_index = 1
                for node in rxn_nodes:
                    for key in self.graph[node].keys():
                        self.graph.edges[node, key]['required_compounds'] = \
                            self.graph.edges[key, rxn_nodes[node_index]]['required_compounds']
                    node_index -= 1

        def add_aggregate(self, aggregate: Union[db.Compound, db.Flask], aggregate_type: db.CompoundOrFlask):
            """
            Add an aggregate to the graph.
            This is only to be used in a fallback strategy to add aggregates only if no reactions are present yet,
            e.g. at the beginning of an exploration.

            Parameters
            ----------
            aggregate : Union[db.Compound, db.Flask]
                The aggregate to be added.
            aggregate_type : db.CompoundOrFlask
                The type of the aggregate.
            """
            aggregate_id = aggregate.id().string()
            aggregate_type_name = aggregate_type.name
            if aggregate_id not in self.graph:
                self.graph.add_node(aggregate_id, type=aggregate_type_name)

        def _get_barriers(self, _: str) -> Optional[Tuple[float, float]]:
            """
            Optional getter for the barrier of the reaction step.
            """
            return None

        def _get_weight(self, reaction: db.Reaction) -> Tuple[float, float]:
            """
            Determining the weights for the edges of the given reaction.

            Parameters
            ----------
            reaction : db.Reaction
                Reaction of interest

            Returns
            -------
            Tuple[float, float]
                Weight for connections to the LHS reaction node, weight for connections to the RHS reaction node
            """
            for step in reaction.get_elementary_steps(self._manager):
                # # # Barrierless weights for barrierless reactions
                if step.get_type() == db.ElementaryStepType.BARRIERLESS:
                    return self.barrierless_weight, self.barrierless_weight
            return 1.0, 1.0

        def get_valid_reaction_ids(self) -> List[db.ID]:
            """
            Basic filter function for reactions.
            Per default, it returns all reactions.

            Returns
            -------
            List[db.ID]
                List of IDs of the filtered reactions.
            """
            valid_ids: List[db.ID] = list()
            for reaction in self._reactions.iterate_all_reactions():
                if self._valid_reaction(reaction):
                    valid_ids.append(reaction.id())
            return valid_ids

        def get_valid_aggregate_ids(self) -> List[Tuple[db.ID, db.CompoundOrFlask]]:
            """
            Basic filter function for aggregates (compounds and flasks)
            Returns a list of valid aggregate IDs.

            Returns
            -------
            List[Tuple[db.ID, db.CompoundOrFlask]]
                A list of tuples containing the valid aggregate IDs and their types.
            """
            valid_ids: List[Tuple[db.ID, db.CompoundOrFlask]] = list()
            for compound in self._compounds.iterate_all_compounds():
                if self._valid_aggregate(compound, db.CompoundOrFlask.COMPOUND):
                    valid_ids.append((compound.id(), db.CompoundOrFlask.COMPOUND))
            for flask in self._flasks.iterate_all_flasks():
                if self._valid_aggregate(flask, db.CompoundOrFlask.FLASK):
                    valid_ids.append((flask.id(), db.CompoundOrFlask.FLASK))
            return valid_ids

        def _get_mapped_es_of_reaction(self, rxn_id_string: str) -> Optional[str]:
            id_ = self._rxn_to_es_map.get(rxn_id_string, None)
            return id_ if id_ is None else str(id_)

        def _valid_reaction(self, reaction: db.Reaction) -> bool:
            """
            Checks if a given reaction is valid.
            A reaction is considered valid if at least one elementary step of the reaction has an electronic energy
            assigned to its first structure, if required calculated with the set db.Model.

            If 'use_structure_model' is set to True, the elementary steps are checked if the first structure of the
            elementary step has the required structure model.

            If 'filter_negative_barriers' is set to True, the elementary step is checked if the electronic energy
            barrier in any direction is None or negative.

            If 'use_only_enabled_aggregates' is set to True, the reaction is checked if all aggregates on both sides
            are enabled.

            Parameters
            ----------
            reaction : db.Reaction
                The reaction which shall be checked for one valid elementary step.

            Returns
            -------
            bool
                Boolean indicating if the reaction is valid.
            """
            reaction.link(self._reactions)
            if self.use_only_enabled_aggregates:
                lhs, rhs = reaction.get_reactants(db.Side.BOTH)
                lhs_types, rhs_types = reaction.get_reactant_types(db.Side.BOTH)
                if not all(
                        get_compound_or_flask(reactant_id, reactant_type, self._compounds, self._flasks).explore()
                        for reactant_id, reactant_type in zip(lhs + rhs, lhs_types + rhs_types)
                ):
                    return False
            for es_id in reaction.get_elementary_steps():
                valid_structures = False
                valid_barriers = True
                es = db.ElementaryStep(es_id, self._elementary_steps)
                first_structure_lhs = db.Structure(es.get_reactants()[0][0], self._structures)
                # # # Model Check
                # Check, if energy of this structure with specified model exists
                if self.model != db.Model("any", "any", "any") or\
                   self.model != db.Model("any", "any", ""):  # type: ignore
                    assert self.model
                    if self.use_structure_model:
                        assert self.structure_model
                        if self.use_thermodynamics_model:  # Extra step for thermodynamics model
                            assert self.thermodynamics_model
                    # Structure model check if wanted here, structure must have model
                    if self.use_structure_model and \
                       self.structure_model != first_structure_lhs.get_model():  # type: ignore
                        # # # Skip structure if the structure model does not fit the set model
                        continue
                    first_structure_lhs_e = get_energy_for_structure(first_structure_lhs, "electronic_energy",
                                                                     self.model, self._structures, self._properties)
                    # # # Structure validity check
                    if first_structure_lhs_e is not None:
                        valid_structures = True
                    if self.use_thermodynamics_model:  # # # validity of the thermodynamics model
                        first_structure_lhs_e2 = get_energy_for_structure(first_structure_lhs, "electronic_energy",
                                                                          self.thermodynamics_model, self._structures,
                                                                          self._properties)
                        if first_structure_lhs_e2 is not None:
                            valid_structures = True
                    if self.filter_negative_barriers:
                        barriers = get_barriers_for_elementary_step_by_type(es, "electronic_energy", self.model,
                                                                            self._structures, self._properties)
                        if None in barriers or (barriers[0] < 0.0 or barriers[1] < 0.0):
                            valid_barriers = False
                        if self.use_thermodynamics_model:  # # # validity of the thermodynamics model
                            barriers2 = get_barriers_for_elementary_step_by_type(es, "electronic_energy",
                                                                                 self.thermodynamics_model,
                                                                                 self._structures, self._properties)
                            if None in barriers2 or (barriers2[0] < 0.0 or barriers2[1] < 0.0):
                                valid_barriers = False
                    # # # Final structure and barrier check
                    if valid_structures and valid_barriers:
                        return True
                # Check if model is not specified (None)
                else:
                    if len(first_structure_lhs.get_properties("electronic_energy")) > 0:
                        return True
            # If all elementary steps of this reaction fail the checks, this reaction is not valid
            return False

        def _valid_aggregate(self, aggregate: Union[db.Compound, db.Flask], aggregate_type: db.CompoundOrFlask) -> bool:
            """
            Checks if a given aggregate is valid.
            An aggregate is considered valid if at least one structure of the aggregate has an electronic energy
            assigned, if required calculated with the set db.Model.

            If 'use_structure_model' is set to True, the structures are checked if they have
            the required structure model.

            If 'use_only_enabled_aggregates' is set to True, the aggregate is checked if it is enabled for
            exploration.

            Parameters
            ----------
            aggregate : Union[db.Compound, db.Flask]
                The aggregate to be checked for validity.
            aggregate_type : db.CompoundOrFlask
                The type of the aggregate (Compound or Flask).

            Returns
            -------
            bool
                True if the aggregate is valid, False otherwise.
            """
            if aggregate_type == db.CompoundOrFlask.COMPOUND:
                aggregate.link(self._compounds)
            else:
                aggregate.link(self._flasks)

            if self.use_only_enabled_aggregates:
                if not aggregate.explore():
                    return False
            # # # Loop over all structures of the aggregate
            for structure_id in aggregate.get_structures():
                tmp_structure = db.Structure(structure_id, self._structures)
                # # # Model Check
                # Check if energy of this structure with specified model exists
                if self.model != db.Model("any", "any", "any") or\
                   self.model != db.Model("any", "any", ""):  # type: ignore
                    assert self.model
                    if self.use_structure_model:
                        assert self.structure_model
                    # Structure model check if wanted here, structure must have model
                    if self.use_structure_model and self.structure_model != tmp_structure.get_model():  # type: ignore
                        # # # Skip structure if the structure model does not fit the set model
                        continue
                    tmp_structure_e = get_energy_for_structure(
                        tmp_structure, "electronic_energy", self.model, self._structures, self._properties)
                    # # # Structure validity check
                    if tmp_structure_e is not None:
                        return True
                # Check if model is not specified (None)
                else:
                    if len(tmp_structure.get_properties("electronic_energy")) > 0:
                        return True
            # If all structures of this aggregate fail the checks, this aggregate is not valid
            return False

        def get_allowed_reaction_sides(self, reaction_id: str) -> db.Side:
            return self._allowed_reaction_sides.get(reaction_id, db.Side.BOTH)

    class BarrierBasedHandler(BasicHandler):
        """
        A class derived from the basic graph handler class to encode the reaction barrier information in the edges.
        The barriers of the elementary step with the minimal TS energy of a reaction are employed.
        The barriers are converted to rate constants, normalized over all rate constants in the graph and then the cost
        function :math:`|log(normalized\\ rate\\ constant)|` is applied to obtain the weight.

        Attributes
        ----------
        temperature : float
            The temperature for calculating the rate constants from the barriers. Default is 298.15 K.
        _rate_constant_normalization : float
            The factor to normalize the rate constant.
        """

        def __init__(self, db_manager: db.Manager, model: db.Model,
                     structure_model=db.Model("any", "any", "any")) -> None:
            super().__init__(db_manager, model, structure_model)
            self.temperature = 298.15
            self._barrier_limit: float = math.inf
            self._rate_constant_normalization = 1.0

        def get_barrier_limit(self) -> float:
            return self._barrier_limit

        def set_barrier_limit(self, barrier_limit: float) -> None:
            self._barrier_limit = barrier_limit

        def set_temperature(self, temperature: float):
            """
            Setting the temperature for determining the rate constants.

            Parameters
            ----------
            temperature : float
                The temperature in Kelvin.
            """
            self.temperature = temperature

        def get_temperature(self):
            """
            Gets the set temperature.

            Returns
            -------
            self.temperature : float
                The set temperature.
            """
            return self.temperature

        def get_valid_reaction_ids(self) -> List[db.ID]:
            return [db.ID(key) for key in self._rxn_to_es_map.keys()]

        def _get_valid_reaction_ids(self) -> List[db.ID]:
            valid_ids: List[db.ID] = list()
            for reaction in self._reactions.iterate_all_reactions():
                if self._valid_reaction(reaction):
                    valid_ids.append(reaction.id())
            return valid_ids

        def _map_elementary_steps_to_reactions(self):
            """
            Loop over all reactions to get the elementary step with the lowest TS energy for each reaction.
            The corresponding db.ID of the elementary step is written to the _rxn_to_es_map.
            """
            for reaction_id in self._get_valid_reaction_ids():
                reaction = db.Reaction(reaction_id, self._reactions)
                # # # Go for gibbs energy first
                es = get_elementary_step_with_min_ts_energy(
                    reaction,
                    "gibbs_free_energy",
                    self.model,
                    self._elementary_steps,
                    self._structures,
                    self._properties,
                    structure_model=self.structure_model)
                # # # If unsuccessful, attempt electronic energy
                if es is None:
                    es = get_elementary_step_with_min_ts_energy(
                        reaction,
                        "electronic_energy",
                        self.model,
                        self._elementary_steps,
                        self._structures,
                        self._properties,
                        structure_model=self.structure_model)
                    # # # If es_id still None, continue; pure safety measure
                    if es is None:
                        continue
                es_id = es.id()
                allowed_side = db.Side.BOTH
                # Enforce non-negative barriers
                if self.filter_negative_barriers or self._barrier_limit < math.inf:
                    es = db.ElementaryStep(es_id, self._elementary_steps)
                    barriers = get_barriers_for_elementary_step_by_type(es, "electronic_energy", self.model,
                                                                        self._structures, self._properties)
                    if self.filter_negative_barriers and (None in barriers or (barriers[0] < 0.0 or barriers[1] < 0.0)):
                        es_id = self._get_elementary_step_with_min_ts_energy_and_non_negative_barriers(reaction)
                        if self._barrier_limit < math.inf:
                            es = db.ElementaryStep(es_id, self._elementary_steps)
                            barriers = get_barriers_for_elementary_step_by_type(es, "electronic_energy", self.model,
                                                                                self._structures, self._properties)
                    if self._barrier_limit < math.inf:
                        if all(b > self._barrier_limit for b in barriers):
                            allowed_side = db.Side.NONE
                        elif barriers[0] > self._barrier_limit:
                            allowed_side = db.Side.RHS
                        elif barriers[1] > self._barrier_limit:
                            allowed_side = db.Side.LHS
                # # # Write elementary step to map
                self._allowed_reaction_sides[reaction_id.string()] = allowed_side
                self._rxn_to_es_map[reaction_id.string()] = es_id

        def _calculate_rate_constant_normalization(self):
            """
            Determine the rate constant normalization factor for calculating the edge weights.
            Loops over the _rxn_to_es_map, converts every barrier to the rate constant and adds it to the sum of rate
            constants.
            For barrierless elementary steps, twice the barrierless_weight is added.
            For elementary steps with negative barriers, for the negative barrier(s) the barrierless_weight is added.

            The rate constant normalization is then the inverse of the final sum.
            """

            k_sum = 0.0
            for es_id in self._rxn_to_es_map.values():
                es = db.ElementaryStep(es_id, self._elementary_steps)
                if es.get_type() == db.ElementaryStepType.BARRIERLESS:
                    k_sum += self.barrierless_weight * 2
                else:
                    if not es.has_transition_state():
                        raise RuntimeError("Elementary step " + es_id.string() + " has label " + es.get_type() +
                                           " but no transition state.")
                    # # # Retrieve barriers of elementary step in kJ/mol
                    if self.use_thermodynamics_model:  # consider model 1 (elec) and model 2 (gibbs correction)
                        barriers2g = get_barriers_for_elementary_step_by_type(es, "gibbs_free_energy",
                                                                              self.thermodynamics_model,
                                                                              self._structures, self._properties)
                        barriers2e = get_barriers_for_elementary_step_by_type(es, "electronic_energy",
                                                                              self.thermodynamics_model,
                                                                              self._structures, self._properties)
                        if None in barriers2g:
                            barriers = barriers2e
                            if None in barriers2e:
                                print("Warning: No electronic nor Gibbs energy with thermodynamics model "
                                      "for elementary step {str_el}".format(str_el=es_id))
                                barriers = get_barriers_for_elementary_step_by_type(es, "electronic_energy", self.model,
                                                                                    self._structures, self._properties)
                        else:
                            gibbs_correction = [g-e for g, e in zip(barriers2g, barriers2e)]
                            barriers1e = get_barriers_for_elementary_step_by_type(es, "electronic_energy", self.model,
                                                                                  self._structures, self._properties)
                            barriers = [b+gc for b, gc in zip(barriers1e, gibbs_correction)]
                    else:  # consider only model 1
                        barriers = get_barriers_for_elementary_step_by_type(es, "gibbs_free_energy", self.model,
                                                                            self._structures, self._properties)
                        if None in barriers:
                            barriers = get_barriers_for_elementary_step_by_type(es, "electronic_energy", self.model,
                                                                                self._structures, self._properties)
                    # # # Use user defined barrierless weight for negative barriers
                    # Check LHS Barrier
                    if barriers[0] < 0.0:
                        k_lhs = self.barrierless_weight
                    else:
                        k_lhs = rate_constant_from_barrier(barriers[0], self.temperature)
                    # Check RHS Barrier
                    if barriers[1] < 0.0:
                        k_rhs = self.barrierless_weight
                    else:
                        k_rhs = rate_constant_from_barrier(barriers[1], self.temperature)
                    k_sum += k_lhs + k_rhs

            if k_sum != 0.0:
                self._rate_constant_normalization = 1 / k_sum

        def _get_weight(self, reaction: db.Reaction) -> Tuple[float, float]:
            """
            Determines the weight for a given reaction.
            The weight is calculated by determining the rate constant from the barrier, normalizing the constant with
            the rate_constant_normalization and taking the abs(log()) of the corresponding product.
            For barrierless reactions and negative barriers, the barrierless_weight is taken as rate constants.

            Parameters
            ----------
            reaction : db.Reaction
               The reaction for which the weights should be determined.

            Returns
            -------
            weights : Tuple(float, float)
                The weight for the LHS -> RxnNode and  for the RHS -> RxnNode.
            """
            # # # Look up elementary step id for reaction
            es_id = self._rxn_to_es_map[reaction.id().string()]
            es_step = db.ElementaryStep(es_id, self._elementary_steps)
            # # # Barrierless weights for barrierless reactions
            if es_step.get_type() == db.ElementaryStepType.BARRIERLESS:
                k_lhs = self.barrierless_weight
                k_rhs = self.barrierless_weight
            else:
                if not es_step.has_transition_state():
                    raise RuntimeError("Elementary step " + es_id.string() + " has label " + es_step.get_type().name +
                                       " but no transition state.")
                assert self.model
                # # # Retrieve barriers of elementary step in kJ/mol
                barriers = get_barriers_for_elementary_step_by_type(
                    es_step, "gibbs_free_energy", self.model, self._structures, self._properties)
                if None in barriers:
                    barriers = get_barriers_for_elementary_step_by_type(
                        es_step, "electronic_energy", self.model, self._structures, self._properties)
                assert barriers[0]
                # Check LHS Barrier
                if barriers[0] < 0.0:
                    k_lhs = self.barrierless_weight
                else:
                    k_lhs = rate_constant_from_barrier(barriers[0], self.temperature)
                # Check RHS Barrier
                if barriers[1] < 0.0:
                    k_rhs = self.barrierless_weight
                else:
                    k_rhs = rate_constant_from_barrier(barriers[1], self.temperature)
            return abs(np.log(k_lhs * self._rate_constant_normalization)), \
                abs(np.log(k_rhs * self._rate_constant_normalization))

        def _get_elementary_step_with_min_ts_energy_and_non_negative_barriers(self, reaction: db.Reaction) -> db.ID:
            """
            Gets the elementary step ID with the lowest energy of the corresponding transition state
            of a valid reaction, without a negative barrier in the forward and backward directions.

            Parameters
            ----------
            reaction : db.Reaction
                The reaction for which the elementary steps shall be analyzed.

            Returns
            -------
            elementary_step_id : db.ID
                The ID of the elementary step with the lowest TS energy without negative barriers.
            """
            # # # Get all es steps of the reaction
            el_steps = reaction.get_elementary_steps()
            es_data: List[Tuple[float, float, float, db.ID]] = list()
            assert self.model
            if self.use_structure_model:
                assert self.structure_model
            # # # Loop over the steps and retrieve TS energy and barrier information
            for es_id in el_steps:
                es = db.ElementaryStep(es_id, self._elementary_steps)
                barriers = get_barriers_for_elementary_step_by_type(
                    es, "electronic_energy", self.model, self._structures, self._properties)
                if None in barriers:
                    continue
                # # # Check for mypy
                assert barriers[0] is not None and barriers[1] is not None
                if es.get_type() == db.ElementaryStepType.BARRIERLESS:
                    return es_id
                # # # TS Retrieval and check structure model
                ts = db.Structure(es.get_transition_state(), self._structures)
                if self.use_structure_model and ts.get_model() != self.structure_model:  # type: ignore
                    continue
                ts_energy = get_energy_for_structure(
                    ts, "electronic_energy", self.model, self._structures, self._properties)
                # # # Check for mypy
                assert ts_energy is not None
                # # # Append tuple of ts energy, lhs barrier, rhs barrier and id to list
                es_data.append((ts_energy, barriers[0], barriers[1], es_id))  # type: ignore
            # # # Sort the es data by lowest TS energy
            es_data_sorted = [item for item in sorted(es_data, key=lambda item: item[0])]
            # # # Return first elementary step id where all barriers are positive (> 0.0)
            for data in es_data_sorted:
                if data[1] > 0.0 and data[2] > 0.0:
                    return data[3]
            raise RuntimeError(f"No elementary step with positive barriers found for reaction {str(reaction.id())}")

    class MinimumReactionBarrierBasedHandler(BasicHandler):
        """
        A class derived from the basic graph handler class to encode the reaction barrier information in the edges.
        The reaction barrier is calculated according to the reaction-wrapper objects
        (@see utilities/db_object_wrappers/reaction_wrapper.py).
        The barriers are converted to rate constants, normalized over all rate constants in the graph and then the cost
        function :math:`|log(normalized\\ rate\\ constant)|` is applied to obtain the weight.

        Attributes
        ----------
        reference_state : ReferenceState
            The reference state for calculating the rate constants from the barriers. Default is  derived from the
            pressure and temperature in the electronic structure model.
        _rate_constant_normalization : float
            The factor to normalize the rate constant.
        """

        def __init__(self, db_manager: db.Manager, model: db.Model, structure_model: Optional[db.Model] = None):
            """
            Constructor.

            Parameters
            ----------
            db_manager : db.Manager
                The database manager.
            model : db.Model
                The electronic structure model for the electronic energies.
            structure_model : db.Model, optional
                The electronic structure for the structures and free energy corrections. If not provided, the electronic
                structure model for the energies is used.
            """
            if structure_model is None:
                structure_model = model
            super().__init__(db_manager, model, structure_model)
            self.reference_state = ReferenceState(float(model.temperature), float(model.pressure))
            self.only_electronic: bool = False
            self._barrier_limit: float = math.inf
            self._rate_constant_normalization = 1.0
            self.__valid_reactions: List[Reaction] = []
            self.__reaction_cache: Optional[ReactionCache] = None
            self._required_collections += ["manager"]
            self._barrier_label = r"$G^{\degree}$"

        def __get_reaction_cache(self) -> ReactionCache:
            if self.__reaction_cache is None:
                model_combination = ModelCombination(self.model, self.structure_model)
                self.__reaction_cache = MultiModelCacheFactory().get_reaction_cache(
                    self.only_electronic, model_combination, self._manager)
                if self.only_electronic:
                    self._barrier_label = "Electronic Energy"
            return self.__reaction_cache

        def get_valid_reaction_ids(self) -> List[db.ID]:
            return [reaction.get_db_id() for reaction in self.__valid_reactions]

        def initialize(self) -> None:
            self.__valid_reactions = []
            for db_reaction in self._reactions.iterate_all_reactions():
                reaction = self.__get_reaction_cache().get_or_produce(db_reaction.id())
                if not reaction.complete():
                    continue
                all_aggregates = reaction.get_lhs_aggregates() + reaction.get_rhs_aggregates()
                if self.use_only_enabled_aggregates and any(not aggregate.explore() for aggregate in all_aggregates):
                    continue
                self.__valid_reactions.append(reaction)
            self._calculate_rate_constant_normalization()

        def __get_pathfinder_rate_constants(self, reaction: Reaction) -> Tuple[float, float]:
            lhs_barrier, rhs_barrier = reaction.get_free_energy_of_activation(self.reference_state)
            lhs_k, rhs_k = reaction.get_ts_theory_rate_constants(self.reference_state)
            assert lhs_k is not None  # satisfy mypy
            assert rhs_k is not None  # satisfy mypy
            assert lhs_barrier is not None
            assert rhs_barrier is not None
            # Determine barrierless reactions
            lhs_k_final = lhs_k if lhs_barrier > 0 else self.barrierless_weight
            rhs_k_final = rhs_k if rhs_barrier > 0 else self.barrierless_weight
            # The logic above should assert that k_lhs and k_rhs are never None here.
            assert lhs_k_final is not None
            assert rhs_k_final is not None
            return lhs_k_final, rhs_k_final

        def _calculate_rate_constant_normalization(self) -> None:
            """
            Determine the rate constant normalization factor for calculating the edge weights.
            Loops over the reactions, converts every barrier to the rate constant and adds it to the sum of rate
            constants.

            The rate constant normalization is then the inverse of the final sum.
            """
            k_sum = sum(sum(self.__get_pathfinder_rate_constants(reaction)) for reaction in self.__valid_reactions)
            if k_sum != 0.0:
                self._rate_constant_normalization = 1 / k_sum

        def _get_weight(self, reaction: db.Reaction) -> Tuple[float, float]:
            """
            Determines the weight for a given reaction.
            The weight is calculated by determining the rate constant from the barrier, normalizing the constant with
            the rate_constant_normalization and taking the abs(log()) of the corresponding product.
            For barrierless reactions and negative barriers, the barrierless_weight is taken as rate constants.

            Parameters
            ----------
            reaction : db.Reaction
               The reaction for which the weights should be determined.

            Returns
            -------
            weights : Tuple(float, float)
                The weight for the LHS -> RxnNode and  for the RHS -> RxnNode.
            """
            reaction_wrapper = self.__get_reaction_cache().get_or_produce(reaction.id())
            k_lhs, k_rhs = self.__get_pathfinder_rate_constants(reaction_wrapper)
            return abs(np.log(k_lhs * self._rate_constant_normalization)), \
                abs(np.log(k_rhs * self._rate_constant_normalization))

        def _get_barriers(self, reaction_str_id: str) -> Optional[Tuple[float, float]]:
            reaction_wrapper = self.__get_reaction_cache().get_or_produce(db.ID(reaction_str_id))
            lhs_barrier, rhs_barrier = reaction_wrapper.get_free_energy_of_activation(self.reference_state)
            if lhs_barrier is None or rhs_barrier is None:
                return None
            return lhs_barrier, rhs_barrier

    class FromRMSFileBasedHandler(BasicHandler):
        """
        A class derived from the basic graph handler that loads the graph information from an RMS input file.
        The barriers are converted to rate constants, normalized over all rate constants in the graph and then the cost
        function :math:`|log(normalized\\ rate\\ constant)|` is applied to obtain the weight.

        Attributes
        ----------
        temperature : float
            The temperature.
        rms_file_name : str
            The RMS input file name.
        """

        def __init__(self, db_manager: db.Manager, rms_file_name: str, temperature: float):
            """
            Constructor.

            Parameters
            ----------
            db_manager : db.Manager
                The database manager.
            rms_file_name : str
                The RMS input file name.
            temperature : float
                The temperature.
            """
            super().__init__(db_manager, construct_place_holder_model(), construct_place_holder_model())
            self.temperature: float = temperature
            self.rms_file_name: str = rms_file_name
            self._barrier_limit: float = math.inf
            self._rate_constant_normalization = 1.0
            self.__valid_reaction_entries: Dict[str, Dict] = {}
            self.__species: Dict[str, Dict] = {}
            self._required_collections += ["manager"]
            self._energy_label = "$G^°$"

        def _valid_reaction(self, _: db.Reaction) -> bool:
            return True

        def get_valid_reaction_ids(self) -> List[db.ID]:
            return [db.ID(str_id) for str_id in self.__valid_reaction_entries.keys()]

        def get_valid_reaction_entries(self) -> Dict[str, Dict]:
            return self.__valid_reaction_entries

        def get_species_entries(self) -> Dict[str, Dict]:
            return self.__species

        def get_aggregate_from_string_id(self, string_id: str) -> Union[db.Compound, db.Flask]:
            aggregate: Union[db.Compound, db.Flask] = db.Compound(db.ID(string_id), self._compounds)
            if not aggregate.exists():
                aggregate = db.Flask(db.ID(string_id), self._flasks)
            if not aggregate.exists():
                raise RuntimeError("The aggregate with id {} does not exist".format(string_id))
            return aggregate

        def initialize(self) -> None:
            self.__valid_reaction_entries = {}
            self.__species = {}
            with open(self.rms_file_name, 'r') as rms_file:
                docs = yaml.safe_load(rms_file)
                if "Phases" not in docs or len(docs["Phases"]) == 0 or "Species" not in docs["Phases"][0]:
                    raise RuntimeError("Unexpected Phases/Species definition in the RMS file.")
                species_doc = docs["Phases"][0]["Species"]

                if "Reactions" not in docs:
                    raise RuntimeError("The 'Reactions' entry is missing the the RMS file!")
                reaction_entries = docs["Reactions"]

                self.__species = {species["name"]: species for species in species_doc}

                for reaction_entry in reaction_entries:
                    if "reactants" not in reaction_entry:
                        raise RuntimeError("No 'reactants' entry provided for a reaction. Make sure that your RMS input"
                                           " file is formatted correctly.")
                    if "products" not in reaction_entry:
                        raise RuntimeError("No 'products' entry provided for a reaction. Make sure that your RMS input"
                                           " file is formatted correctly.")
                    reactant_str_ids = sorted(reaction_entry["reactants"])
                    product_str_ids = sorted(reaction_entry["products"])
                    all_aggregates = [self.get_aggregate_from_string_id(str_id)
                                      for str_id in reactant_str_ids + product_str_ids]
                    possible_reaction_ids = set(r_id.string() for r_id in all_aggregates[0].get_reactions())
                    for aggregate in all_aggregates[1:]:
                        possible_reaction_ids = possible_reaction_ids.intersection(set(r_id.string() for r_id in
                                                                                       aggregate.get_reactions()))
                    matching_reaction_id: Optional[db.ID] = None
                    for reaction_str_id in possible_reaction_ids:
                        reaction = db.Reaction(db.ID(reaction_str_id), self._reactions)
                        candidate_reactants = reaction.get_reactants(db.Side.BOTH)
                        lhs = sorted([a_id.string() for a_id in candidate_reactants[0]])
                        rhs = sorted([a_id.string() for a_id in candidate_reactants[1]])
                        lhs_matches_lhs = lhs == reactant_str_ids
                        rhs_matches_rhs = rhs == product_str_ids
                        lhs_matches_rhs = lhs == product_str_ids
                        rhs_matches_lhs = rhs == reactant_str_ids
                        match = (lhs_matches_lhs and rhs_matches_rhs) or (lhs_matches_rhs and rhs_matches_lhs)
                        if match:
                            matching_reaction_id = db.ID(reaction_str_id)
                            self.__valid_reaction_entries[reaction_str_id] = reaction_entry
                            break
                    if matching_reaction_id is None:
                        raise RuntimeError(f"The reaction with products: {product_str_ids} and reactants"
                                           f" {reactant_str_ids} is not listed in the database!")
            self._calculate_rate_constant_normalization()

        def get_free_energy(self, aggregate_str_id: str) -> float:
            poly = self.__species[aggregate_str_id]["thermo"]["polys"][0]
            h = poly["coefs"][5] * utils.MOLAR_GAS_CONSTANT
            s = poly["coefs"][6] * utils.MOLAR_GAS_CONSTANT
            return h - self.temperature * s

        def _get_barriers(self, reaction_str_id: str) -> Optional[Tuple[float, float]]:
            reaction_entry = self.__valid_reaction_entries[reaction_str_id]
            kinetics = reaction_entry['kinetics']
            ea = kinetics["Ea"]
            g_lhs = sum(self.get_free_energy(a_str_id) for a_str_id in reaction_entry["reactants"])
            g_rhs = sum(self.get_free_energy(a_str_id) for a_str_id in reaction_entry["products"])
            g_ts = g_lhs + ea
            ea_reverse = max(0.0, g_ts - g_rhs)
            return ea * 1e-3, ea_reverse * 1e-3

        def get_rate_constants(self, reaction_str_id: str) -> Tuple[float, float]:
            ea_tuple = self._get_barriers(reaction_str_id)
            assert ea_tuple
            ea = ea_tuple[0]
            ea_reverse = ea_tuple[1]
            reaction_entry = self.__valid_reaction_entries[reaction_str_id]
            kinetics = reaction_entry['kinetics']
            beta = 1e+3 * 1.0 / (self.temperature * utils.MOLAR_GAS_CONSTANT)  # 1e+3 to convert to kJ/mol
            k_lhs2rhs = kinetics['A'] * self.temperature ** kinetics['n'] * math.exp(-ea * beta)
            k_rhs2lhs = kinetics['A'] * self.temperature ** kinetics['n'] * math.exp(-ea_reverse * beta)
            return k_lhs2rhs, k_rhs2lhs

        def _get_mapped_es_of_reaction(self, rxn_id_string: str) -> Optional[str]:
            if rxn_id_string not in self.__valid_reaction_entries:
                raise RuntimeError("The reaction '{}' is not included in the pathfinder graph!"
                                   .format(rxn_id_string))
            reaction = db.Reaction(db.ID(rxn_id_string), self._reactions)
            return reaction.get_elementary_steps()[0].string()

        def _calculate_rate_constant_normalization(self) -> None:
            """
            Determine the rate constant normalization factor for calculating the edge weights.
            Loops over the reactions, converts every barrier to the rate constant and adds it to the sum of rate
            constants.

            The rate constant normalization is then the inverse of the final sum.
            """
            k_sum = sum(sum(self.get_rate_constants(reaction_str_id))
                        for reaction_str_id in self.__valid_reaction_entries.keys())
            if k_sum != 0.0:
                self._rate_constant_normalization = 1 / k_sum

        def _get_weight(self, reaction: db.Reaction) -> Tuple[float, float]:
            """
            Determines the weight for a given reaction.
            The weight is calculated by determining the rate constant from the barrier, normalizing the constant with
            the rate_constant_normalization and taking the abs(log()) of the corresponding product.

            Parameters
            ----------
            reaction : db.Reaction
               The reaction for which the weights should be determined.

            Returns
            -------
            weights : Tuple(float, float)
                The weight for the LHS -> RxnNode and  for the RHS -> RxnNode.
            """
            reaction_str_id = reaction.id().string()
            if reaction_str_id not in self.__valid_reaction_entries:
                raise RuntimeError("The reaction with id {} is not documented in the RMS input file."
                                   .format(reaction_str_id))
            k_lhs, k_rhs = self.get_rate_constants(reaction_str_id)
            return abs(np.log(k_lhs * self._rate_constant_normalization)), \
                abs(np.log(k_rhs * self._rate_constant_normalization))
