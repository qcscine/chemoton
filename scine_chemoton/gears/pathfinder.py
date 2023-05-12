#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import Optional, Iterator, List, Tuple, Union, Dict
import networkx as nx
import numpy as np
import json
import sys
from itertools import islice

# Third party imports
import scine_database as db

# Local application imports
from ..utilities.energy_query_functions import get_energy_for_structure, get_barriers_for_elementary_step_by_type, \
    get_elementary_step_with_min_ts_energy, rate_constant_from_barrier
from ..utilities.get_molecular_formula import get_molecular_formula_of_aggregate


class Pathfinder:
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
        Bool to indicate if the old iterator shall be used querying for paths between a source - target pair.
    _unique_iterator_memory : Tuple[Tuple[List[str], float], Iterator]
        Memory of iterator with the corresponding path and its length as well as the iterator.
    start_compounds : List[str]
        A list containing the compounds which are present at the start.
    start_compounds_set : bool
        Bool to indicate if start_compounds are set.
    _pseudo_inf : float
        Float for edges with infinite weight.
    compound_costs : Dict[str, float]
        A dictionary containing the cost of the compounds with the compounds as keys.
    compound_costs_solved : bool
        Bool to indicate if all compounds have a compound cost.
    """

    def __init__(self, db_manager: db.Manager):
        self.options = self.Options()
        self.manager = db_manager
        # Get required collections
        self._calculations = db_manager.get_collection('calculations')
        self._compounds = db_manager.get_collection('compounds')
        self._flasks = db_manager.get_collection("flasks")
        self._reactions = db_manager.get_collection('reactions')
        self._elementary_steps = db_manager.get_collection('elementary_steps')
        self._structures = db_manager.get_collection('structures')
        self._properties = db_manager.get_collection('properties')

        self.graph_handler: Union[Pathfinder.BasicHandler, Pathfinder.BarrierBasedHandler, None] = None
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

    class Options:
        """
        A class to vary the setup of Pathfinder.
        """
        __slots__ = {"graph_handler", "barrierless_weight", "model", "filter_negative_barriers", "use_structure_model",
                     "structure_model", "energy_threshold"}

        def __init__(self):
            self.graph_handler: str = "basic"  # pylint: disable=no-member
            """
            A string indicating which graph handler to be used.
            """
            self.barrierless_weight: float = 1.0  # 0.01
            """
            The weight for barrierless reactions (basic) and rate constant (barrier), respectively.
            """
            self.model: Union[None, db.Model] = None
            """
            The model for the compounds to be included.
            """
            # in kJ / mol
            self.filter_negative_barriers: bool = False

            self.use_structure_model: bool = False

            self.structure_model: Union[None, db.Model] = None
            self.energy_threshold: float = 100.0

    @staticmethod
    def get_valid_graph_handler_options() -> List[str]:
        return ["basic", "barrier"]

    def _construct_graph_handler(self):
        """
        Constructor for the graph handler.
        Transfers pathfinder.options to graph handler.

        Raises
        ------
        RuntimeError
            Invalid options for graph handler.
        """
        if not self.graph_handler:
            if self.options.graph_handler not in self.get_valid_graph_handler_options():
                raise RuntimeError("Invalid graph handler option.")
            if self.options.graph_handler == "basic":
                self.graph_handler = self.BasicHandler(self.manager, self.options.model, self.options.structure_model)
                self.graph_handler.barrierless_weight = self.options.barrierless_weight
            elif self.options.graph_handler == "barrier":
                self.graph_handler = self.BarrierBasedHandler(
                    self.manager, self.options.model, self.options.structure_model)
                self.graph_handler.barrierless_weight = self.options.barrierless_weight
                self.graph_handler.filter_negative_barriers = self.options.filter_negative_barriers
                self.graph_handler._map_elementary_steps_to_reactions()
                self.graph_handler._calculate_rate_constant_normalization()

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
        self._construct_graph_handler()
        assert self.graph_handler
        for rxn_id in self.graph_handler.get_valid_reaction_ids():
            rxn = db.Reaction(rxn_id, self._reactions)
            self.graph_handler.add_reaction(rxn)

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
            and ``n_skipped_paths=2``, the third, fourth, fifth and sixth path are returned. Therefore, this allows to
            set the starting point of the query.

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

    def find_unique_paths(self, source: str, target: str, number: int = 3) -> List[Tuple[List[str], float]]:
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
            path_iterator = iter(nx.shortest_simple_paths(self.graph_handler.graph, source, target, weight="weight"))
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
                    if self.graph_handler.graph.nodes[aggregate_id]['type'] == db.CompoundOrFlask.COMPOUND.name:
                        aggregate_type = db.CompoundOrFlask.COMPOUND
                    elif self.graph_handler.graph.nodes[aggregate_id]['type'] == db.CompoundOrFlask.FLASK.name:
                        aggregate_type = db.CompoundOrFlask.FLASK

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

    def get_overall_reaction_equation(self, path: List[str]) -> str:
        """
        Summarize a given path to a reaction equation and return its string.
        Count the appearance of compounds in a reaction, -1 for reactants and +1 for products.
        Returns the factor and the compound as molecular formula.

        Parameters
        ----------
        path : List[str]
            Path containing a list of the traversed nodes.

        Returns
        -------
        str
            A string of the overall reaction equation of a given path.
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

        reaction_equation = ""
        for i, side in enumerate([reactants, products]):
            for j, cmp_count in enumerate(side):
                aggregate_id = cmp_count[0]
                # # # Identify Compound or Flask
                if self.graph_handler.graph.nodes[aggregate_id]['type'] == db.CompoundOrFlask.COMPOUND.name:
                    aggregate_type = db.CompoundOrFlask.COMPOUND
                elif self.graph_handler.graph.nodes[aggregate_id]['type'] == db.CompoundOrFlask.FLASK.name:
                    aggregate_type = db.CompoundOrFlask.FLASK

                aggregate_str = get_molecular_formula_of_aggregate(
                    db.ID(aggregate_id), aggregate_type, self._compounds, self._flasks, self._structures)

                reaction_equation += str(cmp_count[1]) + " " + aggregate_str
                if j < len(side) - 1:
                    reaction_equation += " + "
            if i == 0:
                reaction_equation += " = "

        return reaction_equation

    @staticmethod
    def _k_shortest_paths(graph: nx.DiGraph, source: str, target: str, n_paths: int, weight: Union[str, None] = None,
                          path_start: int = 0) -> List[List[str]]:
        """
        Finding k shortest paths between a source and target node in a given graph.
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
        cmps_to_check = [n for n in self.graph_handler.graph.nodes if ';' not in n
                         and n not in self.start_compounds]
        # # # Set all compounds to be checked to pseudo inf cost
        for cmp_id in cmps_to_check:
            self.compound_costs[cmp_id] = self._pseudo_inf
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
        while (not converged):
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
                    # # # Check for value in compound_costs dict and
                    if (self.compound_costs[target] != self._pseudo_inf and
                            10e-6 < self.compound_costs[target] - tmp_cost):
                        compound_costs_opt_change += 1
                    # # # Not already set check
                    if self.compound_costs[target] == self._pseudo_inf or (
                            self.compound_costs != self._pseudo_inf and 10e-6 < self.compound_costs[target] - tmp_cost):
                        # # # Not discovered in current run OR new tmp cost lower than stored cost
                        if (target not in tmp_compound_costs.keys()):
                            tmp_compound_costs[target] = tmp_cost
                        elif (tmp_cost < tmp_compound_costs[target]):
                            tmp_compound_costs[target] = tmp_cost
                # # # Remove targets for no path could be found
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
            sys.stderr.write("Warning: The following compounds have no cost assigned:\n" + str(unsolved_cmp) +
                             "\nGraph will be updated anyway, but maybe reconsider the starting conditions.\n")
        # # # Check if graph has been updated
        if self.graph_updated_with_compound_costs:
            raise Warning("The graph has been updated previously.")
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
        Export graph as dictionary to .json file.

        Parameters
        ----------
        filename : str, optional
            Name of the file to write graph into, by default "graph.json".
        """
        assert self.graph_handler
        graph_as_dict = nx.convert.to_dict_of_dicts(self.graph_handler.graph)
        # # # Add node type to dictionary
        for node in self.graph_handler.graph.nodes:
            graph_as_dict[node]['type'] = self.graph_handler.graph.nodes[node]['type']
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

    class BasicHandler:
        """
        A basic class to handle the construction of the nx.DiGraph.
        A list of reactions can be added differently, depending on the implementation of ``_get_weight`` and
        ``get_valid_reaction_ids``.

        Attributes
        ----------
        graph : nx.DiGraph
            The directed graph.
        db_manager : db.Manager
            The database manager for the connected database.
        barrierless_weight : float
            The weight to be set for barrierless reactions.
        model : Union[None, db.Model]
            A model for filtering the valid reactions.
            Per default None, reactions are included regardless of the model.
        """

        def __init__(self, manager: db.Manager, model: Optional[db.Model] = None,
                     structure_model: Optional[db.Model] = None):
            self.graph = nx.DiGraph()
            self.db_manager = manager
            self.barrierless_weight = 1.0
            self.model: Union[None, db.Model] = model
            self.filter_negative_barriers = False
            self.use_structure_model = False
            self.structure_model: Union[None, db.Model] = structure_model
            if self.structure_model is not None:
                self.use_structure_model = True
            # Collections
            self._compounds = self.db_manager.get_collection("compounds")
            self._flasks = self.db_manager.get_collection("flasks")
            self._structures = self.db_manager.get_collection("structures")
            self._properties = self.db_manager.get_collection("properties")
            self._reactions = self.db_manager.get_collection("reactions")
            self._elementary_steps = self.db_manager.get_collection("elementary_steps")

        def add_reaction(self, reaction: db.Reaction):
            """
            Add a reaction to the graph.
            Each reaction node represents the LHS and RHS.
            Hence every reagent of a reaction is connected to every product of a reaction via one reaction node.

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

            The edges from a compound node to a reaction node contain several information:
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
            rxn_nodes = []
            reaction_id = reaction.id().string()

            for i in range(0, 2):
                # Add rxn node between lhs and rhs compound
                rxn_node = ';'.join([reaction_id, str(i)])
                rxn_node += ';'
                self.graph.add_node(rxn_node, type='rxn_node')
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
                self.graph.add_edge(lhs_cmp, rxn_nodes[0], weight=weights[0], required_compounds=required_cmps_lhs,
                                    required_compound_costs=None)
                self.graph.add_edge(rxn_nodes[1], lhs_cmp, weight=0.0, required_compounds=None)
            # Add rhs aggregates and connect
            for rhs_cmp, rhs_type in zip([i.string() for i in reactants[1]],
                                         [i.name for i in reactant_types[1]]):
                if rhs_cmp not in self.graph:
                    self.graph.add_node(rhs_cmp, type=rhs_type)
                required_cmps_rhs = [s.string() for s in reactants[1]]
                required_cmps_rhs.remove(rhs_cmp)
                self.graph.add_edge(rhs_cmp, rxn_nodes[1], weight=weights[1], required_compounds=required_cmps_rhs,
                                    required_compound_costs=None)
                self.graph.add_edge(rxn_nodes[0], rhs_cmp, weight=0.0, required_compounds=None)

            # # # Loop over reaction nodes to add required compounds info to downwards edges; might be unnecessary
            node_index = 1
            for node in rxn_nodes:
                for key in self.graph[node].keys():
                    self.graph.edges[node, key]['required_compounds'] = \
                        self.graph.edges[key, rxn_nodes[node_index]]['required_compounds']
                node_index -= 1

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
            for step in reaction.get_elementary_steps(self.db_manager):
                # # # Barrierless weights for barrierless reactions
                if step.get_type() == db.ElementaryStepType.BARRIERLESS:
                    return self.barrierless_weight, self.barrierless_weight
            return 1.0, 1.0

        def get_valid_reaction_ids(self) -> List[db.ID]:
            """
            Basic filter function for reactions.
            Per default it returns all reactions.

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

        def _valid_reaction(self, reaction: db.Reaction) -> bool:
            """
            Checks if a given reaction is valid.
            A reaction is considered valid if at least one elementary step of the reaction has an electronic energy
            assigned to its first structure, if required calculated with the set db.Model.

            If 'use_structure_model' is set to True, the elementary steps are checked if the first structure of the
            elementary step has the required structure model.

            If 'filter_negative_barriers' is set to True, the elementary step is checked if the electronic energy
            barrier in any direction is None or negative.

            Parameters
            ----------
            reaction : db.Reaction
                The reaction which shall be checked for one valid elementary step.

            Returns
            -------
            bool
                Bool indicating if the reaction is valid.
            """
            reaction.link(self._reactions)

            for es_id in reaction.get_elementary_steps():
                valid_structures = False
                valid_barriers = True
                es = db.ElementaryStep(es_id, self._elementary_steps)
                first_structure_lhs = db.Structure(es.get_reactants()[0][0], self._structures)
                # # # Model Check
                # Check if model is not specified (None)
                if self.model is None:  # type: ignore
                    if len(first_structure_lhs.get_properties("electronic_energy")) > 0:
                        return True
                # Check, if energy of this structure with specified model exists
                else:
                    assert self.model
                    if self.use_structure_model:
                        assert self.structure_model
                    # Structure model check if wanted here, structure must have model
                    if self.use_structure_model and \
                       self.structure_model != first_structure_lhs.get_model():  # type: ignore
                        # # # Skip structure if the structure model does not fit the set model
                        continue
                    first_structure_lhs_e = get_energy_for_structure(
                        first_structure_lhs, "electronic_energy", self.model, self._structures, self._properties)
                    # # # Structure validity check
                    if first_structure_lhs_e is not None:
                        valid_structures = True
                    if self.filter_negative_barriers:
                        barriers = get_barriers_for_elementary_step_by_type(es, "electronic_energy", self.model,
                                                                            self._structures, self._properties)
                        if None in barriers or (barriers[0] < 0.0 or barriers[1] < 0.0):  # type: ignore
                            valid_barriers = False
                    # # # Final structure and barrier check
                    if valid_structures and valid_barriers:
                        return True
            # If all elementary steps of this reaction fail the checks, this reaction is not valid
            return False

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
        _rxn_to_es_map : Dict[str, db.ID]
            A dictionary holding the ID of the elementary step with the lowest TS energy for each reaction.
        """

        def __init__(self, db_manager: db.Manager, model: db.Model, structure_model: Union[None, db.Model] = None):
            super().__init__(db_manager, model, structure_model)
            self.temperature = 298.15
            self.check_barriers = False
            self._rate_constant_normalization = 1.0
            self._rxn_to_es_map: Dict[str, db.ID] = {}

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

        def get_valid_reaction_ids(self):
            return [db.ID(key) for key in self._rxn_to_es_map.keys()]

        def _get_valid_reaction_ids(self):

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
                es_id = get_elementary_step_with_min_ts_energy(
                    reaction,
                    "gibbs_free_energy",
                    self.model,
                    self._elementary_steps,
                    self._structures,
                    self._properties,
                    self.structure_model)
                # # # If unsuccessful, attempt electronic energy
                if es_id is None:
                    es_id = get_elementary_step_with_min_ts_energy(
                        reaction,
                        "electronic_energy",
                        self.model,
                        self._elementary_steps,
                        self._structures,
                        self._properties,
                        self.structure_model)
                    # # # If es_id still None, continue; pure safety measure
                    if es_id is None:
                        continue
                # Enforce non-negative barriers
                if self.filter_negative_barriers:
                    es = db.ElementaryStep(es_id, self._elementary_steps)
                    barriers = get_barriers_for_elementary_step_by_type(es, "electronic_energy", self.model,
                                                                        self._structures, self._properties)
                    if None in barriers or (barriers[0] < 0.0 or barriers[1] < 0.0):
                        es_id = self._get_elementary_step_with_min_ts_energy_and_non_negative_barriers(reaction)
                # # # Write elementary step to map
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
                    # # # Retrieve barriers of elementary step in kJ/mol
                    barriers = get_barriers_for_elementary_step_by_type(
                        es, "gibbs_free_energy", self.model, self._structures, self._properties)
                    if None in barriers:
                        barriers = get_barriers_for_elementary_step_by_type(
                            es, "electronic_energy", self.model, self._structures, self._properties)
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

        def _get_elementary_step_with_min_ts_energy_and_non_negative_barriers(self, reaction: db.Reaction):
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
