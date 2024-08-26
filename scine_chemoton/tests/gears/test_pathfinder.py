#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import pytest
import unittest
import os
import json
import copy
import numpy as np

# Local application tests imports
from ...gears import HoldsCollections

# Third party imports
import scine_database as db
from scine_database.energy_query_functions import rate_constant_from_barrier
from scine_database import test_database_setup as db_setup

# Local application imports
from ..resources import resources_root_path
from ...gears.pathfinder import Pathfinder as pf


class PathfinderTests(unittest.TestCase, HoldsCollections):

    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "reactions", "compounds", "flasks",
                                      "properties"]
        self.initialize_collections(manager)

    def tearDown(self) -> None:
        self._manager.wipe()

    # Capture std out end err
    @pytest.fixture(autouse=True)
    def capsys(self, capsys):
        self.capsys = capsys

    def _add_complete_reaction(self, n_reagents: int = 2, n_products: int = 2,
                               model: db.Model = db.Model("FAKE", "FAKE", "F-AKE")):
        reagents = []
        products = []
        for _ in range(0, n_reagents):
            cmp_id = db_setup._create_compound(1, self._properties, self._compounds,
                                               self._structures, False, (-15.0, -15.0), model)
            reagents.append(cmp_id)

        for _ in range(0, n_products):
            cmp_id = db_setup._create_compound(1, self._properties, self._compounds,
                                               self._structures, False, (-25.0, -25.0), model)
            products.append(cmp_id)
        # # # Create dummy reaction
        reaction = db.Reaction()
        reaction.link(self._reactions)
        reaction.create(reagents, products)
        for compound_id in reagents + products:
            compound = db.Compound(compound_id, self._compounds)
            compound.add_reaction(reaction.get_id())

        return reaction

    def test_pathfinder_initialization(self):
        manager = db_setup.get_clean_db("test_pathfinder_init")
        self.custom_setup(manager)
        finder = pf(manager)
        # Test, if collections are set up correctly
        assert finder._calculations.count(json.dumps({})) == 0
        assert finder._compounds.count(json.dumps({})) == 0
        assert finder._flasks.count(json.dumps({})) == 0
        assert finder._reactions.count(json.dumps({})) == 0
        assert finder._elementary_steps.count(json.dumps({})) == 0
        assert finder._structures.count(json.dumps({})) == 0
        assert finder._properties.count(json.dumps({})) == 0
        # Test default options
        assert finder.options.graph_handler == "basic"
        assert finder.options.barrierless_weight == 1.0
        assert finder.options.model == db.Model("any", "any", "any")

    def test_basic_graph_handler_core(self):
        manager = db_setup.get_clean_db("test_pathfinder_basic_graph_handler")
        self.custom_setup(manager)
        # # # Create dummy reaction
        # Dummy Reactants
        reactant_struct = []
        reactant_cmp = []
        for _ in range(0, 2):
            cmp_id, struct_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
            reactant_struct.append(struct_id)
            reactant_cmp.append(cmp_id)
        # Dummy Products
        product_struct = []
        product_cmp = []
        for _ in range(0, 1):
            cmp_id, struct_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
            product_struct.append(struct_id)
            product_cmp.append(cmp_id)

        reaction = db.Reaction()
        reaction.link(self._reactions)

        el_step = db.ElementaryStep()
        el_step.link(self._elementary_steps)

        el_step.create([r for r in reactant_struct], product_struct)
        reaction.create(
            [r for r in reactant_cmp],
            product_cmp,
            [db.CompoundOrFlask.COMPOUND for _ in reactant_cmp],
            [db.CompoundOrFlask.COMPOUND],
        )
        el_step.set_reaction(reaction.get_id())
        el_step.set_type(db.ElementaryStepType.REGULAR)

        reaction.add_elementary_step(el_step.get_id())
        # # #  Setup pathfinder with basic graph handler
        finder = pf(manager)
        finder.options.barrierless_weight = 0.5
        finder._construct_graph_handler()

        assert finder.graph_handler._valid_reaction(reaction) is False
        # Assign energy to first reactant of elementary step to be a valid reaction
        reactant = db.Structure(el_step.get_reactants(db.Side.LHS)[0][0], self._structures)
        db_setup.add_random_energy(reactant, (10.0, 20.0), self._properties)
        assert finder.graph_handler._valid_reaction(reaction) is True
        assert finder.graph_handler.get_valid_reaction_ids() == [reaction.id()]
        # # # Check weight for regular reaction
        assert finder.graph_handler._get_weight(reaction) == (1.0, 1.0)
        # # # Check weight for barrierless reaction
        el_step.set_type(db.ElementaryStepType.BARRIERLESS)
        assert finder.graph_handler._get_weight(reaction) == (
            finder.options.barrierless_weight, finder.options.barrierless_weight)
        # # # Set model and check validity again
        finder.graph_handler.model = db.Model("FAKE", "FAKE", "F-AKE")
        assert finder.graph_handler._valid_reaction(reaction) is True

    def test_add_reaction_of_basic_graph_handler(self):
        n_compounds = 7
        n_reactions = 3
        max_r_per_c = 7
        max_n_products_per_r = 3
        max_n_educts_per_r = 3
        max_s_per_c = 1
        max_el_steps_per_r = 1
        barrier_limits = (10, 20)
        n_inserts = 2
        n_flasks = 0
        # Construct random database
        manager = db_setup.get_random_db(
            n_compounds,
            n_flasks,
            n_reactions,
            max_r_per_c,
            "test_pathfinder_basic_graph_handler",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_el_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)

        finder = pf(manager)
        finder._construct_graph_handler()

        test_reaction = self._reactions.random_select_reactions(1)[0]
        n_compounds = len(test_reaction.get_reactants(db.Side.LHS)[
                          0]) + len(test_reaction.get_reactants(db.Side.RHS)[1])

        # # # Adding one reaction to graph
        finder.graph_handler.add_reaction(test_reaction)
        # # # Check correct number of nodes and edges
        assert len(finder.graph_handler.graph.nodes) == n_compounds + 2
        assert len(finder.graph_handler.graph.edges) == n_compounds * 2
        assert len([rxn_node for rxn_node in finder.graph_handler.graph.nodes if ";" in rxn_node]) == 2
        # # # Obtain reactants
        reactants = test_reaction.get_reactants(db.Side.BOTH)
        lhs_ids = [cmp.string() for cmp in reactants[0]]
        rhs_ids = [cmp.string() for cmp in reactants[1]]
        # # # Inspect edge
        for edge in finder.graph_handler.graph.edges:
            if ";" in edge[0]:
                assert finder.graph_handler.graph.edges[edge[0], edge[1]]['weight'] == 0.0
            else:
                assert finder.graph_handler.graph.edges[edge[0], edge[1]]['weight'] == 1.0
                assert finder.graph_handler.graph.edges[edge[0], edge[1]]['required_compound_costs'] is None
                # # # Check for correctly assigned required compounds
                if edge[0] in lhs_ids:
                    tmp_lhs_ids = copy.deepcopy(lhs_ids)
                    tmp_lhs_ids.remove(edge[0])
                    if len(tmp_lhs_ids) == 0:
                        assert len(finder.graph_handler.graph.edges[edge[0], edge[1]]['required_compounds']) == 0
                    else:
                        assert finder.graph_handler.graph.edges[edge[0], edge[1]]['required_compounds'].sort() \
                            == tmp_lhs_ids.sort()
                elif edge[0] in rhs_ids:
                    tmp_rhs_ids = copy.deepcopy(rhs_ids)
                    tmp_rhs_ids.remove(edge[0])
                    if len(tmp_rhs_ids) == 0:
                        assert len(finder.graph_handler.graph.edges[edge[0], edge[1]]['required_compounds']) == 0
                    else:
                        assert finder.graph_handler.graph.edges[edge[0], edge[1]]['required_compounds'].sort() \
                            == tmp_rhs_ids.sort()

    def test_barrier_graph_handler_core(self):
        manager = db_setup.get_clean_db("test_pathfinder_barrier_graph_handler")
        self.custom_setup(manager)
        # # # Create dummy reaction
        reaction = self._add_complete_reaction(2, 2)
        el_step_id = db_setup._add_step(reaction, (10.0, 10.0), self._compounds,
                                        self._structures, self._elementary_steps, self._properties)
        reaction.set_elementary_steps([el_step_id])
        # Setup pathfinder
        finder = pf(manager)
        finder.options.model = db.Model("FAKE", "FAKE", "F-AKE")
        finder.graph_handler = pf.BarrierBasedHandler(manager, db.Model("FAKE", "FAKE", "F-AKE"))
        finder.graph_handler._map_elementary_steps_to_reactions()
        assert finder.graph_handler._rxn_to_es_map == {reaction.id().string(): el_step_id}
        # Check temperature setting
        assert finder.graph_handler.get_temperature() == 298.15
        finder.graph_handler.set_temperature(300.0)
        assert finder.graph_handler.get_temperature() == 300.0
        finder.graph_handler.set_temperature(298.15)
        # Check default normalization constant
        assert finder.graph_handler._rate_constant_normalization == 1.0
        # Determine reference weights for 10 and 30 kJ/mol
        k_sum = rate_constant_from_barrier(10.0, 298.15) + rate_constant_from_barrier(30.0, 298.15)
        inv_k_sum = 1 / k_sum
        ref_weights = (abs(np.log(rate_constant_from_barrier(10.0, 298.15) * inv_k_sum)),
                       abs(np.log(rate_constant_from_barrier(30.0, 298.15) * inv_k_sum)))
        # Check normalization of rate constant
        finder.graph_handler._calculate_rate_constant_normalization()
        assert abs(finder.graph_handler._rate_constant_normalization - inv_k_sum) < 1e-12
        # Check weights obtained
        weights = finder.graph_handler._get_weight(reaction)
        assert abs(ref_weights[0] - weights[0]) < 1e-12
        assert abs(ref_weights[1] - weights[1]) < 1e-12

    def test_barrier_graph_handler_filter_negative_barriers(self):
        manager = db_setup.get_clean_db("test_pathfinder_barrier_filter_negative_barriers")
        self.custom_setup(manager)
        # # # Create dummy reaction
        reaction = self._add_complete_reaction(2, 1)
        el_step_id = db_setup._add_step(reaction, (-10.0, -10.0), self._compounds,
                                        self._structures, self._elementary_steps, self._properties)
        reaction.set_elementary_steps([el_step_id])
        # Setup pathfinder
        finder = pf(manager)
        finder.options.model = db.Model("FAKE", "FAKE", "F-AKE")
        finder.options.filter_negative_barriers = True
        finder.graph_handler = pf.BarrierBasedHandler(manager, db.Model("FAKE", "FAKE", "F-AKE"))
        finder.graph_handler.filter_negative_barriers = finder.options.filter_negative_barriers
        finder.graph_handler.barrierless_weight = finder.options.barrierless_weight

        finder.graph_handler._map_elementary_steps_to_reactions()
        # Check, that nothing is build
        assert finder.graph_handler._rxn_to_es_map == {}
        # Check, that barrierless weight is assumed for negative barriers
        finder.graph_handler.filter_negative_barriers = False
        finder.graph_handler._map_elementary_steps_to_reactions()
        assert finder.graph_handler._rxn_to_es_map == {reaction.id().string(): el_step_id}
        # Check normalization of rate constant
        k_sum = finder.graph_handler.barrierless_weight * 2
        inv_k_sum = 1 / k_sum
        ref_weights = (abs(np.log(finder.graph_handler.barrierless_weight * inv_k_sum)),
                       abs(np.log(finder.graph_handler.barrierless_weight * inv_k_sum)))
        finder.graph_handler._calculate_rate_constant_normalization()
        assert abs(finder.graph_handler._rate_constant_normalization - inv_k_sum) < 1e-12
        # Check weights obtained
        weights = finder.graph_handler._get_weight(reaction)
        assert abs(ref_weights[0] - weights[0]) < 1e-12
        assert abs(ref_weights[1] - weights[1]) < 1e-12

    def test_barrier_graph_handler_use_structure_model(self):
        manager = db_setup.get_clean_db("test_pathfinder_barrier_filter_negative_barriers")
        self.custom_setup(manager)
        # # # Create dummy reaction
        reaction = self._add_complete_reaction(2, 1)
        # Add elementary step with negative barrier
        el_step_id = db_setup._add_step(reaction, (-10.0, -10.0), self._compounds,
                                        self._structures, self._elementary_steps, self._properties)
        reaction.add_elementary_step(el_step_id)
        # Add elementary step with positive barriers
        el_pos_step_id = db_setup._add_step(reaction, (10.0, 20.0), self._compounds,
                                            self._structures, self._elementary_steps, self._properties)
        reaction.add_elementary_step(el_pos_step_id)
        # Add structures with wrong model
        for side in reaction.get_reactants(db.Side.BOTH):
            for c_id in side:
                c = db.Compound(c_id, self._compounds)
                c.add_structure(db_setup._fake_structure(c, self._structures, self._properties, False, (-20.0, -10.0),
                                                         db.Model("wrongFake", "wrongFake", "wrongF-ake")))
        # Add elementary step with wrong structure model
        el_model_step_id = db_setup._add_step(reaction, (5.0, 15.0), self._compounds,
                                              self._structures, self._elementary_steps, self._properties,
                                              db.Model("wrongFake", "wrongFake", "wrongF-ake"))
        reaction.add_elementary_step(el_model_step_id)

        # Setup pathfinder
        finder = pf(manager)
        finder.options.model = db.Model("FAKE", "FAKE", "F-AKE")
        finder.options.graph_handler = "barrier"
        finder.options.filter_negative_barriers = True
        finder.options.use_structure_model = True
        finder.options.structure_model = db.Model("FAKE", "FAKE", "F-AKE")
        finder.build_graph()
        assert finder.graph_handler._rxn_to_es_map == {reaction.id().string(): el_pos_step_id}
        # Check that id is in both reaction nodes
        for i in range(0, 2):
            assert finder.graph_handler.graph.nodes(data=True)[reaction.id().string(
            ) + ";" + str(i) + ";"]['elementary_step_id'] == el_pos_step_id.string()

    def test_barrier_graph_handler_for_barrierless(self):
        manager = db_setup.get_clean_db("test_pathfinder_barrierless_graph_handler")
        self.custom_setup(manager)
        # # # Create dummy reaction
        reaction = self._add_complete_reaction(2, 2)
        el_step_id = db_setup._add_step(reaction, (10.0, 10.0), self._compounds,
                                        self._structures, self._elementary_steps, self._properties)
        reaction.set_elementary_steps([el_step_id])
        # # # Create dummy reaction for flask formation
        db_setup._insert_flask(
            reaction.id(),
            db.Side.LHS,
            self._flasks,
            self._structures,
            self._reactions,
            self._elementary_steps,
            self._properties,
            self._compounds)
        # # # Remove non-barrierless reaction
        rxn_list = [id for id in self._reactions.iterate_all_reactions()]
        first_rxn = rxn_list[0]
        first_rxn.link(self._reactions)
        first_rxn.wipe()

        finder = pf(manager)
        finder.options.model = db.Model("FAKE", "FAKE", "F-AKE")

        finder.graph_handler = pf.BarrierBasedHandler(manager, db.Model("FAKE", "FAKE", "F-AKE"))
        finder.graph_handler.barrierless_weight = 1e12
        finder.graph_handler._map_elementary_steps_to_reactions()

        assert finder.graph_handler.barrierless_weight == 1e12
        inv_k_sum = 1 / (2e12)
        finder.graph_handler._calculate_rate_constant_normalization()
        assert finder.graph_handler._rate_constant_normalization == inv_k_sum
        ref_weights = (abs(np.log(finder.graph_handler.barrierless_weight * inv_k_sum)),
                       abs(np.log(finder.graph_handler.barrierless_weight * inv_k_sum)))
        # Check weights obtained
        second_reaction = rxn_list[1]
        second_reaction.link(self._reactions)
        weights = finder.graph_handler._get_weight(second_reaction)
        assert abs(ref_weights[0] - weights[0]) < 1e-12
        assert abs(ref_weights[1] - weights[1]) < 1e-12

    def test_pathfinder_build_graph_and_find_paths(self):
        n_compounds = 7
        n_reactions = 8
        max_r_per_c = 7
        max_n_products_per_r = 3
        max_n_educts_per_r = 3
        max_s_per_c = 1
        max_el_steps_per_r = 1
        barrier_limits = (10, 80)
        n_inserts = 2
        n_flasks = 1
        manager = db_setup.get_random_db(
            n_compounds,
            n_flasks,
            n_reactions,
            max_r_per_c,
            "test_pathfinder_build_graph",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_el_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)
        dummy_finder = pf(manager)
        # Check for raising of RuntimeError
        dummy_finder.options.graph_handler = "dummy"
        self.assertRaises(RuntimeError, dummy_finder._construct_graph_handler)
        # Loop over both handlers
        for handler in dummy_finder.get_valid_graph_handler_options():
            finder = pf(manager)
            finder.options.graph_handler = handler
            # Set additional settings for barrier handler
            if handler == "barrier":
                finder.options.model = db.Model("FAKE", "FAKE", "F-AKE")
                finder.options.barrierless_weight = 1e3
            # Build the graph
            finder.build_graph()
            # Check that the options were correctly forwarded
            assert finder.graph_handler.barrierless_weight == finder.options.barrierless_weight
            assert finder.graph_handler.model == finder.options.model
            # Check correct number of nodes
            assert len([node for node, data in finder.graph_handler.graph.nodes.data(
                True) if data['type'] == 'rxn_node']) == n_reactions * 2
            assert len([node for node, data in finder.graph_handler.graph.nodes.data(
                True) if data['type'] == 'COMPOUND']) == n_compounds
            assert len([node for node, data in finder.graph_handler.graph.nodes.data(
                True) if data['type'] == 'FLASK']) == n_flasks

            compound_nodes = [node for node, data in finder.graph_handler.graph.nodes.data(
                True) if data['type'] == 'COMPOUND']
            simple_paths = finder.find_paths(compound_nodes[0], compound_nodes[-1], 10)
            if handler == "basic":
                # Check shortest path to have at least length 1.0 in basic handler
                assert simple_paths[0][1] >= 1.0
            # Check start and end node of path are source and target
            assert simple_paths[0][0][0] == compound_nodes[0]
            assert simple_paths[0][0][-1] == compound_nodes[-1]
            # Check for path containing one more compound node than reaction node
            assert len([node for node in simple_paths[0][0] if ';' not in node]) == len(
                [node for node in simple_paths[0][0] if ';' in node]) + 1
            # The following checks are only relevant if more than one path is found
            if len(simple_paths) > 1:
                assert simple_paths[0][1] <= simple_paths[1][1]
                # Check that skip of shortest path
                simple_paths_skip = finder.find_paths(compound_nodes[0], compound_nodes[-1], 2, 1)
                assert simple_paths[1] == simple_paths_skip[0]
            unique_simple_paths = finder.find_unique_paths(compound_nodes[0], compound_nodes[-1], 10)
            # The following checks are only relevant if more than one path is found
            if len(unique_simple_paths) > 1:
                # Check if iterator is correctly stored and retrieved
                finder.find_unique_paths(compound_nodes[0], compound_nodes[-1], 1)
                finder._use_old_iterator = True
                assert finder.find_unique_paths(compound_nodes[0], compound_nodes[-1], 1)[0] == unique_simple_paths[1]
            # Check if asking for more paths than possible works
            assert len(finder.find_paths(compound_nodes[0], compound_nodes[-1], 1000)) < 1000
            assert len(finder.find_unique_paths(compound_nodes[0], compound_nodes[-1], 1000)) < 1000

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pathfinder_compound_costs(self):
        manager = db_setup.get_clean_db("test_pathfinder_compound_costs")
        self.custom_setup(manager)

        reaction = self._add_complete_reaction(2, 1)
        el_step_id = db_setup._add_step(reaction, (10.0, 10.0), self._compounds,
                                        self._structures, self._elementary_steps, self._properties)
        reaction.set_elementary_steps([el_step_id])

        # Add one isolated reaction
        isolated_reaction = self._add_complete_reaction(1, 1)
        el_step_id = db_setup._add_step(isolated_reaction, (10.0, 10.0), self._compounds,
                                        self._structures, self._elementary_steps, self._properties)
        isolated_reaction.set_elementary_steps([el_step_id])

        # Trigger RuntimeError if no starting conditions set or graph without nodes
        dummy_finder = pf(manager)
        dummy_finder._construct_graph_handler()
        self.assertRaises(RuntimeError, dummy_finder.calculate_compound_costs)
        dummy_finder.start_compounds_set = True
        self.assertRaises(RuntimeError, dummy_finder.calculate_compound_costs)

        finder = pf(manager)
        finder.build_graph()
        n_compounds = [node for node in finder.graph_handler.graph.nodes if ';' not in node]
        n_rxn_nodes = [node for node in finder.graph_handler.graph.nodes if ';' in node]

        # Check default for start_compounds_set
        assert finder.start_compounds_set is False
        start_conditions = {
            n_compounds[0]: 0.7,
        }
        # Check setting of starting conditions
        finder.set_start_conditions(start_conditions)
        assert finder.start_compounds_set is True
        assert finder.start_compounds == n_compounds[0:1]
        assert finder.compound_costs == start_conditions

        # Check failed attempt for determining compound costs
        finder.calculate_compound_costs()
        assert finder.compound_costs_solved is False
        assert finder.compound_costs == {n_compounds[0]: 0.7,
                                         n_compounds[1]: finder._pseudo_inf,
                                         n_compounds[2]: finder._pseudo_inf,
                                         n_compounds[3]: finder._pseudo_inf,
                                         n_compounds[4]: finder._pseudo_inf}

        # Check resetting graph without compound costs raises error
        assert finder.graph_updated_with_compound_costs is False
        self.assertRaises(RuntimeError, finder.reset_graph_compound_costs)

        # Check updating of graph with compound costs warns thee user if not all solved
        with self.assertWarns(UserWarning) as w_handler:
            finder.update_graph_compound_costs()
        assert ("The following compounds have no cost assigned:" in str(w_handler.warning))
        assert finder.graph_updated_with_compound_costs
        finder.reset_graph_compound_costs()
        assert finder.graph_updated_with_compound_costs is False

        # Check updating of start conditions
        start_conditions[n_compounds[1]] = 0.4
        finder.set_start_conditions(start_conditions)
        assert finder.start_compounds == n_compounds[0:2]
        assert finder.compound_costs == start_conditions

        # Check extraction of connected subgraph, second reaction should not be in it anymore
        connected_subgraph = finder.extract_connected_graph(list(start_conditions.keys()))
        assert len(connected_subgraph.nodes) == 5
        assert n_compounds[3] not in list(connected_subgraph.nodes)
        assert n_rxn_nodes[2] not in list(connected_subgraph.nodes)

        # Check successful attempt for determining compound costs, the third compound has the cost of 1 + 0.4 + 0.7
        finder.calculate_compound_costs()
        assert finder.compound_costs_solved is False
        assert finder.compound_costs[n_compounds[2]] == 1.0 + 0.4 + 0.7

        # Check correct update of weights in graph
        finder.update_graph_compound_costs()
        assert finder.graph_updated_with_compound_costs is True
        assert finder.graph_handler.graph.edges[n_compounds[0], n_rxn_nodes[0]]["required_compound_costs"] == 0.4
        assert finder.graph_handler.graph.edges[n_compounds[0], n_rxn_nodes[0]]["weight"] == 1.4
        assert finder.graph_handler.graph.edges[n_compounds[1], n_rxn_nodes[0]]["required_compound_costs"] == 0.7
        assert finder.graph_handler.graph.edges[n_compounds[1], n_rxn_nodes[0]]["weight"] == 1.7

        # Check correct conversion of path to elementary step string with colors and line break
        path = finder.find_paths(n_compounds[0], n_compounds[2], 1)[0]
        ref_string = "\033[31mH2O(c:0, m:1)\033[0m + H2O(c:0, m:1) -> \033[36mH2O(c:0, m:1)\033[0m\n"
        assert finder.get_elementary_step_sequence(path[0]) == ref_string
        ref_overall_equation = "1 H2O(c:0, m:1) + 1 H2O(c:0, m:1) = 1 H2O(c:0, m:1)"
        assert finder.get_overall_reaction_equation(path[0]) == ref_overall_equation

        # Check correct resetting of starting conditions
        start_conditions = {
            n_compounds[0]: 1.4,
            n_compounds[1]: 1.7
        }
        assert finder.start_compounds_set
        finder.set_start_conditions(start_conditions)
        finder.calculate_compound_costs()
        assert finder.compound_costs_solved is False
        assert finder.compound_costs[n_compounds[2]] == 1.0 + 1.4 + 1.7
        # Check correct update of weights in graph
        finder.update_graph_compound_costs()
        assert finder.graph_handler.graph.edges[n_compounds[0], n_rxn_nodes[0]]["required_compound_costs"] == 1.7
        assert abs(finder.graph_handler.graph.edges[n_compounds[0], n_rxn_nodes[0]]["weight"] - 2.7) < 1e-12

        # Check resetting of graph
        finder.reset_graph_compound_costs()
        assert finder.compound_costs_solved is False
        assert finder.graph_updated_with_compound_costs is False
        assert abs(finder.graph_handler.graph.edges[n_compounds[0], n_rxn_nodes[0]]["required_compound_costs"]
                   - 0.0) < 1e-12
        assert abs(finder.graph_handler.graph.edges[n_compounds[0], n_rxn_nodes[0]]["weight"] - 1.0) < 1e-12

        # Check that correct warnings are raised when calling update twice
        finder.update_graph_compound_costs()

        # # Check that 2 warnings are raised when updating the graph again
        with self.assertWarns(UserWarning) as w_handler:
            finder.update_graph_compound_costs()
        assert ("The following compounds have no cost assigned:" in str(w_handler.warnings[0]))
        assert ("The graph has been updated with compound costs previously" in str(w_handler.warnings[1]))

    def test_pathfinder_export_to_files(self):
        manager = db_setup.get_clean_db("test_pathfinder_export")
        self.custom_setup(manager)

        reaction = self._add_complete_reaction(2, 1)
        el_step_id = db_setup._add_step(reaction, (10.0, 10.0), self._compounds,
                                        self._structures, self._elementary_steps, self._properties)
        reaction.set_elementary_steps([el_step_id])

        finder = pf(manager)
        finder.options.graph_handler = "barrier"
        finder.options.model = db.Model("FAKE", "FAKE", "F-AKE")
        finder.options.barrierless_weight = 1e12
        finder.build_graph()

        n_compounds = [node for node in finder.graph_handler.graph.nodes if ';' not in node]
        # Check correct resetting of starting conditions
        start_conditions = {
            n_compounds[0]: 0.4,
            n_compounds[1]: 0.7
        }
        finder.set_start_conditions(start_conditions)
        finder.calculate_compound_costs()
        assert finder.compound_costs_solved is True
        finder.update_graph_compound_costs()
        assert finder.graph_updated_with_compound_costs is True
        graph_path = os.path.join(resources_root_path(), "test_graph.json")
        # Export to file
        finder.export_graph(graph_path)
        assert os.path.isfile(graph_path)

        loaded_graph = finder._import_graph(graph_path)
        loaded_graph_nodes = [node for node in loaded_graph.nodes(data=True)]
        loaded_graph_edges = [edge for edge in loaded_graph.edges(data=True)]

        # Copy to compare with exported graph
        finder_export = copy.copy(finder)
        finder_export.reset_graph_compound_costs()

        # Check nodes in graph
        for index, node in enumerate(finder_export.graph_handler.graph.nodes(data=True)):
            assert node == loaded_graph_nodes[index]
        # Check edges in graph
        for index, edge in enumerate(finder_export.graph_handler.graph.edges(data=True)):
            assert edge == loaded_graph_edges[index]

        cc_path = os.path.join(resources_root_path(), "test_cc.json")
        finder.export_compound_costs(cc_path)
        assert os.path.isfile(cc_path)
        assert finder.compound_costs == finder._import_dictionary(cc_path)

        # # # Test loading into Pathfinder

        loaded_finder = pf(manager)
        loaded_finder.load_graph(graph_path)
        # Check for default basic graph handler
        assert loaded_finder.graph_handler.graph
        assert loaded_finder.options.graph_handler == "basic"
        assert loaded_finder.options.model == db.Model("any", "any", "any")  # None

        for node_org, node_loaded in zip(finder.graph_handler.graph.nodes(data=True),
                                         loaded_finder.graph_handler.graph.nodes(data=True)):
            assert node_org == node_loaded

        for edge_org, edge_loaded in zip(finder.graph_handler.graph.edges(data=True),
                                         loaded_finder.graph_handler.graph.edges(data=True)):
            assert edge_org == edge_loaded

        # Check that loaded graph is really copied
        assert loaded_graph is not loaded_finder.graph_handler.graph

        # Test loading graph and compound costs directly

        loaded_complete_finder = pf(manager)
        loaded_complete_finder.load_graph(graph_path, cc_path)

        assert loaded_complete_finder.graph_handler.graph
        assert loaded_complete_finder.compound_costs
        assert loaded_complete_finder.compound_costs_solved is True
        assert loaded_complete_finder.compound_costs == loaded_complete_finder._import_dictionary(cc_path)
        assert loaded_complete_finder.graph_updated_with_compound_costs is True

        os.remove(graph_path)
        os.remove(cc_path)
