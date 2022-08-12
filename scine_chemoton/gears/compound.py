#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
from typing import Union, Tuple, Dict, List
from warnings import warn

# Third party imports
import scine_database as db
import scine_utilities as utils
import scine_molassembler as masm

# Local application imports
from . import Gear
from ..utilities.queries import stop_on_timeout, calculation_exists_in_structure
from ..utilities.calculation_creation_helpers import finalize_calculation


class BasicAggregateHousekeeping(Gear):
    """
    This Gear updates all relevant Structures stored in the database with
    bond orders and graph representations.
    This data is then used to sort each Structure into existing Compounds/Flasks or to
    create a new Compound/Flask if no appropriate one exists.

    Attributes
    ----------
    options : BasicAggregateHousekeeping.Options
        The options for the BasicAggregateHousekeeping Gear.

    Notes
    -----
    Checks for all 'user_optimized', 'minimum_optimized', and 'complex_optimized' Structures
    that do not have an Aggregate assigned. The Gear then generates bond orders and molecular graphs
    ('masm_cbor_graph', and 'masm_decision_list') if they are not yet present.
    Using the molecular graphs the Structures are then sorted into Compounds/Flasks.
    """

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self._required_collections = ["calculations", "compounds", "flasks",
                                      "properties", "structures", "reactions"]
        self._compound_map: Dict[int, Dict[Tuple[int, int, int], List[db.ID]]] = dict()
        """
        Caching map from n_atoms, n_molecules, charge, multiplicity to compound ids.
        """
        self._flask_map: Dict[int, Dict[Tuple[int, int, int], List[db.ID]]] = dict()
        """
        Caching map from n_atoms, n_molecules, charge, multiplicity to flask ids.
        """
        self._unique_structures: Dict[str, List[Tuple[db.ID, str]]] = dict()
        """
        Caching map from compound id to list of structure ids and decision lists.
        """

    class Options:
        """
        The options for the BasicAggregateHouseKeeping Gear.
        """

        __slots__ = [
            "cycle_time",
            "model",
            "bond_order_job",
            "bond_order_settings",
            "graph_job",
            "graph_settings",
            "exclude_misguided_conformer_optimizations"
        ]

        def __init__(self):
            self.cycle_time = 10
            """
            int
                The minimum number of seconds between two cycles of the Gear.
                Cycles are finished independent of this option, thus if a cycle
                takes longer than the cycle_time will effectively lead to longer
                cycle times and not cause multiple cycles of the same Gear.
            """
            self.model: db.Model = db.Model("PM6", "PM6", "")
            """
            db.Model (Scine::Database::Model)
                The Model used for the bond order calculations.
                The default is: PM6 using Sparrow.
            """
            self.bond_order_job: db.Job = db.Job("scine_bond_orders")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for the bond order calculations.
                The default is: the 'scine_bond_orders' order on a single core.
            """
            self.bond_order_settings: utils.ValueCollection = utils.ValueCollection()
            """
            utils.ValueCollection
                Additional settings passed to the bond order calculations.
                Empty by default.
            """
            self.graph_job: db.Job = db.Job("graph")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for the graph calculations.
                The default is: the 'graph' order on a single core.
            """
            self.graph_settings: utils.ValueCollection = utils.ValueCollection()
            """
            Dict[str, str]
                Additional settings passed to the graph calculations.
                Empty by default.
            """
            self.exclude_misguided_conformer_optimizations: bool = True
            """
            bool
                If true, no additional aggregate is created if the structure was generated only by geometry
                optimization.
            """

    def _create_unique_structure_map(self):
        for compound in stop_on_timeout(self._compounds.iterate_all_compounds()):
            compound.link(self._compounds)
            structure_ids = compound.get_structures()
            key = compound.id().string()
            self._unique_structures[key] = []
            for s_id in structure_ids:
                structure = db.Structure(s_id, self._structures)
                if structure.get_label() in [db.Label.DUPLICATE, db.Label.MINIMUM_GUESS]:
                    continue
                self._unique_structures[key].append((s_id, structure.get_graph("masm_decision_list")))

    def _add_to_unique_structures(self, structure: db.Structure):
        key = structure.get_aggregate().string()
        if key in self._unique_structures:
            self._unique_structures[key].append((structure.id(), structure.get_graph("masm_decision_list")))
        else:
            self._unique_structures[key] = [(structure.id(), structure.get_graph("masm_decision_list"))]

    def _create_compound_map(self):
        for compound in stop_on_timeout(self._compounds.iterate_all_compounds()):
            compound.link(self._compounds)
            self._add_to_map(compound, self._compound_map)

    def _create_flask_map(self):
        for flask in stop_on_timeout(self._flasks.iterate_all_flasks()):
            flask.link(self._flasks)
            self._add_to_map(flask, self._flask_map)

    def _add_to_map(self, aggregate: Union[db.Compound, db.Flask], cache_map: Dict):
        n_atoms, n_molecules, charge, multiplicity, graph = self._get_aggregate_info(aggregate)
        key = (n_molecules, charge, multiplicity)
        if n_atoms in cache_map:
            if key in cache_map[n_atoms]:
                cache_map[n_atoms][key].append((aggregate.id(), graph))
                return
            else:
                cache_map[n_atoms][key] = [(aggregate.id(), graph)]
                return
        cache_map[n_atoms] = {key: [(aggregate.id(), graph)]}

    def _get_aggregate_info(self, aggregate: Union[db.Compound, db.Flask]) -> Tuple[int, int, int, int, str]:
        # We could even think about adding the graph here too ...
        centroid = db.Structure(aggregate.get_centroid(), self._structures)
        graph = centroid.get_graph("masm_cbor_graph")
        return len(centroid.get_atoms()), len(graph.split(";")), centroid.get_charge(), centroid.get_multiplicity(),\
            graph

    @staticmethod
    def _get_aggregate_from_map(n_atoms: int, n_molecules: int, charge: int, multiplicity: int, graph,
                                cache_map: Dict) -> Union[db.ID, None]:
        if n_atoms in cache_map:
            key = (n_molecules, charge, multiplicity)
            if key in cache_map[n_atoms]:
                for a_id, ref_graph in cache_map[n_atoms][key]:
                    if masm.JsonSerialization.equal_molecules(ref_graph, graph):
                        return a_id
        return None

    def _loop_impl(self):
        if not self._compound_map:
            self._create_compound_map()
        if not self._flask_map:
            self._create_flask_map()
        if not self._unique_structures:
            self._create_unique_structure_map()
        self._check_for_aggregates()
        self._check_compounds_in_flasks()

    def _check_compounds_in_flasks(self):
        """
        Check for Flasks without Compounds.
        Complete them if ALL sub-structures are assigned a Compound.
        """
        # Check for flasks without compounds
        selection = {
            "compounds": {"$size": 0}
        }
        for flask in stop_on_timeout(self._flasks.iterate_flasks(dumps(selection))):
            flask.link(self._flasks)
            flask_centroid = db.Structure(flask.get_centroid(), self._structures)
            if not flask_centroid.has_graph("masm_cbor_graph") or not flask_centroid.has_graph("masm_idx_map"):
                warn(f"Centroid {str(flask_centroid.id())} of flask {str(flask.id())} is missing a graph")
                continue
            graphs = flask_centroid.get_graph("masm_cbor_graph").split(';')
            compounds = []
            n_atoms_per_molecule = self.get_n_atoms(flask_centroid)
            n_graphs_missing = len(n_atoms_per_molecule)
            for i, n_atoms in enumerate(n_atoms_per_molecule):
                if n_atoms not in self._compound_map:
                    break
                for key in self._compound_map[n_atoms]:
                    compound_list = self._compound_map[n_atoms][key]
                    for c_id, current in compound_list:
                        # we are looping in reverse, so we can pop the element if identical while looping
                        # we loop through because one compound could be part of flask multiple times
                        # e.g. solute with 2 water molecules
                        if masm.JsonSerialization.equal_molecules(graphs[i], current):
                            compounds.append(c_id)
                            n_graphs_missing -= 1
                        if n_graphs_missing == 0:
                            flask.set_compounds(compounds)
                            break

    def _check_for_aggregates(self):
        # Setup query for optimized structures without aggregate
        selection = {
            "$and": [
                {
                    "$or": [
                        {"label": "complex_optimized"},
                        {"label": "minimum_optimized"},
                        {"label": "user_optimized"},
                    ]
                },
                {"aggregate": ""},
                {"analysis_disabled": {"$ne": True}},
            ]
        }
        # Loop over all results
        for structure in stop_on_timeout(self._structures.iterate_structures(dumps(selection))):
            structure.link(self._structures)
            if structure.has_graph("masm_cbor_graph"):
                # Check if graph exists
                graph = structure.get_graph("masm_cbor_graph")
                label = structure.get_label()
                if ";" in graph:
                    if label != db.Label.COMPLEX_OPTIMIZED:
                        warn(f"Structure '{str(structure.id())}' received incorrect label '{str(label)}', "
                             f"according to its graph, it is actually a complex.")
                else:
                    if label == db.Label.COMPLEX_OPTIMIZED:
                        warn(f"Structure '{str(structure.id())}' received incorrect label '{str(label)}', "
                             f"according to its graph, it is actually NOT a complex.")
            elif structure.has_properties("bond_orders"):
                self._setup_graph_job(structure.id())
                continue
            else:
                self._setup_bond_order_job(structure.id())
                continue

            # If a graph exists but no aggregate is registered, generate a new aggregate
            # or append the structure to an existing compound
            matching_aggregate = self._get_matching_aggregate(structure)
            if matching_aggregate is not None:
                structure.set_aggregate(matching_aggregate.id())
                matching_aggregate.add_structure(structure.id())
                # Check if duplicate in same aggregate
                duplicate = self._query_duplicate(structure, matching_aggregate)
                if duplicate is not None:
                    structure.set_label(db.Label.DUPLICATE)
                    structure.set_comment("Structure is a duplicate of {:s}.".format(str(duplicate.id())))
                    structure.set_as_duplicate_of(duplicate.id())
            else:
                # Create a new aggregate but only if the structure is not a misguided conformer optimization
                # TODO this query may become a bottleneck for huge databases.
                #      avoiding a query into the calculations would be much preferred
                selection = {
                    "$and": [
                        {"results.structures": {"$oid": structure.id().string()}},
                        {"results.elementary_steps": {"$size": 0}},
                    ]
                }
                hit = None
                if self.options.exclude_misguided_conformer_optimizations:
                    hit = self._calculations.get_one_calculation(dumps(selection))
                if hit is not None:
                    # Check if there is a calculation that generated the compound
                    minimization = hit
                    minimization.link(self._calculations)
                    starting_structure = db.Structure(minimization.get_structures()[0], self._structures)
                    if starting_structure.has_aggregate():
                        # If the starting structure has an aggregate, then the
                        #   result should also be part of the same aggregate
                        structure.set_label(db.Label.IRRELEVANT)
                        structure.set_comment(
                            f"Result of misguided optimization not staying within the origin aggregate "
                            f"{str(starting_structure.get_aggregate())}."
                        )
                        continue
                if ";" in graph:
                    flask = db.Flask(db.ID(), self._flasks)
                    flask.create([structure.id()], [])
                    flask.disable_exploration()
                    structure.set_aggregate(flask.id())
                    self._add_to_map(flask, self._flask_map)
                else:
                    compound = db.Compound(db.ID(), self._compounds)
                    compound.create([structure.id()])
                    compound.disable_exploration()
                    structure.set_aggregate(compound.id())
                    self._add_to_map(compound, self._compound_map)

    def _setup_bond_order_job(self, structure_id: db.ID) -> None:
        """
        Checks for already existing bond_order job and if none present, sets up one for given structure
        """
        if calculation_exists_in_structure(self.options.bond_order_job.order, [structure_id], self.options.model,
                                           self._structures, self._calculations):
            return
        # Setup calculation of a graph
        calculation = db.Calculation(db.ID(), self._calculations)
        calculation.create(self.options.model, self.options.bond_order_job, [structure_id])
        if self.options.bond_order_settings:
            calculation.set_settings(self.options.bond_order_settings)
        finalize_calculation(calculation, self._structures)

    def _setup_graph_job(self, structure_id: db.ID) -> None:
        """
        Checks for already existing graph job and if none present, sets up one for given structure
        """
        if calculation_exists_in_structure(self.options.graph_job.order, [structure_id], self.options.model,
                                           self._structures, self._calculations):
            return
        # Setup calculation of a graph
        calculation = db.Calculation(db.ID(), self._calculations)
        calculation.create(self.options.model, self.options.graph_job, [structure_id])
        if self.options.graph_settings:
            calculation.set_settings(self.options.graph_settings)
        finalize_calculation(calculation, self._structures)

    def _get_matching_aggregate(self, structure: db.Structure) -> Union[db.Compound, db.Flask, None]:
        """
        Queries database for matching aggregate of the given structure based on:
          * graph (irrespective of order for flasks)
          * charge
          * multiplicity
        Returns None if no matching aggregate in DB yet
        """
        graph = structure.get_graph("masm_cbor_graph")
        charge = structure.get_charge()
        multiplicity = structure.get_multiplicity()
        n_atoms = len(structure.get_atoms())
        n_molecules = len(graph.split(";"))
        if n_molecules > 1:
            f_id = self._get_aggregate_from_map(n_atoms, n_molecules, charge, multiplicity, graph, self._flask_map)
            if f_id:
                return db.Flask(f_id, self._flasks)
        else:
            c_id = self._get_aggregate_from_map(n_atoms, n_molecules, charge, multiplicity, graph, self._compound_map)
            if c_id:
                return db.Compound(c_id, self._compounds)
        return None

    def _query_duplicate(self, structure: db.Structure, aggregate: Union[db.Compound, db.Flask]) \
            -> Union[db.Structure, None]:
        """
        Check based on decision lists if the given aggregate has a duplicate structure to the given structure
        NOTE: not implemented for flasks yet
        """
        if isinstance(aggregate, db.Flask):
            # currently not possible
            return None
        decision_list = structure.get_graph("masm_decision_list")
        key = aggregate.id().string()
        if key in self._unique_structures:
            similar_structures = self._unique_structures[key]
            for sid, ref_decision_list in similar_structures:
                if not masm.JsonSerialization.equal_decision_lists(decision_list, ref_decision_list):
                    continue
                similar_structure = db.Structure(sid, self._structures)
                if similar_structure.get_model() == structure.get_model():
                    return similar_structure

        self._add_to_unique_structures(structure)
        return None

    @staticmethod
    def get_n_atoms(structure: db.Structure) -> List[int]:
        import ast
        idx_map = ast.literal_eval(structure.get_graph("masm_idx_map"))
        n_mols = 2
        for iMol, _ in idx_map:
            n_mols = max(n_mols, iMol + 1)
        n_atoms_per_mol = [0 for _ in range(n_mols)]
        for iMol, _ in idx_map:
            n_atoms_per_mol[iMol] += 1
        return n_atoms_per_mol
