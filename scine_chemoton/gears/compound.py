#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
from typing import Union, Tuple, Dict, List

# Third party imports
import scine_database as db
import scine_utilities as utils
import scine_molassembler as masm

# Local application imports
from . import Gear
from ..utilities.queries import model_query, stop_on_timeout


class BasicCompoundHousekeeping(Gear):
    """
    This Gear updates all relevant Structures stored in the database with
    bond orders and graph representations.
    This data is then used to sort each Structure into existing Compounds or to
    create a new Compound if no appropriate one exist.

    Attributes
    ----------
    options : BasicCompoundHousekeeping.Options
        The options for the BasicCompoundHousekeeping Gear.

    Notes
    -----
    Checks for all 'minimum_optimized' Structures that do not have a Compound
    assigned. The Gear then generates bond orders and molecular graphs
    ('masm_cbor_graph', and 'masm_decision_list') if they
    are not yet present.
    Using the molecular graphs the Structures are then sorted into Compounds.
    """

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self._calculations = "required"
        self._structures = "required"
        self._properties = "required"
        self._compounds = "required"
        self._compound_map: Dict[Tuple[int, int, int], List[db.ID]] = dict()
        """
        Caching map from n_atoms, n_molecules, charge, multiplicity to compound ids.
        """
        self._unique_structures: Dict[str, List[Tuple[db.ID, str]]] = dict()
        """
        Caching map from compound id to list of structure ids and decision lists.
        """

    class Options:
        """
        The options for the BasicCompoundHousekeeping Gear.
        """

        __slots__ = [
            "cycle_time",
            "model",
            "bond_order_job",
            "bond_order_settings",
            "graph_job",
            "graph_settings",
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
            self.model: db.Model = db.Model("PM6", "", "")
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
        key = structure.get_compound().string()
        if key in self._unique_structures:
            self._unique_structures[key].append((structure.id(), structure.get_graph("masm_decision_list")))
        else:
            self._unique_structures[key] = [(structure.id(), structure.get_graph("masm_decision_list"))]

    def _create_compound_map(self):
        for compound in stop_on_timeout(self._compounds.iterate_all_compounds()):
            compound.link(self._compounds)
            self._add_to_map(compound, self._compound_map)

    def _add_to_map(self, compound: db.Compound, cache_map: Dict):
        key, graph = self._get_compound_info(compound)
        if key in cache_map:
            cache_map[key].append((compound.id(), graph))
        else:
            cache_map[key] = [(compound.id(), graph)]

    def _get_compound_info(self, compound: db.Compound) -> Tuple[Tuple[int, int, int], str]:
        # We could even think about adding the graph here too ...
        centroid = db.Structure(compound.get_centroid(), self._structures)
        graph = centroid.get_graph("masm_cbor_graph")
        return tuple((len(centroid.get_atoms()), centroid.get_charge(), centroid.get_multiplicity())), graph

    def _loop_impl(self):
        if not self._compound_map:
            self._create_compound_map()
        if not self._unique_structures:
            self._create_unique_structure_map()

        # Setup query for optimized structures without compound
        selection = {
            "$and": [
                {
                    "$or": [
                        {"label": {"$eq": "minimum_optimized"}},
                        {"label": {"$eq": "user_optimized"}},
                    ]
                },
                {"compound": {"$eq": ""}},
                {"analysis_disabled": {"$ne": True}},
            ]
        }
        # Loop over all results
        for structure in stop_on_timeout(self._structures.iterate_structures(dumps(selection))):
            structure.link(self._structures)
            if structure.has_graph("masm_cbor_graph"):
                # Check if graph exists
                graph = structure.get_graph("masm_cbor_graph")
                if len(graph.split(";")) > 1:
                    # If there is more than one molecule this is not a structure but a reactive complex
                    structure.set_label(db.Label.IRRELEVANT)
                    continue
            elif structure.has_properties("bond_orders"):
                # Generate graph
                # Check if a calculation for this is already scheduled
                selection = {
                    "$and": [
                        {"job.order": {"$eq": self.options.graph_job.order}},
                        {"structures": {"$oid": structure.id().string()}},
                    ]
                    + model_query(self.options.model)
                }
                if self._calculations.get_one_calculation(dumps(selection)) is not None:
                    continue
                # Setup calculation of a graph
                calculation = db.Calculation()
                calculation.link(self._calculations)
                calculation.create(self.options.model, self.options.graph_job, [structure.id()])
                if self.options.graph_settings:
                    calculation.set_settings(self.options.graph_settings)
                calculation.set_status(db.Status.HOLD)
                continue
            else:
                # Calculate bond orders
                # Check if a calculation for this is already scheduled
                selection = {
                    "$and": [
                        {"job.order": {"$eq": self.options.bond_order_job.order}},
                        {"structures": {"$oid": structure.id().string()}},
                    ]
                    + model_query(self.options.model)
                }
                if self._calculations.get_one_calculation(dumps(selection)) is not None:
                    continue
                # Setup calculation of a graph
                calculation = db.Calculation()
                calculation.link(self._calculations)
                calculation.create(self.options.model, self.options.bond_order_job, [structure.id()])
                if self.options.bond_order_settings:
                    calculation.set_settings(self.options.bond_order_settings)
                calculation.set_status(db.Status.HOLD)
                continue

            # If a graph exists but no compound is registered, generate a new compound
            # or append the structure to an existing compound
            matching_compound = self._get_matching_compound(structure)
            if matching_compound is not None:
                # Structure belongs to an existing compound
                structure.set_compound(matching_compound.id())
                matching_compound.add_structure(structure.id())
                # Check if this structure is a duplicate
                duplicate = self._query_duplicate(structure, matching_compound)
                if duplicate is not None:
                    structure.set_label(db.Label.DUPLICATE)
                    structure.set_comment("Structure is a duplicate of {:s}.".format(str(duplicate.id())))

            else:
                # Create a new compound but only if the structure is not a misguided conformer optimization
                # TODO this query may become a bottleneck for huge databases.
                #      avoiding a query into the calculations would be much preferred
                selection = {
                    "$and": [
                        {"results.structures": {"$oid": structure.id().string()}},
                        {"results.elementary_steps": {"$size": 0}},
                    ]
                }
                hit = self._calculations.get_one_calculation(dumps(selection))
                if hit is not None:
                    # Check if there is a calculation that generated the compound
                    minimization = hit
                    minimization.link(self._calculations)
                    starting_structure = db.Structure(minimization.get_structures()[0])
                    starting_structure.link(self._structures)
                    if starting_structure.has_compound():
                        # If the starting structure has a compound, then the
                        #   result should also be part of the same compound
                        structure.set_label(db.Label.IRRELEVANT)
                        structure.set_comment(
                            "Result of misguided optimization not staying within the origin compound."
                        )
                        continue
                compound = db.Compound()
                compound.link(self._compounds)
                compound.create([structure.id()])
                compound.disable_exploration()
                structure.set_compound(compound.id())
                self._add_to_map(compound, self._compound_map)

    def _get_matching_compound(self, structure: db.Structure) -> Union[db.Compound, None]:
        graph = structure.get_graph("masm_cbor_graph")
        charge = structure.get_charge()
        multiplicity = structure.get_multiplicity()
        n_atoms = len(structure.get_atoms())
        key = (n_atoms, charge, multiplicity)
        if key in self._compound_map:
            for c_id, ref_graph in self._compound_map[key]:
                if masm.JsonSerialization.equal_molecules(ref_graph, graph):
                    return db.Compound(c_id, self._compounds)
        return None

    def _query_duplicate(self, structure: db.Structure, compound: db.Compound) -> Union[db.Structure, None]:
        """
        Check based on decision lists if the given compound has a duplicate structure to the given structure
        """
        decision_list = structure.get_graph("masm_decision_list")
        key = compound.id().string()
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
