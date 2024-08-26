#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
from typing import Union, Tuple, Dict, List, Optional
from warnings import warn
from copy import copy
import numpy as np

# Third party imports
import scine_database as db
from scine_database.queries import (
    stop_on_timeout,
    calculation_exists_in_structure,
    get_calculation_id_from_structure,
    optimized_labels,
)
import scine_utilities as utils
import scine_molassembler as masm

# Local application imports
from . import Gear
from ..utilities.calculation_creation_helpers import finalize_calculation
from ..utilities.masm import mol_from_cbor, mols_from_properties
from .network_refinement.enabling import AggregateEnabling
from scine_chemoton.utilities.place_holder_model import (
    construct_place_holder_model,
    PlaceHolderModelType
)


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

    def __init__(self) -> None:
        super().__init__()
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
        self.aggregate_enabling: AggregateEnabling = AggregateEnabling()
        """
        AggregateEnabling
            If a structure is added to an aggregate, the analysis of the corresponding aggregate is enabled
            according to this policy.
        """

    class Options(Gear.Options):
        """
        The options for the BasicAggregateHousekeeping Gear.
        """

        __slots__ = [
            "bond_order_job",
            "bond_order_settings",
            "graph_job",
            "graph_settings",
            "exclude_misguided_conformer_optimizations",
        ]

        def __init__(self) -> None:
            super().__init__()
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
                optimization. The default policy does nothing.
            """

    options: Options

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
                if not structure.has_graph("masm_decision_list"):
                    continue
                self._unique_structures[key].append((s_id, structure.get_graph("masm_decision_list")))

    def _add_to_unique_structures(self, key: str, structure: db.Structure):
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
        sum_formula, n_molecules, charge, multiplicity, mols = self._get_aggregate_info(aggregate)
        key = (n_molecules, charge, multiplicity)
        if sum_formula in cache_map:
            if key in cache_map[sum_formula]:
                if (aggregate.id(), mols) in cache_map[sum_formula][key]:
                    return
                cache_map[sum_formula][key].append((aggregate.id(), mols))
                return
            else:
                cache_map[sum_formula][key] = [(aggregate.id(), mols)]
                return
        cache_map[sum_formula] = {key: [(aggregate.id(), mols)]}

    def _get_sum_formula(self, mols: List[masm.Molecule]) -> str:
        sum_formulas = []
        for m in mols:
            sum_formulas.append(m.graph.__repr__().split()[-1])
        sum_formulas.sort()
        return ','.join(sum_formulas)

    @staticmethod
    def _get_aggregate_from_map(sum_formula: str, n_molecules: int, charge: int, multiplicity: int,
                                mols: List[masm.Molecule], cache_map: Dict) -> Union[db.ID, None]:
        if sum_formula in cache_map:
            key = (n_molecules, charge, multiplicity)
            if key in cache_map[sum_formula]:
                for a_id, ref_mols in cache_map[sum_formula][key]:
                    remaining_mols = copy(ref_mols)
                    for m in mols:
                        idx = remaining_mols.index(m) if m in remaining_mols else None
                        if idx is not None:
                            remaining_mols.pop(idx)
                        else:
                            break
                    else:
                        return a_id
        return None

    def _check_graph(self, structure: db.Structure) -> Optional[str]:
        if structure.has_graph("masm_cbor_graph") and structure.has_graph("masm_decision_list"):
            # Check if graph exists
            graph = structure.get_graph("masm_cbor_graph")
            label = structure.get_label()
            if ";" in graph:
                if label != db.Label.COMPLEX_OPTIMIZED and label != db.Label.USER_COMPLEX_OPTIMIZED:
                    warn(f"Structure '{str(structure.id())}' received incorrect label '{str(label)}', "
                         f"according to its graph, it is actually a complex.")
            else:
                if label == db.Label.COMPLEX_OPTIMIZED or label == db.Label.USER_COMPLEX_OPTIMIZED:
                    warn(f"Structure '{str(structure.id())}' received incorrect label '{str(label)}', "
                         f"according to its graph, it is actually NOT a complex.")
            return graph

        if structure.has_properties("bond_orders"):
            self._setup_graph_job(structure.id())
            return None

        self._setup_bond_order_job(structure.id())
        return None

    def _label_structure_as_duplicate(self, structure: db.Structure, duplicate_id: db.ID):
        structure.set_label(db.Label.DUPLICATE)
        structure.set_comment("Structure is a duplicate of {:s}.".format(str(duplicate_id)))
        structure.set_original(duplicate_id)

    def _check_for_misguided_conformer_optimization(self, structure: db.Structure) -> bool:
        # TODO this query may become a bottleneck for huge databases.
        #      avoiding a query into the calculations would be much preferred
        selection = {
            "$and": [
                {"status": {"$in": ["complete", "failed", "analyzed"]}},
                {"results.structures": {"$oid": structure.id().string()}},
                {"results.elementary_steps": {"$size": 0}},
            ]
        }
        hit = None
        irrelevant_structure = False
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
                    irrelevant_structure = True

        return irrelevant_structure

    def _store_as_compound(self, structure: db.Structure):
        compound = db.Compound(db.ID(), self._compounds)
        compound.create([structure.id()], exploration_disabled=True)
        structure.set_aggregate(compound.id())
        self._add_to_map(compound, self._compound_map)

    def _store_as_flask(self, structure: db.Structure):
        flask = db.Flask(db.ID(), self._flasks)
        flask.create([structure.id()], [], exploration_disabled=True)
        structure.set_aggregate(flask.id())
        self._add_to_map(flask, self._flask_map)

    def _loop_impl(self):
        if not self._compound_map:
            self._create_compound_map()
        if not self._flask_map:
            self._create_flask_map()
        if not self._unique_structures:
            self._create_unique_structure_map()
        self._check_for_aggregates()

    def _check_for_aggregates(self):
        # Setup query for optimized structures without aggregate
        selection = {
            "$and": [
                {"label": {"$in": optimized_labels()}},
                {"aggregate": ""},
                {"analysis_disabled": {"$ne": True}},
            ]
        }
        # Loop over all results
        for structure in stop_on_timeout(self._structures.iterate_structures(dumps(selection))):
            structure.link(self._structures)
            if self.have_to_stop_at_next_break_point():
                return
            # # # Continue if a job for has been setup
            graph = self._check_graph(structure)
            if graph is None:
                continue
            # If a graph exists but no aggregate is registered, generate a new aggregate
            # or append the structure to an existing compound
            try:
                matching_aggregate = self._get_matching_aggregate(structure)
            except IndexError:
                print("Structure: " + structure.id().string() + " has invalid Molassember serialization.")
                print("The structure is set as IRRELEVANT for further analysis.")
                structure.set_label(db.Label.IRRELEVANT)
            if matching_aggregate is not None:
                # Check if duplicate in same aggregate
                duplicate = self._query_duplicate(structure, matching_aggregate)
                if duplicate is not None:
                    self._label_structure_as_duplicate(structure, duplicate.id())
                else:
                    # only add aggregate info for non-duplicates
                    structure.set_aggregate(matching_aggregate.id())
                    matching_aggregate.add_structure(structure.id())
                self.aggregate_enabling.process(matching_aggregate)
            else:
                # Create a new aggregate but only if the structure is not a misguided conformer optimization
                # TODO this query may become a bottleneck for huge databases.
                #      avoiding a query into the calculations would be much preferred
                irrelevant_structure = self._check_for_misguided_conformer_optimization(structure)
                if irrelevant_structure:
                    continue
                if ";" in graph:
                    self._store_as_flask(structure)
                else:
                    self._store_as_compound(structure)

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

    def _get_aggregate_info(
        self, aggregate: Union[db.Compound, db.Flask]
    ) -> Tuple[str, int, int, int, List[masm.Molecule]]:
        # We could even think about adding the graph here too ...
        centroid = db.Structure(aggregate.get_centroid(), self._structures)
        try:
            graph = centroid.get_graph("masm_cbor_graph")
            mols = [mol_from_cbor(m) for m in graph.split(";")]
        except RuntimeError as e:
            tmp = mols_from_properties(centroid, self._properties)
            if tmp is None:
                raise RuntimeError(
                    f'Error: Graph deserialization and recreation failed for {centroid.id().string}'
                ) from e
            mols = tmp
        sum_formula = self._get_sum_formula(mols)
        return sum_formula, len(graph.split(";")), centroid.get_charge(), centroid.get_multiplicity(), \
            mols

    def _get_matching_aggregate(self, structure: db.Structure) -> Union[db.Compound, db.Flask, None]:
        """
        Queries database for matching aggregate of the given structure based on:
          * graph (irrespective of order for flasks)
          * charge
          * multiplicity
        Returns None if no matching aggregate in DB yet
        """
        try:
            graph = structure.get_graph("masm_cbor_graph")
            mols = [mol_from_cbor(m) for m in graph.split(";")]
        except RuntimeError as e:
            tmp = mols_from_properties(structure, self._properties)
            if tmp is None:
                raise RuntimeError(
                    f'Error: Graph deserialization and recreation failed for {structure.id().string}'
                ) from e
            mols = tmp
        charge = structure.get_charge()
        multiplicity = structure.get_multiplicity()
        sum_formula = self._get_sum_formula(mols)
        n_molecules = len(mols)
        if n_molecules > 1:
            f_id = self._get_aggregate_from_map(
                sum_formula, n_molecules, charge, multiplicity, mols, self._flask_map
            )
            if f_id:
                return db.Flask(f_id, self._flasks)
        else:
            c_id = self._get_aggregate_from_map(
                sum_formula, n_molecules, charge, multiplicity, mols, self._compound_map
            )
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
        key = aggregate.id().string()
        if key in self._unique_structures:
            similar_structures = self._unique_structures[key]
            decision_list = structure.get_graph("masm_decision_list")
            model = structure.get_model()
            for sid, ref_decision_list in similar_structures:
                # Check that similar structure is not identical to the given ID
                if sid == structure.id():
                    continue
                if not masm.JsonSerialization.equal_decision_lists(decision_list, ref_decision_list):
                    continue
                similar_structure = db.Structure(sid, self._structures)
                if similar_structure.get_model() == model:
                    return similar_structure

        self._add_to_unique_structures(key, structure)
        return None


class ThermoAggregateHousekeeping(BasicAggregateHousekeeping):
    """
    This Gear updates all relevant Structures stored in the database with frequencies,
    bond orders and graph representations.
    This data is then used to sort each Structure into existing Compounds/Flasks or to
    create a new Compound/Flask if no appropriate one exists.
    To be sorted into an Aggregate, a Structure must have no imaginary frequencies
    above the given absolute frequency threshold.
    Otherwise, a validation job is setup or, if this has been attempted already,
    the exploration and analysis of this Structure are disabled.

    Attributes
    ----------
    options : ThermoAggregateHousekeeping.Options
        The options for the ThermoAggregateHousekeeping Gear.

    Notes
    -----
    Checks for all 'user_optimized', 'minimum_optimized', and 'complex_optimized' Structures
    that do not have an Aggregate assigned and are enabled for analysis.
    The Gear then generates bond orders and molecular graphs ('masm_cbor_graph', and 'masm_decision_list')
    if they are not yet present.
    Using the molecular graphs, the Structures are then sorted into Compounds/Flasks.
    """

    class Options(BasicAggregateHousekeeping.Options):
        """
        The options for the ThermoAggregateHouseKeeping Gear.
        """
        __slots__ = [
            "validation_job",
            "validation_settings",
            "structure_model",
            "absolute_frequency_threshold"
        ]

        def __init__(self) -> None:
            super().__init__()
            self.validation_job: db.Job = db.Job("scine_geometry_validation")
            """
            db.Job
                The Job used for the geometry validation calculations.
                The default is: the 'scine_geometry_validation' order on a single core.
            """
            self.validation_settings: utils.ValueCollection = utils.ValueCollection()
            """
            utils.ValueCollection
                Additional settings passed to the geometry validation order calculation.
                Empty by default.
            """
            self.structure_model: db.Model = construct_place_holder_model()
            """
            Optional[db.Model]
                Validation calculations are only started for structures with the given model.
            """

            self.absolute_frequency_threshold: float = 50.0
            """
            float
                Abs Frequency threshold in cm^-1.
            """

    options: Options

    def _check_for_aggregates(self):
        # Setup query for optimized structures without aggregate
        selection = {"$and": [
            {"label": {"$in": optimized_labels()}},
            {"aggregate": ""},
            {"analysis_disabled": {"$ne": True}},
        ]
        }
        # Loop over all results
        for structure in stop_on_timeout(self._structures.iterate_structures(dumps(selection))):
            structure.link(self._structures)
            if self.have_to_stop_at_next_break_point():
                return
            # Structure Model Check, might be cached
            if not isinstance(self.options.structure_model, PlaceHolderModelType):
                if structure.get_model() != self.options.structure_model:
                    continue
            # # # Start Graph Check
            # # # Continue if a job for has been setup
            graph = self._check_graph(structure)
            if graph is None:
                continue

            # If a graph exists but no aggregate is registered, generate a new aggregate
            # or append the structure to an existing compound

            # # # Look, if aggregate is there
            try:
                matching_aggregate = self._get_matching_aggregate(structure)
            except IndexError:
                print("Structure: " + structure.id().string() + " has invalid Molassember serialization.")
                print("The structure is set as IRRELEVANT for further analysis.")
                structure.set_label(db.Label.IRRELEVANT)
            if matching_aggregate is not None:
                # # # Frequency check
                keep_going, valid_minimum = self._check_frequencies_and_validation_job(structure)
                # Move on to next structure as either a validation job has been setup or
                # or the structure was attempted to be validated already
                if not keep_going:
                    continue
                # Check if duplicate in same aggregate
                duplicate = self._query_duplicate(structure, matching_aggregate)
                if duplicate is not None:
                    self._label_structure_as_duplicate(structure, duplicate.id())
                elif valid_minimum:
                    # only add aggregate info for non-duplicates
                    structure.set_aggregate(matching_aggregate.id())
                    matching_aggregate.add_structure(structure.id())

            # # # Few more checks and make new aggregate
            else:
                # # # Frequency check
                keep_going, valid_minimum = self._check_frequencies_and_validation_job(structure)
                # Move on to next structure as either a validation job has been setup or
                # or the structure was attempted to be validated already
                if not keep_going:
                    continue

                # Create a new aggregate but only if the structure is not a misguided conformer optimization

                # # # Check for misguided conformation jobs
                irrelevant_structure = self._check_for_misguided_conformer_optimization(structure)
                if irrelevant_structure:
                    continue
                # # # End misguided conformation job

                # # # Sort to correct aggregate, only if it is a valid minimum
                if ";" in graph and valid_minimum:
                    self._store_as_flask(structure)
                elif valid_minimum:
                    self._store_as_compound(structure)

    def _check_frequencies_and_validation_job(self, structure: db.Structure) -> Tuple[bool, bool]:
        """
        Checks if structure is minimum, sets up a validation job if not setup yet and disables the structure,
        if it is not a valid minimum and a validation has been attempted already.
        The latter two should continue the loop.

        Parameters
        ----------
        structure : db.Structure
            The structure to be checked.

        Returns
        -------
        keep_going, valid_minimum : Tuple[bool, bool]
            keep_going indicates if the parent loop should continue or look at the next structure.
            valid_minimum indicates if the structure is a valid minimum.
        """
        # Default return values
        valid_minimum = False
        keep_going = True

        if structure.has_properties("frequencies"):
            valid_minimum = self._is_valid_minium(structure)

        # # # Setting up validation job
        # Single atom is a valid minium, but has no frequencies
        if len(structure.get_atoms()) == 1:
            valid_minimum = True
        elif not valid_minimum and len(structure.get_atoms()) > 1:
            # Not attempted validation
            validation_setup = self._setup_validation_job(structure.id())
            if validation_setup:
                keep_going = False
            # Disable structure from being analyzed
            # Only if validation job ran through and frequencies not proper for a minimum
            elif not validation_setup and structure.has_properties("frequencies"):
                structure.disable_analysis()
                structure.disable_exploration()
                structure.set_comment(
                    "Structure has imaginary frequencies larger than {:.0f} cm^-1".format(
                        self.options.absolute_frequency_threshold) +
                    "\nDisabled.")
                keep_going = False

        return keep_going, valid_minimum

    def _is_valid_minium(self, structure: db.Structure) -> bool:
        frequencies = self._properties.get_vector_property(structure.get_property("frequencies"))
        frequencies.link(self._properties)
        # Get minimal frequency
        min_frequency = np.min(frequencies.get_data()[0])
        if min_frequency < 0.0 and abs(
                min_frequency) > self.options.absolute_frequency_threshold *\
                utils.HARTREE_PER_INVERSE_CENTIMETER / (2 * utils.PI):
            return False
        else:
            return True

    def _setup_validation_job(self, structure_id: db.ID) -> bool:
        """
        Checks for already existing validation_order job and if none present, sets up one for given structure
        """
        success = False
        val_calc_id = get_calculation_id_from_structure(self.options.validation_job.order, [structure_id],
                                                        self.options.model, self._structures, self._calculations)
        # Setup calculation of validation
        if val_calc_id is None:
            calculation = db.Calculation(db.ID(), self._calculations)
            calculation.create(self.options.model, self.options.validation_job, [structure_id])
            if self.options.validation_settings:
                calculation.set_settings(self.options.validation_settings)
            finalize_calculation(calculation, self._structures)

            success = True
        else:
            val_calc = db.Calculation(val_calc_id, self._calculations)
            val_calc_status = val_calc.get_status()
            # If the calculation is complete, setup failed
            if val_calc_status in [db.Status.COMPLETE, db.Status.FAILED]:
                success = False
            # Any other states indicates
            else:
                success = True

        return success
