#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import time
from copy import deepcopy
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union
from json import dumps
from itertools import combinations
from warnings import warn

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application imports
from ....utilities.queries import stop_on_timeout, get_calculation_id, select_calculation_by_structures
from ....utilities.energy_query_functions import get_energy_sum_of_elementary_step_side, get_energy_for_structure
from ..reactive_site_filters import ReactiveSiteFilter
from .connectivity_analyzer import ReactionType, ConnectivityAnalyzer
from .bond_based import BondBased
from ..trial_generator import TrialGenerator


def single_structure_assertion(fun: Callable):
    """
    Makes sure that the first argument given to the function `fun` is either a list of db.Structure of len 1 or
    a single db.Structure and then calls the function with the list argument, throws TypeError otherwise
    """
    @wraps(fun)
    def _impl(self, structures: Union[List[db.Structure], db.Structure], *args):
        arg = structures
        if isinstance(structures, list):
            if len(structures) != 1:
                raise TypeError(f"The method {fun.__name__} of the filter {self.__class__.__name__} is only supported"
                                f"for single structure lists.")
        elif isinstance(structures, db.Structure):
            arg = [structures]
        else:
            raise TypeError(f"The method {fun.__name__} of the filter {self.__class__.__name__} received an invalid "
                            f"type {type(structures)} for its input argument")
        return fun(self, arg, *args)

    return _impl


class FurtherExplorationFilter(ReactiveSiteFilter):
    """
    Class to evaluate if detailed dissociation exploration trials should be setup.
    This base class does not filter anything out.
    """

    def __init__(self):
        super().__init__()
        self._setting_key = 'dissociations'
        self._structure_property_key = 'dissociated_structures'

    def filter_atoms(self, _: List[db.Structure], atom_indices: List[int]) -> List[int]:
        return atom_indices

    def filter_atom_pairs(self, _: List[db.Structure], pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return pairs

    def filter_reaction_coordinates(self, _: List[db.Structure], coordinates: List[List[Tuple[int, int]]]) \
            -> List[List[Tuple[int, int]]]:
        return coordinates

    @staticmethod
    def dissociation_setting_from_reaction_coordinate(coordinate: List[Tuple[int, int]]) -> List[int]:
        return [item for pair in coordinate for item in pair]

    def _query_dissociation_energy(self, structure: db.Structure, dissociations_list: List[List[int]], energy_type: str,
                                   model: db.Model, job_order: str) -> List[Union[float, None]]:
        # prepare data structures to only loop calculations once
        step_dissociation_energies: Dict[str, List[float]] = {}
        plain_dissociation_energies: Dict[str, List[float]] = {}
        for data in [step_dissociation_energies, plain_dissociation_energies]:
            for dissociations in dissociations_list:
                data[str(dissociations)] = []

        selection = select_calculation_by_structures(job_order, [structure.id()], model)
        for calculation in stop_on_timeout(self._calculations.iterate_calculations(dumps(selection))):
            calculation.link(self._calculations)
            dissociation = calculation.get_settings().get(self._setting_key)
            if dissociation is not None and dissociation in dissociations_list:
                results = calculation.get_results()
                if results.elementary_step_ids:
                    for step_id in results.elementary_step_ids:
                        step = db.ElementaryStep(step_id, self._elementary_steps)
                        reactant_energy = get_energy_sum_of_elementary_step_side(step, db.Side.LHS, energy_type,
                                                                                 model, self._structures,
                                                                                 self._properties)
                        if reactant_energy is None:
                            continue
                        product_energy = get_energy_sum_of_elementary_step_side(step, db.Side.RHS, energy_type,
                                                                                model, self._structures,
                                                                                self._properties)
                        if product_energy is None:
                            continue
                        step_dissociation_energies[str(dissociation)].append(product_energy - reactant_energy)
                elif results.property_ids:
                    reactant_energy = get_energy_for_structure(structure, energy_type, model, self._structures,
                                                               self._properties)
                    if reactant_energy is None:
                        continue
                    try:  # if property is not present the db wrapper throws an exception
                        dissocations_property_id = results.get_property(self._structure_property_key, self._properties)
                        dissocations_property = db.StringProperty(dissocations_property_id, self._properties)
                    except BaseException:
                        continue
                    sids = dissocations_property.get_data().split(",")
                    product_energies = []
                    for sid in sids:
                        structure = db.Structure(db.ID(sid), self._structures)
                        product_energies.append(get_energy_for_structure(structure, energy_type, model,
                                                                         self._structures, self._properties))
                    if None in product_energies:
                        continue
                    plain_dissociation_energies[str(dissociation)].append(
                        sum(product_energies) - reactant_energy)  # type: ignore
        energies: List[Union[float, None]] = []
        for dissociation in dissociations_list:
            step_energies = step_dissociation_energies[str(dissociation)]
            if step_energies:
                energies.append(min(step_energies) * utils.KJPERMOL_PER_HARTREE)
                continue
            plain_energies = plain_dissociation_energies[str(dissociation)]
            if plain_energies:
                energies.append(min(plain_energies) * utils.KJPERMOL_PER_HARTREE)
                continue
            energies.append(None)
        return energies


class AllBarrierLessDissociationsFilter(FurtherExplorationFilter):
    def __init__(self, model: db.Model, job_order: str):
        super().__init__()
        self.model = model
        self.job_order = job_order

    @single_structure_assertion
    def filter_reaction_coordinates(self, structure_list: List[db.Structure],
                                    coordinates: List[List[Tuple[int, int]]]) \
            -> List[List[Tuple[int, int]]]:
        structure = structure_list[0]
        dissociation_lists = [self.dissociation_setting_from_reaction_coordinate(coordinate)
                              for coordinate in coordinates]
        filtered = []
        # search for barrierless steps with structure on LHS
        # first go over reactions to query less
        aggregate_id = structure.get_aggregate()
        if not structure.has_graph("masm_cbor_graph"):
            return []
        aggregate_type = "flask" if ";" in structure.get_graph("masm_cbor_graph") else "compound"
        lhs = {"id": {"$oid": str(aggregate_id)}, "type": aggregate_type}
        selection = {
            "$and": [
                {"exploration_disabled": False},
                {"lhs": {"$size": 1, "$all": [lhs]}},
            ]
        }
        relevant_step_ids = []
        structure_id = structure.id()
        for reaction in self._reactions.query_reactions(dumps(selection)):
            step_ids = reaction.get_elementary_steps()
            for sid in step_ids:
                step = db.ElementaryStep(sid, self._elementary_steps)
                if step.get_type() != db.ElementaryStepType.BARRIERLESS:
                    # caveat: this relies on logic in reaction gear that we
                    # disable barrierless elementary steps if there is a regular one for the same reaction
                    continue
                if structure_id in step.get_reactants(db.Side.LHS)[0]:
                    relevant_step_ids.append(sid)
        step_sele = {"results.elementary_steps": {"$in": [{"$oid": str(sid)} for sid in relevant_step_ids]}}
        selection = select_calculation_by_structures(self.job_order, [structure.id()], self.model)
        selection["$and"].append(step_sele)
        for calculation in stop_on_timeout(self._calculations.iterate_calculations(dumps(selection))):
            calculation.link(self._calculations)
            calc_dissociation = calculation.get_settings().get(self._setting_key)
            if calc_dissociation is not None:
                for coordinate, dissociation in zip(coordinates, dissociation_lists):
                    if calc_dissociation == dissociation:
                        filtered.append(coordinate)
        return filtered


class ReactionCoordinateMaxDissociationEnergyFilter(FurtherExplorationFilter):
    def __init__(self, max_dissociation_energy: float, energy_type: str,
                 model: db.Model, job_order: str):
        super().__init__()
        self.max_dissociation_energy = max_dissociation_energy
        self.energy_type = energy_type
        self.model = model
        self.job_order = job_order

    @single_structure_assertion
    def filter_reaction_coordinates(self, structure_list: List[db.Structure],
                                    coordinates: List[List[Tuple[int, int]]]) \
            -> List[List[Tuple[int, int]]]:
        structure = structure_list[0]
        dissociation_lists = [self.dissociation_setting_from_reaction_coordinate(coordinate)
                              for coordinate in coordinates]
        energies = self._query_dissociation_energy(structure, dissociation_lists, self.energy_type, self.model,
                                                   self.job_order)
        if len(energies) != len(coordinates):
            raise RuntimeError("Something went wrong while fetching the dissociation energies for the coordinates "
                               f"{coordinates}")
        filtered = []
        for energy, coordinate in zip(energies, coordinates):
            if energy is None or energy > self.max_dissociation_energy:
                continue
            filtered.append(coordinate)
        return filtered


class FastDissociations(BondBased):
    """
    Class to evaluate pure dissociations via bond-based approaches.

    Attributes
    ----------
    options : FastDissociations.Options
        The options for generating dissociations.
    reactive_site_filter : ReactiveSiteFilter
        The filter applied to determine reactive sites, reactive pairs and trial
        reaction coordinates.
    """
    __slots__ = ("options", "reactive_site_filter")

    class Options:
        """
        The options for generating dissociations.
        """

        def __init__(self, parent: Optional[TrialGenerator] = None):
            self._parent = parent  # best be first member to be set, due to __setattr__
            self.model: db.Model = db.Model("PM6", "PM6", "")
            """
            db.Model (Scine::Database::Model)
                The Model used to evaluate the possible reactions.
                The default is: PM6 using Sparrow.
            """
            self.cutting_job: db.Job = db.Job("scine_dissociation_cut")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used to evaluate the possible reactions.
                The default is: the `scine_dissociation_cut`
                order on a single core.
            """
            self.cutting_job_settings: utils.ValueCollection = utils.ValueCollection({})
            """
            utils.ValueCollection
                Additional settings passed to Calculation evaluating the possible
                reactions. These settings are passed to all calculations for which
                all of the reactive atom pairs are bound in the start structure
                and cutting them apart would result into at least two separate molecules.
                Empty by default.
            """
            self.min_bond_dissociations = 1
            """
            int
                The minimum number of bond dissociations to be
                encouraged simultaneously during an elementary steps trial calculation.
                By default 1.
            """
            self.max_bond_dissociations = 1
            """
            int
                The maximum number of bond dissociations to be encouraged
                simultaneously during an elementary steps trial calculation. By default 1.
            """
            self.enable_further_explorations: bool = False
            """
            bool
                Whether the dissociations should also be explored with another more detailed job.
            """
            self.always_further_explore_dissociative_reactions: bool = True
            """
            bool
                Whether reaction coordinates that only lead to a dissociative reaction type, i.e. bonds are broken,
                but no two (or more) separate molecules are formed (based on the reaction coordinate), should
                automatically be explored with the further exploration job.
                This overrules 'enable_further_explorations' (but only for these reaction types)
                and the further_exploration_filter is not applied to them.
            """
            self.further_exploration_filter = FurtherExplorationFilter()
            """
            FurtherExplorationFilter
                An optional additional filter that can limit further exploration e.g. based on dissociation energies
                evaluated in the fast dissociation job. The default filter does not filter any reaction coordinate.
            """
            self.further_job: db.Job = db.Job("scine_react_complex_nt2")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used to evaluate the possible reactions with more detailed PES scans.
                Jobs with the order `scine_react_complex_nt2` are
                supported. The default is: the `scine_react_complex_nt2`
                order on a single core.
            """
            self.further_job_settings: utils.ValueCollection = utils.ValueCollection({})
            """
            utils.ValueCollection
                Additional settings passed to Calculation evaluating the possible
                reactions. These settings are passed to all further explorations and the dissociative
                reaction coordinates.
            """

        def __setattr__(self, item, value):
            """
            Overwritten standard method to clear cache of options holding parent whenever an option is changed
            """
            super().__setattr__(item, value)
            if self._parent is not None:
                self._parent.clear_cache()

    def __init__(self):
        super().__init__()
        self.options = self.Options(parent=self)
        self.reactive_site_filter: ReactiveSiteFilter = ReactiveSiteFilter()
        self._cache: Dict[str, List[int]] = {}
        self._required_collections = ["manager", "calculations", "structures"]

    def clear_cache(self):
        self._cache = {}

    def _sanity_check_configuration(self) -> None:
        if not isinstance(self.reactive_site_filter, ReactiveSiteFilter):
            raise TypeError("Expected a ReactiveSiteFilter (or a class derived "
                            "from it) in FastDissociations.reactive_site_filter.")
        if not isinstance(self.options.further_exploration_filter, FurtherExplorationFilter):
            raise TypeError("Expected a FurtherExplorationFilter (or a class derived "
                            "from it) in FastDissociations.options.further_exploration_filter.")
        if not isinstance(self._manager, db.Manager):
            raise TypeError("Propagation of db information failed")
        self.options.further_exploration_filter.initialize_collections(self._manager)

    def bimolecular_reactions(self, _: List[db.Structure]) -> None:
        """
        Bimolecular reactions are not supported by this TrialGenerator.

        Parameters
        ----------
        _ :: List[db.Structure]
            List of the two structures to be considered.
        """
        warn(f"Bimolecular reactions are not supported by the {self.__class__.__name__} TrialGenerator.")

    def unimolecular_reactions(self, structure: db.Structure) -> None:
        """
        Creates reactive complex calculations corresponding to the unimolecular
        reactions of the structure if there is not already a calculation to
        search for a reaction of the same structure with the same job order, model and settings.

        Parameters
        ----------
        structure :: db.Structure
            The structure to be considered. The Structure has to
            be linked to a database.
        """
        self._sanity_check_configuration()
        if self.options.max_bond_dissociations < 1:
            return
        # Rule out compounds too small for intramolecular reactions right away
        atoms = structure.get_atoms()
        structure_id = structure.id()
        str_structure_id = str(structure_id)
        # No intramolecular reactions for monoatomic compounds
        if atoms.size() == 1:
            return

        # We are filling the cache with structure as key.
        # The value is a list of the so far maximum explored dissociations
        # e.g. 1 at once (break 5-7), 2 at once (break 5-7 and 3-6 in one step), and so on
        # the ordering is fast disconnective [0], slow dissociative [1], and slow disconnective [2]
        if str_structure_id not in self._cache:
            self._cache[str_structure_id] = [0, 0, 0]
        current_cache_entry = self._cache[str_structure_id]
        allowed_types: List[bool] = [True,
                                     self.options.always_further_explore_dissociative_reactions,
                                     self.options.enable_further_explorations]

        # Get all reactive atoms
        reactive_atoms = self.reactive_site_filter.filter_atoms([structure], list(range(atoms.size())))
        further_reactive_atoms = self.options.further_exploration_filter.filter_atoms(structure, reactive_atoms)
        # Get all ordered pairs of reactive atoms
        connectivity_analyzer = ConnectivityAnalyzer(structure)
        fast_diss_pairs: List[Tuple[int, int]] = self._get_filtered_intraform_and_diss(
            structure, reactive_atoms, connectivity_analyzer)[1]
        further_diss_pairs: List[Tuple[int, int]] = self._get_filtered_intraform_and_diss(
            structure, further_reactive_atoms, connectivity_analyzer)[1]

        # Loop over possible options of dissociations
        for n_diss in range(self.options.min_bond_dissociations, self.options.max_bond_dissociations + 1):
            if n_diss < 1:
                continue
            # determine allowed number of simultaneous dissociations for each subtype based on cache and settings
            allowed_for_n = [i < n_diss and active for i, active in zip(current_cache_entry, allowed_types)]
            if not any([allowed for allowed in allowed_for_n]):
                # skip already if none of the three types are allowed for this number of dissociations
                continue
            dissociative_coordinates = []
            all_coordinates = [list(x) for x in combinations(fast_diss_pairs, n_diss)]  # type: ignore
            """ Fast dissociations """
            filtered_coordinates = self.reactive_site_filter.filter_reaction_coordinates([structure], all_coordinates)
            if allowed_for_n[0]:
                calculation_ids = []
                # TODO in first cache run, we are constructing the settings and then check if calculation
                # with these settings exists
                # a faster strategy would be to construct all possible settings (based on coordinates)
                # and then query calculations once for all still uncovered settings and create calculations for those
                for diss_pairs in filtered_coordinates:
                    # Get reaction type:
                    reaction_type = connectivity_analyzer.get_reaction_type(diss_pairs)
                    if reaction_type == ReactionType.Dissociative:
                        dissociative_coordinates.append(diss_pairs)
                    if reaction_type != ReactionType.Disconnective:
                        continue
                    cid = self._add_disconnective_calculation(structure_id, diss_pairs,
                                                              self.options.cutting_job,
                                                              self.options.cutting_job_settings)
                    if cid is not None:
                        calculation_ids.append(cid)
                if calculation_ids:
                    # only add first due to limited entry space for mongodb document
                    structure.add_calculation(self.options.cutting_job.order, calculation_ids[0])
                current_cache_entry[0] = n_diss

            """ Slow dissociative coordinates """
            if allowed_for_n[1]:
                filtered_coordinates = self.reactive_site_filter.filter_reaction_coordinates([structure],
                                                                                             dissociative_coordinates)
                calculation_ids = []
                # TODO see fast dissociations
                for diss_pairs in filtered_coordinates:
                    cid = self._add_reactive_complex_calculation([structure_id], [], diss_pairs,
                                                                 self.options.further_job,
                                                                 self.options.further_job_settings,
                                                                 check_for_existing=True)
                    if cid is not None:
                        calculation_ids.append(cid)
                if calculation_ids:
                    # only add first due to limited entry space for mongodb document
                    structure.add_calculations(self.options.further_job.order, calculation_ids)
                current_cache_entry[1] = n_diss

            """ Slow dissociations """
            if allowed_for_n[2]:
                pairs = self.options.further_exploration_filter.filter_atom_pairs(structure, further_diss_pairs)
                calculation_ids = []
                # TODO see fast dissociations
                for slow_diss_pairs in combinations(pairs, n_diss):
                    diss_pairs = list(slow_diss_pairs)
                    if current_cache_entry[1] >= n_diss and diss_pairs in dissociative_coordinates:
                        # we already explored this because it is solely dissociative
                        continue
                    # Get reaction type:
                    reaction_type = connectivity_analyzer.get_reaction_type(diss_pairs)
                    if reaction_type != ReactionType.Disconnective:
                        continue
                    # Check whether trial coordinate is to be considered
                    if self.reactive_site_filter.filter_reaction_coordinates([structure], [diss_pairs]) and \
                            self.options.further_exploration_filter.filter_reaction_coordinates(structure,
                                                                                                [diss_pairs]):
                        cid = self._add_reactive_complex_calculation([structure_id], [], diss_pairs,
                                                                     self.options.further_job,
                                                                     self.options.further_job_settings,
                                                                     check_for_existing=True)
                        if cid is not None:
                            calculation_ids.append(cid)
                if calculation_ids:
                    # only add first due to limited entry space for mongodb document
                    structure.add_calculations(self.options.further_job.order, calculation_ids)
                current_cache_entry[2] = n_diss

    def _add_disconnective_calculation(
        self,
        reactive_structure: db.ID,
        dissociation_pairs: List[Tuple[int, int]],
        job: db.Job,
        settings: utils.ValueCollection,
    ) -> Union[db.ID, None]:
        """
        Adds a reactive calculation for a bond-based approach (i.e., one during which it is tried to form/break
        bonds between designated atom pairs) to the database and puts it on hold.

        Parameters
        ----------

        reactive_structure :: db.ID
            The ID of the reactant.
        dissociation_pairs:: List[Tuple[int, int]]
            List of atom index pairs in between which a bond dissociation is to
            be encouraged.
        job :: scine_database.Job
        settings :: scine_utilities.ValueCollection
        """

        if job.order == "scine_dissociation_cut":
            # job takes lists of integer where elements 0/1, 2/3 ... N-1, N are combined with each other
            # Flatten pair lists
            settings["dissociations"] = FurtherExplorationFilter.dissociation_setting_from_reaction_coordinate(
                dissociation_pairs)
        else:
            raise RuntimeError("Only 'scine_dissociation_cut' order supported for fast dissociations")

        if get_calculation_id(job.order, [reactive_structure], self.options.model,
                              self._calculations, settings=settings) is not None:
            return None
        calculation = db.Calculation(db.ID(), self._calculations)
        calculation.create(self.options.model, job, [reactive_structure])
        # Sleep a bit in order not to make the DB choke
        time.sleep(0.001)
        calculation.set_settings(deepcopy(settings))
        calculation.set_status(db.Status.HOLD)
        return calculation.id()
