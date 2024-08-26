#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Set
from itertools import combinations
from warnings import warn

# Third party imports
from numpy import ndarray
import scine_database as db
from scine_database.queries import get_calculation_id
import scine_utilities as utils

# Local application imports
from scine_chemoton.filters.reactive_site_filters import ReactiveSiteFilter
from scine_chemoton.filters.further_exploration_filters import FurtherExplorationFilter
from .connectivity_analyzer import ReactionType, ConnectivityAnalyzer
from .bond_based import BondBased
from ..trial_generator import TrialGenerator, _sanity_check_wrapper


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
    further_exploration_filter : FurtherExplorationFilter
        An optional additional filter that can limit further exploration,
        e.g. based on dissociation energies evaluated in the fast dissociation job.
        The default filter does not filter any reaction coordinate.
    """

    class Options(BondBased.Options):
        """
        The options for generating dissociations.
        """
        __slots__ = ()

        class UnimolOptions(BondBased.Options.UnimolOptions):
            __slots__ = (
                "cutting_job_settings",
                "enable_further_explorations",
                "always_further_explore_dissociative_reactions",
                "further_job",
                "further_job_settings",
                "_parent",
                "_unusable_settings",
            )

            def __init__(self, parent: Optional[TrialGenerator] = None) -> None:
                super().__init__()
                self._parent = parent
                self._unusable_settings: Set[str] = {
                    "job_settings_associative",
                    "job_settings_dissociative",
                    "job_settings_disconnective",
                    "min_bond_formations",
                    "max_bond_formations",
                }
                self.job: db.Job = db.Job("scine_dissociation_cut")
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

            def __setattr__(self, item, value) -> None:
                """
                Overwritten standard method to clear cache of options holding parent whenever an option is changed
                """
                super().__setattr__(item, value)
                if hasattr(self, "_parent") and self._parent is not None:
                    self._parent.clear_cache()
                if hasattr(self, "_unusable_settings") and item in self._unusable_settings \
                        and hasattr(self, item) and getattr(self, item) != value:
                    raise NotImplementedError(f"The setting {item} is not implemented for "
                                              f"{self._parent.__class__.__name__}")
                # couple together modifications and dissociations
                elif item == "min_bond_modifications":
                    super().__setattr__("min_bond_dissociations", value)
                elif item == "max_bond_modifications":
                    super().__setattr__("max_bond_dissociations", value)
                elif item == "min_bond_dissociations":
                    super().__setattr__("min_bond_modifications", value)
                elif item == "max_bond_dissociations":
                    super().__setattr__("max_bond_modifications", value)

        unimolecular_options: UnimolOptions

        def __setattr__(self, item, value) -> None:
            """
            Overwritten standard method to clear cache of options holding parent whenever an option is changed
            """
            # should include all options of the parent class
            if item in ["_parent", "model", "unimolecular_options", "bimolecular_options", "base_job_settings",
                        "results_enabling_policy"]:
                super().__setattr__(item, value)
            else:
                self.unimolecular_options.__setattr__(item, value)
            if hasattr(self, "_parent") and self._parent is not None:
                self._parent.clear_cache()

    options: Options

    def __init__(self) -> None:
        super().__init__()
        self.further_exploration_filter = FurtherExplorationFilter()
        self._cache: Dict[str, List[int]] = {}
        self._required_collections = ["manager", "calculations", "structures"]

    def clear_cache(self) -> None:
        self._cache = {}

    def _sanity_check_configuration(self) -> None:
        if not isinstance(self.reactive_site_filter, ReactiveSiteFilter):
            raise TypeError("Expected a ReactiveSiteFilter (or a class derived "
                            "from it) in FastDissociations.reactive_site_filter.")
        if not isinstance(self.further_exploration_filter, FurtherExplorationFilter):
            raise TypeError(f"Expected a FurtherExplorationFilter (or a class derived "
                            f"from it) in FastDissociations.further_exploration_filter, not type "
                            f"{type(self.further_exploration_filter)}.")
        if not isinstance(self._manager, db.Manager):
            raise TypeError("Propagation of db information failed")
        self.further_exploration_filter.initialize_collections(self._manager)

    def bimolecular_coordinates(self, structure_list: List[db.Structure], with_exact_settings_check: bool = False) \
            -> Dict[
        Tuple[List[Tuple[int, int]], int],
        List[Tuple[ndarray, ndarray, float, float]]
    ]:
        return {}

    def bimolecular_reactions(self,
                              structure_list: List[db.Structure],
                              with_exact_settings_check: bool = False) -> None:
        """
        Bimolecular reactions are not supported by this TrialGenerator.

        Parameters
        ----------
        structure_list : List[db.Structure]
            List of the two structures to be considered.
        with_exact_settings_check : bool
            If True, more expensive queries are carried out to check if the settings of the
            calculations are exactly the same as the settings of the trial generator. This allows to add more
            inclusive additional reaction trials but the queries are less efficient, therefore this option
            should be only toggled if necessary.
        """
        warn(f"Bimolecular reactions are not supported by the {self.__class__.__name__} TrialGenerator.")

    @_sanity_check_wrapper
    def unimolecular_coordinates(self, structure: db.Structure, with_exact_settings_check: bool = False) \
            -> List[Tuple[List[List[Tuple[int, int]]], int]]:
        return self.unimolecular_fast_coordinates(structure, with_exact_settings_check) + \
            self.unimolecular_further_explore_coordinates(structure, with_exact_settings_check)

    def unimolecular_fast_coordinates(self, structure: db.Structure, with_exact_settings_check: bool = False) \
            -> List[Tuple[List[List[Tuple[int, int]]], int]]:
        if self.options.unimolecular_options.max_bond_dissociations < 1:
            return []
        # Rule out compounds too small for intramolecular reactions right away
        atoms = structure.get_atoms()
        structure_id = structure.id()
        str_structure_id = str(structure_id)
        # No intramolecular reactions for monoatomic compounds
        if atoms.size() == 1:
            return []
        # We are filling the cache with structure as key.
        # The value is a list of the so far maximum explored dissociations
        # e.g. 1 at once (break 5-7), 2 at once (break 5-7 and 3-6 in one step), and so on
        # the ordering is fast disconnective [0], slow dissociative [1], and slow disconnective [2]
        if str_structure_id not in self._cache:
            self._cache[str_structure_id] = [0, 0, 0]
        current_cache_entry = self._cache[str_structure_id]
        allowed_types: List[bool] = [True,
                                     self.options.unimolecular_options.always_further_explore_dissociative_reactions,
                                     self.options.unimolecular_options.enable_further_explorations]

        # Get all reactive atoms
        reactive_atoms = self.reactive_site_filter.filter_atoms([structure], list(range(atoms.size())))
        # Get all ordered pairs of reactive atoms
        connectivity_analyzer = ConnectivityAnalyzer(structure)
        fast_diss_pairs: List[Tuple[int, int]] = self._get_filtered_intraform_and_diss(
            structure, reactive_atoms, connectivity_analyzer)[1]

        result: List[Tuple[List[List[Tuple[int, int]]], int]] = []
        # Loop over possible options of dissociations
        for n_diss in range(self.options.unimolecular_options.min_bond_dissociations,
                            self.options.unimolecular_options.max_bond_dissociations + 1):
            if n_diss < 1:
                continue
            # determine allowed number of simultaneous dissociations for each subtype based on cache and settings
            allowed_for_n = [i < n_diss and active for i, active in zip(current_cache_entry[:2], allowed_types[:2])]
            if not with_exact_settings_check and not any([allowed for allowed in allowed_for_n]):
                # skip already if none of the three types are allowed for this number of dissociations
                continue
            all_coordinates = [list(x) for x in combinations(fast_diss_pairs, n_diss)]  # type: ignore
            filtered_coordinates = self.reactive_site_filter.filter_reaction_coordinates([structure], all_coordinates)
            result.append((filtered_coordinates, n_diss))

        return result

    def unimolecular_further_explore_coordinates(self, structure: db.Structure,
                                                 with_exact_settings_check: bool = False) \
            -> List[Tuple[List[List[Tuple[int, int]]], int]]:
        if self.options.unimolecular_options.max_bond_dissociations < 1 \
                or not self.options.unimolecular_options.enable_further_explorations:
            return []
        # Rule out compounds too small for intramolecular reactions right away
        atoms = structure.get_atoms()
        structure_id = structure.id()
        str_structure_id = str(structure_id)
        # No intramolecular reactions for monoatomic compounds
        if atoms.size() == 1:
            return []
        # We are filling the cache with structure as key.
        # The value is a list of the so far maximum explored dissociations
        # e.g. 1 at once (break 5-7), 2 at once (break 5-7 and 3-6 in one step), and so on
        # the ordering is fast disconnective [0], slow dissociative [1], and slow disconnective [2]
        if str_structure_id not in self._cache:
            self._cache[str_structure_id] = [0, 0, 0]
        current_cache_entry = self._cache[str_structure_id]

        # Get all reactive atoms
        reactive_atoms = self.reactive_site_filter.filter_atoms([structure], list(range(atoms.size())))
        further_reactive_atoms = self.further_exploration_filter.filter_atoms([structure], reactive_atoms)
        # Get all ordered pairs of reactive atoms
        connectivity_analyzer = ConnectivityAnalyzer(structure)
        further_diss_pairs: List[Tuple[int, int]] = self._get_filtered_intraform_and_diss(
            structure, further_reactive_atoms, connectivity_analyzer)[1]
        result: List[Tuple[List[List[Tuple[int, int]]], int]] = []
        # Loop over possible options of dissociations
        for n_diss in range(self.options.unimolecular_options.min_bond_dissociations,
                            self.options.unimolecular_options.max_bond_dissociations + 1):
            if n_diss < 1:
                continue
            if not with_exact_settings_check and n_diss < current_cache_entry[2]:
                # skip already if further exploration is not allowed for this number of dissociations
                continue
            # determine allowed number of simultaneous dissociations for each subtype based on cache and settings
            pairs = self.further_exploration_filter.filter_atom_pairs([structure], further_diss_pairs)
            for slow_diss_pairs in combinations(pairs, n_diss):
                diss_pairs = list(slow_diss_pairs)
                # we assume here that
                # The further filter will be more restrictive than the fast filter, hence we don't need to add
                # dissociative coordinate, since they are covered by the standard way
                reaction_type = connectivity_analyzer.get_reaction_type(diss_pairs)
                if reaction_type != ReactionType.Disconnective:
                    continue
                # Check whether trial coordinate is to be considered
                if self.reactive_site_filter.filter_reaction_coordinates([structure], [diss_pairs]) and \
                        self.further_exploration_filter.filter_reaction_coordinates([structure], [diss_pairs]):
                    result.append(([diss_pairs], n_diss))

        return result

    @_sanity_check_wrapper
    def unimolecular_reactions(self, structure: db.Structure, with_exact_settings_check: bool = False) -> None:
        """
        Creates reactive complex calculations corresponding to the unimolecular
        reactions of the structure if there is not already a calculation to
        search for a reaction of the same structure with the same job order, model and settings.

        Parameters
        ----------
        structure : db.Structure
            The structure to be considered. The Structure has to
            be linked to a database.
        with_exact_settings_check : bool
            If True, more expensive queries are carried out to check if the settings of the
            calculations are exactly the same as the settings of the trial generator. This allows to add more
            inclusive additional reaction trials but the queries are less efficient, therefore this option
            should be only toggled if necessary.
        """
        fast_coordinates = self.unimolecular_fast_coordinates(structure, with_exact_settings_check)
        further_coordinates = self.unimolecular_further_explore_coordinates(structure, with_exact_settings_check)
        if not fast_coordinates and not further_coordinates:
            return
        connectivity_analyzer = ConnectivityAnalyzer(structure)
        structure_id = structure.id()
        str_structure_id = str(structure.id())
        allowed_types: List[bool] = [True,
                                     self.options.unimolecular_options.always_further_explore_dissociative_reactions,
                                     self.options.unimolecular_options.enable_further_explorations]
        current_cache_entry = self._cache[str_structure_id]

        for filtered_coordinates, n_diss in fast_coordinates:
            dissociative_coordinates = []
            """ Fast dissociations """
            if allowed_types[0]:
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
                                                              self.options.unimolecular_options.job,
                                                              self.options.unimolecular_options.cutting_job_settings,
                                                              check_for_existing=True)
                    if cid is not None:
                        calculation_ids.append(cid)
                if calculation_ids:
                    # only add first due to limited entry space for mongodb document
                    structure.add_calculation(self.options.unimolecular_options.job.order, calculation_ids[0])
                # update cache
                current_cache_entry[0] = n_diss

            """ Slow dissociative coordinates """
            if allowed_types[1]:
                calculation_ids = []
                # TODO see fast dissociations
                for diss_pairs in dissociative_coordinates:
                    cid = self._add_reactive_complex_calculation([structure_id], [], diss_pairs,
                                                                 self.options.unimolecular_options.further_job,
                                                                 self.options.unimolecular_options.further_job_settings,
                                                                 check_for_existing=True)
                    if cid is not None:
                        calculation_ids.append(cid)
                if calculation_ids:
                    # only add first due to limited entry space for mongodb document
                    structure.add_calculations(self.options.unimolecular_options.further_job.order, calculation_ids)
                # update cache
                current_cache_entry[1] = n_diss

        for filtered_coordinates, n_diss in further_coordinates:
            """ Slow dissociations """
            if allowed_types[2]:  # should not be needed, but just to be sure
                calculation_ids = []
                # TODO see fast dissociations
                for slow_diss_pairs in filtered_coordinates:
                    cid = self._add_reactive_complex_calculation([structure_id], [], slow_diss_pairs,
                                                                 self.options.unimolecular_options.further_job,
                                                                 self.options.unimolecular_options.further_job_settings,
                                                                 check_for_existing=True)
                    if cid is not None:
                        calculation_ids.append(cid)
                if calculation_ids:
                    # only add first due to limited entry space for mongodb document
                    structure.add_calculations(self.options.unimolecular_options.further_job.order, calculation_ids)
                # update cache
                current_cache_entry[2] = n_diss

    def _add_disconnective_calculation(
            self,
            reactive_structure: db.ID,
            dissociation_pairs: List[Tuple[int, int]],
            job: db.Job,
            settings: utils.ValueCollection,
            check_for_existing: bool = False
    ) -> Optional[db.ID]:
        """
        Adds a reactive calculation for a bond-based approach (i.e., one during which it is tried to form/break
        bonds between designated atom pairs) to the database and puts it on hold.

        Parameters
        ----------

        reactive_structure : db.ID
            The ID of the reactant.
        dissociation_pairs : List[Tuple[int, int]]
            List of atom index pairs in between which a bond dissociation is to
            be encouraged.
        job : scine_database.Job
        settings : scine_utilities.ValueCollection
        """
        this_settings = self._get_settings(settings)

        if job.order == "scine_dissociation_cut":
            # job takes lists of integer where elements 0/1, 2/3 ... N-1, N are combined with each other
            # Flatten pair lists
            this_settings["dissociations"] = FurtherExplorationFilter.dissociation_setting_from_reaction_coordinate(
                dissociation_pairs)
        else:
            raise RuntimeError("Only 'scine_dissociation_cut' order supported for fast dissociations")

        if check_for_existing and get_calculation_id(job.order, [reactive_structure], self.options.model,
                                                     self._calculations, settings=this_settings) is not None:
            return None
        calculation = db.Calculation(db.ID(), self._calculations)
        calculation.create(self.options.model, job, [reactive_structure])
        # Sleep a bit in order not to make the DB choke
        time.sleep(0.001)
        calculation.set_settings(deepcopy(this_settings))
        calculation.set_status(db.Status.HOLD)
        return calculation.id()

    def get_unimolecular_job_order(self) -> str:
        return self.options.unimolecular_options.job.order

    def get_bimolecular_job_order(self) -> str:
        raise RuntimeError("Bimolecular reactions are not supported by this TrialGenerator.")
