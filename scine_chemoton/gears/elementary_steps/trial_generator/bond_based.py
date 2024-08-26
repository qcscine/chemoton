#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import time
from collections import defaultdict
from typing import List, Optional, Tuple, Union, Dict
from itertools import combinations, product
from scipy.special import comb

# Third party imports
from numpy import ndarray
from scine_database.queries import calculation_exists_in_structure, get_calculation_id
import scine_database as db
import scine_utilities as utils
import scine_molassembler as masm

# Local application imports
from scine_chemoton.utilities.options import BaseOptions
from .connectivity_analyzer import ReactionType, ConnectivityAnalyzer
from . import TrialGenerator, _sanity_check_wrapper
from ...network_refinement.enabling import (
    EnableCalculationResults, PlaceHolderCalculationEnabling, EnableJobSpecificCalculations
)


class BondBased(TrialGenerator):
    """
    Class to generate reactive complex calculations via bond-based approaches.

    Attributes
    ----------
    options : BondBased.Options
        The options for generating reactive complex calculations.
    reactive_site_filter : ReactiveSiteFilter
        The filter applied to determine reactive sites, reactive pairs and trial
        reaction coordinates.
    """

    def __init__(self):
        super().__init__()
        self.results_enabling_policy: EnableCalculationResults = PlaceHolderCalculationEnabling()
        """
        If given (must be of class EnableJobSpecificCalculations), the results enabling policy is applied.
        """

    class Options(TrialGenerator.Options):
        """
        The options for bond-based reactive complex enumeration.
        """

        __slots__ = ("unimolecular_options", "bimolecular_options")

        class BimolOptions(BaseOptions):
            """
            The options for the generation and exploration of bimolecular reactions.

            NOTE: The minimum and maximum bounds regarding the number of bonds to
            be modified/formed/broken have to be understood as strict limitations.
            If different options are in conflict with each other the stricter limit
            is applied. For example, if you set `max_bond_dissociations` to 2
            but `max_bond_modifications` to 1 you will never get a trial
            reaction coordinate containing two dissociations.
            """

            __slots__ = (
                "job",
                "job_settings",
                "min_bond_modifications",
                "max_bond_modifications",
                "min_inter_bond_formations",
                "max_inter_bond_formations",
                "min_intra_bond_formations",
                "max_intra_bond_formations",
                "min_bond_dissociations",
                "max_bond_dissociations",
                "complex_generator",
                "minimal_spin_multiplicity",
            )

            def __init__(self) -> None:
                self.job: db.Job = db.Job("scine_react_complex_nt2")
                """
                db.Job (Scine::Database::Calculation::Job)
                    The Job used to evaluate the elementary step trial calculations.
                    The default is: the `scine_react_complex_nt2` order on a single core.
                """
                self.job_settings: utils.ValueCollection = utils.ValueCollection({})
                """
                Dict[str, Union[bool, int, float]]
                    Additional settings added to the elementary step trial calculation.
                    Empty by default.
                """
                self.min_bond_modifications = 1
                """
                int
                    The minimum number of bond modifications (formation or
                    dissociation) to be encouraged simultaneously during an
                    elementary step trial calculation. By default 1.
                """
                self.max_bond_modifications = 1
                """
                int
                    The maximum number of bond modifications (formation or
                    dissociation) to be encouraged simultaneously during an
                    elementary step trial calculation. By default 1.
                """
                self.min_inter_bond_formations = 1
                """
                int
                    The minimum number of bond formations between the two reactants
                    to be encouraged simultaneously during an elementary step trial calculation.
                    Has to be equal or greater than 1.
                    By default 1.
                """
                self.max_inter_bond_formations = 1
                """
                int
                    The maximum number of bond formations between the two reactants
                    to be encouraged simultaneously during an elementary step trial calculation.
                    More than two intermolecular bond formations are currently not supported.
                    By default 1.
                """
                self.min_intra_bond_formations = 0
                """
                int
                    The minimum number of intramolecular bond formations to be
                    encouraged simultaneously during an elementary step trial calculation.
                    By default 0.
                """
                self.max_intra_bond_formations = 0
                """
                int
                    The maximum number of bond formations to be encouraged
                    simultaneously during an elementary step trial calculation.
                    By default 0.
                """
                self.min_bond_dissociations = 0
                """
                int
                    The minimum number of intramolecular bond dissociations to
                    be encouraged simultaneously during an elementary step trial calculation.
                    By default 0.
                """
                self.max_bond_dissociations = 0
                """
                int
                    The maximum number of bond dissociations to be encouraged
                    simultaneously during an elementary step trial calculation.
                    By default 1.
                """
                from ....utilities.reactive_complexes.inter_reactive_complexes import InterReactiveComplexes

                self.complex_generator = InterReactiveComplexes()
                """
                InterReactiveComplexes
                    The generator used for the composition of reactive complexes.
                """
                self.minimal_spin_multiplicity = False
                """
                bool
                    Whether to assume max spin recombination, thus assuming minimal resulting spin, or take combination
                    of input spin multiplicities. (default: False)
                    True: | multiplicity1 - multiplicity2 | - 1
                    False: sum(multiplicities) - 1
                """

        class UnimolOptions(BaseOptions):
            """
            The options for the generation and exploration of unimolecular
            reactions.
            """

            __slots__ = (
                "job",
                "min_bond_modifications",
                "max_bond_modifications",
                "min_bond_formations",
                "max_bond_formations",
                "min_bond_dissociations",
                "max_bond_dissociations",
                "job_settings_associative",
                "job_settings_dissociative",
                "job_settings_disconnective",
            )

            def __init__(self) -> None:
                self.job: db.Job = db.Job("scine_react_complex_nt2")
                """
                db.Job (Scine::Database::Calculation::Job)
                    The Job used to evaluate the possible reactions.
                    Jobs with the order `scine_react_complex_nt2` are
                    supported. The default is: the `scine_react_complex_nt2`
                    order on a single core.
                """
                self.min_bond_modifications = 1
                """
                int
                    The minimum number of bond modifications (formation or
                    dissociation) to be encouraged simultaneously during an elementary step trial calculation.
                    By default 1.
                """
                self.max_bond_modifications = 1
                """
                int
                    The maximum number of bond modifications (formation or
                    dissociation) to be encouraged simultaneously during an elementary step trial calculation.
                    By default 1.
                """
                self.min_bond_formations = 1
                """
                int
                    The minimum number of bond formations to be encouraged
                    simultaneously during an elementary step trial calculation. By default 1.
                """
                self.max_bond_formations = 1
                """
                int
                    The maximum number of bond formations to be encouraged
                    simultaneously during an elementary step trial calculation. By default 1.
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
                self.job_settings_associative: utils.ValueCollection = utils.ValueCollection({})
                """
                utils.ValueCollection
                    Additional settings passed to Calculation evaluating the possible
                    reactions. These settings are passed to all calculations that are
                    expected to result in the formations of at least one bond, i.e.,
                    that at least one of the reactive atom pairs is not bound in the start
                    structure.
                    Empty by default.
                """
                self.job_settings_dissociative: utils.ValueCollection = utils.ValueCollection({})
                """
                utils.ValueCollection
                    Additional settings passed to Calculation evaluating the possible
                    reactions. These settings are passed to all calculations for which
                    all of the reactive atom pairs are bound in the start structure
                    but cutting them apart would not result into two separate molecules.
                    Empty by default.
                """
                self.job_settings_disconnective: utils.ValueCollection = utils.ValueCollection({})
                """
                utils.ValueCollection
                    Additional settings passed to Calculation evaluating the possible
                    reactions. These settings are passed to all calculations for which
                    all of the reactive atom pairs are bound in the start structure
                    and cutting them apart would result into at least two separate molecules.
                    Empty by default.
                """

        unimolecular_options: UnimolOptions
        bimolecular_options: BimolOptions

        def __init__(self, parent: Optional[TrialGenerator] = None) -> None:
            super().__init__(parent)
            self.unimolecular_options = self.UnimolOptions()
            """
            UnimolOptions
                The options for reactions involving a single molecule.
            """
            self.bimolecular_options = self.BimolOptions()
            """
            BimolOptions
                The options for reactions involving two molecules, that are set up
                to be associative in nature.
            """

    options: Options

    def clear_cache(self):
        pass

    @_sanity_check_wrapper
    def bimolecular_coordinates(self, structure_list: List[db.Structure], with_exact_settings_check: bool = False) \
            -> Dict[
        Tuple[List[Tuple[int, int]], int],
        List[Tuple[ndarray, ndarray, float, float]]
    ]:

        if self._quick_bimolecular_already_exists(structure_list, do_fast_query=not with_exact_settings_check):
            return {}

        # Get reactive atoms
        n_atoms1 = structure_list[0].get_atoms().size()
        n_atoms2 = structure_list[1].get_atoms().size()
        reactive_atoms = self.reactive_site_filter.filter_atoms(
            structure_list, list(range(n_atoms1)) + [i + n_atoms1 for i in range(n_atoms2)]
        )
        reactive_atoms1 = [i for i in reactive_atoms if i < n_atoms1]
        reactive_atoms2 = [i - n_atoms1 for i in reactive_atoms if i >= n_atoms1]

        # If needed get reactive intrastructural coordinates for both structures
        # split into formation and dissociation
        allows_intra_reactions = (self.options.bimolecular_options.max_intra_bond_formations > 0 or
                                  self.options.bimolecular_options.max_bond_dissociations > 0)
        will_probe_intra_reactions = (self.options.bimolecular_options.max_bond_modifications >
                                      self.options.bimolecular_options.min_inter_bond_formations)
        if allows_intra_reactions and will_probe_intra_reactions:
            filtered_form_pairs1, filtered_diss_pairs1 = self._get_filtered_intraform_and_diss(
                structure_list[0], reactive_atoms1
            )
            filtered_form_pairs2, filtered_diss_pairs2 = self._get_filtered_intraform_and_diss(
                structure_list[1], reactive_atoms2
            )
            all_filtered_intra_form_pairs = filtered_form_pairs1 + [
                (pair[0] + n_atoms1, pair[1] + n_atoms1) for pair in filtered_form_pairs2
            ]
            all_filtered_diss_pairs = filtered_diss_pairs1 + [
                (pair[0] + n_atoms1, pair[1] + n_atoms1) for pair in filtered_diss_pairs2
            ]
        else:
            all_filtered_intra_form_pairs = []
            all_filtered_diss_pairs = []

        # Generate reactive pair combinations across the two structures from reactive sites
        shifted_inter_pairs = list(product(reactive_atoms1, (idx + n_atoms1 for idx in reactive_atoms2)))
        filtered_shifted_inter_pairs = self.reactive_site_filter.filter_atom_pairs(
            structure_list, shifted_inter_pairs)
        filtered_unshifted_inter_pairs = [(pair[0], pair[1] - n_atoms1) for pair in filtered_shifted_inter_pairs]
        # Loop through all allowed combinations of inter/intra bond formations and dissociations
        true_max_inter_bond_formations = min(self.options.bimolecular_options.max_inter_bond_formations,
                                             self.options.bimolecular_options.max_bond_modifications)
        true_max_intra_bond_formations = min(self.options.bimolecular_options.max_intra_bond_formations,
                                             self.options.bimolecular_options.max_bond_modifications)
        true_max_bond_dissociations = min(self.options.bimolecular_options.max_bond_dissociations,
                                          self.options.bimolecular_options.max_bond_modifications)
        range_inter = range(
            self.options.bimolecular_options.min_inter_bond_formations,
            true_max_inter_bond_formations + 1)
        range_intra = range(
            self.options.bimolecular_options.min_intra_bond_formations,
            true_max_intra_bond_formations + 1)
        range_diss = range(self.options.bimolecular_options.min_bond_dissociations,
                           true_max_bond_dissociations + 1)

        result: Dict[
            Tuple[List[Tuple[int, int]], int],
            List[Tuple[ndarray, ndarray, float, float]]
        ] = {}
        for n_inter_form, n_intra_form, n_diss in product(range_inter, range_intra, range_diss):
            s = n_inter_form + n_intra_form + n_diss
            if s > self.options.bimolecular_options.max_bond_modifications:
                continue
            elif s < self.options.bimolecular_options.min_bond_modifications:
                continue

            # Batch all trials before doing a final filter run and adding
            #  the remaining trials. This reduces the number of filter calls
            #  and thus the number of DB look-ups.

            # keys: List of pairs
            # values: information required to build different reactive complexes
            batch: Dict[
                Tuple[Tuple[int, int], ...],
                List[Tuple[ndarray, ndarray, float, float]]
            ] = defaultdict(list)

            reactive_inter_coords = list(combinations(filtered_unshifted_inter_pairs, n_inter_form))
            for (inter_coord, align1, align2, rot, spread, ) in \
                    self.options.bimolecular_options.complex_generator.generate_reactive_complexes(
                        structure_list[0], structure_list[1], reactive_inter_coords):  # type: ignore

                # Shift to complex indexing
                shifted_inter_coord = tuple((pair[0], pair[1] + n_atoms1) for pair in inter_coord)

                for intra_form_pairs in combinations(all_filtered_intra_form_pairs, n_intra_form):
                    for diss_pairs in combinations(all_filtered_diss_pairs, n_diss):

                        form_pairs = shifted_inter_coord + intra_form_pairs
                        key = form_pairs + diss_pairs
                        batch[key].append((align1, align2, rot, spread))

            filter_result = self.reactive_site_filter.filter_reaction_coordinates(
                structure_list,
                list(batch.keys())  # type: ignore
            )
            for filtered in filter_result:
                result[(filtered, n_diss)] = batch[filtered]  # type: ignore
        return result

    @_sanity_check_wrapper
    def bimolecular_reactions(self, structure_list: List[db.Structure], with_exact_settings_check: bool = False):
        """
        Creates reactive complex calculations corresponding to the bimolecular
        reactions between the structures if there is not already a calculation
        to search for a reaction of the same structures with the same job order.

        Parameters
        ----------
        structure_list : List[db.Structure]
            List of the two structures to be considered.
            The Structures have to be linked to a database.
        with_exact_settings_check : bool
            If True, more expensive queries are carried out to check if the settings of the
            calculations are exactly the same as the settings of the trial generator. This allows to add more
            inclusive additional reaction trials but the queries are less efficient, therefore this option
            should be only toggled if necessary.
        """
        # already includes quick check for existing calculation
        coordinate_complex_build_info = self.bimolecular_coordinates(structure_list, with_exact_settings_check)
        if not coordinate_complex_build_info:
            return
        structure_id_list = [s.id() for s in structure_list]
        new_calculation_ids = []
        for (pairs, n_diss), complexes in coordinate_complex_build_info.items():
            for align1, align2, rot, spread in complexes:
                cid = self._add_reactive_complex_calculation(
                    structure_id_list,
                    pairs[:len(pairs) - n_diss],
                    pairs[len(pairs) - n_diss:],
                    self.options.bimolecular_options.job,
                    self.options.bimolecular_options.job_settings,
                    list(align1),
                    list(align2),
                    rot,
                    spread,
                    0.0,
                    check_for_existing=with_exact_settings_check
                )
                if cid is not None:
                    new_calculation_ids.append(cid)
        if new_calculation_ids:
            for s in structure_list:
                s.add_calculations(self.options.bimolecular_options.job.order, [new_calculation_ids[0]])

    @_sanity_check_wrapper
    def unimolecular_reactions(self, structure: db.Structure, with_exact_settings_check: bool = False):
        """
        Creates reactive complex calculations corresponding to the unimolecular
        reactions of the structure if there is not already a calculation to
        search for a reaction of the same structure with the same job order.

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
        # already includes quick check for existing calculation
        filtered_coordinates_per_ndiss = self.unimolecular_coordinates(structure, with_exact_settings_check)
        if not filtered_coordinates_per_ndiss:
            return
        structure_id = structure.id()
        connectivity_analyzer = ConnectivityAnalyzer(structure)
        new_calculation_ids = []
        for filter_result, n_diss in filtered_coordinates_per_ndiss:
            for pairs in filter_result:
                # Get reaction type:
                reaction_type = connectivity_analyzer.get_reaction_type(pairs)
                # Set up reactive complex calculation
                if reaction_type in (ReactionType.Associative, ReactionType.Mixed):
                    settings = self.options.unimolecular_options.job_settings_associative
                elif reaction_type == ReactionType.Dissociative:
                    settings = self.options.unimolecular_options.job_settings_dissociative
                elif reaction_type == ReactionType.Disconnective:
                    settings = self.options.unimolecular_options.job_settings_disconnective
                else:
                    raise RuntimeError(f"Unknown reaction type {reaction_type}")

                cid = self._add_reactive_complex_calculation(
                    [structure_id],
                    pairs[:len(pairs) - n_diss],
                    pairs[len(pairs) - n_diss:],
                    self.options.unimolecular_options.job,
                    settings,
                    check_for_existing=with_exact_settings_check
                )
                if cid is not None:
                    new_calculation_ids.append(cid)
        if new_calculation_ids:
            structure.add_calculations(self.options.unimolecular_options.job.order, [new_calculation_ids[0]])

    def _quick_unimolecular_already_exists(self, structure: db.Structure, do_fast_query: bool) -> bool:
        if not do_fast_query:
            return False
        # Rule out compounds too small for intramolecular reactions right away
        atoms = structure.get_atoms()
        if atoms.size() == 1:
            return False
        if atoms.size() == 2 and self.options.unimolecular_options.max_bond_dissociations < 1:
            return False
        if not calculation_exists_in_structure(self.options.unimolecular_options.job.order, [structure.id()],
                                               self.options.model, self._structures, self._calculations):
            return False
        if isinstance(self.results_enabling_policy, EnableJobSpecificCalculations):
            self.results_enabling_policy.process_calculations_of_structures([structure.id()])
        return True

    def _quick_bimolecular_already_exists(self, structure_list: List[db.Structure], do_fast_query: bool) -> bool:
        # Check number of compounds
        if len(structure_list) != 2:
            raise RuntimeError("Exactly two structures are needed for setting up a bimolecular reaction.")
        if not do_fast_query:
            return False
        structure_ids = [s.id() for s in structure_list]
        if not calculation_exists_in_structure(self.options.bimolecular_options.job.order,
                                               structure_ids, self.options.model,
                                               self._structures, self._calculations):
            return False
        if isinstance(self.results_enabling_policy, EnableJobSpecificCalculations):
            self.results_enabling_policy.process_calculations_of_structures(structure_ids)
        return True

    @_sanity_check_wrapper
    def unimolecular_coordinates(self, structure: db.Structure, with_exact_settings_check: bool = False) \
            -> List[Tuple[List[List[Tuple[int, int]]], int]]:
        if self._quick_unimolecular_already_exists(structure, do_fast_query=not with_exact_settings_check):
            return []

        atoms = structure.get_atoms()
        # Get all reactive atoms
        reactive_atoms = self.reactive_site_filter.filter_atoms([structure], list(range(atoms.size())))
        # Get all ordered pairs of reactive atoms
        connectivity_analyzer = ConnectivityAnalyzer(structure)
        filtered_form_pairs, filtered_diss_pairs = self._get_filtered_intraform_and_diss(
            structure, reactive_atoms, connectivity_analyzer
        )

        # Loop over all allowed combinations of association and dissociation counts
        true_max_bond_formations = min(self.options.unimolecular_options.max_bond_formations,
                                       self.options.unimolecular_options.max_bond_modifications)
        true_max_bond_dissociations = min(self.options.unimolecular_options.max_bond_dissociations,
                                          self.options.unimolecular_options.max_bond_modifications)
        range_asso = range(self.options.unimolecular_options.min_bond_formations, true_max_bond_formations + 1)
        range_diss = range(self.options.unimolecular_options.min_bond_dissociations,
                           true_max_bond_dissociations + 1)

        result = []
        for n_asso, n_diss in product(range_asso, range_diss):
            s = n_asso + n_diss
            if s > self.options.unimolecular_options.max_bond_modifications:
                continue
            elif s < self.options.unimolecular_options.min_bond_modifications:
                continue

            # Batch all trials before doing a final filter run and adding the
            #  remaining trials. This reduces the number of filter calls and
            #  thus the number of DB look-ups.
            batch: List[tuple] = []
            # Loop over all allowed combinations of reaction coordinates within
            #  the current count of associations and dissociations
            for asso_pairs in combinations(filtered_form_pairs, n_asso):
                for diss_pairs in combinations(filtered_diss_pairs, n_diss):
                    batch += [asso_pairs + diss_pairs]
            filter_result = self.reactive_site_filter.filter_reaction_coordinates(
                [structure],
                batch  # type: ignore
            )
            result.append((filter_result, n_diss))

        return result

    @_sanity_check_wrapper
    def estimate_n_unimolecular_trials(
        self, structure_file: str,
        n_reactive_bound_pairs: Optional[int] = None,
        n_reactive_unbound_pairs: Optional[int] = None
    ):
        """
        Estimates the number of unimolecular reactive coordinate trials expected
        to arise directly (i.e. as single step elementary reaction trials) from
        the given structure without taking reactive site filters into account.
        If the number of reactive pairs shall be limited for this estimate,
        specify `n_reactive_bound_pairs` and/or `n_reactive_unbound_pairs`.
        The structure's connectivity is perceived from interatomic distances.

        Note: The number of trials increases quickly with the number of allowed
              bond modifications. Be aware!

        Parameters
        ----------
        structure_file : str
            An xyz or mol file with the structure of interest.
        n_reactive_bound_pairs : int, optional
            The number of bound reactive pairs to consider.
            If None, all bound atom pairs are included.
        n_reactive_unbound_pairs : int, optional
            The number of unbound reactive pairs to consider.
            If None, all unbound atom pairs are included.

        Returns
        -------
        int
            The number of reactive trial coordinates expected to arise from this
            structure.
        """
        # Read in atom collection and interpret connectivity from interatomic distances
        atoms = utils.io.read(structure_file)[0]
        n_unbound_pairs, n_bound_pairs = self._get_bound_unbound_pairs_from_atoms(
            atoms, n_reactive_bound_pairs, n_reactive_unbound_pairs
        )

        # Get all types of bond modification patterns that are compliant with the options
        # and add the associated number of trials
        n_trials = 0
        for n_form in range(
            self.options.unimolecular_options.min_bond_formations,
                self.options.unimolecular_options.max_bond_formations + 1
        ):
            for n_diss in range(
                self.options.unimolecular_options.min_bond_dissociations,
                    self.options.unimolecular_options.max_bond_dissociations + 1
            ):
                if not (
                    self.options.unimolecular_options.min_bond_modifications
                    <= n_form + n_diss
                    <= self.options.unimolecular_options.max_bond_modifications
                ):
                    continue
                n_form_combis = comb(n_unbound_pairs, n_form)
                n_diss_combis = comb(n_bound_pairs, n_diss)
                n_form_diss_combis = n_form_combis * n_diss_combis
                n_trials += n_form_diss_combis

        return n_trials

    @_sanity_check_wrapper
    def estimate_n_bimolecular_trials(
        self,
        structure_file1: str,
        structure_file2: str,
        attack_points_per_site: int = 3,
        n_inter_reactive_pairs: Optional[int] = None,
        n_reactive_bound_pairs1: Optional[int] = None,
        n_reactive_unbound_pairs1: Optional[int] = None,
        n_reactive_bound_pairs2: Optional[int] = None,
        n_reactive_unbound_pairs2: Optional[int] = None,
    ):
        """
        Estimates the number of bimolecular reactive coordinate trials expected
        to arise directly (i.e. as single step elementary reaction trials) from
        the given structures without taking reactive site filters into account.
        If the number of reactive pairs shall be limited for this estimate,
        specify `n_inter_reactive_pairs`, `n_reactive_bound_pairs` and/or
        `n_reactive_unbound_pairs`.
        Note that this method only estimates the number of trials but is not
        necessarily resulting in the exact number that would be generated by the
        trial generator:
        The structures connectivities are perceived from interatomic distances.
        Multiple attack points are not calculated from the structure's geometry
        but only included based on a fixed input value
        (`attack_points_per_site`).
        The number of rotamers is not accounted for correctly for
        intermolecular coordinates involving two reactive pairs with one
        monoatomic fragment.

        Note: The number of trials increases quickly with the number of allowed
              bond modifications. Be aware!

        Parameters
        ----------
        structure_file1 : str
            An xyz or mol file with the structure of interest.
        structure_file2 : str
            An xyz or mol file with the structure of interest.
        attack_points_per_site : int
            The number of attack points per intermolecular reactive site to
            consider if multiple attack points are enabled in the reactive
            complex generator.
            Note that these will not be calculated from the structures'
            geometries in this estimator.
            By default 3.
        n_inter_reactive_pairs : int, optional
            The number of intermolecular reactive pairs to consider.
            If None, all interstructural atom pairs are included.
        n_reactive_bound_pairs1 : int, optional
            The number of bound reactive pairs in the first structure to
            consider. If None, all bound atom pairs of structure 1
            are included.
        n_reactive_bound_pairs2 : int, optional
            The number of bound reactive pairs in the second structure to
            consider. If None, all bound atom pairs of structure 2
            are included.
        n_reactive_unbound_pairs1 : int, optional
            The number of unbound reactive pairs in the first structure to
            consider. If None, all unbound atom pairs are included.
        n_reactive_unbound_pairs2 : int, optional
            The number of unbound reactive pairs in the second structure to
            consider. If None, all unbound atom pairs are included.

        Returns
        -------
        int
            The number of reactive trial coordinates expected to arise from this
            structure.
        """
        # Read in atom collections and get intramolecular bound and unbound pairs
        atoms1 = utils.io.read(structure_file1)[0]
        n_unbound_pairs1, n_bound_pairs1 = self._get_bound_unbound_pairs_from_atoms(
            atoms1, n_reactive_bound_pairs1, n_reactive_unbound_pairs1
        )
        atoms2 = utils.io.read(structure_file2)[0]
        n_unbound_pairs2, n_bound_pairs2 = self._get_bound_unbound_pairs_from_atoms(
            atoms2, n_reactive_bound_pairs2, n_reactive_unbound_pairs2
        )
        n_inter_pairs = atoms1.size() * atoms2.size() if n_inter_reactive_pairs is None else n_inter_reactive_pairs
        n_bound_pairs = n_bound_pairs1 + n_bound_pairs2
        n_unbound_pairs = n_unbound_pairs1 + n_unbound_pairs2

        # Get all types of bond modification patterns that are compliant with the options
        # and add the associated number of trials
        n_trials = 0
        # At least on intermolecular coord required
        if self.options.bimolecular_options.max_inter_bond_formations < 1:
            return 0

        for n_inter in range(
            self.options.bimolecular_options.min_inter_bond_formations,
                self.options.bimolecular_options.max_inter_bond_formations + 1
        ):
            for n_form in range(
                self.options.bimolecular_options.min_intra_bond_formations,
                    self.options.bimolecular_options.max_intra_bond_formations + 1,
            ):
                for n_diss in range(
                    self.options.bimolecular_options.min_bond_dissociations,
                        self.options.bimolecular_options.max_bond_dissociations + 1
                ):
                    if not (
                        self.options.bimolecular_options.min_bond_modifications
                        <= n_form + n_diss + n_inter
                        <= self.options.bimolecular_options.max_bond_modifications
                    ):
                        continue
                    n_inter_combis = comb(n_inter_pairs, n_inter)
                    n_form_combis = comb(n_unbound_pairs, n_form)
                    n_diss_combis = comb(n_bound_pairs, n_diss)
                    n_form_diss_combis = n_form_combis * n_diss_combis * n_inter_combis

                    # Add rotamers
                    if n_inter == 1:
                        n_form_diss_combis *= self.options.bimolecular_options.complex_generator.options.number_rotamers
                    # There can be two intermolecular coordinates involving the same atom twice (e.g. "atom on bond")
                    # These are weighted with number_rotamers in the real enumeration but with
                    # number_rotamers_two_on_two here
                    elif n_inter == 2:
                        n_form_diss_combis *= (
                            self.options.bimolecular_options.complex_generator.options.number_rotamers_two_on_two
                        )

                    n_trials += n_form_diss_combis

        # Take into account estimate of attack points per site
        if self.options.bimolecular_options.complex_generator.options.multiple_attack_points:
            n_trials *= attack_points_per_site * attack_points_per_site
        return n_trials

    def _add_reactive_complex_calculation(
        self,
        reactive_structures: List[db.ID],
        association_pairs: List[Tuple[int, int]],
        dissociation_pairs: List[Tuple[int, int]],
        job: db.Job,
        settings: utils.ValueCollection,
        lhs_alignment: Optional[List[float]] = None,
        rhs_alignment: Optional[List[float]] = None,
        x_rotation: Optional[float] = None,
        spread: Optional[float] = None,
        displacement: Optional[float] = None,
        charge: Optional[int] = None,
        multiplicity: Optional[int] = None,
        check_for_existing: bool = False,
    ) -> Union[db.ID, None]:
        """
        Adds a reactive calculation for a bond-based approach
        (i.e., one during which it is tried to form/break
        bonds between designated atom pairs) to the database and puts it on hold.
        Note: This function does not add the calculation into the calculation
        list of the structures used in the calculation. This can and has to be
        done in batches afterwards.

        Parameters
        ----------

        reactive_structures : List[db.ID]
            List of the IDs of the reactants.
        association_pairs:: List[Tuple(int, int))]
            List of atom index pairs in between which a bond formation is to be
            encouraged.
        dissociation_pairs:: List[Tuple(int, int))]
            List of atom index pairs in between which a bond dissociation is to
            be encouraged.
        job : scine_database.Job
        settings : scine_utilities.ValueCollection
        lhs_alignment : List[float], length=9
            In case of two structures building the reactive complex, this option
            describes a rotation of the first structure (index 0) that aligns
            the reaction coordinate along the x-axis (pointing towards +x).
            The rotation assumes that the geometric mean position of all
            atoms in the reactive site (``lhs_list``) is shifted into the
            origin.
        rhs_alignment : List[float], length=9
            In case of two structures building the reactive complex, this option
            describes a rotation of the second structure (index 1) that aligns
            the reaction coordinate along the x-axis (pointing towards -x).
            The rotation assumes that the geometric mean position of all
            atoms in the reactive site (``rhs_list``) is shifted into the
            origin.
        x_rotation : float
            In case of two structures building the reactive complex, this option
            describes a rotation angle around the x-axis of one of the two
            structures after ``lhs_alignment`` and ``rhs_alignment`` have
            been applied.
        spread : float
            In case of two structures building the reactive complex, this option
            gives the distance by which the two structures are moved apart along
            the x-axis after ``lhs_alignment``, ``rhs_alignment``, and
            ``x_rotation`` have been applied.
        displacement : float
            In case of two structures building the reactive complex, this option
            adds a random displacement to all atoms (random direction, random
            length). The maximum length of this displacement (per atom) is set to
            be the value of this option.
        multiplicity : int
            This option sets the ``spin_multiplicity`` of the reactive complex.
        charge : int
            This option sets the ``molecular_charge`` of the reactive complex.
        check_for_existing : bool
            Whether it should be checked if a calculation with these exact
            settings and model already exists or not (default: False)

        Returns
        -------
        calculation : scine_database.ID
            A calculation that is on hold.
        """
        this_settings = self._get_settings(settings)
        if lhs_alignment is not None:
            this_settings["rc_x_alignment_0"] = lhs_alignment
        if rhs_alignment is not None:
            this_settings["rc_x_alignment_1"] = rhs_alignment
        if x_rotation is not None:
            this_settings["rc_x_rotation"] = x_rotation
        if spread is not None:
            this_settings["rc_x_spread"] = spread
        if displacement is not None:
            this_settings["rc_displacement"] = displacement
        if multiplicity is not None:
            this_settings["rc_spin_multiplicity"] = multiplicity
        if charge is not None:
            this_settings["rc_molecular_charge"] = charge

        if "nt2" in job.order:
            # nt2 job takes lists of integer where elements 0/1, 2/3 ... N-1, N are combined with each other
            # Flatten pair lists
            this_settings["nt_nt_associations"] = [idx for idx_pair in association_pairs for idx in idx_pair]
            this_settings["nt_nt_dissociations"] = [idx for idx_pair in dissociation_pairs for idx in idx_pair]
        else:
            raise RuntimeError(
                "Only 'NT2'-based job orders are supported for bond-based reactive complex calculations."
            )

        if len(reactive_structures) > 1:
            this_settings["rc_minimal_spin_multiplicity"] = bool(
                self.options.bimolecular_options.minimal_spin_multiplicity)

        if check_for_existing:
            calc_id = get_calculation_id(job.order, reactive_structures, self.options.model, self._calculations,
                                         settings=this_settings)
            if calc_id is not None:
                if isinstance(self.results_enabling_policy, EnableJobSpecificCalculations):
                    self.results_enabling_policy.process(db.Calculation(calc_id, self._calculations))
                return None

        calculation = db.Calculation()
        calculation.link(self._calculations)
        calculation.create(self.options.model, job, reactive_structures)
        # Sleep a bit in order not to make the DB choke
        time.sleep(0.001)
        calculation.set_settings(this_settings)
        calculation.set_status(db.Status.HOLD)
        return calculation.id()

    def _get_filtered_intraform_and_diss(
        self,
        structure: db.Structure,
        reactive_atoms: List[int],
        connectivity_analyzer: Optional[ConnectivityAnalyzer] = None,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Get all pairs of reactive atoms within the given structure, that pass
        the pair filter, sorted into a list of dissociative pairs (if the atoms
        in the pair are adjacent) and a list of associative pairs (if they are
        not adjacent)

        Parameters
        ----------
        structure : db.Structure
            The structure of interest.
        reactive_atoms : List[int]
            The atoms of the structure that shall be considered.
        connectivity_analyzer : ConnectivityAnalyzer
            The connectivity analyzer of the structure.

        Returns
        -------
        List[Tuple[int]], List[Tuple[int]]
            Two lists of reactive pairs, the first with associative pairs
            (not adjacent in the structure), the second with dissociative pairs
            (adjacent in the structure).
        """
        if connectivity_analyzer is None:
            connectivity_analyzer = ConnectivityAnalyzer(structure)
        adjacency_matrix = connectivity_analyzer.get_adjacency_matrix()

        filtered_intra_reactive_atom_pairs = self.reactive_site_filter.filter_atom_pairs(
            [structure], list(combinations(reactive_atoms, 2))
        )
        # Split into bound and unbound (i.e. resulting in diss and form respectively)
        filtered_diss_pairs = []
        filtered_form_pairs = []
        for pair in filtered_intra_reactive_atom_pairs:
            if adjacency_matrix[pair[0], pair[1]]:
                filtered_diss_pairs.append(pair)
            else:
                filtered_form_pairs.append(pair)

        return filtered_form_pairs, filtered_diss_pairs

    @staticmethod
    def _get_bound_unbound_pairs_from_atoms(
        atoms: utils.AtomCollection,
        n_reactive_bound_pairs: Optional[int] = None,
        n_reactive_unbound_pairs: Optional[int] = None
    ):
        """
        Counts how many bound and unbound atom pairs there are in the given atom collection according to distance-based
        bond orders.
        If n_reactive_bound_pairs and/or n_reactive_unbound_pairs is equal or larger than zero these values are returned
        instead.

        Parameters
        ----------
        atoms : utils.AtomCollection
            The atom collection of interest.
        n_reactive_bound_pairs : int, optional
            The number of bound reactive pairs to consider.
            If None, all bound atom pairs are included.
        n_reactive_unbound_pairs : int, optional
            The number of bound reactive pairs to consider.
            If None, all unbound atom pairs are included.

        Returns
        -------
        Tuple[int, int]
            The number of unbound and bound atom pairs.

        Raises
        ------
        RuntimeError
            Raises if the structure is not corresponding to one connected molecule.
        """
        if n_reactive_bound_pairs is None or n_reactive_unbound_pairs is None:
            bond_orders = utils.BondDetector.detect_bonds(atoms)
            # Get the number of connected and unconnected atom pairs
            graph_result = masm.interpret.graphs(atoms, bond_orders, masm.interpret.BondDiscretization.Binary)
            n_atoms = atoms.size()
            n_pairs = comb(n_atoms, 2)

        n_bound_pairs = sum([g.E for g in graph_result.graphs]) if n_reactive_bound_pairs is None \
            else max(n_reactive_bound_pairs, 0)
        if n_reactive_unbound_pairs is None:
            # pylint: disable-next=(possibly-used-before-assignment)
            n_unbound_pairs = n_pairs - sum([g.E for g in graph_result.graphs])
        else:
            n_unbound_pairs = max(n_reactive_unbound_pairs, 0)
        return n_unbound_pairs, n_bound_pairs

    def get_unimolecular_job_order(self) -> str:
        return self.options.unimolecular_options.job.order

    def get_bimolecular_job_order(self) -> str:
        return self.options.bimolecular_options.job.order
