#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import time
from copy import deepcopy
from typing import List, Optional, Tuple
from json import dumps
from itertools import combinations, product
from scipy.special import comb

# Third party imports
import scine_database as db
import scine_utilities as utils
import scine_molassembler as masm

# Local application imports
from ....utilities.queries import model_query, select_calculation_by_structures
from ..reactive_site_filters import ReactiveSiteFilter
from .connectivity_analyzer import ReactionType, ConnectivityAnalyzer
from . import TrialGenerator


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

    class Options:
        """
        The options for bond-based reactive complex enumeration.
        """

        __slots__ = ("model", "unimolecular", "bimolecular")

        class BimolOptions:
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

            def __init__(self):
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

        class UnimolOptions:
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

            def __init__(self):
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

        def __init__(self):
            self.model: db.Model = db.Model("PM6", "", "")
            """
            db.Model (Scine::Database::Model)
                The Model used to evaluate the possible reactions.
                The default is: PM6 using Sparrow.
            """
            self.unimolecular = self.UnimolOptions()
            """
            UnimolOptions
                The options for reactions involving a single molecule.
            """
            self.bimolecular = self.BimolOptions()
            """
            BimolOptions
                The options for reactions involving two molecules, that are set up
                to be associative in nature.
            """

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self.reactive_site_filter: ReactiveSiteFilter = ReactiveSiteFilter()
        self._calculations = "required"

    def _sanity_check_configuration(self):
        if not isinstance(self.reactive_site_filter, ReactiveSiteFilter):
            raise TypeError("Expected a ReactiveSiteFilter (or a class derived "
                            "from it) in BondBased.reactive_site_filter.")

    def bimolecular_reactions(self, structure_list: List[db.Structure]):
        """
        Creates reactive complex calculations corresponding to the bimolecular
        reactions between the structures if there is not already a calculation
        to search for a reaction of the same structures with the same job order.

        Parameters
        ----------
        structure_list :: List[db.Structure]
            List of the two structures to be considered.
            The Structures have to be linked to a database.
        """
        self._sanity_check_configuration()
        # Check number of compounds
        if len(structure_list) != 2:
            raise RuntimeError("Exactly two structures are needed for setting up a bimolecular reaction.")

        structure_id_list = [s.id() for s in structure_list]

        # If there is a reactive complex calculation for the same structures, return
        selection = select_calculation_by_structures(
            self.options.bimolecular.job.order, structure_id_list, self.options.model)
        if self._calculations.get_one_calculation(dumps(selection)) is not None:
            return

        # Get reactive atoms
        reactive_atoms1 = self.reactive_site_filter.filter_atoms(
            [structure_list[0]], list(range(structure_list[0].get_atoms().size()))
        )
        reactive_atoms2 = self.reactive_site_filter.filter_atoms(
            [structure_list[1]], list(range(structure_list[1].get_atoms().size()))
        )

        n_atoms1 = structure_list[0].get_atoms().size()
        # If needed get reactive intrastructural coordinates for both structures
        # splitted into formation and dissociation
        if (
            self.options.bimolecular.max_intra_bond_formations > 0
            or self.options.bimolecular.max_bond_dissociations > 0
        ) and (self.options.bimolecular.max_bond_modifications > self.options.bimolecular.min_inter_bond_formations):
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

        # Generate reactive pair combinations across the two structures from reactive sites
        shifted_inter_pairs = tuple(product(reactive_atoms1, (idx + n_atoms1 for idx in reactive_atoms2)))
        filtered_shifted_inter_pairs = self.reactive_site_filter.filter_atom_pairs(structure_list, shifted_inter_pairs)

        # TODO Avoid enumeration of all coords by rewriting reactive complex generator
        reactive_inter_coords = []
        filtered_unshifted_inter_pairs = [(pair[0], pair[1] - n_atoms1) for pair in filtered_shifted_inter_pairs]
        max_inter = min(
            self.options.bimolecular.max_inter_bond_formations, self.options.bimolecular.max_bond_modifications
        )
        for n_inter in range(self.options.bimolecular.min_inter_bond_formations, max_inter + 1):
            inter_pair_combis = list(combinations(filtered_unshifted_inter_pairs, n_inter))
            reactive_inter_coords += inter_pair_combis

        # TODO  Avoid using different conventions regarding indices of second structure
        for (
            inter_coord,
            align1,
            align2,
            rot,
            spread,
        ) in self.options.bimolecular.complex_generator.generate_reactive_complexes(
            structure_list[0], structure_list[1], reactive_inter_coords
        ):

            # Shift to complex indexing
            shifted_inter_coord = tuple((pair[0], pair[1] + n_atoms1) for pair in inter_coord)

            # Add intramolecular coordinates
            n_inter_pairs = len(inter_coord)
            if n_inter_pairs == self.options.bimolecular.max_bond_modifications or (
                self.options.bimolecular.max_bond_dissociations == 0
                and self.options.bimolecular.max_intra_bond_formations == 0
            ):
                if n_inter_pairs < self.options.bimolecular.min_bond_modifications:
                    # Should not happen with reasonable options
                    continue
                if (
                    self.options.bimolecular.min_bond_dissociations > 0
                    or self.options.bimolecular.min_intra_bond_formations > 0
                ):
                    # Do not set up an inter only calculation if intramolecular coordinates are explicitly required
                    continue
                # If there is either no room for further intramolecular contributions or no such contributions are
                # wanted, submit calc with intermolecular only and continue
                diss_pairs = []
                if (
                    len(self.reactive_site_filter.filter_reaction_coordinates(structure_list, [shifted_inter_coord]))
                    > 0
                ):
                    self._add_reactive_complex_calculation(
                        structure_id_list,
                        shifted_inter_coord,
                        diss_pairs,
                        self.options.bimolecular.job,
                        self.options.bimolecular.job_settings,
                        list(align1),
                        list(align2),
                        rot,
                        spread,
                        0.0,
                    )
                continue

            # Get remaining numbers of bonds to form or break
            min_intra_mod = max(
                self.options.bimolecular.min_bond_modifications - n_inter_pairs,
                self.options.bimolecular.min_intra_bond_formations + self.options.bimolecular.min_bond_dissociations,
            )
            max_intra_mod = min(
                self.options.bimolecular.max_bond_modifications - n_inter_pairs,
                self.options.bimolecular.max_bond_dissociations + self.options.bimolecular.max_intra_bond_formations,
            )

            for n_intra_mod in range(min_intra_mod, max_intra_mod + 1):
                # If no intra mod just set up inter
                if n_intra_mod == 0:
                    diss_pairs = []
                    if (
                        len(
                            self.reactive_site_filter.filter_reaction_coordinates(structure_list, [shifted_inter_coord])
                        )
                        > 0
                    ):
                        self._add_reactive_complex_calculation(
                            structure_id_list,
                            shifted_inter_coord,
                            diss_pairs,
                            self.options.bimolecular.job,
                            self.options.bimolecular.job_settings,
                            list(align1),
                            list(align2),
                            rot,
                            spread,
                            0.0,
                        )
                    continue

                # Get all n_diss/n_intra_form combis that are possible for this n_intra_mod
                for n_diss in range(
                    self.options.bimolecular.min_bond_dissociations, self.options.bimolecular.max_bond_dissociations + 1
                ):
                    n_intra_form = n_intra_mod - n_diss
                    if not (
                        self.options.bimolecular.min_intra_bond_formations
                        <= n_intra_form
                        <= self.options.bimolecular.max_intra_bond_formations
                    ):
                        continue
                    # Get all possible combinations of groups of this size out of filtered intrastructural
                    #  dissociations and association coordinates
                    for diss_pairs in combinations(all_filtered_diss_pairs, n_diss):
                        for intra_form_pairs in combinations(all_filtered_intra_form_pairs, n_intra_form):
                            form_pairs = shifted_inter_coord + intra_form_pairs
                            if (
                                len(
                                    self.reactive_site_filter.filter_reaction_coordinates(
                                        structure_list, [form_pairs + diss_pairs]
                                    )
                                )
                                > 0
                            ):
                                self._add_reactive_complex_calculation(
                                    structure_id_list,
                                    form_pairs,
                                    diss_pairs,
                                    self.options.bimolecular.job,
                                    self.options.bimolecular.job_settings,
                                    list(align1),
                                    list(align2),
                                    rot,
                                    spread,
                                    0.0,
                                )
        return

    def unimolecular_reactions(self, structure: db.Structure):
        """
        Creates reactive complex calculations corresponding to the unimolecular
        reactions of the structure if there is not already a calculation to
        search for a reaction of the same structure with the same job order.

        Parameters
        ----------
        structure :: db.Structure
            The structure to be considered. The Structure has to
            be linked to a database.
        """
        self._sanity_check_configuration()
        # Rule out compounds too small for intramolecular reactions right away
        atoms = structure.get_atoms()
        structure_id = structure.id()
        # No intramolecular reactions for monoatomic compounds
        if atoms.size() == 1:
            return
        #  considered
        if atoms.size() == 2 and self.options.unimolecular.max_bond_dissociations < 1:
            return

        # Check whether there is a reactive complex calculation for the same structure already
        selection = {
            "$and": [
                {"job.order": {"$eq": self.options.unimolecular.job.order}},
                {"structures": [{"$oid": structure_id.string()}]},
            ]
            + model_query(self.options.model)
        }
        if self._calculations.get_one_calculation(dumps(selection)) is not None:
            return

        # Get all reactive atoms
        reactive_atoms = self.reactive_site_filter.filter_atoms([structure], list(range(atoms.size())))
        # Get all ordered pairs of reactive atoms
        connectivity_analyzer = ConnectivityAnalyzer(structure)
        filtered_form_pairs, filtered_diss_pairs = self._get_filtered_intraform_and_diss(
            structure, reactive_atoms, connectivity_analyzer
        )

        # Get requested combinations of reactive pairs
        for n_mod in range(
            self.options.unimolecular.min_bond_modifications, self.options.unimolecular.max_bond_modifications + 1
        ):
            # Loop over possible options of dissociations
            for n_diss in range(
                self.options.unimolecular.min_bond_dissociations, self.options.unimolecular.max_bond_dissociations + 1
            ):
                n_form = n_mod - n_diss
                if not (
                    self.options.unimolecular.min_bond_formations
                    <= n_form
                    <= self.options.unimolecular.max_bond_formations
                ):
                    continue
                for diss_pairs in combinations(filtered_diss_pairs, n_diss):
                    for form_pairs in combinations(filtered_form_pairs, n_form):
                        # Check whether trial coordinate is to be considered
                        if (
                            len(
                                self.reactive_site_filter.filter_reaction_coordinates(
                                    [structure], [form_pairs + diss_pairs]
                                )
                            )
                            > 0
                        ):
                            # Get reaction type:
                            reaction_type = connectivity_analyzer.get_reaction_type(form_pairs + diss_pairs)
                            # Set up reactive complex calculation
                            if reaction_type in (ReactionType.Associative, ReactionType.Mixed):
                                self._add_reactive_complex_calculation(
                                    [structure_id],
                                    form_pairs,
                                    diss_pairs,
                                    self.options.unimolecular.job,
                                    self.options.unimolecular.job_settings_associative,
                                )
                            elif reaction_type == ReactionType.Dissociative:
                                self._add_reactive_complex_calculation(
                                    [structure_id],
                                    form_pairs,
                                    diss_pairs,
                                    self.options.unimolecular.job,
                                    self.options.unimolecular.job_settings_dissociative,
                                )
                            elif reaction_type == ReactionType.Disconnective:
                                self._add_reactive_complex_calculation(
                                    [structure_id],
                                    form_pairs,
                                    diss_pairs,
                                    self.options.unimolecular.job,
                                    self.options.unimolecular.job_settings_disconnective,
                                )
        return

    def estimate_n_unimolecular_trials(
        self, structure_file: str, n_reactive_bound_pairs: int = -1, n_reactive_unbound_pairs: int = -1
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
            If smaller than zero, all bound atom pairs are included.
            By default -1.
        n_reactive_unbound_pairs : int, optional
            The number of unbound reactive pairs to consider.
            If smaller than zero, all unbound atom pairs are included.
            By default -1.

        Returns
        -------
        int
            The number of reactive trial coordinates expected to arise from this
            structure.
        """
        self._sanity_check_configuration()
        # Read in atom collection and interpret connectivity from interatomic distances
        atoms = utils.io.read(structure_file)[0]
        n_unbound_pairs, n_bound_pairs = self._get_bound_unbound_pairs_from_atoms(
            atoms, n_reactive_bound_pairs, n_reactive_unbound_pairs
        )

        # Get all types of bond modification patterns that are compliant with the options
        # and add the associated number of trials
        n_trials = 0
        for n_form in range(
            self.options.unimolecular.min_bond_formations, self.options.unimolecular.max_bond_formations + 1
        ):
            for n_diss in range(
                self.options.unimolecular.min_bond_dissociations, self.options.unimolecular.max_bond_dissociations + 1
            ):
                if not (
                    self.options.unimolecular.min_bond_modifications
                    <= n_form + n_diss
                    <= self.options.unimolecular.max_bond_modifications
                ):
                    continue
                n_form_combis = comb(n_unbound_pairs, n_form)
                n_diss_combis = comb(n_bound_pairs, n_diss)
                n_form_diss_combis = n_form_combis * n_diss_combis
                n_trials += n_form_diss_combis

        return n_trials

    def estimate_n_bimolecular_trials(
        self,
        structure_file1: str,
        structure_file2: str,
        attack_points_per_site: int = 3,
        n_inter_reactive_pairs: int = -1,
        n_reactive_bound_pairs1: int = -1,
        n_reactive_unbound_pairs1: int = -1,
        n_reactive_bound_pairs2: int = -1,
        n_reactive_unbound_pairs2: int = -1,
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
        Multiple attack points are not calculated from the structures geometry
        but only included based on a fixed input value
        (`attack_points_per_site`).
        The number of rotamers is not accounted for correctly for
        intermolecular coordinates involving two reactive pairs with one
        monoatomic fragment.

        Note: The number of trials increases quickly with the number of allowed
              bond modifications. Be aware!

        Parameters
        ----------
        structure_file : str
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
            If smaller than zero, all interstructural atom pairs are included.
            By default -1.
        n_reactive_bound_pairs1 : int, optional
            The number of bound reactive pairs in the first structure to
            consider. If smaller than zero, all bound atom pairs of structure 1
            are included.
            By default -1.
        n_reactive_bound_pairs2 : int, optional
            The number of bound reactive pairs in the second structure to
            consider. If smaller than zero, all bound atom pairs of structure 2
            are included.
            By default -1.
        n_reactive_unbound_pairs1 : int, optional
            The number of unbound reactive pairs in the first structure to
            consider. If smaller than zero, all unbound atom pairs are included.
            By default -1.
        n_reactive_unbound_pairs2 : int, optional
            The number of unbound reactive pairs in the second structure to
            consider. If smaller than zero, all unbound atom pairs are included.
            By default -1.

        Returns
        -------
        int
            The number of reactive trial coordinates expected to arise from this
            structure.
        """
        self._sanity_check_configuration()
        # Read in atom collections and get intramolecular bound and unbound pairs
        atoms1 = utils.io.read(structure_file1)[0]
        n_unbound_pairs1, n_bound_pairs1 = self._get_bound_unbound_pairs_from_atoms(
            atoms1, n_reactive_bound_pairs1, n_reactive_unbound_pairs1
        )
        atoms2 = utils.io.read(structure_file2)[0]
        n_unbound_pairs2, n_bound_pairs2 = self._get_bound_unbound_pairs_from_atoms(
            atoms2, n_reactive_bound_pairs2, n_reactive_unbound_pairs2
        )
        n_inter_pairs = atoms1.size() * atoms2.size() if n_inter_reactive_pairs < 0 else n_inter_reactive_pairs
        n_bound_pairs = n_bound_pairs1 + n_bound_pairs2
        n_unbound_pairs = n_unbound_pairs1 + n_unbound_pairs2

        # Get all types of bond modification patterns that are compliant with the options
        # and add the associated number of trials
        n_trials = 0
        # At least on intermolecular coord required
        if self.options.bimolecular.max_inter_bond_formations < 1:
            return 0

        for n_inter in range(
            self.options.bimolecular.min_inter_bond_formations, self.options.bimolecular.max_inter_bond_formations + 1
        ):
            for n_form in range(
                self.options.bimolecular.min_intra_bond_formations,
                self.options.bimolecular.max_intra_bond_formations + 1,
            ):
                for n_diss in range(
                    self.options.bimolecular.min_bond_dissociations, self.options.bimolecular.max_bond_dissociations + 1
                ):
                    if not (
                        self.options.bimolecular.min_bond_modifications
                        <= n_form + n_diss + n_inter
                        <= self.options.bimolecular.max_bond_modifications
                    ):
                        continue
                    n_inter_combis = comb(n_inter_pairs, n_inter)
                    n_form_combis = comb(n_unbound_pairs, n_form)
                    n_diss_combis = comb(n_bound_pairs, n_diss)
                    n_form_diss_combis = n_form_combis * n_diss_combis * n_inter_combis

                    # Add rotamers
                    if n_inter == 1:
                        n_form_diss_combis *= self.options.bimolecular.complex_generator.options.number_rotamers
                    # There can be two intermolecular coordinates involving the same atom twice (e.g. "atom on bond")
                    # These are weighted with number_rotamers in the real enumeration but with
                    # number_rotamers_two_on_two here
                    elif n_inter == 2:
                        n_form_diss_combis *= (
                            self.options.bimolecular.complex_generator.options.number_rotamers_two_on_two
                        )

                    n_trials += n_form_diss_combis

        # Take into account estimate of attack points per site
        if self.options.bimolecular.complex_generator.options.multiple_attack_points:
            n_trials *= attack_points_per_site * attack_points_per_site
        return n_trials

    def _add_reactive_complex_calculation(
        self,
        reactive_structures: List[db.ID],
        association_pairs: List[Tuple[int]],
        dissociation_pairs: List[Tuple[int]],
        job: db.Job,
        settings: utils.ValueCollection,
        lhs_aligment: Optional[List[float]] = None,
        rhs_aligment: Optional[List[float]] = None,
        x_rotation: Optional[float] = None,
        spread: Optional[float] = None,
        displacement: Optional[float] = None,
        charge: Optional[int] = None,
        multiplicity: Optional[int] = None,
    ):
        """
        Adds a reactive calculation for a bond-based approach (i.e., one during which it is tried to form/break
        bonds between designated atom pairs) to the database and puts it on hold.

        Parameters
        ----------

        reactive_structures :: List[db.ID]
            List of the IDs of the reactants.
        association_pairs:: List[Tuple(int, int))]
            List of atom index pairs in between which a bond formation is to be
            encouraged.
        dissociation_pairs:: List[Tuple(int, int))]
            List of atom index pairs in between which a bond dissociation is to
            be encouraged.
        job :: scine_database.Job
        settings :: scine_utilities.ValueCollection
        lhs_alignment :: List[float], length=9
            In case of two structures building the reactive complex, this option
            describes a rotation of the first structure (index 0) that aligns
            the reaction coordinate along the x-axis (pointing towards +x).
            The rotation assumes that the geometric mean position of all
            atoms in the reactive site (``lhs_list``) is shifted into the
            origin.
        rhs_alignment :: List[float], length=9
            In case of two structures building the reactive complex, this option
            describes a rotation of the second structure (index 1) that aligns
            the reaction coordinate along the x-axis (pointing towards -x).
            The rotation assumes that the geometric mean position of all
            atoms in the reactive site (``rhs_list``) is shifted into the
            origin.
        x_rotation :: float
            In case of two structures building the reactive complex, this option
            describes a rotation angle around the x-axis of one of the two
            structures after ``lhs_aligment`` and ``rhs_aligment`` have
            been applied.
        spread :: float
            In case of two structures building the reactive complex, this option
            gives the distance by which the two structures are moved apart along
            the x-axis after ``lhs_aligment``, ``rhs_aligment``, and
            ``x_rotation`` have been applied.
        displacement :: float
            In case of two structures building the reactive complex, this option
            adds a random displacement to all atoms (random direction, random
            length). The maximum length of this displacement (per atom) is set to
            be the value of this option.
        multiplicity :: int
            This option sets the ``spin_multiplicity`` of the reactive complex.
        charge :: int
            This option sets the ``molecular_charge`` of the reactive complex.
        """
        model = self.options.model
        calculation = db.Calculation()
        calculation.link(self._calculations)
        calculation.create(model, job, reactive_structures)
        # Sleep a bit in order not to make the DB choke
        time.sleep(0.001)
        if lhs_aligment is not None:
            settings["rc_x_alignment_0"] = lhs_aligment
        if rhs_aligment is not None:
            settings["rc_x_alignment_1"] = rhs_aligment
        if x_rotation is not None:
            settings["rc_x_rotation"] = x_rotation
        if spread is not None:
            settings["rc_x_spread"] = spread
        if displacement is not None:
            settings["rc_displacement"] = displacement
        if multiplicity is not None:
            settings["rc_spin_multiplicity"] = multiplicity
        if charge is not None:
            settings["rc_molecular_charge"] = charge

        if job.order == "scine_react_complex_nt2":
            # nt2 job takes lists of integer where elements 0/1, 2/3 ... N-1, N are combined with each other
            # Flatten pair lists
            settings["nt_nt_associations"] = [idx for idx_pair in association_pairs for idx in idx_pair]
            settings["nt_nt_dissociations"] = [idx for idx_pair in dissociation_pairs for idx in idx_pair]
        else:
            raise RuntimeError(
                "Only 'scine_react_complex_nt2' order supported for bond-based reactive complex calculations."
            )

        if len(reactive_structures) > 1:
            settings["rc_minimal_spin_multiplicity"] = bool(self.options.bimolecular.minimal_spin_multiplicity)

        calculation.set_settings(deepcopy(settings))
        calculation.set_status(db.Status.HOLD)

    def _get_filtered_intraform_and_diss(
        self,
        structure: db.Structure,
        reactive_atoms: List[int],
        connectivity_analyzer: Optional[ConnectivityAnalyzer] = None,
    ):
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
        atoms: utils.AtomCollection, n_reactive_bound_pairs: int = -1, n_reactive_unbound_pairs: int = -1
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
            If smaller than zero, all bound atom pairs are included.
            By default -1.
        n_reactive_unbound_pairs : int, optional
            The number of bound reactive pairs to consider.
            If smaller than zero, all unbound atom pairs are included.
            By default -1.

        Returns
        -------
        Tuple[int, int]
            The number of unbound and bound atom pairs.

        Raises
        ------
        RuntimeError
            Raises if the structure is not corresponding to one connected molecule.
        """
        if n_reactive_bound_pairs < 0 or n_reactive_unbound_pairs < 0:
            bond_orders = utils.BondDetector.detect_bonds(atoms)
            # Get the number of connected and unconnected atom pairs
            graph_result = masm.interpret.graphs(atoms, bond_orders, masm.interpret.BondDiscretization.Binary)
            n_atoms = atoms.size()
            n_pairs = comb(n_atoms, 2)
            if len(graph_result.graphs) != 1:
                raise RuntimeError(
                    "Atom collection contains more than one molecule according to distance-based bond orders."
                )
            graph = graph_result.graphs[0]

        n_bound_pairs = graph.E if n_reactive_bound_pairs < 0 else n_reactive_bound_pairs
        n_unbound_pairs = n_pairs - graph.E if n_reactive_unbound_pairs < 0 else n_reactive_unbound_pairs
        return n_unbound_pairs, n_bound_pairs
