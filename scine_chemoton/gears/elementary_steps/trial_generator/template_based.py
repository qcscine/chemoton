#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import time
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Tuple, Union, Dict, Any
from itertools import combinations
from scipy.special import comb

# Third party imports
from numpy import ndarray
from scine_database.queries import calculation_exists_in_structure, get_calculation_id
import scine_database as db
import scine_utilities as utils
import scine_molassembler as masm

# Local application imports
from .connectivity_analyzer import ReactionType, ConnectivityAnalyzer
from . import TrialGenerator, _sanity_check_wrapper

from ....utilities.masm import deserialize_molecules
from ....utilities.options import BaseOptions
from scine_art.database import ReactionTemplateDatabase
import ast


class TemplateBased(TrialGenerator):
    """
    Class to generate reactive complex calculations via template-based approaches.

    Attributes
    ----------
    options : BondBased.Options
        The options for generating reactive complex calculations.
    reactive_site_filter : ReactiveSiteFilter
        The filter applied to determine reactive sites, reactive pairs and trial
        reaction coordinates.
    """

    class Options(TrialGenerator.Options):
        """
        The options for template-basedreactive complex enumeration.
        """

        __slots__ = (
            "unimolecular_options",
            "bimolecular_options",
            "reaction_template_file",
            "energy_cutoff",
            "enforce_atom_shapes"
        )

        class BimolOptions(BaseOptions):
            """
            The options for the generation and exploration of bimolecular reactions.
            """

            __slots__ = (
                "job",
                "job_settings",
                "complex_generator",
                "minimal_spin_multiplicity",
            )

            def __init__(self) -> None:
                super().__init__()
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
                "job_settings_associative",
                "job_settings_dissociative",
                "job_settings_disconnective",
            )

            def __init__(self) -> None:
                super().__init__()
                self.job: db.Job = db.Job("scine_react_complex_nt2")
                """
                db.Job (Scine::Database::Calculation::Job)
                    The Job used to evaluate the possible reactions.
                    Jobs with the order `scine_react_complex_nt2` are
                    supported. The default is: the `scine_react_complex_nt2`
                    order on a single core.
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
            self.reaction_template_file: str = 'templates.rtdb.pickle.obj'
            """
            str
                The path to a SCINE Art reaction template file to be loaded before
                starting the loop(s).
            """
            self.energy_cutoff: float = 100.0
            """
            str
                Only apply the template if a barrier in the trialed direction of
                less than this value has been reported, by default 100 (kJ/mol)
            """
            self.enforce_atom_shapes: bool = True
            """
            str
                If true only allow atoms with the same coordination sphere
                shapes to be considered matching, by default True
            """

    options: Options

    def __init__(self, energy_cutoff: float = 100, enforce_atom_shapes: bool = True) -> None:
        super().__init__()
        self.rtdb = ReactionTemplateDatabase()
        self.__loaded_templates = False
        self.options.energy_cutoff = energy_cutoff
        self.options.enforce_atom_shapes = enforce_atom_shapes
        self._last_data: Dict[str, Any] = {'ids': []}

    @_sanity_check_wrapper
    def bimolecular_coordinates(self,
                                structure_list: List[db.Structure],
                                with_exact_settings_check: bool = False
                                ) -> Dict[
        Tuple[List[Tuple[int, int]], int],
        List[Tuple[ndarray, ndarray, float, float]]
    ]:
        if not self.__loaded_templates:
            self.rtdb.load(self.options.reaction_template_file)
            self.__loaded_templates = True

        if self._quick_bimolecular_already_exists(structure_list, do_fast_query=not with_exact_settings_check):
            return {}

        # Check number of compounds
        if len(structure_list) != 2:
            raise RuntimeError("Exactly two structures are needed for setting up a bimolecular reaction.")

        ids = [s.get_id().string() for s in structure_list]
        if self._last_data['ids'] == ids:
            ids = self._last_data['ids']
            masm_idx_maps = self._last_data['masm_idx_maps']
            idx_map = self._last_data['idx_map']
            atom_offset = self._last_data['atom_offset']
            allowed = self._last_data['allowed']
        else:
            # Generate masm.Molecules
            molecules = []
            for s in structure_list:
                molecules.append(deserialize_molecules(s)[0])
            # Generate a few maps
            masm_idx_maps = [ast.literal_eval(s.get_graph("masm_idx_map")) for s in structure_list]
            idx_map = []
            atom_offset = [sum([m.graph.V for m in molecules][:i]) for i in range(len(molecules))]
            for i, mol in enumerate(molecules):
                for _ in range(mol.graph.V):
                    idx_map.append(i)
            # Get RTDB allowed reactions
            allowed = self.rtdb.find_matching_templates(
                molecules,
                energy_cutoff=self.options.energy_cutoff,
                enforce_atom_shapes=self.options.enforce_atom_shapes
            )
            # Store to cache
            self._last_data['ids'] = ids
            self._last_data['masm_idx_maps'] = masm_idx_maps
            self._last_data['idx_map'] = idx_map
            self._last_data['atom_offset'] = atom_offset
            self._last_data['allowed'] = allowed

        result: Dict[
            Tuple[List[Tuple[int, int]], int],
            List[Tuple[ndarray, ndarray, float, float]]
        ] = {}

        if allowed is None:
            return result

        for template in allowed:
            n_diss = len(template['dissos'])
            # n_assos = len(template['assos'])
            n_assos_inter = 0
            n_assos_intra = 0
            reactive_inter_coords_unshifted = []
            reactive_inter_coords = []
            reactive_intra_coords = []
            for a in template['assos']:
                atom_idx_one = masm_idx_maps[a[0][0]].index((0, a[0][1]))
                atom_idx_two = masm_idx_maps[a[1][0]].index((0, a[1][1]))
                atom_offset_one = atom_offset[a[0][0]]
                atom_offset_two = atom_offset[a[1][0]]
                if a[0][0] != a[1][0]:
                    n_assos_inter += 1
                    if a[0][0] == 0:
                        reactive_inter_coords_unshifted.append((atom_idx_one, atom_idx_two))
                        reactive_inter_coords.append((atom_idx_one + atom_offset_one, atom_idx_two + atom_offset_two))
                    else:
                        reactive_inter_coords_unshifted.append((atom_idx_two, atom_idx_one))
                        reactive_inter_coords.append((atom_idx_two + atom_offset_two, atom_idx_one + atom_offset_one))
                else:
                    n_assos_intra += 1
                    reactive_intra_coords.append((atom_idx_one + atom_offset_one, atom_idx_two + atom_offset_two))
            reactive_diss_coords = []
            for d in template['dissos']:
                atom_idx_one = masm_idx_maps[d[0][0]].index((0, d[0][1]))
                atom_idx_two = masm_idx_maps[d[1][0]].index((0, d[1][1]))
                atom_offset_one = atom_offset[d[0][0]]
                atom_offset_two = atom_offset[d[1][0]]
                reactive_diss_coords.append((atom_idx_one + atom_offset_one, atom_idx_two + atom_offset_two))

            # Batch all trials before doing a final filter run and adding
            #  the remaining trials. This reduces the number of filter calls
            #  and thus the number of DB look-ups.

            # keys: List of pairs
            # values: information required to build different reactive complexes
            batch: Dict[
                Tuple[Tuple[int, int], ...],
                List[Tuple[ndarray, ndarray, float, float]]
            ] = defaultdict(list)
            for (_, align1, align2, rot, spread, ) in \
                    self.options.bimolecular_options.complex_generator.generate_reactive_complexes(
                        structure_list[0], structure_list[1], [reactive_inter_coords_unshifted]):

                form_pairs = reactive_inter_coords + reactive_intra_coords
                key = tuple(form_pairs + reactive_diss_coords)
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
        """
        # Check number of compounds
        if len(structure_list) != 2:
            raise RuntimeError("Exactly two structures are needed for setting up a bimolecular reaction.")
        structure_id_list = [s.id() for s in structure_list]
        coordinate_complex_build_info = self.bimolecular_coordinates(structure_list, with_exact_settings_check)
        if not coordinate_complex_build_info:
            return
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
        """
        structure_id = structure.id()
        filtered_coordinates_per_ndiss = self.unimolecular_coordinates(structure, with_exact_settings_check)
        if not filtered_coordinates_per_ndiss:
            return
        connectivity_analyzer = ConnectivityAnalyzer(structure)
        for filter_result, n_diss in filtered_coordinates_per_ndiss:
            new_calculation_ids = []
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
                    settings
                )
                if cid is not None:
                    new_calculation_ids.append(cid)
            if new_calculation_ids:
                structure.add_calculations(self.options.unimolecular_options.job.order, [new_calculation_ids[0]])

    @_sanity_check_wrapper
    def unimolecular_coordinates(
        self,
        structure: db.Structure,
        with_exact_settings_check: bool = False
    ) -> List[Tuple[List[List[Tuple[int, int]]], int]]:
        if self._quick_unimolecular_already_exists(structure, do_fast_query=not with_exact_settings_check):
            return []

        if not self.__loaded_templates:
            self.rtdb.load(self.options.reaction_template_file)
            self.__loaded_templates = True

        ids = [structure.get_id().string()]
        if self._last_data['ids'] == ids:
            ids = self._last_data['ids']
            masm_idx_maps = self._last_data['masm_idx_maps']
            allowed = self._last_data['allowed']
        else:
            # Generate masm.Molecules
            molecule = deserialize_molecules(structure)[0]
            # Generate a few maps
            masm_idx_maps = [ast.literal_eval(structure.get_graph("masm_idx_map"))]
            # Get RTDB allowed reactions
            allowed = self.rtdb.find_matching_templates(
                [molecule],
                energy_cutoff=self.options.energy_cutoff,
                enforce_atom_shapes=self.options.enforce_atom_shapes
            )
            # Store to cache
            self._last_data['ids'] = ids
            self._last_data['masm_idx_maps'] = masm_idx_maps
            self._last_data['allowed'] = allowed

        result: List[Tuple[List[List[Tuple[int, int]]], int]] = []

        if allowed is None:
            return result

        for template in allowed:
            n_diss = len(template['dissos'])
            reactive_asso_coords = []
            for a in template['assos']:
                atom_idx_one = masm_idx_maps[0].index((0, a[0][1]))
                atom_idx_two = masm_idx_maps[0].index((0, a[1][1]))
                reactive_asso_coords.append((atom_idx_one, atom_idx_two))
            reactive_diss_coords = []
            for d in template['dissos']:
                atom_idx_one = masm_idx_maps[0].index((0, d[0][1]))
                atom_idx_two = masm_idx_maps[0].index((0, d[1][1]))
                reactive_diss_coords.append((atom_idx_one, atom_idx_two))
            result.append(([reactive_diss_coords + reactive_diss_coords], n_diss))
        return result

    def _quick_unimolecular_already_exists(self, structure: db.Structure, do_fast_query: bool) -> bool:
        # Rule out compounds too small for intramolecular reactions right away
        atoms = structure.get_atoms()
        if atoms.size() == 1:
            return False
        return do_fast_query and calculation_exists_in_structure(
            self.options.unimolecular_options.job.order,
            [structure.id()],
            self.options.model,
            self._structures,
            self._calculations)

    def _quick_bimolecular_already_exists(self, structure_list: List[db.Structure], do_fast_query: bool) -> bool:
        # Check number of compounds
        if len(structure_list) != 2:
            raise RuntimeError("Exactly two structures are needed for setting up a bimolecular reaction.")
        return do_fast_query and calculation_exists_in_structure(
            self.options.bimolecular_options.job.order,
            [s.id() for s in structure_list],
            self.options.model,
            self._structures,
            self._calculations)

    @_sanity_check_wrapper
    def estimate_n_unimolecular_trials(
        self, structure_file: str, n_reactive_bound_pairs: int = -1, n_reactive_unbound_pairs: int = -1
    ):
        raise NotImplementedError

    @_sanity_check_wrapper
    def estimate_n_bimolecular_trials(
        self,
        structure_file1: str,
        structure_file2: str,
        attack_points_per_site: int = 3
    ):
        raise NotImplementedError

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
        Adds a reactive calculation for a template-based approach
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

        if job.order == "scine_react_complex_nt2":
            # nt2 job takes lists of integer where elements 0/1, 2/3 ... N-1, N are combined with each other
            # Flatten pair lists
            this_settings["nt_nt_associations"] = [idx for idx_pair in association_pairs for idx in idx_pair]
            this_settings["nt_nt_dissociations"] = [idx for idx_pair in dissociation_pairs for idx in idx_pair]
        else:
            raise RuntimeError(
                "Only 'scine_react_complex_nt2' order supported for template-based reactive complex calculations."
            )

        if len(reactive_structures) > 1:
            this_settings["rc_minimal_spin_multiplicity"] = bool(
                self.options.bimolecular_options.minimal_spin_multiplicity)

        if check_for_existing and get_calculation_id(job.order, reactive_structures, self.options.model,
                                                     self._calculations, settings=this_settings) is not None:
            return None
        calculation = db.Calculation()
        calculation.link(self._calculations)
        calculation.create(self.options.model, job, reactive_structures)
        # Sleep a bit in order not to make the DB choke
        time.sleep(0.001)
        calculation.set_settings(deepcopy(this_settings))
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

        # pylint: disable-next=possibly-used-before-assignment
        n_bound_pairs = graph.E if n_reactive_bound_pairs < 0 else n_reactive_bound_pairs
        # pylint: disable-next=possibly-used-before-assignment
        n_unbound_pairs = n_pairs - graph.E if n_reactive_unbound_pairs < 0 else n_reactive_unbound_pairs
        return n_unbound_pairs, n_bound_pairs

    def get_unimolecular_job_order(self) -> str:
        return self.options.unimolecular_options.job.order

    def get_bimolecular_job_order(self) -> str:
        return self.options.bimolecular_options.job.order

    def clear_cache(self):
        pass
