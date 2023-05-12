#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from collections import Counter
from json import dumps
from typing import Dict, List, Tuple, Union, Set, Optional
from warnings import warn
import os
import pickle

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application imports
from ..gears import Gear
from ..utilities.queries import (
    identical_reaction,
    model_query,
    stop_on_timeout,
    get_calculation_id_from_structure,
    query_calculation_in_id_set
)
from ..utilities.energy_query_functions import get_energy_for_structure, get_barriers_for_elementary_step_by_type
from ..utilities.calculation_creation_helpers import finalize_calculation


class NetworkRefinement(Gear):
    """
    This Gear can improve an existing network built with some model (e.g., semi-empirics) with additional calculations
    with a different model (e.g., DFT).
    The level of refinement is determined via its options.

    Attributes
    ----------
    options :: NetworkRefinement.Options
        The options for the NetworkRefinement Gear.

    Notes
    -----
    Six different levels of refinement:
      'refine_single_points':
        New single point calculations for all minima and TS in the network with the refinement model
      'refine_optimizations'
        New optimizations of all minima and TS in the network with the refinement model.
        Minima are checked via the CompoundGear to be within the same compound
        TS should be checked for validity with an IRC within the optimization job in Puffin
      'double_ended_refinement'
        Check successful single ended react jobs and try to find a TS for these reactions with a double ended search
      'double_ended_new_connections'
        Check structures of same PES without an unimolecular reaction combining their compounds to be connected via a
        double ended search. This can also be done with the same model with which the structures were generated.
      'refine_single_ended_search'
        Perform single ended searches again with the refinement model, if they were already successful with a different
        model. Equality of products is not checked.
      'refine_structures_and_irc'
        Perform single ended searches again starting with the transition state of the previous transition state
        search.
    """

    class Options(Gear.Options):
        """
        The options for the NetworkRefinement Gear.
        """

        __slots__ = (
            "cycle_time",
            "pre_refine_model",
            "post_refine_model",
            "use_calculation_model",
            "calculation_model",
            "refinements",
            "sp_job",
            "sp_job_settings",
            "opt_job",
            "opt_job_settings",
            "tsopt_job",
            "tsopt_job_settings",
            "double_ended_job",
            "double_ended_job_settings",
            "single_ended_job",
            "single_ended_job_settings",
            "single_ended_step_refinement_job",
            "single_ended_step_refinement_settings",
            "max_barrier",
            "exclude_barrierless",
            "transition_state_energy_window",
            "refine_n_per_reaction",
            "reaction_based_loop",
            "jobs_to_wait_for",
            "caching_file_name",
            "manual_reaction_selection",
            "reaction_ids_to_refine",
            "elementary_step_index_file_name"
        )

        def __init__(self):
            super().__init__()
            self.cycle_time = 101
            """
            int
                The minimum number of seconds between two cycles of the Gear.
                Cycles are finished independent of this option, thus if a cycle
                takes longer than the cycle_time will effectively lead to longer
                cycle times and not cause multiple cycles of the same Gear.
            """
            self.pre_refine_model: db.Model = db.Model("PM6", "", "")
            """
            db.Model (Scine::Database::Model)
                The Model that indicates that the structure(s) and/or calculation(s) should be refined
                The default is: PM6
            """
            self.post_refine_model: db.Model = db.Model("DFT", "", "")
            """
            db.Model (Scine::Database::Model)
                The Model used for the refinement
                The default is: DFT
            """
            self.use_calculation_model = False
            """
            bool
                If true, the option calculation_model is used for the loop over previous calculations.
            """
            self.calculation_model: db.Model = db.Model("DFT", "", "")
            """
            db.Model
                The model used for the loop over previous calculations. This is only used if use_calculation_model
                is True.
            """
            self.refinements: Dict[str, bool] = {
                "refine_single_points": False,
                "refine_optimizations": False,
                "double_ended_refinement": False,
                "double_ended_new_connections": False,
                "refine_single_ended_search": False,
                "refine_structures_and_irc": False,
            }
            """
            Dict[str, bool]
                A dictionary specifying the wanted refinement(s)
                'refine_single_points': Calculate energies of all minima and transition states if they belong to a
                calculation that produced an elementary step with a barrier less than 'max_barrier'. The job for the
                calculations can be selected with 'options.sp_job' and its settings with 'options.sp_job_settings'.
                'refine_optimizations': Perform optimizations of minima and transition states. The same maximum barrier
                condition applies as for 'refine_single_points'. The job for the minima optimizations can be selected
                with 'options.opt_job' and its settings with 'options.opt_job_settings'. The job for the transition
                state optimizations can be selected with 'options.tsopt_job' and its settings with
                'options.tsopt_job_settings'.
                'double_ended_refinement': Perform double ended TS searches for compounds that
                are connected. The 'max_barrier' conditions applies as above. The job for this search can be
                selected with 'options.double_ended_job' and its settings with 'options.double_ended_job_settings'.
                'double_ended_new_connections': Perform double ended TS searches for compounds that
                might be connected. The job for this search can be selected with 'options.double_ended_job' and its
                settings with 'options.double_ended_job_settings'.
                'refine_single_ended_search': Perform single ended searches again with new model if they were
                already successful with another model. The 'max_barrier' conditions applies as above. The job for this
                search can be selected with 'options.single_ended_job' and its settings with
                'options.single_ended_job_settings'.
                'refine_structures_and_irc': Reoptimize an elementary step that was found previously. The previous
                transition state is used as an initial guess. A complete IRC is performed. The 'max_barrier' conditions
                applies as above.  The job for this search can be selected with
                'options.single_ended_step_refinement_job' and its settings with
                'options.single_ended_step_refinement_settings'.
            """
            self.sp_job: db.Job = db.Job("scine_single_point")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for calculating new single point energies.
                The default is: the 'scine_single_point' order on a single core.
            """
            self.sp_job_settings: utils.ValueCollection = utils.ValueCollection()
            """
            utils.ValueCollection
                Additional settings for calculating new single point energies.
                Empty by default.
            """
            self.opt_job: db.Job = db.Job("scine_geometry_optimization")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for optimizing all minima.
                The default is: the 'scine_geometry_optimization' order on a single core.
            """
            self.opt_job_settings: utils.ValueCollection = utils.ValueCollection()
            """
            utils.ValueCollection
                Additional settings for optimizing all minima.
                Empty by default.
            """
            self.tsopt_job: db.Job = db.Job("scine_ts_optimization")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for optimizing all transition states.
                The default is: the 'scine_ts_optimization' order on a single core.
            """
            self.tsopt_job_settings: utils.ValueCollection = utils.ValueCollection()
            """
            utils.ValueCollection
                Additional settings for optimizing all transition states.
                Empty by default.
            """
            self.double_ended_job: db.Job = db.Job("scine_bspline_optimization_job")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for searching for a transition state between two compounds.
                The default is: the 'scine_react_double_ended' order on a single core.
            """
            self.double_ended_job_settings: utils.ValueCollection = utils.ValueCollection()
            """
            utils.ValueCollection
                Additional settings for searching for a transition state between two compounds.
                Empty by default.
            """
            self.single_ended_job: db.Job = db.Job("scine_react_complex_nt")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for searching for redoing previously successful single ended searches.
                The default is: the 'scine_react_complex_nt' order on a single core. This job implies the approximation
                that the structures of the old model can be used for the single ended calculation with the new model.
            """
            self.single_ended_job_settings: utils.ValueCollection = utils.ValueCollection()
            """
            utils.ValueCollection
                Additional settings for single ended reaction search.
                Empty by default.
            """
            self.single_ended_step_refinement_job: db.Job = db.Job("scine_step_refinement")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for searching for refining previously successful single ended searches.
                The default is: the 'scine_step_refinement' order.
            """
            self.single_ended_step_refinement_settings: utils.ValueCollection = utils.ValueCollection()
            """
            utils.ValueCollection
                Additional settings for refining single ended reaction searches.
                Empty by default.
            """
            self.max_barrier: float = 262.5
            """
            float
                Maximum electronic energy barrier in kJ/mol for which elementary reaction steps are refined if
                structures_on_elementary_steps = True.
            """
            self.exclude_barrierless = False
            """
            bool
                If true, barrier-less reactions/elementary steps are not refined.
            """
            self.transition_state_energy_window = 1000.0
            """
            float
                Energy window for the elementary step selection in kJ/mol for reaction based refinement
                (see reaction_based_loop).
            """
            self.refine_n_per_reaction = 1000
            """
            int
                The maximum number of elementary steps to refine for a given reaction.
            """
            self.reaction_based_loop = False
            """
            bool
                If true, the elementary steps are traversed reaction wise and the elementary
                steps that are refined can be narrowed down by the transition_state_energy_window, e.g.,
                transition states with an energy higher than this window with respect to the lowest
                energy transition state for the reaction are not considered.
            """
            self.jobs_to_wait_for = ["scine_react_complex_nt2", "scine_single_point"]
            """
            List[str]
                Wait for these jobs to finish for each reaction before setting up reaction-wise refinement calculations.
            """
            self.caching_file_name = ".chemoton_refinement_calculation_id_cache.pickle"
            """
            str
                The name of the file used to save the already considered calculation ids.
            """
            self.manual_reaction_selection = False
            """
            bool
                If true the list "reaction_ids_to_refine" is used to select the reactions to refine. By default, False.
            """
            self.reaction_ids_to_refine = list()
            """
            List[db.ID]
                A list of reaction ids which should be refined if a manual selection is required. By default an empty
                list.
            """
            self.elementary_step_index_file_name = ".chemoton_refinement_elementary_step_index.pickle"
            """
            str
                The file name for the elementary step index.
            """

    def __init__(self):
        super().__init__()
        self._required_collections = ["calculations", "compounds", "elementary_steps", "flasks", "reactions",
                                      "properties", "structures"]
        self._calculation_id_cache = {
            "refine_single_points": set(),
            "refine_optimizations": set(),
            "double_ended_refinement": set(),
            "double_ended_new_connections": set(),
            "refine_single_ended_search": set(),
            "refine_structures_and_irc": set(),
        }
        self._lhs_optimization_calculations = dict()
        self._elementary_step_to_calculation_map: Dict[str, str] = dict()

    def _loop_impl(self):
        if self.options.pre_refine_model == self.options.post_refine_model and (
            sum(self.options.refinements.values()) != 1 or not self.options.refinements["double_ended_new_connections"]
        ) and not (self.options.use_calculation_model and
                   self.options.calculation_model != self.options.pre_refine_model):
            # new_connections would make sense to have identical model --> allow it if this is the only activated
            # refinement
            # If a calculation_model is given which differs from the pre_refine_model, allow it too.
            raise RuntimeError("pre_refine_model and post_refine_model must be different!")
        if self.options.refinements["refine_single_points"]:
            self._loop_steps("refine_single_points")
        if self.options.refinements["refine_optimizations"]:
            warn("WARNING: optimized TS verification after refinement is not implemented by default, yet")
            self._loop_steps("refine_optimizations")
        if self.options.refinements["double_ended_refinement"]:
            # warn("WARNING: double ended job is not implemented by default, yet")
            self._loop_steps("double_ended_refinement")
        if self.options.refinements["double_ended_new_connections"]:
            warn("WARNING: double ended job is not implemented by default, yet")
            self._double_ended_new_connections_loop()
        if self.options.refinements["refine_single_ended_search"]:
            self._loop_steps("refine_single_ended_search")
        if self.options.refinements["refine_structures_and_irc"]:
            self._loop_steps("refine_structures_and_irc")

    def _loop_steps(self, job_label: str):
        if self.options.reaction_based_loop or self.options.manual_reaction_selection:
            self._loop_reactions_with_barrier_screening(job_label)
        else:
            self._loop_elementary_step_with_barrier_screening(job_label)

    def _refine_existing_react_jobs(
        self, job: db.Job, settings: utils.ValueCollection, keys_to_take_over: List[str],
            structure_ids, old_settings, elementary_step=None
    ):
        new_settings = utils.ValueCollection({k: v for k, v in old_settings.items() if k in keys_to_take_over})
        # add new settings defined for refinement
        new_settings.update(settings.as_dict())
        if elementary_step:
            if elementary_step.get_type() != db.ElementaryStepType.REGULAR:
                return
            transition_state_id = elementary_step.get_transition_state()
            auxiliaries = {"transition-state-id": transition_state_id,
                           "elementary-step-id": elementary_step.id()}

            transition_state = db.Structure(transition_state_id, self._structures)
            # Check if the calculation already exists. If so, it will have an entry on the transition state of the
            # elementary step. This calculation is uniquely defined by
            # 1. The fact that it is assigned to the TS.
            # 2. The model.
            # 3. The job order
            # 4. The settings + auxiliaries.
            calc_id_strs = set([c_id.string() for c_id in transition_state.query_calculations(
                job.order, self.options.post_refine_model, self._calculations)])
            calc_ids = query_calculation_in_id_set(calc_id_strs, 2, self._calculations,
                                                   settings=new_settings, auxiliaries=auxiliaries)
            if not calc_ids:
                spline = elementary_step.get_spline()
                _, atoms_start = spline.evaluate(0)
                _, atoms_end = spline.evaluate(1)
                model = transition_state.get_model()
                charge = transition_state.get_charge()
                spin = transition_state.get_multiplicity()
                step_reactants = elementary_step.get_reactants(db.Side.BOTH)
                reactant_label = db.Label.COMPLEX_GUESS if len(step_reactants[0]) > 1 else db.Label.MINIMUM_GUESS
                product_label = db.Label.COMPLEX_GUESS if len(step_reactants[1]) > 1 else db.Label.MINIMUM_GUESS

                reactant_structure = db.Structure.make(atoms_start, charge, spin, model, reactant_label,
                                                       self._structures)
                product_structure = db.Structure.make(atoms_end, charge, spin, model, product_label, self._structures)
                calc_id = self._create_refinement_calculation([reactant_structure.id(), product_structure.id()], job,
                                                              new_settings, auxiliaries)
                transition_state.add_calculation(job.order, calc_id)
            return

        if not self._refinement_calculation_already_setup(structure_ids, job, new_settings):
            self._create_refinement_calculation(structure_ids, job, new_settings)

    def _refine_structures_and_irc(self, structure_ids, ts_id, old_settings):
        new_settings = utils.ValueCollection({k: v for k, v in old_settings.items()})
        # add new settings defined for refinement
        new_settings.update(self.options.single_ended_step_refinement_settings.as_dict())
        # start the calculations
        auxiliaries = {"transition-state-id": ts_id}
        if not self._refinement_calculation_already_setup(structure_ids,
                                                          self.options.single_ended_step_refinement_job,
                                                          new_settings, auxiliaries):
            self._create_refinement_calculation(structure_ids, self.options.single_ended_step_refinement_job,
                                                new_settings, auxiliaries)

    def _refine_single_points(self, structure_ids):
        for structure in structure_ids:
            # Check if a calculation for this is already scheduled
            already_set_up = self._refinement_calculation_already_setup(
                [structure], self.options.sp_job, self.options.sp_job_settings
            )
            if not already_set_up:
                _ = self._create_refinement_calculation([structure], self.options.sp_job,
                                                        self.options.sp_job_settings)

    def _refine_optimizations(self, structure_ids):
        calculation_ids = list()
        for sid in structure_ids:
            structure = db.Structure(sid, self._structures)
            structure.link(self._structures)
            refine_job, refine_settings = self._get_opt_refinement(structure.get_label())
            original_id = self._refinement_calculation_already_setup([sid], refine_job, refine_settings)
            # Check if a calculation for this is already scheduled
            if original_id is None:
                calculation_ids.append(self._create_refinement_calculation([sid], refine_job, refine_settings))
            else:
                calculation_ids.append(original_id)
        return calculation_ids

    def _double_ended_new_connections_loop(self):
        # cycle all minimum structures
        selection_i = {
            "$and": [
                {"exploration_disabled": {"$ne": True}},
                {
                    "$or": [
                        {"label": "minimum_optimized"},
                        {"label": "user_optimized"},
                        {"label": "complex_optimized"},
                    ]
                },
                {"aggregate": {"$ne": ""}},
            ]
            + model_query(self.options.pre_refine_model)
        }
        for structure_i in stop_on_timeout(iter(self._structures.query_structures(dumps(selection_i)))):
            if self.stop_at_next_break_point:
                return

            structure_i.link(self._structures)
            aggregate_i = structure_i.get_aggregate()
            # get PES data
            charge = structure_i.get_charge()
            multiplicity = structure_i.get_multiplicity()
            atoms_i = structure_i.get_atoms()
            n_atoms = len(atoms_i)
            elements_i = [str(x) for x in atoms_i.elements]

            # search for minimum structures of different aggregates that are on same PES
            selection_j = {
                "$and": [
                    {"_id": {"$gt": {"$oid": str(structure_i.id())}}},  # avoid double count
                    {"exploration_disabled": {"$ne": True}},
                    # PES minimum requirements
                    {"nAtoms": {"$eq": n_atoms}},
                    {"charge": {"$eq": charge}},
                    {"multiplicity": {"$eq": multiplicity}},
                    # has aggregate but different one
                    {"aggregate": {"$ne": ""}},
                    {"aggregate": {"$ne": str(aggregate_i)}},
                    # minimum structure
                    {
                        "$or": [
                            {"label": "minimum_optimized"},
                            {"label": "user_optimized"},
                            {"label": "complex_optimized"},
                        ]
                    },
                ]
                + model_query(self.options.pre_refine_model)
            }
            for structure_j in stop_on_timeout(iter(self._structures.query_structures(dumps(selection_j)))):
                structure_j.link(self._structures)
                # final check for identical PES
                elements_j = [str(x) for x in structure_j.get_atoms().elements]
                if Counter(elements_i) != Counter(elements_j):
                    continue
                # structures are on same PES, now check for unimolecular reactions between their aggregates
                aggregate_j = structure_j.get_aggregate()
                cl = [db.CompoundOrFlask.COMPOUND]
                if identical_reaction([aggregate_i], [aggregate_j], cl, cl, self._reactions) is not None:
                    continue
                # check for existing refinement calculation
                if not self._refinement_calculation_already_setup(
                    [structure_i.id(), structure_j.id()],
                    self.options.double_ended_job,
                    self.options.double_ended_job_settings,
                ):
                    self._create_refinement_calculation(
                        [structure_i.id(), structure_j.id()],
                        self.options.double_ended_job,
                        self.options.double_ended_job_settings,
                    )

    def _create_refinement_calculation(self, structure_ids: List[db.ID], job: db.Job, settings: utils.ValueCollection,
                                       auxiliaries: Optional[dict] = None) -> db.ID:
        calc = db.Calculation()
        calc.link(self._calculations)
        calc.create(self.options.post_refine_model, job, structure_ids)
        if auxiliaries is not None:
            calc.set_auxiliaries(auxiliaries)
        calc.set_settings(settings)
        finalize_calculation(calc, self._structures)
        return calc.id()

    def _refinement_calculation_already_setup(self, structure_ids: List[db.ID], job: db.Job,
                                              settings: utils.ValueCollection, auxiliaries: Optional[dict] = None) \
            -> Union[db.ID, None]:
        return get_calculation_id_from_structure(job.order, structure_ids, self.options.post_refine_model,
                                                 self._structures, self._calculations, settings, auxiliaries)

    def _get_opt_refinement(self, label: db.Label) -> Tuple[db.Job, utils.ValueCollection]:
        if label == db.Label.TS_OPTIMIZED:
            return self.options.tsopt_job, self.options.tsopt_job_settings
        else:
            return self.options.opt_job, self.options.opt_job_settings

    @staticmethod
    def _rc_keys() -> List[str]:
        return [
            "rc_x_alignment_0",
            "rc_x_alignment_1",
            "rc_x_rotation",
            "rc_x_spread",
            "rc_x_spread",
            "rc_displacement",
            "rc_spin_multiplicity",
            "rc_molecular_charge",
        ]

    @staticmethod
    def _double_ended_keys_to_take_over() -> List[str]:
        return [
            "opt_convergence_max_iterations",
            "opt_convergence_step_max_coefficient",
            "opt_convergence_step_rms",
            "opt_convergence_gradient_max_coefficient",
            "opt_convergence_gradient_rms",
            "opt_convergence_requirement",
            "opt_convergence_delta_value",
            "opt_geoopt_coordinate_system",
            "opt_bfgs_use_trust_radius",
            "opt_bfgs_trust_radius",
            "ircopt_convergence_max_iterations",
            "ircopt_convergence_step_max_coefficient",
            "ircopt_convergence_step_rms",
            "ircopt_convergence_gradient_max_coefficient",
            "ircopt_convergence_gradient_rms",
            "ircopt_convergence_requirement",
            "ircopt_convergence_delta_value",
            "ircopt_geoopt_coordinate_system",
            "ircopt_bfgs_use_trust_radius",
            "ircopt_bfgs_trust_radius",
            "irc_convergence_max_iterations",
            "irc_sd_factor",
            "irc_irc_initial_step_size",
            "irc_stop_on_error",
            "irc_convergence_step_max_coefficient",
            "irc_convergence_step_rms",
            "irc_convergence_gradient_max_coefficient",
            "irc_convergence_gradient_rms",
            "irc_convergence_delta_value",
            "irc_irc_coordinate_system",
            "tsopt_convergence_max_iterations",
            "tsopt_convergence_step_max_coefficient",
            "tsopt_convergence_step_rms",
            "tsopt_convergence_gradient_max_coefficient",
            "tsopt_convergence_gradient_rms",
            "tsopt_convergence_requirement",
            "tsopt_convergence_delta_value",
            "tsopt_optimizer",
            "tsopt_geoopt_coordinate_system",
            "tsopt_bofill_trust_radius",
            "tsopt_bofill_follow_mode",
        ]

    @staticmethod
    def _single_ended_keys_to_take_over() -> List[str]:
        return [
            "nt_nt_rhs_list",
            "nt_nt_lhs_list",
            "nt_nt_attractive",
            "nt_nt_associations",
            "nt_nt_dissociations",
            "afir_afir_rhs_list",
            "afir_afir_lhs_list",
            "afir_afir_attractive",
            "afir_afir_use_max_fragment_distance",
        ]

    def _save_calculation_id_cache(self):
        # save dictionary to pickle file
        with open(self.options.caching_file_name, 'wb') as file:
            pickle.dump(self._calculation_id_cache, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_calculation_id_cache(self):
        if os.path.exists(self.options.caching_file_name) and os.path.getsize(self.options.caching_file_name) > 0:
            with open(self.options.caching_file_name, "rb") as file:
                load_cache = pickle.load(file)
                if load_cache:
                    self._calculation_id_cache.update(load_cache)

    def _loop_elementary_step_with_barrier_screening(self, job_label: str):
        """
        Create refinement calculations under the condition that there is a calculation that produced an elementary
        step with a barrier less than self.options.max_barrier.

        Parameters
        ----------
        job_label: str
            The label for the refinement to be executed.
        """
        self._load_calculation_id_cache()
        cache = self._calculation_id_cache[job_label]
        model_to_search_with = self.options.pre_refine_model
        if self.options.use_calculation_model:
            model_to_search_with = self.options.calculation_model
        selection = {
            "$and": [
                {"status": "complete"},
                {"results.elementary_steps.0": {"$exists": True}},
            ]
            + model_query(model_to_search_with)  # type: ignore
        }
        cache_update: List[str] = list()
        for calculation in stop_on_timeout(self._calculations.iterate_calculations(dumps(selection))):
            if self.stop_at_next_break_point:
                cache.update(set(cache_update))
                self._save_calculation_id_cache()
                return
            str_id = calculation.id().string()
            if str_id in cache:
                continue
            calculation.link(self._calculations)
            elementary_steps = [db.ElementaryStep(step_id, self._elementary_steps) for step_id in
                                calculation.get_results().get_elementary_steps()]
            all_barrierless = True
            for step in elementary_steps:
                if step.get_type() != db.ElementaryStepType.BARRIERLESS:
                    all_barrierless = False
            if self.options.exclude_barrierless and all_barrierless:
                continue
            calculation_fully_done = self._set_up_calculation(job_label, calculation)
            if calculation_fully_done:
                cache_update.append(str_id)

        cache.update(set(cache_update))
        self._save_calculation_id_cache()

    def _create_elementary_step_index(self):
        indexing_model = self.options.pre_refine_model
        if self.options.use_calculation_model:
            indexing_model = self.options.calculation_model
        selection = {
            "$and": [
                {"status": "complete"},
                {"results.elementary_steps.0": {"$exists": True}},
            ]
            + model_query(indexing_model)
        }
        self._elementary_step_to_calculation_map = dict()
        for calculation in stop_on_timeout(self._calculations.iterate_calculations(dumps(selection))):
            calculation.link(self._calculations)
            self._update_elementary_step_index(calculation)
        self._save_elementary_step_index()

    def _save_elementary_step_index(self):
        # save dictionary to pickle file
        with open(self.options.elementary_step_index_file_name, 'wb') as file:
            pickle.dump(self._elementary_step_to_calculation_map, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_elementary_step_index(self):
        if os.path.exists(self.options.elementary_step_index_file_name)\
                and os.path.getsize(self.options.elementary_step_index_file_name) > 0:
            with open(self.options.elementary_step_index_file_name, "rb") as file:
                load_cache = pickle.load(file)
                if load_cache:
                    self._elementary_step_to_calculation_map.update(load_cache)

    def _update_elementary_step_index(self, calculation: db.Calculation):
        elementary_step_ids = calculation.get_results().elementary_step_ids
        for step_id in elementary_step_ids:
            self._elementary_step_to_calculation_map[step_id.string()] = calculation.id().string()

    def _get_calculation_id_for_step(self, elementary_step_id: db.ID) -> Optional[db.ID]:
        if elementary_step_id.string() in self._elementary_step_to_calculation_map:
            return db.ID(self._elementary_step_to_calculation_map[elementary_step_id.string()])
        # TODO: This will be slow and could time out. But I do not see a better way to do it at the moment.
        selection = {
            "$and": [
                {"results.elementary_steps": {"$oid": elementary_step_id.string()}}
            ]
        }
        calc = self._calculations.get_one_calculation(dumps(selection))
        if calc is None:
            print(f"Failed to find calculation that resulted in elementary step "
                  f"{str(elementary_step_id)}")
            return None
        calc.link(self._calculations)
        self._update_elementary_step_index(calc)
        return calc.id()

    def _loop_reactions_with_barrier_screening(self, job_label: str):
        """
        Create refinement calculations for only a selection of the elementary
        steps for each reaction (e.g. the most favorable one).

        Parameters
        ----------
        job_label: The label for the refinement to be executed.
        """
        # TODO: Maybe better some preselection or caching of already refined reaction ids?
        if self._exploration_calculations_still_running():
            return
        # Create an elementary step id to calculation id map if necessary.
        self._load_elementary_step_index()
        if not self._elementary_step_to_calculation_map:
            self._create_elementary_step_index()
        if not self.options.manual_reaction_selection:
            for reaction in stop_on_timeout(self._reactions.iterate_all_reactions()):
                if self.stop_at_next_break_point:
                    return
                reaction.link(self._reactions)
                self._refine_reaction(reaction, job_label)
        else:
            for r_id in self.options.reaction_ids_to_refine:
                reaction = db.Reaction(r_id, self._reactions)
                self._refine_reaction(reaction, job_label)
        self._save_elementary_step_index()

    def _refine_reaction(self, reaction: db.Reaction, job_label: str) -> None:
        elementary_steps = self._get_all_elementary_steps(reaction)
        if not elementary_steps:
            return
        model_to_search_with = self.options.pre_refine_model
        if self.options.use_calculation_model:
            model_to_search_with = self.options.calculation_model

        eligible_steps = self._select_elementary_steps(elementary_steps, model_to_search_with)
        for step_id in eligible_steps:
            calc_id = None
            if job_label in ["refine_structures_and_irc", "refine_single_ended_search", "double_ended_refinement"]:
                calc_id = self._get_calculation_id_for_step(step_id)
            if calc_id:
                calculation = db.Calculation(calc_id, self._calculations)
            self._set_up_calculation(job_label, calculation, step_id)

    def _get_job_order(self, job_label: str):
        if job_label == "refine_structures_and_irc":
            return self.options.single_ended_step_refinement_job.order
        elif job_label == "refine_single_points":
            return self.options.sp_job.order
        elif job_label == "refine_optimizations":
            return self.options.opt_job.order
        elif job_label == "refine_single_ended_search":
            return self.options.single_ended_job.order
        elif job_label == "double_ended_refinement":
            return self.options.double_ended_job.order
        else:
            raise RuntimeError("The job label is not resolved for elementary step refinement.")

    def _exploration_calculations_still_running(self) -> bool:
        """
        Check if some calculations corresponding to the job order in the options, are still running.
        ----------

        Returns
        -------
        bool
            Returns True if calculations are still running, False, otherwise.
        """
        selection = {
            "$and": [
                {"status": {"$in": ["hold", "new", "pending"]}},
                {"job.order": {"$in": self.options.jobs_to_wait_for}}
            ]
        }
        return self._calculations.get_one_calculation(dumps(selection)) is not None

    def _get_all_elementary_steps(self, reaction: db.Reaction) -> List[db.ID]:
        """
        Get all elementary steps and calculations that produced these steps for a given reaction.
        Parameters
        ----------
        reaction :: db.Reaction
            The reaction.
        Returns
        -------
            The list of elementary steps and the list of the corresponding calculations.
        """
        elementary_steps = []
        elementary_steps_in_reaction = reaction.get_elementary_steps()
        for step_id in elementary_steps_in_reaction:
            elementary_step = db.ElementaryStep(step_id, self._elementary_steps)
            # Check the TS for non-barrierless reactions.
            if elementary_step.get_type() != db.ElementaryStepType.BARRIERLESS:
                ts = db.Structure(elementary_step.get_transition_state())
                energy = get_energy_for_structure(ts, "electronic_energy", self.options.pre_refine_model,
                                                  self._structures, self._properties)
                if energy is not None:
                    elementary_steps.append(step_id)
            else:
                if self.options.exclude_barrierless:
                    continue
                lhs_barrier, rhs_barrier = get_barriers_for_elementary_step_by_type(
                    elementary_step, "electronic_energy", self.options.pre_refine_model,
                    self._structures, self._properties)
                if lhs_barrier is not None and rhs_barrier is not None:
                    elementary_steps.append(step_id)
        return elementary_steps

    def _all_structures_match_model(self, structure_ids: List[db.ID], model: db.Model) -> bool:
        for s_id in structure_ids:
            structure = db.Structure(s_id, self._structures)
            structure_model = structure.get_model()
            if structure_model != model:
                return False
        return True

    def _select_elementary_steps(self, elementary_steps: List[db.ID], model: db.Model) -> List[db.ID]:
        """
        Select the most favorable (the lowest energy transition state) elementary step(s) for a given reaction according
        to the given energy window.
        Parameters
        ----------
        elementary_steps :: List[db.ID]
            The list of all elementary steps with the given model.
        Returns
        -------
        List[db.ID]
            The list of the selected elementary steps.
        """
        # Get minimum energy ts-elementary step + some extra
        energy_step_tuples: List[Tuple[float, db.ID]] = list()
        structure_model = self.options.pre_refine_model
        if self.options.use_calculation_model:
            structure_model = self.options.calculation_model
        for step_id in elementary_steps:
            elementary_step = db.ElementaryStep(step_id, self._elementary_steps)
            lhs_rhs_structure_ids = elementary_step.get_reactants(db.Side.BOTH)
            if not self._all_structures_match_model(lhs_rhs_structure_ids[0] + lhs_rhs_structure_ids[1],
                                                    structure_model):
                continue
            energy: Optional[float] = 0.0
            if elementary_step.get_type() != db.ElementaryStepType.BARRIERLESS:
                ts = db.Structure(elementary_step.get_transition_state(), self._structures)
                if ts.get_model() != model:
                    continue
                energy = get_energy_for_structure(ts, "electronic_energy", self.options.pre_refine_model,
                                                  self._structures, self._properties)
                if energy is None:
                    continue
                energy_step_tuples.append(tuple((energy, step_id)))  # type: ignore

        if len(energy_step_tuples) < 1:
            return []
        sorted_energy_step_tuples = sorted(energy_step_tuples, key=lambda tup: tup[0])
        minimum_energy: float = min(sorted_energy_step_tuples, key=lambda tup: tup[0])[0]
        threshold = self.options.transition_state_energy_window * utils.HARTREE_PER_KJPERMOL
        eligible_step_ids = [sorted_energy_step_tuples[0][1]]
        sorted_energy_step_tuples.pop(0)
        for energy, step_id in sorted_energy_step_tuples:
            if abs(energy - minimum_energy) < threshold\
                    and len(eligible_step_ids) <= self.options.refine_n_per_reaction:
                eligible_step_ids.append(step_id)
            else:
                break
        return eligible_step_ids

    def _get_optimized_structure_ids(self, original_structures_ids: List[db.ID]):
        """
        Getter for the IDs of the re-optimized structures corresponding to the original structure ID list.
        Parameters
        ----------
        original_structures_ids :: List[db.ID]
            The original structure ID list.

        Returns
        -------
        List[db.ID]
            Returns an empty list if not all structures were optimized yet. Otherwise, the list of the optimized
            structure IDs is returned.
        """
        optimized_lhs_s_ids = list()
        for s_id in original_structures_ids:
            str_id = s_id.string()
            # If the calculation is not cached or was not set up yet, set it up or get the old calculation id
            if str_id not in self._lhs_optimization_calculations:
                self._lhs_optimization_calculations[str_id] = self._refine_optimizations([s_id])[0]
            # Check the calculation for result structures.
            optimization_calculation = db.Calculation(self._lhs_optimization_calculations[str_id],
                                                      self._calculations)
            result_structures = optimization_calculation.get_results().get_structures()
            if result_structures:
                optimized_lhs_s_ids.append(result_structures[0])
        if len(optimized_lhs_s_ids) == len(original_structures_ids):
            return optimized_lhs_s_ids
        else:
            return list()

    def _all_structures_have_energy(self, structure_id_list: List[db.ID]):
        for s_id in structure_id_list:
            s = db.Structure(s_id)
            energy = get_energy_for_structure(s, "electronic_energy", self.options.post_refine_model, self._structures,
                                              self._properties)
            if energy is None:
                return False
        return True

    def _set_up_calculation(self, job_label: str, calculation: db.Calculation,
                            targeted_step_id: Optional[db.ID] = None) -> bool:
        """
        Set up the refinement calculation or if needed geometry optimizations to be run before the refinement.
        Parameters
        ----------
        job_label :: str
            The job label for the refinement.
        calculation :: db.Calculation
            The original calculation that needs to be refined. Settings may be required from the calculation object.
        Returns
        -------
        bool
            Return True if the input calculation is fully handled. Returns False if the input calculation
            needs to be revisited by the gear.
        """
        assert calculation
        if targeted_step_id:
            elementary_steps = [db.ElementaryStep(targeted_step_id, self._elementary_steps)]
        else:
            elementary_steps = [db.ElementaryStep(step_id, self._elementary_steps) for step_id in
                                calculation.get_results().get_elementary_steps()]

        transition_state_id = None
        rhs_lhs_structure_id_set: Set[str] = set()
        for elementary_step in elementary_steps:
            # If all elementary steps are sorted into a reaction, there should no duplicates be present anymore.
            # If duplicates are still there, we may waste a lot of time calculating energies/geometries etc. for
            # duplicate elementary steps.
            if not elementary_step.has_reaction() or not elementary_step.analyze():
                continue

            reactants = elementary_step.get_reactants(db.Side.BOTH)
            rhs_lhs_structure_id_set = rhs_lhs_structure_id_set.union(
                set([s_id.string() for s_id in reactants[0] + reactants[1]]))
            if elementary_step.has_transition_state():
                transition_state_id = elementary_step.get_transition_state()
                if self._barrier_exceeded(elementary_step):
                    return True

        rhs_lhs_structure_list = [db.ID(str_id) for str_id in rhs_lhs_structure_id_set]
        if transition_state_id is not None:
            all_structure_ids_list = rhs_lhs_structure_list + [transition_state_id]
        else:
            all_structure_ids_list = rhs_lhs_structure_list
        if job_label == "refine_structures_and_irc" and transition_state_id is not None:
            reactant_structure_ids = self._get_optimized_structure_ids(calculation.get_structures())
            if reactant_structure_ids and self._all_structures_have_graph(reactant_structure_ids):
                self._refine_structures_and_irc(reactant_structure_ids, transition_state_id,
                                                calculation.get_settings())
            else:
                return False
        elif job_label == "refine_single_points":
            if self._all_structures_have_energy(rhs_lhs_structure_list):
                return True
            self._refine_single_points(all_structure_ids_list)
        elif job_label == "refine_optimizations":
            self._refine_optimizations(all_structure_ids_list)
        elif job_label == "refine_single_ended_search":
            reactant_structure_ids = self._get_optimized_structure_ids(calculation.get_structures())
            if reactant_structure_ids and self._all_structures_have_graph(reactant_structure_ids):
                self._refine_existing_react_jobs(self.options.single_ended_job, self.options.single_ended_job_settings,
                                                 self._rc_keys() + self._single_ended_keys_to_take_over(),
                                                 reactant_structure_ids, calculation.get_settings())
            else:
                return False
        elif job_label == "double_ended_refinement":
            for elementary_step in elementary_steps:
                if self.options.double_ended_job.order != "scine_bspline_optimization_job":
                    raise RuntimeError("Only the logic for the scine_bspline_optimization job is implemented!")
                if not elementary_step.has_reaction():
                    return False
                if elementary_step.get_type() != db.ElementaryStepType.REGULAR:
                    continue
                self._refine_existing_react_jobs(self.options.double_ended_job,
                                                 self.options.double_ended_job_settings,
                                                 self._double_ended_keys_to_take_over(),
                                                 [], calculation.get_settings(), elementary_step)
        else:
            raise RuntimeError("The job label is not resolved for elementary step refinement.")
        return True

    def _barrier_exceeded(self, elementary_step: db.ElementaryStep) -> bool:
        """
        Check the barrier of the elementary step.
        Parameters
        ----------
        elementary_step :: db.ElementaryStep
            The barrier.

        Returns
        -------
        bool
            True if the barrier is exceeded or not defined for the given model. False, otherwise.
        """
        barrier, _ = get_barriers_for_elementary_step_by_type(elementary_step, 'electronic_energy',
                                                              self.options.pre_refine_model, self._structures,
                                                              self._properties)
        # If the barrier is None, one of the compounds needs to be recalculated with the prerefine model.
        return barrier is None or barrier > self.options.max_barrier

    def _all_structures_have_graph(self, structure_ids: List[db.ID]):
        """
        Check if all structures in the list have a graph assigned.
        Parameters
        ----------
        structure_ids :: List[db.ID]
            The structure ID list.
        Returns
        -------
        bool
            True if all structures have a "masm_cbor_graph". False otherwise.
        """
        for s_id in structure_ids:
            structure = db.Structure(s_id, self._structures)
            if not structure.has_graph("masm_cbor_graph"):
                return False
        return True
