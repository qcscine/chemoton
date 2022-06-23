#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from collections import Counter
from json import dumps
from typing import Dict, List, Tuple
from warnings import warn

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application imports
from ..gears import Gear
from ..utilities.queries import (
    identical_reaction,
    model_query,
    stop_on_timeout,
)
from ..gears.energy_query_functions import get_single_barrier_for_elementary_step


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
    Five different levels of refinement:
      'redo_single_points':
        New single point calculations for all minima and TS in the network with the refinement model
      'redo_optimizations'
        New optimizations of all minima and TS in the network with the refinement model.
        Minima are checked via the CompoundGear to be within the same compound
        TS should be checked for validity with an IRC within the optimization job in Puffin
      'double_ended_refinement'
        Check successful single ended react jobs and try to find a TS for these reactions with a double ended search
      'double_ended_new_connections'
        Check structures of same PES without a unimolecular reaction combining their compounds to be connected via a
        double ended search. This can also be done with the same model with which the structures were generated.
      'redo_single_ended_search'
        Perform single ended searches again with the refinement model, if they were already successful with a different
        model. Equality of products is not checked.
    """

    class Options:
        """
        The options for the NetworkRefinement Gear.
        """

        __slots__ = (
            "cycle_time",
            "pre_refine_model",
            "post_refine_model",
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
            "max_barrier"
        )

        def __init__(self):
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
            self.double_ended_job: db.Job = db.Job("scine_react_double_ended")  # TODO Write puffin job
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

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self._calculations = "required"
        self._structures = "required"
        self._properties = "required"
        self._elementary_steps = "required"
        self._reactions = "required"

    def _loop_impl(self):
        if self.options.pre_refine_model == self.options.post_refine_model and (
            sum(self.options.refinements.values()) != 1 or not self.options.refinements["double_ended_new_connections"]
        ):
            # new_connections would make sense to have identical model --> allow it if this is the only activated
            # refinement
            raise RuntimeError("pre_refine_model and post_refine_model must be different!")
        if self.options.refinements["refine_single_points"]:
            self._loop_elementary_step_with_barrier_screening("refine_single_points")
        if self.options.refinements["refine_optimizations"]:
            warn("WARNING: optimized TS verification after refinement is not implemented by default, yet")
            self._loop_elementary_step_with_barrier_screening("refine_optimizations")
        if self.options.refinements["double_ended_refinement"]:
            warn("WARNING: double ended job is not implemented by default, yet")
            self._loop_elementary_step_with_barrier_screening("double_ended_refinement")
        if self.options.refinements["double_ended_new_connections"]:
            warn("WARNING: double ended job is not implemented by default, yet")
            self._double_ended_new_connections_loop()
        if self.options.refinements["refine_single_ended_search"]:
            self._loop_elementary_step_with_barrier_screening("refine_single_ended_search")
        if self.options.refinements["refine_structures_and_irc"]:
            self._loop_elementary_step_with_barrier_screening("refine_structures_and_irc")

    def _refine_existing_react_jobs(
        self, job: db.Job, settings: utils.ValueCollection, keys_to_take_over: List[str], add_products: bool,
            structure_ids, old_settings, products
    ):
        new_settings = utils.ValueCollection({k: v for k, v in old_settings.items() if k in keys_to_take_over})
        # add new settings defined for refinement
        new_settings.update(settings.as_dict())
        if add_products:
            new_settings.update({"bspline_bspline_products": [str(sid) for sid in products]})
            structure_ids += products
        if not self._refinement_calculation_already_setup(structure_ids, job, new_settings):
            self._create_refinement_calculation(structure_ids, job, new_settings)

    def _refine_structures_and_irc(self, structure_ids, old_settings):
        new_settings = utils.ValueCollection({k: v for k, v in old_settings.items()})
        # add new settings defined for refinement
        new_settings.update(self.options.single_ended_step_refinement_settings.as_dict())
        # start the calculations
        if not self._refinement_calculation_already_setup(structure_ids,
                                                          self.options.single_ended_step_refinement_job,
                                                          new_settings):
            self._create_refinement_calculation(structure_ids, self.options.single_ended_step_refinement_job,
                                                new_settings)

    def _refine_single_points(self, structure_ids):
        for structure in structure_ids:
            # Check if a calculation for this is already scheduled
            if not self._refinement_calculation_already_setup(
                [structure], self.options.sp_job, self.options.sp_job_settings
            ):
                self._create_refinement_calculation([structure], self.options.sp_job, self.options.sp_job_settings)

    def _refine_optimizations(self, structure_ids):
        for sid in structure_ids:
            structure = db.Structure(sid, self._structures)
            structure.link(self._structures)
            refine_job, refine_settings = self._get_opt_refinement(structure.get_label())
            # Check if a calculation for this is already scheduled
            if not self._refinement_calculation_already_setup([sid], refine_job, refine_settings):
                self._create_refinement_calculation([sid], refine_job, refine_settings)

    def _double_ended_new_connections_loop(self):
        # cycle all minimum structures
        selection_i = {
            "$and": [
                {"exploration_disabled": {"$ne": True}},
                {
                    "$or": [
                        {"label": {"$eq": "minimum_optimized"}},
                        {"label": {"$eq": "user_optimized"}},
                    ]
                },
                {"compound": {"$ne": ""}},
            ]
            + model_query(self.options.pre_refine_model)
        }
        for structure_i in stop_on_timeout(iter(self._structures.query_structures(dumps(selection_i)))):
            structure_i.link(self._structures)
            compound_i = structure_i.get_compound()
            # get PES data
            charge = structure_i.get_charge()
            multiplicity = structure_i.get_multiplicity()
            atoms_i = structure_i.get_atoms()
            n_atoms = len(atoms_i)
            elements_i = [str(x) for x in atoms_i.elements]

            # search for minimum structures of different compounds that are on same PES
            selection_j = {
                "$and": [
                    {"_id": {"$gt": {"$oid": str(structure_i.id())}}},  # avoid double count
                    {"exploration_disabled": {"$ne": True}},
                    # PES minimum requirements
                    {"nAtoms": {"$eq": n_atoms}},
                    {"charge": {"$eq": charge}},
                    {"multiplicity": {"$eq": multiplicity}},
                    # has compound but different one
                    {"compound": {"$ne": ""}},
                    {"compound": {"$ne": str(compound_i)}},
                    # minimum structure
                    {
                        "$or": [
                            {"label": {"$eq": "minimum_optimized"}},
                            {"label": {"$eq": "user_optimized"}},
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
                # structures are on same PES, now check for unimolecular reactions between their compounds
                compound_j = structure_j.get_compound()
                if identical_reaction([compound_i], [compound_j], self._reactions) is not None:
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

    def _create_refinement_calculation(self, structure_ids: List[db.ID], job: db.Job, settings: utils.ValueCollection):
        calc = db.Calculation()
        calc.link(self._calculations)
        calc.create(self.options.post_refine_model, job, structure_ids)
        calc.set_settings(settings)
        calc.set_status(db.Status.HOLD)

    def _create_reoptimze_structure_job(self, structure: db.Structure):
        refine_job, refine_settings = self._get_opt_refinement(structure.get_label())
        # Check if a calculation for this is already scheduled
        if not self._refinement_calculation_already_setup([structure.id()], refine_job, refine_settings):
            self._create_refinement_calculation([structure.id()], refine_job, refine_settings)

    def _refinement_calculation_already_setup(
        self, structure_ids: List[db.ID], job: db.Job, settings: utils.ValueCollection
    ) -> bool:
        structures_string_ids = [{"$oid": str(s_id)} for s_id in structure_ids]
        selection = {
            "$and": [
                {"job.order": {"$eq": job.order}},
                {"structures": {"$all": structures_string_ids, "$size": len(structures_string_ids)}},
            ]
            + model_query(self.options.post_refine_model)
        }
        # TODO replace setting comparison with elaborate query
        # (direct setting comparison in query is dependent on order in dict and string-double comparison has problems)
        for calculation in stop_on_timeout(iter(self._calculations.query_calculations(dumps(selection)))):
            calculation.link(self._calculations)
            if calculation.get_settings() == settings:
                return True
        return False

    def _already_refined_step(self, step_id: db.ID) -> bool:
        step = db.ElementaryStep(step_id, self._elementary_steps)
        ts = db.Structure(step.get_transition_state(), self._structures)
        return ts.get_model() == self.options.post_refine_model

    def _already_refined_reaction(self, step_id: db.ID) -> bool:
        step = db.ElementaryStep(step_id)
        step.link(self._elementary_steps)
        selection = {"reaction": {"$oid": str(step.get_reaction())}}
        for possible_step in stop_on_timeout(iter(self._elementary_steps.query_elementary_steps(dumps(selection)))):
            possible_step.link(self._elementary_steps)
            ts = db.Structure(possible_step.get_transition_state())
            ts.link(self._structures)
            if ts.get_model() == self.options.post_refine_model:
                return True
        return False

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
            "rc_displacement",
            "rc_spin_multiplicity",
            "rc_molecular_charge",
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

    def _loop_elementary_step_with_barrier_screening(self, job_label):
        """
        Create refinement calculations under the condition that there is a calculation that produced an elementary
        step with a barrier less than self.options.max_barrier.

        Parameters
        ----------
        job_label: The label for the refinement to be executed.
        """
        selection = {
            "$and": [
                {"status": {"$eq": "complete"}},
                {"results.elementary_steps": {"$size": 1}},
            ]
            + model_query(self.options.pre_refine_model)
        }
        for calculation in stop_on_timeout(self._calculations.iterate_calculations(dumps(selection))):
            calculation.link(self._calculations)
            elementary_step_id = calculation.get_results().get_elementary_step(0)
            elementary_step = db.ElementaryStep(elementary_step_id, self._elementary_steps)
            barrier = get_single_barrier_for_elementary_step(elementary_step, self.options.pre_refine_model,
                                                             self._structures, self._properties)
            # If the barrier is None, one of the compounds needs to be recalculated with the prerefine model.
            if barrier is None:
                continue
            # check the barrier
            if barrier > self.options.max_barrier:
                continue
            reactants_products = elementary_step.get_reactants(db.Side.BOTH)
            transition_state_id = elementary_step.get_transition_state()
            structure_ids = reactants_products[0] + [transition_state_id] + reactants_products[1]
            if job_label == "refine_structures_and_irc":
                self._refine_structures_and_irc(reactants_products[0] + [transition_state_id],
                                                calculation.get_settings())
            elif job_label == "refine_single_points":
                self._refine_single_points(structure_ids)
            elif job_label == "refine_optimizations":
                self._refine_optimizations(structure_ids)
            elif job_label == "refine_single_ended_search":
                self._refine_existing_react_jobs(self.options.single_ended_job, self.options.single_ended_job_settings,
                                                 self._rc_keys() + self._single_ended_keys_to_take_over(), False,
                                                 reactants_products[0] + [transition_state_id],
                                                 calculation.get_settings(), reactants_products[1])
            elif job_label == "double_ended_refinement":
                self._refine_existing_react_jobs(self.options.double_ended_job, self.options.double_ended_job_settings,
                                                 self._rc_keys(), True, reactants_products[0] + [transition_state_id],
                                                 calculation.get_settings(), reactants_products[1])
            else:
                raise RuntimeError("The job label is not resolved for elementary step refinement.")
