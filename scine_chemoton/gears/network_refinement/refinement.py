#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from collections import Counter
from json import dumps
from typing import Dict, List, Tuple, Set, Optional
from warnings import warn
from abc import ABC, abstractmethod

# Third party imports
import scine_database as db
import scine_utilities as utils
from scine_database.queries import (
    identical_reaction,
    model_query,
    stop_on_timeout,
    get_calculation_id_from_structure,
    query_calculation_in_id_set
)
from scine_database.energy_query_functions import get_energy_for_structure

# Local application imports
from ...gears import Gear, HoldsCollections
from scine_chemoton.filters.elementary_step_filters import (
    ElementaryStepBarrierFilter,
    ElementaryStepFilter,
    BarrierlessElementaryStepFilter,
    PlaceHolderElementaryStepFilter
)
from scine_chemoton.filters.reaction_filters import (
    ReactionFilter,
    ReactionBarrierFilter,
    BarrierlessReactionFilter,
    StopDuringExploration,
    PlaceHolderReactionFilter
)
from ...utilities.calculation_creation_helpers import finalize_calculation
from ...utilities.model_combinations import ModelCombination
from scine_chemoton.utilities.place_holder_model import (
    construct_place_holder_model,
    PlaceHolderModelType
)
from .disabling import ReactionDisabling, StepDisabling
from .enabling import (
    ReactionEnabling, EnableCalculationResults, PlaceHolderReactionEnabling, PlaceHolderCalculationEnabling
)
from ...utilities.db_object_wrappers.thermodynamic_properties import ReferenceState, PlaceHolderReferenceState
from scine_chemoton.default_settings import default_opt_settings, default_nt_settings


class NetworkRefinement(Gear, ABC):
    """
    This Gear can improve an existing network built with some model (e.g., semi-empirics) with additional calculations
    with a different model (e.g., DFT). The level of refinement is determined by its options.

    Note that this is only the base class for the classes ReactionBasedRefinement and CalculationBasedRefinement.

    Attributes
    ----------
    options : NetworkRefinement.Options
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
            "post_refine_model",
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
            "transition_state_energy_window",
            "aggregate_energy_window",
            "refine_n_per_reaction",
            "caching_file_name",
            "elementary_step_index_file_name",
            "hessian_model",
            "only_electronic_energies",
            "reference_state"
        )

        def __init__(self) -> None:
            super().__init__()
            self.post_refine_model: db.Model = db.Model("DFT", "", "")
            """
            db.Model (Scine::Database::Model)
                The Model used for the refinement
                The default is: DFT
            """
            self.calculation_model: db.Model = construct_place_holder_model()
            """
            db.Model
                The model used for the loop over previous calculations. If just a place-holder is provided,
                the pre_refine_model is used.
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
            self.opt_job_settings: utils.ValueCollection = default_opt_settings()
            """
            utils.ValueCollection
                Additional settings for optimizing all minima.
                Chemoton's default optimization settings by default.
            """
            self.tsopt_job: db.Job = db.Job("scine_ts_optimization")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for optimizing all transition states.
                The default is: the 'scine_ts_optimization' order on a single core.
            """
            self.tsopt_job_settings: utils.ValueCollection = utils.ValueCollection(
                {k: v for k, v in default_opt_settings().as_dict().items() if "bfgs" not in k}
            )
            """
            utils.ValueCollection
                Additional settings for optimizing all transition states.
                Chemoton's default optimization settings by default without the BFGS settings.
            """
            self.double_ended_job: db.Job = db.Job("scine_bspline_optimization")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for searching for a transition state between two compounds.
                The default is: the 'scine_react_double_ended' order on a single core.
            """
            self.double_ended_job_settings: utils.ValueCollection = utils.ValueCollection()
            # ToDo: Is this really a good idea since the settings are taken from the old calculation.
            #     utils.ValueCollection(
            #     {k: v for k, v in default_nt_settings().as_dict().items()
            #      if "nt_" not in k and "rcopt" not in k}
            # )
            """
            utils.ValueCollection
                Additional settings for searching for a transition state between two compounds.
                Chemoton's default NT settings without the NT and RCOpt parts by default.
            """
            self.single_ended_job: db.Job = db.Job("scine_react_complex_nt2")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for searching for redoing previously successful single ended searches.
                The default is: the 'scine_react_complex_nt' order on a single core. This job implies the approximation
                that the structures of the old model can be used for the single ended calculation with the new model.
            """
            self.single_ended_job_settings: utils.ValueCollection = default_nt_settings()
            """
            utils.ValueCollection
                Additional settings for single ended reaction search.
                Chemoton's default NT job settings by default.
            """
            self.single_ended_step_refinement_job: db.Job = db.Job("scine_step_refinement")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for searching for refining previously successful single ended searches.
                The default is: the 'scine_step_refinement' order.
            """
            self.single_ended_step_refinement_settings: utils.ValueCollection = utils.ValueCollection(
                {k: v for k, v in default_nt_settings().as_dict().items() if "nt_" not in k and "rcopt" not in k}
            )
            """
            utils.ValueCollection
                Additional settings for refining single ended reaction searches.
                Chemoton's default NT settings without the NT and RCOpt part by default.
            """
            self.transition_state_energy_window = 1000.0
            """
            float
                Energy window for the elementary step selection in kJ/mol for reaction based refinement
                (see reaction_based_loop).
            """
            self.aggregate_energy_window = 1000.0
            """
            float
                Energy window for the aggregate selection for single point refinement.
            """
            self.refine_n_per_reaction = 1000
            """
            int
                The maximum number of elementary steps to refine for a given reaction.
            """
            self.caching_file_name = ".chemoton_refinement_calculation_id_cache.pickle"
            """
            str
                The name of the file used to save the already considered calculation ids.
            """
            self.elementary_step_index_file_name = ".chemoton_refinement_elementary_step_index.pickle"
            """
            str
                The file name for the elementary step index.
            """
            self.hessian_model: db.Model = construct_place_holder_model()
            """
            db.Model
                A second electronic structure model along side the pre_refine_model. With this model, the free
                energy corrections are evaluated. If none is given, the pre_refine_model is used.
            """
            self.only_electronic_energies: bool = False
            """
            bool
                If true, free energies are approximated only by their electronic energy contribution.
            """
            self.reference_state: ReferenceState = PlaceHolderReferenceState()
            """
            Optional[ReferenceState]
                The reference state to calculate free energies with. If none is given, the reference state is deduced
                from the pre_refine model.
            """

    options: Options

    def __init__(self) -> None:
        super().__init__()
        self._required_collections = ["calculations", "compounds", "elementary_steps", "flasks", "reactions",
                                      "properties", "structures"]
        self._lhs_optimization_calculations: Dict[str, db.ID] = dict()
        self.reaction_filter: ReactionFilter = PlaceHolderReactionFilter()
        """
        ReactionFilter
            The filter for the reaction selection for refinement.
            By default max barrier of 262.5 kJ/mol, exclude all barrier-less, and only refine after all
            exploration calculations are done.
        """
        self.reaction_enabling: ReactionEnabling = PlaceHolderReactionEnabling()
        """
        ReactionEnabling
            If given, reactions/elementary steps already found with the refined model are enabled and not new
            calculations are created. This fakes the refinement and is useful for rapid testing of meta algorithms
            in the refinement.
        """
        self.reaction_validation: ReactionFilter = PlaceHolderReactionFilter()
        """
        Optional[ReactionFilter]
            If reactions are enabled instead of refined (see option reaction_enabling), and this filter returns
            true, no further refinement calculations are done for the given reaction.
        """
        self.result_enabling: EnableCalculationResults = PlaceHolderCalculationEnabling()
        """
        EnableCalculationResults
            If this calculation result enabling policy is given, the result of an already existing calculation
            is enabled again (if disabled previously).
        """
        self.reaction_disabling: ReactionDisabling = ReactionDisabling()
        """
        ReactionDisabling
            Reaction post-refinement-processing to disable them until the refinement is done.
        """
        self.step_disabling: StepDisabling = StepDisabling()
        """
        StepDisabling
            Elementary step post-refinement-processing to disable them until the refinement is done.
        """
        self.elementary_step_filter: ElementaryStepFilter = PlaceHolderElementaryStepFilter()
        """
        ElementaryStepFilter
            The filter for the elementary step selection for refinement.
            By default max barrier of 262.5 kJ/mol and exclude all barrier-less.
        """
        self.model_combination: Optional[ModelCombination] = None
        """
        Optional[ModelCombination]
            The electronic structure model combination combining the pre-refine model and the hessian model.
        """
        self.__default_max_barrier = 262.5  # kJ/mol

    def _propagate_db_manager(self, manager: db.Manager) -> None:
        self.__initialize_helper_attributes_and_filters()

        # Initialize the collections in the filters.
        self.elementary_step_filter.initialize_collections(manager)
        self.reaction_filter.initialize_collections(manager)
        self.reaction_disabling.initialize_collections(manager)
        self.reaction_validation.initialize_collections(manager)
        self.result_enabling.initialize_collections(manager)
        self.step_disabling.initialize_collections(manager)
        if isinstance(self.reaction_enabling, HoldsCollections):
            self.reaction_enabling.initialize_collections(manager)

    def __initialize_helper_attributes_and_filters(self) -> None:
        """
        Initialize some attributes that are only known as soon as the _loop_impl is called and all options are final for
        the loop.
        """
        only_electronic_energies = self.options.only_electronic_energies
        if isinstance(self.options.hessian_model, PlaceHolderModelType):
            self.options.hessian_model = self.options.model
        if self.model_combination is None:
            self.model_combination = ModelCombination(self.options.model, self.options.hessian_model)
        if isinstance(self.reaction_filter, PlaceHolderReactionFilter):
            f1 = StopDuringExploration()
            f2 = BarrierlessReactionFilter(self.model_combination, only_electronic_energies=only_electronic_energies)
            f2.initialize_collections(self._manager)
            f3 = ReactionBarrierFilter(self.__default_max_barrier, self.model_combination, only_electronic_energies)
            f3.set_cache(f2.get_cache())
            self.reaction_filter = f1 and f2 and f3
        if isinstance(self.elementary_step_filter, PlaceHolderElementaryStepFilter):
            self.elementary_step_filter = BarrierlessElementaryStepFilter() and ElementaryStepBarrierFilter(
                self.__default_max_barrier, self.model_combination, only_electronic_energies=only_electronic_energies)
        if isinstance(self.options.calculation_model, PlaceHolderModelType):
            self.options.calculation_model = self.options.model
        if (not isinstance(self.reaction_enabling, PlaceHolderReactionEnabling) and
                isinstance(self.reaction_validation, PlaceHolderReactionFilter)):
            raise RuntimeError(
                "Error: If reactions should be enabled to avoid refinement, a validation policy in"
                " the option reaction_validation must be given.")
        if isinstance(self.options.reference_state, PlaceHolderReferenceState):
            self.options.reference_state = ReferenceState(float(self.options.model.temperature),
                                                          float(self.options.model.pressure))

    def _loop_impl(self):
        if self.options.model == self.options.post_refine_model and (
            sum(self.options.refinements.values()) != 1 or not self.options.refinements["double_ended_new_connections"]
        ) and not self.options.calculation_model != self.options.model:
            # new_connections would make sense to have identical model --> allow it if this is the only activated
            # refinement
            # If a calculation_model is given which differs from the pre_refine_model, allow it too.
            raise RuntimeError("model and post_refine_model must be different!")
        if self.options.refinements["refine_single_points"]:
            self._loop("refine_single_points")
        if self.options.refinements["refine_optimizations"]:
            warn("WARNING: optimized TS verification after refinement is not implemented by default, yet")
            self._loop("refine_optimizations")
        if self.options.refinements["double_ended_refinement"]:
            self._loop("double_ended_refinement")
        if self.options.refinements["double_ended_new_connections"]:
            warn("WARNING: A double ended job creating new connections between aggregates is not implemented by"
                 " default, yet")
            self._double_ended_new_connections_loop()
        if self.options.refinements["refine_single_ended_search"]:
            self._loop("refine_single_ended_search")
        if self.options.refinements["refine_structures_and_irc"]:
            self._loop("refine_structures_and_irc")
        if self.have_to_stop_at_next_break_point():
            return

    @abstractmethod
    def _loop(self, job_label: str):
        raise NotImplementedError

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
            else:
                if not isinstance(self.result_enabling, PlaceHolderCalculationEnabling):
                    self.result_enabling.process(db.Calculation(calc_ids, self._calculations))
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

    def _refine_optimizations(self, structure_ids) -> List[db.ID]:
        calculation_ids = []
        for sid in structure_ids:
            structure = db.Structure(sid, self._structures)
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
                        {"label": "user_complex_optimized"},
                    ]
                },
                {"aggregate": {"$ne": ""}},
            ]
            + model_query(self.options.model)
        }
        for structure_i in stop_on_timeout(iter(self._structures.query_structures(dumps(selection_i)))):
            if self.have_to_stop_at_next_break_point():
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
                            {"label": "user_complex_optimized"},
                        ]
                    },
                ]
                + model_query(self.options.model)
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
            -> Optional[db.ID]:
        calc_id = get_calculation_id_from_structure(job.order, structure_ids, self.options.post_refine_model,
                                                    self._structures, self._calculations, settings, auxiliaries)
        if calc_id is not None and not isinstance(self.result_enabling, PlaceHolderCalculationEnabling):
            self.result_enabling.process(db.Calculation(calc_id, self._calculations))
        return calc_id

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
            "spin_propensity_check",
            "spin_propensity_check_for_unimolecular_reaction",
            "spin_propensity_energy_range_to_save",
            "spin_propensity_optimize_all",
            "spin_propensity_energy_range_to_optimize",
            "store_full_mep",
            "store_all_structures",
            "n_surface_atom_threshold",
            "imaginary_wavenumber_threshold",
            "sp_expect_charge_separation",
            "sp_charge_separation_threshold"
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
            "spin_propensity_check",
            "spin_propensity_check_for_unimolecular_reaction",
            "spin_propensity_energy_range_to_save",
            "spin_propensity_optimize_all",
            "spin_propensity_energy_range_to_optimize",
            "store_full_mep",
            "store_all_structures",
            "n_surface_atom_threshold",
            "imaginary_wavenumber_threshold",
            "sp_expect_charge_separation",
            "sp_charge_separation_threshold"
        ]

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

    def _get_optimized_structure_ids(self, original_structures_ids: List[db.ID]):
        """
        Getter for the IDs of the re-optimized structures corresponding to the original structure ID list.
        Parameters
        ----------
        original_structures_ids : List[db.ID]
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

    def _set_up_calculation(self, job_label: str, calculation: Optional[db.Calculation],
                            targeted_step_id: Optional[db.ID] = None) -> bool:
        """
        Set up the refinement calculation or if needed geometry optimizations to be run before the refinement.
        Parameters
        ----------
        job_label : str
            The job label for the refinement.
        calculation : db.Calculation
            The original calculation that needs to be refined. Settings may be required from the calculation object.
        Returns
        -------
        bool
            Return True if the input calculation is fully handled. Returns False if the input calculation
            needs to be revisited by the gear.
        """
        if targeted_step_id:
            elementary_steps = [db.ElementaryStep(targeted_step_id, self._elementary_steps)]
        elif calculation is not None:
            elementary_steps = [db.ElementaryStep(step_id, self._elementary_steps) for step_id in
                                calculation.get_results().get_elementary_steps()]
        else:
            raise RuntimeError("Error: Calculation or elementary step ID must be supplied.")
        if calculation is None and job_label in ["refine_structures_and_irc", "double_ended_refinement",
                                                 "refine_single_ended_search"]:
            raise RuntimeError("Error: The old calculation is a necessary requirement for new react jobs."
                               "However, it was not found in the database.")

        transition_state_id = None
        rhs_lhs_structure_id_set: Set[str] = set()
        for elementary_step in elementary_steps:
            if not self.elementary_step_filter.filter(elementary_step):
                continue

            reactants = elementary_step.get_reactants(db.Side.BOTH)
            rhs_lhs_structure_id_set = rhs_lhs_structure_id_set.union(
                set([s_id.string() for s_id in reactants[0] + reactants[1]]))
            if elementary_step.has_transition_state():
                transition_state_id = elementary_step.get_transition_state()
                rhs_lhs_structure_id_set.add(transition_state_id.string())

        all_structure_ids_list = [db.ID(str_id) for str_id in rhs_lhs_structure_id_set]
        if job_label == "refine_structures_and_irc":
            if transition_state_id is not None:
                assert calculation and isinstance(calculation, db.Calculation)
                reactant_structure_ids = self._get_optimized_structure_ids(calculation.get_structures())
                if reactant_structure_ids and self._all_structures_have_graph(reactant_structure_ids):
                    self._refine_structures_and_irc(reactant_structure_ids, transition_state_id,
                                                    calculation.get_settings())
                else:
                    return False
        elif job_label == "refine_single_points":
            if self._all_structures_have_energy(all_structure_ids_list):
                return True
            self._refine_single_points(all_structure_ids_list)
        elif job_label == "refine_optimizations":
            self._refine_optimizations(all_structure_ids_list)
        elif job_label == "refine_single_ended_search":
            assert calculation and isinstance(calculation, db.Calculation)
            reactant_structure_ids = self._get_optimized_structure_ids(calculation.get_structures())
            if reactant_structure_ids \
               and self._all_structures_have_graph(reactant_structure_ids) \
               and self._all_structures_have_aggregate(reactant_structure_ids):
                self._refine_existing_react_jobs(
                    self.options.single_ended_job,
                    self.options.single_ended_job_settings,
                    self._rc_keys() +
                    self._single_ended_keys_to_take_over(),
                    reactant_structure_ids,
                    calculation.get_settings())
            else:
                return False
        elif job_label == "double_ended_refinement":
            for elementary_step in elementary_steps:
                if self.options.double_ended_job.order != "scine_bspline_optimization":
                    raise RuntimeError("Only the logic for the scine_bspline_optimization job is implemented!")
                if not elementary_step.has_reaction():
                    return False
                if elementary_step.get_type() != db.ElementaryStepType.REGULAR:
                    continue
                assert calculation
                self._refine_existing_react_jobs(self.options.double_ended_job,
                                                 self.options.double_ended_job_settings,
                                                 self._double_ended_keys_to_take_over(),
                                                 [], calculation.get_settings(), elementary_step)
        else:
            raise RuntimeError(f"The job label '{job_label}' is not resolved for elementary step refinement.")
        for elementary_step in elementary_steps:
            self.step_disabling.process(elementary_step, job_label)
        return True

    def _all_structures_have_graph(self, structure_ids: List[db.ID]):
        """
        Check if all structures in the list have a graph assigned.
        Parameters
        ----------
        structure_ids : List[db.ID]
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

    def _all_structures_have_energy(self, structure_id_list: List[db.ID]):
        """
        Check if all structures in the list have an electronic energy with the post-refinement model.
        Parameters
        ----------
        structure_ids : List[db.ID]
            The structure ID list.
        Returns
        -------
        bool
            True if all structures have an electronic energy with the post-refinement model. False otherwise.
        """
        for s_id in structure_id_list:
            s = db.Structure(s_id)
            energy = get_energy_for_structure(s, "electronic_energy", self.options.post_refine_model, self._structures,
                                              self._properties)
            if energy is None:
                return False
        return True

    def _all_structures_have_aggregate(self, structure_ids: List[db.ID]):
        """
        Check if all structures in the list have an aggregate assigned.
        Parameters
        ----------
        structure_ids : List[db.ID]
            The structure ID list.
        Returns
        -------
        bool
            True if all structures have an aggregate assigned. False otherwise.
        """
        for s_id in structure_ids:
            structure = db.Structure(s_id, self._structures)
            if not structure.has_aggregate():
                return False
        return True
