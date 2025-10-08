#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import copy
from json import dumps
from typing import List, Set

# Third party imports
import scine_utilities as utils
import scine_database as db
from scine_database.queries import (
    model_query,
    stop_on_timeout,
    calculation_exists_in_structure,
)
from scine_database.energy_query_functions import get_elementary_step_with_min_ts_energy, get_energy_for_structure

# Local application imports
from scine_chemoton.default_settings import default_ts_irc_ircopt_settings
from scine_chemoton.gears import Gear
from scine_chemoton.filters.structure_filters import StructureFilter
from scine_chemoton.utilities.place_holder_model import (
    ModelNotSetError,
    construct_place_holder_model,
    PlaceHolderModelType
)
from scine_chemoton.utilities.calculation_creation_helpers import finalize_calculation


class TSRefinement(Gear):
    """
    A class representing the TSRefinement gear.

    This gear is responsible for refining transition state (TS) structures in a chemical reaction network.
    It provides options for selecting the refinement model, structure model, and various job settings.
    The refinement process involves iterating over the elementary steps in the reaction network and refining
    the TS structures within a specified energy window.
    """

    class Options(Gear.Options):
        """
        The options for the TSRefinement Gear.
        """

        __slots__ = (
            "refine_model",
            "structure_model",
            "lowest_per_reaction",
            "ts_job",
            "dissociation_job_rerun",
            "bspline_job_rerun",
            "ts_job_settings",
            "ts_label",
            "ts_energy_window",
            "use_reactive_atoms"
        )

        def __init__(self) -> None:
            super().__init__()
            self.refine_model: db.Model = construct_place_holder_model()
            self.structure_model: db.Model = construct_place_holder_model()
            self.lowest_per_reaction: bool = False
            self.ts_label = db.Label.TS_OPTIMIZED.name.lower()
            self.ts_energy_window: float = 5.0  # in kJ/mol
            self.use_reactive_atoms: bool = False
            self.ts_job = db.Job("scine_react_ts_guess")
            self.ts_job_settings = default_ts_irc_ircopt_settings()
            self.dissociation_job_rerun = (db.Job("scine_dissociation_cut"),
                                           db.Job("scine_dissociation_cut_with_optimization"))
            self.bspline_job_rerun = db.Job("scine_bspline_optimization")

    options: Options

    def __init__(self) -> None:
        super().__init__()
        self._required_collections = ["calculations", "compounds", "elementary_steps", "flasks", "reactions",
                                      "properties", "structures"]
        self._ts_cache: Set[str] = set()
        self._barrierless_step_cache: Set[str] = set()
        self.structure_filter: StructureFilter = StructureFilter()

    def _propagate_db_manager(self, manager: db.Manager) -> None:
        self._sanity_check_configuration()
        self.structure_filter.initialize_collections(manager)

    def _sanity_check_configuration(self) -> None:
        if not isinstance(self.structure_filter, StructureFilter):
            raise TypeError(f"Expected a StructureFilter (or a class derived "
                            f"from it) in {self.name}.options.structure_filter.")

    def clear_cache(self) -> None:
        self._ts_cache.clear()
        self._barrierless_step_cache.clear()

    def _loop_impl(self) -> None:
        if isinstance(self.options.model, PlaceHolderModelType) or \
           isinstance(self.options.refine_model, PlaceHolderModelType):
            raise ModelNotSetError("Either model or refine_model is not specified.")
        if self.options.model == self.options.refine_model:
            raise RuntimeError("Model and refine_model must be different!")
        if isinstance(self.options.structure_model, PlaceHolderModelType):
            self.options.structure_model = copy.deepcopy(self.options.model)
        if self.options.lowest_per_reaction:
            self._reaction_loop()
        else:
            self._ts_loop()
        self._barrierless_loop()

    def _barrierless_loop(self) -> None:
        # loop steps for barrierless
        # if both sides are only compounds (strong code logic coupling)
        # the step comes from either dissociation cut or bspline
        # search for calculation
        # bspline --> copy job
        # dissociation cut --> new job with pre-optimization
        # rest ignore because it should be only complexation steps
        selection = {'type': 'barrierless'}
        for step in stop_on_timeout(self._elementary_steps.iterate_elementary_steps(dumps(selection))):
            step.link(self._elementary_steps)
            if self.have_to_stop_at_next_break_point():
                return
            if step.id().string() in self._barrierless_step_cache:
                continue
            lhs, rhs = step.get_reactants()
            structures = [db.Structure(s, self._structures) for s in lhs + rhs]
            if not all(structure.has_aggregate() for structure in structures):
                continue
            # Filter Check
            if not all(self.structure_filter.filter(structure) for structure in structures):
                continue
            compounds = [db.Compound(s.get_aggregate(), self._compounds) for s in structures]
            if not all(compound.exists() for compound in compounds):
                # some reactants are flasks --> should only come from complexation --> should be found again
                # with regular job, nothing to do here
                self._barrierless_step_cache.add(step.id().string())
                continue
            if calculation_exists_in_structure(self.options.bspline_job_rerun.order,
                                               lhs + rhs, self.options.refine_model,
                                               self._structures, self._calculations):
                self._barrierless_step_cache.add(step.id().string())
                continue
            # all reactants are compounds, find calculation that created step
            calculation_selection = {"$and": [
                {"status": "complete"},
                {"results.elementary_steps": {"$in": [
                    {"$oid": step.id().string()}
                ]}},
            ]}
            calculation = self._calculations.get_one_calculation(dumps(calculation_selection))
            if calculation is None:
                raise RuntimeError(f"Could not find calculation that created barrierless step {step.id()}")
            calculation.link(self._calculations)
            if calculation.get_job().order == self.options.bspline_job_rerun.order:
                self._copy_calculation(calculation, self.options.bspline_job_rerun, lhs + rhs)
            elif calculation.get_job().order == self.options.dissociation_job_rerun[0].order:
                if len(lhs) > 1 and len(rhs) > 1:
                    raise RuntimeError(f"Elementary step {step.id()} has more than one reactant and product and was "
                                       f"created by the calculation {calculation.id()} with the job "
                                       f"{calculation.get_job().order}. Multiple reactant dissociations are not "
                                       f"supported by the {self.name}")
                elif len(lhs) == 1:
                    self._copy_calculation(calculation, self.options.dissociation_job_rerun[1], lhs)
                else:
                    self._copy_calculation(calculation, self.options.dissociation_job_rerun[1], rhs)
            else:
                raise RuntimeError(f"Elementary step {step.id()} has only compound reactants and was created by "
                                   f"the calculation {calculation.id()} with the job {calculation.get_job().order} "
                                   f"which is not supported by the {self.name}")
            self._barrierless_step_cache.add(step.id().string())

    def _reaction_loop(self) -> None:
        selection = {'exploration_disabled': False}
        for reaction in stop_on_timeout(self._reactions.iterate_reactions(dumps(selection))):
            if self.have_to_stop_at_next_break_point():
                return
            reaction.link(self._reactions)
            min_step = get_elementary_step_with_min_ts_energy(reaction, 'electronic_energy', self.options.model,
                                                              self._elementary_steps, self._structures,
                                                              self._properties,
                                                              structure_model=self.options.structure_model)
            if min_step is None or not min_step.has_transition_state():
                continue
            min_ts = db.Structure(min_step.get_transition_state(), self._structures)
            # Checks
            ts_id = min_ts.id().string()
            if ts_id in self._ts_cache:
                continue
            if min_ts.get_model() != self.options.structure_model or\
               not self.structure_filter.filter(min_ts):
                self._ts_cache.add(ts_id)
                continue
            self._refine_ts(min_ts)

            # List of TS within window for this reaction
            min_ts_energy = get_energy_for_structure(
                min_ts, 'electronic_energy', self.options.model, self._structures, self._properties)

            # Loop over all other steps
            for tmp_step_id in reaction.get_elementary_steps():
                # Skip the minimum step
                if tmp_step_id == min_step.id():
                    continue

                tmp_step = db.ElementaryStep(tmp_step_id, self._elementary_steps)
                # Check if step has TS
                if not tmp_step.has_transition_state():
                    continue

                tmp_ts = db.Structure(tmp_step.get_transition_state(), self._structures)
                # Check TS structure
                tmp_ts_id = tmp_ts.id().string()
                if tmp_ts_id in self._ts_cache:
                    continue
                if tmp_ts.get_model() != self.options.structure_model or\
                   not self.structure_filter.filter(tmp_ts):
                    self._ts_cache.add(tmp_ts_id)
                    continue
                tmp_ts_energy = get_energy_for_structure(
                    tmp_ts, 'electronic_energy', self.options.model, self._structures, self._properties)
                if tmp_ts_energy is None or min_ts_energy is None:
                    continue

                # Check if TS is within window
                if (tmp_ts_energy - min_ts_energy) - self.options.ts_energy_window * utils.HARTREE_PER_KJPERMOL\
                   > 1e-12:
                    continue
                else:
                    self._refine_ts(tmp_ts)

    def _ts_loop(self) -> None:
        selection = {"$and": [
            {'exploration_disabled': False},
            {'label': self.options.ts_label},
        ] + model_query(self.options.structure_model)
        }
        for ts in stop_on_timeout(self._structures.iterate_structures(dumps(selection))):
            if self.have_to_stop_at_next_break_point():
                return
            ts.link(self._structures)
            # Checks
            ts_id = ts.id().string()
            if ts_id in self._ts_cache:
                continue
            if not self.structure_filter.filter(ts):
                self._ts_cache.add(ts_id)
                continue
            self._refine_ts(ts)

    def _refine_ts(self, ts: db.Structure) -> None:
        # # # Set default ts_job_settings
        calc_settings = default_ts_irc_ircopt_settings().as_dict()
        # # # Check for reactive atoms
        if self.options.use_reactive_atoms:
            if ts.has_property("reactive_atoms"):
                reactive_atoms = db.VectorProperty(ts.get_property("reactive_atoms"), self._properties)
                calc_settings["tsopt_automatic_mode_selection"] = [int(i) for i in reactive_atoms.get_data()]
            else:
                return
        # # # Add to cache after successful checks
        self._ts_cache.add(ts.id().string())
        # # # Overwrite with settings given in options
        for key, value in self.options.ts_job_settings.as_dict().items():
            calc_settings[key] = value
        # Check, if calculation already exists
        if calculation_exists_in_structure(self.options.ts_job.order, [ts.id()], self.options.refine_model,
                                           self._structures, self._calculations,
                                           calc_settings):
            return
        calc = db.Calculation.make(self.options.refine_model, self.options.ts_job, [ts.id()], self._calculations)
        calc.set_settings(utils.ValueCollection(calc_settings))
        finalize_calculation(calc, self._structures, [ts.id()])

    def _copy_calculation(self, original_calculation: db.Calculation, new_job: db.Job,
                          input_structures: List[db.ID]) -> None:
        new_calculation = db.Calculation.make(self.options.refine_model, new_job, input_structures,
                                              self._calculations)
        new_calculation.set_settings(original_calculation.get_settings())
        new_calculation.set_auxiliaries(original_calculation.get_auxiliaries())
        finalize_calculation(new_calculation, self._structures, input_structures)
