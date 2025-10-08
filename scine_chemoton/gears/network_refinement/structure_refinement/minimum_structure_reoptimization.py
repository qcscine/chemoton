#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""
# Standard library imports
from json import dumps
from typing import Set

# Third party imports
import scine_utilities as utils
import scine_database as db
from scine_database.queries import (
    stop_on_timeout,
    calculation_exists_in_structure,
)

# Local application imports
from scine_chemoton.gears import Gear
from scine_chemoton.utilities.calculation_creation_helpers import finalize_calculation
from scine_chemoton.default_settings import default_opt_settings
from scine_chemoton.utilities.place_holder_model import (
    ModelNotSetError,
    construct_place_holder_model,
    PlaceHolderModelType
)


class MinimumStructureReoptimization(Gear):
    """
    A class representing the MinimumStructureReoptimization gear.
    This is a simpler version of the AggregateBasedRefinement gear with fewer options and straightforward logic.

    This gear is responsible for reoptimizing the minimum energy structures.
    It provides options for selecting the refinement model, reoptimization job, and various job settings.
    The reoptimization process involves iterating over the compounds and flasks in the reaction network and
    reoptimizing the minimum energy structures using the specified model and job settings.
    """

    class Options(Gear.Options):
        __slots__ = (
            "refine_model",
            "reoptimization_job",
            "reoptimization_job_settings",
            "reoptimize_flasks",
            "reoptimize_compounds",
            "reoptimize_centroid_only",
        )

        def __init__(self) -> None:
            super().__init__()
            self.refine_model: db.Model = construct_place_holder_model()
            self.reoptimization_job = db.Job("scine_geometry_optimization")
            self.reoptimization_job_settings = utils.ValueCollection()
            self.reoptimize_flasks: bool = True
            self.reoptimize_compounds: bool = True
            self.reoptimize_centroid_only: bool = False

    def __init__(self) -> None:
        super().__init__()
        self._required_collections = ["compounds", "flasks", "properties", "structures", "calculations"]  # only needed
        self._structure_cache: Set[str] = set()

    options: Options

    def clear_cache(self) -> None:
        self._structure_cache.clear()

    def _loop_impl(self) -> None:
        if isinstance(self.options.model, PlaceHolderModelType) or \
           isinstance(self.options.refine_model, PlaceHolderModelType):
            raise ModelNotSetError("Either model or refine_model is not specified.")
        if self.options.model == self.options.refine_model:
            raise RuntimeError("Model and refine_model must be different!")
        if self.options.reoptimize_flasks:
            self._flask_loop()
        if self.options.reoptimize_compounds:
            self._compound_loop()

    def _flask_loop(self) -> None:
        selection = {'analysis_disabled': False}
        for flask in stop_on_timeout(self._flasks.iterate_flasks(dumps(selection))):
            if self.have_to_stop_at_next_break_point():
                return
            flask.link(self._flasks)
            structures = flask.get_structures()
            for struc in structures:
                structure = db.Structure(struc, self._structures)
                if structure.id().string() in self._structure_cache:
                    continue
                self._reoptimize_structure(structure)

    def _compound_loop(self) -> None:
        selection = {'analysis_disabled': False}
        for compound in stop_on_timeout(self._compounds.iterate_compounds(dumps(selection))):
            if self.have_to_stop_at_next_break_point():
                return
            compound.link(self._compounds)
            if self.options.reoptimize_centroid_only:
                structure = db.Structure(compound.get_centroid(), self._structures)
                if structure.id().string() in self._structure_cache:
                    continue
                self._reoptimize_structure(structure)
            else:
                structures = compound.get_structures()
                for struc in structures:
                    structure = db.Structure(struc, self._structures)
                    if structure.id().string() in self._structure_cache:
                        continue
                    self._reoptimize_structure(structure)

    def _reoptimize_structure(self, structure: db.Structure) -> None:
        self._structure_cache.add(structure.id().string())
        calc_settings = {k: v for k, v in default_opt_settings().as_dict().items()}
        for key, value in self.options.reoptimization_job_settings.as_dict().items():
            calc_settings[key] = value
        if calculation_exists_in_structure(self.options.reoptimization_job.order, [structure.id()],
                                           self.options.refine_model,
                                           self._structures, self._calculations,
                                           calc_settings):
            return
        calc = db.Calculation.make(
            self.options.refine_model, self.options.reoptimization_job,
            [structure.id()], self._calculations)
        calc.set_settings(utils.ValueCollection(calc_settings))
        finalize_calculation(calc, self._structures, [structure.id()])
