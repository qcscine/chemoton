#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import scine_database as db
import scine_utilities as utils
from scine_database.queries import stop_on_timeout, calculation_exists_in_structure

from typing import List, Union
from json import dumps

# Local application imports
from . import Gear
from scine_chemoton.filters.structure_filters import StructureFilter
from scine_chemoton.utilities.calculation_creation_helpers import finalize_calculation


class SinglePoint(Gear):
    class Options(Gear.Options):
        """
        The options for the SinglePoint Gear.

        This gear writes calculation entries to the database for every structure fitting the given label list and
        filter settings.
        Application example: Run single point calculations for gradients and forces for structures during
        training of a machine learning model.

        Notes
        -----
        This gear per default does not consider the model of the structures and setups calculations with its model.
        If you want to consider only certain structures, you can give it a StructureFilter.
        """
        __slots__ = ("allowed_labels", "job", "job_settings")

        def __init__(self) -> None:
            super().__init__()
            self.allowed_labels: List[Union[str, db.Label]] = []
            """
            allowed_labels : List[Union[str, db.Label]]
                Only structures with a label in the given list are considered by this gear. If none are provided,
                structures are considered independently of their label.
            """
            self.job: db.Job = db.Job("scine_single_point")
            """
            job : db.Job
                The calculation job.
            """
            self.job_settings: utils.ValueCollection = utils.ValueCollection()
            """
            job_settings : utils.ValueCollection
                The calculation settings.
            """

    options: Options

    def __init__(self) -> None:
        super().__init__()
        self._required_collections = ["structures", "calculations"]
        self.structure_filter: StructureFilter = StructureFilter()

    def _propagate_db_manager(self, manager: db.Manager):
        self.structure_filter.initialize_collections(manager)

    def _sanity_check_configuration(self):
        if not isinstance(self.structure_filter, StructureFilter):
            raise TypeError(f"Expected a StructureFilter (or a class derived "
                            f"from it) in {self.name}.options.structure_filter.")

    def __get_query_selection(self):
        """
        Returns a dictionary that is used to query the database for structures to be considered by this gear.
        """
        selection = {}
        if self.options.allowed_labels:
            label_strs = [label.name.lower() if isinstance(label, db.Label) else label.lower()
                          for label in self.options.allowed_labels]
            selection["label"] = {"$in": label_strs}
        return selection

    def __calculation_already_set_up(self, structure: db.Structure) -> bool:
        """
        Checks whether a calculation for the given structure already exists.
        Relies on an entry of the setup calculations in the structure document.
        """
        return calculation_exists_in_structure(self.options.job.order, [structure.id()], self.options.model,
                                               self._structures, self._calculations,
                                               self.options.job_settings.as_dict())

    def __set_up_calculation(self, structure: db.Structure):
        structure_id = structure.id()
        calc = db.Calculation()
        calc.link(self._calculations)
        calc.create(self.options.model, self.options.job, [structure_id])
        calc.set_settings(self.options.job_settings)
        finalize_calculation(calc, self._structures, [structure_id])

    def _loop_impl(self):
        selection = self.__get_query_selection()
        for structure in stop_on_timeout(self._structures.iterate_structures(dumps(selection))):
            if self.have_to_stop_at_next_break_point():
                return
            structure.link(self._structures)
            if not self.structure_filter.filter(structure):
                continue
            if self.__calculation_already_set_up(structure):
                continue
            self.__set_up_calculation(structure)
