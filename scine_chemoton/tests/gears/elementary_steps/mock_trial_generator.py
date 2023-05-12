#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import Dict, List, Tuple, Optional

# Third party imports
import numpy as np
import scine_database as db
from scine_utilities import ValueCollection

# Local application imports
from ....gears.elementary_steps.trial_generator import TrialGenerator, _sanity_check_wrapper
from scine_chemoton.utilities.queries import get_calculation_id, calculation_exists_in_structure


class MockGenerator(TrialGenerator):
    """
    A mock trial generator that counts how often the `unimolecular_reactions`
    and `bimolecular_reactions` methods are called.

    Attributes
    ----------
    unimol_counter : int
        How often  `unimolecular_reactions` was called.
    bimol_counter : ReactiveSiteFilter
        How often  `bimolecular_reactions` was called.
    """
    class Options(TrialGenerator.Options):

        class MockOptions:
            def __init__(self) -> None:
                self.job = db.Job('fake')

        def __init__(self, parent: Optional[TrialGenerator] = None):
            super().__init__(parent)
            self.unimolecular_options = self.MockOptions()
            self.bimolecular_options = self.MockOptions()
            self.settings: Dict[str, str] = {}

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self.unimol_counter = 0
        self.bimol_counter = 0
        self._required_collections = ["calculations", "structures"]

    def clear_cache(self):
        pass

    def _quick_already_exists(self, structure_list: List[db.Structure], with_exact_settings_check: bool = False) \
            -> bool:
        """
            If there is a reactive complex calculation for the same structures, return True
        """
        if with_exact_settings_check:
            return False
        return calculation_exists_in_structure(
            self.options.unimolecular_options.job.order,
            [s.id() for s in structure_list],
            self.options.model,
            self._structures,
            self._calculations
        )

    @_sanity_check_wrapper
    def unimolecular_reactions(self, structure: db.Structure, with_exact_settings_check: bool = False) -> None:
        if self._quick_already_exists([structure], with_exact_settings_check):
            return
        if with_exact_settings_check and \
                get_calculation_id(self.options.unimolecular_options.job.order, [structure.id()], self.options.model,
                                   self._calculations, settings=self.options.settings) is not None:
            return
        self.unimol_counter += 1
        calculation = db.Calculation()
        calculation.link(self._calculations)
        calculation.create(self.options.model, self.options.unimolecular_options.job, [structure.id()])
        calculation.set_settings(ValueCollection(self.options.settings))
        calculation.set_status(db.Status.HOLD)
        structure.add_calculation(self.options.unimolecular_options.job.order, calculation.id())

    @_sanity_check_wrapper
    def bimolecular_reactions(self, structure_list: List[db.Structure], with_exact_settings_check: bool = False) \
            -> None:
        if self._quick_already_exists(structure_list, with_exact_settings_check):
            return
        if with_exact_settings_check and \
                get_calculation_id(self.options.bimolecular_options.job.order, [s.id() for s in structure_list],
                                   self.options.model, self._calculations, settings=self.options.settings) is not None:
            return
        self.bimol_counter += 1
        calculation = db.Calculation()
        calculation.link(self._calculations)
        calculation.create(self.options.model, self.options.bimolecular_options.job, [s.id() for s in structure_list])
        calculation.set_settings(ValueCollection(self.options.settings))
        calculation.set_status(db.Status.HOLD)
        for structure in structure_list:
            structure.add_calculation(self.options.bimolecular_options.job.order, calculation.id())

    @_sanity_check_wrapper
    def unimolecular_coordinates(self, structure: db.Structure, with_exact_settings_check: bool = False) \
            -> List[Tuple[List[List[Tuple[int, int]]], int]]:
        if self._quick_already_exists([structure], with_exact_settings_check):
            return []
        if with_exact_settings_check and \
                get_calculation_id(self.options.unimolecular_options.job.order, [structure.id()], self.options.model,
                                   self._calculations, settings=self.options.settings) is not None:
            return []
        return [([[(0, 1)]], 1)]

    @_sanity_check_wrapper
    def bimolecular_coordinates(self, structure_list: List[db.Structure], with_exact_settings_check: bool = False) \
            -> Dict[
        Tuple[List[Tuple[int, int]], int],
        List[Tuple[np.ndarray, np.ndarray, float, float]]
    ]:
        if self._quick_already_exists(structure_list, with_exact_settings_check):
            return {}
        if with_exact_settings_check and \
                get_calculation_id(self.options.bimolecular_options.job.order, [s.id() for s in structure_list],
                                   self.options.model, self._calculations, settings=self.options.settings) is not None:
            return {}
        return {
            (((0, 1), (0, 2)), 1): [  # type: ignore
                (np.zeros(3), np.zeros(3), 0.0, 0.0)
            ]
        }

    def get_unimolecular_job_order(self) -> str:
        return self.options.unimolecular_options.job.order

    def get_bimolecular_job_order(self) -> str:
        return self.options.bimolecular_options.job.order
