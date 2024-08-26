#!/usr/bin/env python3
from __future__ import annotations
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""
from typing import List, Optional, Union, Dict

import scine_database as db

from . import SafeFirstSelection
from ..datastructures import SelectionResult, LogicCoupling
from scine_chemoton.filters.aggregate_filters import AggregateFilter, CatalystFilter
from scine_chemoton.filters.reactive_site_filters import ReactiveSiteFilter, CentralSiteFilter
from scine_chemoton.filters.further_exploration_filters import \
    FurtherExplorationFilterAndArray


class CentralMetalSelection(SafeFirstSelection):

    class Options(SafeFirstSelection.Options):
        metal: str
        ligand_without_metal_reactive: bool
        additional_catalyst_elements: Optional[Dict[str, int]]

        def __init__(self, model: db.Model, metal: str, ligand_without_metal_reactive: bool,
                     additional_catalyst_elements: Optional[Dict[str, int]], *args, **kwargs):
            super().__init__(model, *args, **kwargs)
            self.metal = metal
            self.ligand_without_metal_reactive = ligand_without_metal_reactive
            self.additional_catalyst_elements = additional_catalyst_elements

    options: CentralMetalSelection.Options  # required for mypy checks, so it knows which options object to check

    def __init__(self, model: db.Model,  # pylint: disable=keyword-arg-before-vararg
                 central_metal_species: str,
                 ligand_without_metal_reactive: bool,
                 additional_catalyst_elements: Optional[Dict[str, int]] = None,
                 additional_aggregate_filters: Optional[List[AggregateFilter]] = None,
                 additional_reactive_site_filters: Optional[List[ReactiveSiteFilter]] = None,
                 logic_coupling: Union[str, LogicCoupling] = LogicCoupling.AND,
                 *args, **kwargs
                 ):
        super().__init__(model, additional_aggregate_filters, additional_reactive_site_filters, logic_coupling,
                         central_metal_species, ligand_without_metal_reactive, additional_catalyst_elements,
                         *args, **kwargs)
        self.options = self.Options(model, central_metal_species, ligand_without_metal_reactive,
                                    additional_catalyst_elements, *args, **kwargs)

    def _select(self) -> SelectionResult:
        central_site = CentralSiteFilter(
            self.options.metal,
            ligand_without_central_atom_reactive=self.options.ligand_without_metal_reactive
        )
        add_elements = {} if self.options.additional_catalyst_elements is None \
            else self.options.additional_catalyst_elements
        return SelectionResult(
            aggregate_filter=CatalystFilter({self.options.metal: 1, **add_elements},
                                            restrict_unimolecular_to_catalyst=True),
            reactive_site_filter=central_site,
            further_exploration_filter=FurtherExplorationFilterAndArray([central_site])
        )
