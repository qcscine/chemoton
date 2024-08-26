#!/usr/bin/env python3
from __future__ import annotations
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, Optional

import scine_database as db
from scine_database.queries import optimized_labels_enums

from . import NetworkExpansion, thermochemistry_job_wrapper
from ..datastructures import ProtocolEntry, NetworkExpansionResult
from scine_chemoton.default_settings import default_opt_settings
from scine_chemoton.gears.conformers.brute_force import BruteForceConformers


class ConformerCreation(NetworkExpansion):
    """
    Network expansion that adds conformers to the given selection result.
    """

    _conformer_gear: Optional[BruteForceConformers] = None  # so we can address the gear for results parsing
    options: ConformerCreation.Options

    @thermochemistry_job_wrapper
    def _relevant_puffin_jobs(self) -> List[str]:
        return ['conformers', 'scine_geometry_optimization'] + self._aggregation_necessary_jobs()

    def _set_protocol(self, credentials: db.Credentials) -> None:
        self.protocol.append(ProtocolEntry(credentials, self._prepare_scheduler()))
        self._conformer_gear = BruteForceConformers()
        self._conformer_gear.options.minimization_settings = default_opt_settings()
        self.protocol.append(ProtocolEntry(credentials, self._conformer_gear, wait_for_calculation_finish=True))
        # run conformer gear double to avoid race conditions
        self._extra_manual_cycles_to_avoid_race_condition(
            credentials, aggregate_reactions=False, additional_entries=[
                ProtocolEntry(credentials, self._conformer_gear, fork=False, n_runs=2, wait_for_calculation_finish=True)
            ])
        self._extra_manual_cycles_to_avoid_race_condition(credentials, aggregate_reactions=False)

    def _execute(self, n_already_executed_protocol_steps: int) -> NetworkExpansionResult:
        result = NetworkExpansionResult()
        if self._conformer_gear is None:
            raise RuntimeError(f"Error in {self.name}, conformer gear was not saved in class")
        valid_compounds = self._conformer_gear.valid_compounds()
        self._basic_execute(n_already_executed_protocol_steps)
        result.compounds = valid_compounds

        desired_labels = optimized_labels_enums()
        for compound_id in valid_compounds:
            compound = db.Compound(compound_id, self._compounds)
            for sid in compound.get_structures():
                structure = db.Structure(sid, self._structures)
                if structure.get_label() not in desired_labels:
                    continue
                if not structure.has_graph("masm_cbor_graph"):
                    raise RuntimeError(f"{self.name} failed, a optimized structure '{str(sid)}' is missing a graph.")
                result.structures.append(sid)

        return result
