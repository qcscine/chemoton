#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, Dict

# Third party imports
import scine_database as db
import scine_utilities as utils

from .prepare_kinetic_modeling_job import KineticModelingJobFactory
from ...utilities.model_combinations import ModelCombination


class KinetxKineticModelingJobFactory(KineticModelingJobFactory):
    """
    A class that creates KiNetX kinetic modeling jobs.
    """

    def __init__(self, model_combinations: List[ModelCombination], model_combinations_reactions: List[ModelCombination],
                 manager: db.Manager,
                 only_electronic: bool = False) -> None:
        super().__init__(model_combinations=model_combinations,
                         model_combinations_reactions=model_combinations_reactions,
                         manager=manager, only_electronic=only_electronic)

    def create_kinetic_modeling_job(self, settings: utils.ValueCollection) -> bool:
        reactions, aggregates = self._setup_general_settings(settings)
        if aggregates is None or reactions is None:
            return False
        # Take the maximum rate found for an elementary step.
        all_rates = [r.get_ts_theory_rate_constants(self.reference_state) for r in reactions]
        lhs = [k[0] for k in all_rates]
        rhs = [k[1] for k in all_rates]
        assert None not in lhs
        assert None not in rhs
        settings["lhs_rates"] = lhs  # type: ignore
        settings["rhs_rates"] = rhs  # type: ignore

        all_structure_ids = [a.get_db_object().get_centroid() for a in aggregates.values()]
        if self._calc_already_set_up(all_structure_ids, settings):
            return False

        return self._finalize_calculation(settings, [a.get_db_object().get_centroid() for a in aggregates.values()])

    @staticmethod
    def get_job():
        return db.Job('kinetx_kinetic_modeling')

    @staticmethod
    def get_default_settings():
        return utils.ValueCollection({
            "time_step": 1e-8,
            "solver": "cash_karp_5",
            "batch_interval": int(1e+3),
            "n_batches": int(1e+5),
            "convergence": 1e-10,
            "concentration_label_postfix": ""
        })

    def _identical_model_definition(self, _: utils.ValueCollection, __: Dict):
        return True

    @staticmethod
    def order_dependent_setting_keys() -> List[str]:
        return []
