#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import scine_database as db
import scine_utilities as utils

from .prepare_kinetic_modeling_job import KineticModelingJobFactory


class RMSKineticModelingJobFactory(KineticModelingJobFactory):
    """
    A class that creates RMS kinetic modeling jobs.
    """

    def __init__(self, electronic_model: db.Model, hessian_model: db.Model, manager: db.Manager,
                 only_electronic: bool = False):
        super().__init__(electronic_model=electronic_model, hessian_model=hessian_model, manager=manager,
                         only_electronic=only_electronic)

    def create_kinetic_modeling_job(self, settings: utils.ValueCollection) -> bool:
        reactions, aggregates = self._setup_general_settings(settings)
        if aggregates is None or reactions is None:
            return False
        hartree_to_J_per_mol = utils.KJPERMOL_PER_HARTREE * 1e+3
        enthalpies = [a.get_enthalpy(self.reference_state) for a in aggregates.values()]
        entropies = [a.get_entropy(self.reference_state) for a in aggregates.values()]
        ea = [r.get_free_energy_of_activation(self.reference_state)[0] for r in reactions]
        assert None not in enthalpies
        assert None not in entropies
        assert None not in ea
        settings["enthalpies"] = [e * hartree_to_J_per_mol for e in enthalpies]  # type: ignore
        settings["entropies"] = [e * hartree_to_J_per_mol for e in entropies]  # type: ignore
        settings["ea"] = [e * hartree_to_J_per_mol for e in ea]  # type: ignore
        settings["arrhenius_prefactors"] = [r.get_arrhenius_prefactor(self.reference_state) for r in reactions]
        settings["arrhenius_temperature_exponents"] = [0 for _ in reactions]
        return self._finalize_calculation(settings, [a.get_db_object().get_centroid() for a in aggregates.values()])

    @staticmethod
    def get_job():
        return db.Job('rms_kinetic_modeling')

    @staticmethod
    def get_default_settings():
        return utils.ValueCollection({
            "solver": "CVODE_BDF",
            "phase_type": "ideal_gas",  # other: ideal_dilute_solution
            "max_time": 3600.0,  # in s
            "viscosity": "none",  # taken from tabulated values corresponding to the solvent in the model if not given
            "reactor_solvent": "none",  # taken from the electronic structure model if not given
            "diffusion_limited": False,  # Only viable for phase_type: ideal_dilute_solution. May destabilize ODE.
            "reactor_temperature": "none",  # taken from the the electronic structure model if not given
            "reactor_pressure": "none",  # taken from the the electronic structure model if not given
            "absolute_tolerance": 1e-20,  # 1e-20 should be very conservative. 1e-18 could be fine as well.
            "relative_tolerance": 1e-6,
            "solvent_aggregate_str_id": "none",  # allow the solvent concentration to change by providing the solvent id
            "solvent_concentration": 14.3,
        })
