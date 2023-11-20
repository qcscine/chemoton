#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Dict, Union, Optional

import scine_database as db
from scine_database.concentration_query_functions import query_concentration_with_object

from .ensemble_wrapper import Ensemble
from .thermodynamic_properties import ReferenceState


class AggregateCache:
    """
    A cache for aggregate objects.
    """

    def __init__(self, manager: db.Manager, electronic_model: db.Model, hessian_model: db.Model,
                 only_electronic: bool = False):
        self._manager = manager
        self._electronic_model = electronic_model
        self._hessian_model = hessian_model
        self._only_electronic = only_electronic
        self._aggregates: Dict[int, Aggregate] = {}

    def get_or_produce(self, aggregate_id: db.ID):
        int_id = int(aggregate_id.string(), 16)
        if int_id not in self._aggregates:
            self._aggregates[int_id] = Aggregate(aggregate_id, self._manager, self._electronic_model,
                                                 self._hessian_model, self._only_electronic)
        return self._aggregates[int_id]


class Aggregate(Ensemble):
    """
    A class that wraps around db.Compound and db.Flask to provide easy access to all thermodynamic contributions.
    Free energies are calculated with the harmonic oscillator, rigid rotor, particle in a box model according to the
    given thermodynamic reference state (allows caching or works on the fly if necessary).
    """

    def __init__(self, aggregate_id: db.ID, manager: db.Manager, electronic_model: db.Model, hessian_model: db.Model,
                 only_electronic: bool = False):
        """
        Parameters
        ----------
        aggregate_id : db.ID
            The database ID.
        manager
            The database manager.
        electronic_model
            The electronic structure model from which the electronic energy is taken.
        hessian_model
            The electronic structure model with which the geometrical hessian was calculated.
        only_electronic
            If true, only the electronic energies are used to determine the thermodynamics.
        """
        self._is_flask = False
        super().__init__(aggregate_id, manager, electronic_model, hessian_model, only_electronic)
        self._start_concentration_property_name = "start_concentration"
        self._concentration_flux_property_name = "concentration_flux"

    def _initialize_db_object(self, manager: db.Manager) -> Union[db.Compound, db.Flask]:
        compounds = manager.get_collection("compounds")
        db_object = db.Compound(self._db_id, compounds)  # type: ignore
        if not db_object.exists():
            flasks = manager.get_collection("flasks")
            db_object = db.Flask(self._db_id, flasks)  # type: ignore
            self._is_flask = True
        return db_object

    def _update_thermodynamics(self):
        structure_ids = self.get_db_object().get_structures()
        for s_id in structure_ids:
            _ = self._structure_thermodynamics.get_or_produce(s_id)

    def get_free_energy(self, reference_state: ReferenceState) -> Optional[float]:
        """
        Getter for the aggregate's free energy.
        """
        n_structures = len(self.get_db_object().get_structures())
        if self._structure_thermodynamics.minimum_values_need_update(reference_state, n_structures):
            self._update_thermodynamics()
        return self._structure_thermodynamics.get_ensemble_gibbs_free_energy(reference_state, n_structures)

    def get_enthalpy(self, reference_state: ReferenceState) -> Optional[float]:
        """
        Getter for the aggregate's enthalpy.
        """
        self.get_free_energy(reference_state)  # update if necessary
        return self._structure_thermodynamics.get_ensemble_enthalpy(reference_state)

    def get_entropy(self, reference_state: ReferenceState) -> Optional[float]:
        """
        Getter for the aggregate's entropy.
        """
        self.get_free_energy(reference_state)  # update if necessary
        return self._structure_thermodynamics.get_ensemble_entropy(reference_state)

    def get_starting_concentration(self) -> float:
        """
        Getter for the aggregates starting concentration.
        """
        return query_concentration_with_object(self._start_concentration_property_name, self.get_db_object(),
                                               self._properties, self._structures)

    def get_concentration_flux(self) -> float:
        """
        Getter for the aggregates vertex flux.
        """
        return query_concentration_with_object(self._concentration_flux_property_name, self.get_db_object(),
                                               self._properties, self._structures)

    def get_aggregate_type(self) -> db.CompoundOrFlask:
        """
        Getter for the aggregate type.
        """
        return db.CompoundOrFlask.FLASK if self._is_flask else db.CompoundOrFlask.COMPOUND
